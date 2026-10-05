"""LoRA training on whole conversation threads, learning ONLY from replies marked train=True.

Everything else in a thread (the human's messages, switched-off assistant replies,
the system prompt) is context: it shapes the reply but carries no loss.

Designed for one DGX Spark (BF16, plain LoRA, no quantization) or a pod GPU.

GLM run :
  python train_lora.py --preset glm \
      --model /workspace/glm53-fp8 --tokenizer /workspace/glm_tokenizer \
      --chat-template /workspace/glm_tokenizer/chat_template.jinja \
      --train data/train.jsonl --heldout data/heldout.jsonl \
      --system system_prompt.txt --out /workspace/runs/glm
Add --dry-run to tokenize and report only (no GPU needed).

Input rows: {"id", "tools"?: [...], "messages": [
    {"role": "user"|"assistant"|"tool", "content", "reasoning_content"?, "tool_calls"?,
     "tool_call_id"?, "train"?}]}

FORMAT GUARANTEE: for every learned assistant turn, the text before it must be
exactly what the server renders with add_generation_prompt=True (same template,
same kwargs, same tools). If not, the turn is skipped and counted - never guessed.
The learned span = everything the model generates: thinking, reply or tool call,
plus the stop marker that ends the turn.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # glm_common next to this file

from transformers import AutoTokenizer

IGNORE = -100
# Turn terminators the server stops on. GLM-5.3 has no end-of-turn token: a reply
# ends where the next role begins. Qwen ends turns with <|im_end|> (already inside).
STOP_MARKERS = ("<|user|>", "<|observation|>", "<|endoftext|>")


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def render(tok, msgs, tools=None, **kw):
    return tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, **kw)


def clean_msg(m):
    out = {"role": m["role"], "content": m.get("content", "")}
    for k in ("reasoning_content", "tool_calls", "tool_call_id", "name"):
        if m.get(k):
            out[k] = m[k]
    return out


def encode_thread(tok, row, system, max_len, stats, tkw, require_thinking=False):
    system = row.get("system") or system          # rows may carry their own system prompt
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        clean_msg(m) for m in row["messages"]]
    offset = 1 if system else 0
    tools = row.get("tools")
    full = render(tok, msgs, tools, **tkw)
    # a thread that ends on a learned turn still needs that turn's terminator
    end_marker = next((s for s in STOP_MARKERS if s in tok.get_vocab()), "")
    if end_marker and not full.endswith(STOP_MARKERS) and "<|im_end|>" not in full[-20:]:
        full += "<|user|>" if "<|user|>" in tok.get_vocab() else end_marker

    spans = []
    for k, m in enumerate(row["messages"]):
        if m["role"] != "assistant":
            continue
        stats["replies"] += 1
        if not m.get("train", True):
            stats["context_only"] += 1
            continue
        reply_only = m.get("learn") == "reply_only"      # June: learn his words, not the misfired thinking
        if require_thinking and not reply_only and not m.get("reasoning_content"):
            stats["no_thinking_context_only"] += 1   # would teach an empty <think></think>
            continue
        i = k + offset
        gen = render(tok, msgs[:i], tools, add_generation_prompt=True, **tkw)
        upto = render(tok, msgs[: i + 1], tools, **tkw)
        if not (upto.startswith(gen) and full.startswith(upto) and len(upto) > len(gen)):
            stats["template_mismatch"] += 1  # training != serving format: skip, never guess
            continue
        start = len(gen)
        if reply_only:
            close = upto.find("</think>", len(gen))
            if close < 0:
                stats["template_mismatch"] += 1
                continue
            start = close + len("</think>")          # thinking stays in the input, carries no loss
            stats["reply_only_turns"] += 1
        end = len(upto)
        rest = full[end:]
        for s in STOP_MARKERS:              # learn to stop: include the next role marker
            if rest.startswith(s):
                end += len(s)
                break
        spans.append((start, end))
        stats["tool_turns"] += bool(m.get("tool_calls"))

    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False,
              truncation=True, max_length=max_len)
    labels = []
    for (a, b), tid in zip(enc["offset_mapping"], enc["input_ids"]):
        learn = any(s <= a < e for s, e in spans)
        labels.append(tid if learn else IGNORE)
    n_learn = sum(1 for l in labels if l != IGNORE)
    stats["learned_tokens"] += n_learn
    stats["total_tokens"] += len(labels)
    if len(tok(full, add_special_tokens=False)["input_ids"]) > max_len:
        stats["truncated_threads"] += 1
    return {"input_ids": enc["input_ids"], "labels": labels} if n_learn else None


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id)
        lab = torch.full((len(batch), n), IGNORE)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def build(tok, path, system, max_len, name, tkw, require_thinking=False):
    stats = {k: 0 for k in ("replies", "context_only", "no_thinking_context_only", "reply_only_turns",
                            "template_mismatch", "tool_turns", "learned_tokens", "total_tokens", "truncated_threads")}
    data = [e for r in load_rows(path)
            if (e := encode_thread(tok, r, system, max_len, stats, tkw, require_thinking))]
    print(f"[{name}] threads {len(data)} | " + " | ".join(f"{k} {v}" for k, v in stats.items()))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=["glm", "qwen-legacy"],
                    help="glm = the pinned GLM-5.3 recipe ; every setting is fixed, nothing defaults "
                         "silently. qwen-legacy = an older Qwen preset, explicit flags only.")
    ap.add_argument("--model", required=True, help="model dir (glm: the orcarouter FP8 checkout)")
    ap.add_argument("--tokenizer", help="glm: REQUIRED, the pinned zai-org tokenizer dir")
    ap.add_argument("--chat-template", help="glm: REQUIRED, official template (md5-checked)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout")
    ap.add_argument("--system", required=True, help="system prompt used at serving time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tokenize + report, no model load")
    # qwen-legacy only (ignored / refused with --preset glm)
    ap.add_argument("--targets")
    ap.add_argument("--template-kwargs")
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--grad-accum", type=int)
    args = ap.parse_args()

    if args.preset == "glm":
        import glm_common as G
        if not (args.tokenizer and args.chat_template):
            raise SystemExit("--preset glm needs --tokenizer and --chat-template")
        if any(v is not None for v in (args.targets, args.template_kwargs, args.max_len, args.grad_accum)):
            raise SystemExit("--preset glm fixes targets/template-kwargs/max-len/grad-accum; don't pass them")
        tok = G.load_tokenizer(args.tokenizer, args.chat_template)
        tkw, max_len, grad_accum = dict(G.TEMPLATE_KWARGS), G.MAX_LEN, G.GRAD_ACCUM
        require_thinking = True
        print(f"[preset glm] template md5 OK | kwargs {tkw} | max_len {max_len} | grad_accum {grad_accum} "
              f"| lr {G.LR} | epochs {G.EPOCHS} | r {G.RANK} alpha {G.ALPHA} | seed {G.SEED}")
    else:
        if not (args.targets and args.max_len and args.grad_accum):
            raise SystemExit("--preset qwen-legacy needs explicit --targets, --max-len, --grad-accum")
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        if args.chat_template:
            tok.chat_template = Path(args.chat_template).read_text(encoding="utf-8")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tk = args.template_kwargs or "{}"
        tkw = json.loads(Path(tk).read_text(encoding="utf-8") if tk.endswith(".json") and Path(tk).exists() else tk)
        max_len, grad_accum, require_thinking = args.max_len, args.grad_accum, False

    system = Path(args.system).read_text(encoding="utf-8")
    train = build(tok, args.train, system, max_len, "train", tkw, require_thinking)
    held = build(tok, args.heldout, system, max_len, "heldout", tkw, require_thinking) if args.heldout else None
    if args.dry_run:
        return

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    if args.preset == "glm":
        set_seed(G.SEED)
        model = G.attach_lora(G.load_model(args.model))     # fresh adapter, asserts 180 modules / 35.3M params
        lr, epochs, guard = G.LR, G.EPOCHS, G.attention_guard
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                 target_modules=args.targets, task_type="CAUSAL_LM"))
        lr, epochs, guard = 1e-4, 2, __import__("contextlib").nullcontext
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,     # eval default is 8 -> OOM risk
        gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0, bf16=True, logging_steps=1,
        eval_strategy="epoch" if held else "no",
        save_strategy="epoch", save_only_model=True,                       # adapter only, no optimizer dump
        save_total_limit=2, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=held,
                      data_collator=Collate(tok.pad_token_id))
    with guard():
        trainer.train()
        if held:
            loss = trainer.evaluate()["eval_loss"]
            print(f"heldout loss {loss:.3f}  perplexity {math.exp(loss):.2f}")
    model.save_pretrained(Path(args.out) / "adapter")                     # adapter only - never merge (MTP)
    tok.save_pretrained(Path(args.out) / "adapter")
    print("adapter saved to", Path(args.out) / "adapter")


if __name__ == "__main__":
    main()
