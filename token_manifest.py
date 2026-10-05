"""Post-tokenization manifest for a training set (Voss review, point 1 + 3).

For every example: total tokens, learned tokens, truncation, and every learned span
checked at TOKEN-ID level: the token before the span must be the generation prompt's
last token, the span must end on a stop-token id, no role token may appear inside.
Also reports optimizer-step arithmetic. Numbers only; no text printed.

  python token_manifest.py --tok DIR --template FILE --kwargs FILE --system FILE --data A.jsonl [B.jsonl] --max-len N
"""
import argparse
import json
import sys
from collections import Counter
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora as T  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--kwargs", required=True)
ap.add_argument("--system", required=True)
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=4)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tok)
tok.chat_template = Path(a.template).read_text(encoding="utf-8")
tkw = json.loads(Path(a.kwargs).read_text(encoding="utf-8"))
system = Path(a.system).read_text(encoding="utf-8")
voc = tok.get_vocab()
STOP_IDS = {voc[s] for s in ("<|user|>", "<|observation|>", "<|endoftext|>") if s in voc}
ROLE_IDS = {voc[s] for s in ("<|user|>", "<|assistant|>", "<|system|>", "<|observation|>") if s in voc}
ASSIST = voc.get("<|assistant|>")
THINK = voc.get("<think>")
THINK_CLOSE = voc.get("</think>")
print("tokenizer:", a.tok, "| template bytes:", len(Path(a.template).read_bytes()),
      "| stop ids:", sorted(STOP_IDS), "| <|assistant|>:", ASSIST, "| <think>:", THINK)

import hashlib
seen_spans = Counter()
for path in a.data:
    rows = T.load_rows(path)
    lens, c = [], Counter()
    for row in rows:
        st = Counter()
        full_len = len(tok(T.render(tok, [{"role": "system", "content": row.get("system") or system}] +
                                    [T.clean_msg(m) for m in row["messages"]], row.get("tools"), **tkw),
                           add_special_tokens=False)["input_ids"])
        e = T.encode_thread(tok, row, system, a.max_len, st, tkw, require_thinking=True)
        c["reply_only_turns"] += st["reply_only_turns"]
        c["examples"] += e is not None
        c["truncated"] += full_len > a.max_len
        c["template_mismatch"] += st["template_mismatch"]
        lens.append(full_len)
        if not e:
            continue
        ids, lab = e["input_ids"], e["labels"]
        c["learned_tokens"] += sum(l != -100 for l in lab)
        i = 0
        while i < len(lab):
            if lab[i] == -100:
                i += 1
                continue
            j = i
            while j < len(lab) and lab[j] != -100:
                j += 1
            span = ids[i:j]
            seen_spans[hashlib.sha1(str(span).encode()).hexdigest()] += 1
            c["spans"] += 1
            c["prev_is_think_open"] += ids[i - 1] == THINK                 # full turn: gen prompt ends <|assistant|><think>
            c["prev_is_think_close"] += ids[i - 1] == THINK_CLOSE          # reply-only turn: starts right after </think>
            c["prev2_is_assistant"] += i >= 2 and ids[i - 2] == ASSIST
            c["ends_on_stop_id"] += span[-1] in STOP_IDS
            c["role_id_inside"] += any(t in ROLE_IDS for t in span[:-1])
            i = j
    lens.sort()
    n = len(lens)
    updates = a.epochs * ceil(c["examples"] / a.grad_accum)
    print(f"\n[{Path(path).name}] examples {c['examples']} | tokens/example min {lens[0]} median {lens[n // 2]} max {lens[-1]} "
          f"| truncated at {a.max_len}: {c['truncated']} | template_mismatch {c['template_mismatch']}")
    print(f"  learned tokens {c['learned_tokens']} | spans {c['spans']} | span preceded by <think> {c['prev_is_think_open']} "
          f"| <|assistant|><think> {c['prev2_is_assistant']} | ends on stop id {c['ends_on_stop_id']} "
          f"| role token inside span {c['role_id_inside']}")
    print(f"  full turns (after <think>) {c['prev_is_think_open']} + reply-only (after </think>) {c['prev_is_think_close']} "
          f"= {c['prev_is_think_open'] + c['prev_is_think_close']} of {c['spans']} spans | reply_only turns flagged {c['reply_only_turns']}")
    print(f"  optimizer updates at epochs={a.epochs}, grad_accum={a.grad_accum}: {updates}")


dups = sum(1 for v in seen_spans.values() if v > 1)
print(f"\nALL FILES: unique learned spans {len(seen_spans)} | spans appearing more than once (train/holdout leakage or duplicates): {dups}")