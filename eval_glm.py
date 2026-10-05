"""G5 / G7 evaluation for the GLM-5.3 LoRA test . Imported by run_glm.py.

Everything here is an OBSERVATION for a human reviewer, not a pass/fail gate:
  - held-out loss (clear_thinking=false only: with true, earlier thinking is dropped and loss is undefined)
  - voice: the assistant's next turn at fixed held-out points; 2 of them also with clear_thinking=true (generation only)
  - one synthetic image prompt (vision is non-negotiable; frozen vision != unchanged answers)
  - 5 tool tasks with automatic format checks
Base and adapter use the same seed per prompt so differences come from the weights.
"""
import json
import re
from pathlib import Path

STOP_IDS = [154820, 154827, 154829]          # <|endoftext|> <|user|> <|observation|>
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)

TOOL_TASKS = [
    {"id": "T1_single_call", "expect": "call:search_memories",
     "user": "Can you look in your memories for what we decided about the garden project?"},
    {"id": "T2_two_step", "expect": "chain:search_memories>add_calendar_event",
     "user": "Find what we saved about the bike repair, then put a calendar event on October 8 to pick up the part.",
     "fake_results": ["Memory: bike repair - rear derailleur hanger bent; replacement part ordered at the shop; "
                      "pickup when it arrives.",
                      "Calendar event created: 2026-10-08 'Pick up bike part'."]},
    {"id": "T3_tool_error", "expect": "report_error",
     "user": "Please check my calendar for next week.",
     "fake_results": ["ERROR: calendar service unavailable (connection refused)."]},
    {"id": "T4_no_tool", "expect": "no_call",
     "user": "Quick one: what is 17 times 23?"},
    {"id": "T5_voice_after_result", "expect": "reply_no_call",
     "user": "Did you save the thing about the bread recipe?",
     "fake_results": ["Memory found: 2026-09-30 - sourdough came out flat; still tasted good."]},
]


def parse_turn(text):
    """Split a generated turn into thinking / reply / tool calls (GLM-native format)."""
    for s in ("<|endoftext|>", "<|user|>", "<|observation|>"):
        text = text.split(s)[0]
    thinking, _, rest = text.partition("</think>") if "</think>" in text else ("", "", text)
    calls = []
    for body in TOOL_RE.findall(rest):
        name = body.split("<arg_key>")[0].strip()
        calls.append({"name": name, "arguments": {k.strip(): v.strip() for k, v in ARG_RE.findall(body)}})
    reply = TOOL_RE.sub("", rest).strip()
    return {"thinking": thinking.strip(), "reply": reply, "tool_calls": calls}


def repetition(text, n=4):
    w = text.split()
    grams = [" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))]
    return max((grams.count(g) for g in set(grams)), default=0)


def third_person_hits(text):
    return len(re.findall(r"\b(the user|the human)\b", text))


def generate(model, tok, msgs, tools, tkw, max_new, seed, processor=None, images=None):
    import torch
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, **tkw)
    if images:
        enc = processor(text=[text], images=images, return_tensors="pt")
    else:
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
    dev = model.get_input_embeddings().weight.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95,
                             eos_token_id=STOP_IDS, pad_token_id=STOP_IDS[0])
    new = out[0][enc["input_ids"].shape[1]:]
    return parse_turn(tok.decode(new, skip_special_tokens=False))


def heldout_loss(model, data):
    """Token-weighted mean loss over learned tokens (clear_thinking=false format)."""
    import torch
    dev = model.get_input_embeddings().weight.device
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for ex in data:
            ids = torch.tensor([ex["input_ids"]], device=dev)
            lab = torch.tensor([ex["labels"]], device=dev)
            k = int((lab[:, 1:] != -100).sum())
            if k:
                tot += float(model(input_ids=ids, labels=lab).loss) * k
                n += k
    return tot / max(n, 1), n


def voice_points(heldout_rows, k=6):
    """Fixed points: the last 2 user turns per held-out segment that are answered by a plain assistant reply."""
    pts = []
    for row in heldout_rows:
        m = row["messages"]
        cand = [i for i in range(len(m) - 1) if m[i]["role"] == "user"
                and m[i + 1]["role"] == "assistant" and not m[i + 1].get("tool_calls")]
        for i in cand[-2:]:
            pts.append((row["id"], i, row.get("tools")))
    return pts[:k]


def clean(m):
    out = {"role": m["role"], "content": m.get("content", "")}
    for key in ("reasoning_content", "tool_calls", "tool_call_id"):
        if m.get(key):
            out[key] = m[key]
    return out


def synthetic_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (448, 336), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 200], fill="red")
    d.ellipse([260, 60, 400, 200], fill="blue")
    d.text((60, 260), "HELLO", fill="black")
    return img


def run_eval(model, tok, processor, system, heldout_rows, heldout_enc, tools, tkw, tag, out_dir,
             seed=42, max_new_voice=600, max_new_tool=400, max_new_image=300, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"tag": tag}

    loss, n = heldout_loss(model, heldout_enc)
    res["heldout_loss"], res["heldout_tokens"] = loss, n
    log(f"[{tag}] held-out loss {loss:.4f} over {n} tokens")

    sysmsg = [{"role": "system", "content": system}]
    by_id = {r["id"]: r for r in heldout_rows}
    res["voice"] = []
    for j, (rid, i, rtools) in enumerate(voice_points(heldout_rows)):
        own = by_id[rid].get("system")                    # rows may carry their own system prompt
        ctx = ([{"role": "system", "content": own}] if own else sysmsg) + [clean(m) for m in by_id[rid]["messages"][: i + 1]]
        g = generate(model, tok, ctx, rtools, tkw, max_new_voice, seed + j)
        entry = {"point": f"{rid}:{i}", "clear_thinking": False, **g,
                 "third_person_in_thinking": third_person_hits(g["thinking"]),
                 "max_4gram_repeat": repetition(g["reply"])}
        res["voice"].append(entry)
        if j < 2:
            g2 = generate(model, tok, ctx, rtools, {**tkw, "clear_thinking": True}, max_new_voice, seed + j)
            res["voice"].append({"point": f"{rid}:{i}", "clear_thinking": True, **g2,
                                 "third_person_in_thinking": third_person_hits(g2["thinking"]),
                                 "max_4gram_repeat": repetition(g2["reply"])})
        log(f"[{tag}] voice {j + 1} done")

    img_msgs = sysmsg + [{"role": "user", "content": [{"type": "image"},
                                                       {"type": "text", "text": "What do you see in this picture?"}]}]
    try:
        res["image"] = generate(model, tok, img_msgs, None, tkw, max_new_image, seed,
                                processor=processor, images=[synthetic_image()])
        res["image"]["mentions"] = {w: w in res["image"]["reply"].lower()
                                    for w in ("red", "blue", "hello", "square", "circle")}
    except Exception as e:                                  # recorded, never fatal for the run
        res["image"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    log(f"[{tag}] image: {res['image'].get('mentions', res['image'].get('error'))}")

    res["tools"] = [run_tool_task(model, tok, sysmsg, tools, tkw, t, max_new_tool, seed) for t in TOOL_TASKS]
    log(f"[{tag}] tools: " + ", ".join(f"{t['id']}={t['pass']}" for t in res["tools"]))

    (out_dir / f"eval_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def run_tool_task(model, tok, sysmsg, tools, tkw, task, max_new, seed):
    names = {t["function"]["name"]: t["function"] for t in tools}
    msgs = sysmsg + [{"role": "user", "content": task["user"]}]
    turns, results = [], list(task.get("fake_results", []))
    for step in range(3):
        g = generate(model, tok, msgs, tools, tkw, max_new, seed + step)
        turns.append(g)
        if not g["tool_calls"] or not results:
            break
        calls = [{"id": f"t{step}_{k}", "type": "function",
                  "function": {"name": c["name"], "arguments": c["arguments"]}} for k, c in enumerate(g["tool_calls"])]
        msgs = msgs + [{"role": "assistant", "content": "", "reasoning_content": g["thinking"], "tool_calls": calls}]
        for c in calls:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": results.pop(0) if results else "OK"})

    called = [c["name"] for t in turns for c in t["tool_calls"]]
    valid = all(n in names and all(r in c["arguments"] for r in names[n]["parameters"].get("required", []))
                for t in turns for c in t["tool_calls"] for n in [c["name"]])
    last = turns[-1]
    exp = task["expect"]
    if exp.startswith("call:"):
        ok = bool(called) and called[0] == exp[5:] and valid
    elif exp.startswith("chain:"):
        want = exp[6:].split(">")
        ok = called[: len(want)] == want and valid
    elif exp == "report_error":
        ok = bool(called) and not last["tool_calls"] and bool(last["reply"])
    elif exp == "no_call":
        ok = not called and bool(last["reply"])
    else:                                                  # reply_no_call after a result
        ok = bool(called) and not last["tool_calls"] and bool(last["reply"])
    return {"id": task["id"], "expect": exp, "called": called, "args_valid": valid, "pass": ok, "turns": turns}


def adapter_key_report(adapter_dir):
    """Which projections the saved adapter touches, mapped to vLLM PR #56327's packed groups."""
    from safetensors import safe_open
    leafs = set()
    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = re.search(r"\.self_attn\.(\w+)\.lora_[AB]", k)
            leafs.add(m.group(1) if m else "OTHER:" + k)
    vllm = {"q_proj": "in_proj_qkvbfg_a", "k_proj": "in_proj_qkvbfg_a", "v_proj": "in_proj_qkvbfg_a",
            "q_a_proj": "fused_qkv_a_proj", "kv_a_proj_with_mqa": "fused_qkv_a_proj",
            "q_b_proj": "q_b_proj (ColumnParallelLinear)", "o_proj": "o_proj (RowParallelLinear)"}
    return {"leafs": sorted(leafs), "vllm_groups": {l: vllm.get(l, "UNMAPPED") for l in sorted(leafs)},
            "unmapped": [l for l in leafs if l not in vllm]}
