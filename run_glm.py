"""One-process runner for the GLM-5.3 LoRA test: G3 -> G4 -> G4b -> G5 -> G6 -> G7 .

The 600 GB model is loaded ONCE. G4/G4b use a diagnostic adapter that is fully removed
(unload + fresh optimizer) before G5; G6 trains a brand-new adapter from the fixed seed.
Every gate writes results.json; a failed gate or the wall-clock cap stops the run.

Pod:   python run_glm.py --model /workspace/glm53-fp8 --tokenizer /workspace/glm_tokenizer \
           --system system_prompt.txt --train data/train.jsonl --heldout data/heldout.jsonl \
           --out /workspace/runs/glm
Tiny CPU rehearsal (notebook, random 4-layer GLM, same code paths):
       python run_glm.py --tiny ... --out runs/tiny
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_common as G          # noqa: E402
import eval_glm as E            # noqa: E402
import train_lora as TR    # noqa: E402

GATE_CAPS_MIN = {"G3": 25, "G4": 20, "G4b": 10, "G5": 25, "G6": 60, "G7": 25}
RESERVE_MIN = 15      # always kept free for: pull adapter+results to the notebook, verify, wipe the pod


class Run:
    def __init__(self, out, cap_min):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        start_file = Path("/workspace/logs/run_start_epoch")
        self.t0 = float(start_file.read_text()) if start_file.exists() else time.time()
        self.cap = cap_min * 60
        self.res = {"gates": {}, "status": "running"}
        self.hard_stop = self.t0 + self.cap - RESERVE_MIN * 60

    def start_watchdog(self):
        """Interrupts the main thread mid-gate if the hard stop (cap minus reserve) is reached."""
        import _thread
        import threading

        def watch():
            while True:
                time.sleep(15)
                if time.time() >= self.hard_stop:
                    self.log(f"WATCHDOG: hard stop at {self.elapsed_min():.0f} min (cap {self.cap / 60:.0f} "
                             f"- reserve {RESERVE_MIN}) - interrupting")
                    _thread.interrupt_main()
                    return
        threading.Thread(target=watch, daemon=True).start()

    def save(self):
        (self.out / "results.json").write_text(json.dumps(self.res, indent=1, default=str), encoding="utf-8")

    def elapsed_min(self):
        return (time.time() - self.t0) / 60

    def log(self, msg):
        line = f"[{self.elapsed_min():6.1f} min] {msg}"
        print(line, flush=True)
        with open(self.out / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def gate(self, name, ok, **info):
        self.res["gates"][name] = {"pass": bool(ok), "elapsed_min": round(self.elapsed_min(), 1), **info}
        self.save()
        self.log(f"{name} {'PASS' if ok else 'FAIL'} {json.dumps(info, default=str)[:400]}")
        if not ok:
            raise SystemExit(f"{name} failed - stop the pod, debug offline")

    def budget(self, name):
        need = GATE_CAPS_MIN[name] * 60
        if time.time() + need > self.hard_stop:
            self.gate(name, False, reason=f"wall-clock: {self.elapsed_min():.0f} min used + {GATE_CAPS_MIN[name]} "
                                          f"needed + {RESERVE_MIN} reserve > cap {self.cap / 60:.0f}")


# ---------------------------------------------------------------- tiny model
def tiny_model(tok_dir):
    import torch
    from transformers import AutoConfig, Glm5NextForConditionalGeneration
    c = AutoConfig.from_pretrained(tok_dir)
    if hasattr(c, "quantization_config"):
        delattr(c, "quantization_config")
    t = c.text_config
    t.num_hidden_layers = 4
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention", "indexed_attention"]
    t.indexer_types = ["full"] * 4
    t.mlp_layer_types = ["dense", "sparse", "sparse", "sparse"]
    t.first_k_dense_replace = 1
    for k, v in dict(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, num_attention_heads=4,
                     num_key_value_heads=4, q_lora_rank=64, kv_lora_rank=32, v_head_dim=32, qk_nope_head_dim=32,
                     qk_head_dim=32, index_n_heads=2, index_head_dim=32, index_topk=64, linear_head_dim=32,
                     linear_num_heads=4, n_routed_experts=8, num_experts_per_tok=2).items():
        setattr(t, k, v)
    v = c.vision_config
    for k, val in dict(depth=1, hidden_size=64, num_heads=2, intermediate_size=128, out_hidden_size=256,
                       projection_intermediate_size=256).items():
        setattr(v, k, val)
    torch.manual_seed(0)
    return Glm5NextForConditionalGeneration._from_config(c, dtype=torch.float32)


# ---------------------------------------------------------------- G3 helpers
def fp8_numeric_check(model, model_dir, max_tensors=6):
    """Loaded BF16 params vs an independent FP8 x block-scale dequant of the raw checkpoint."""
    import torch
    from safetensors import safe_open
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    params = dict(model.named_parameters())
    wanted = [r"layers.0.self_attn.q_proj.weight", r"layers.3.self_attn.q_a_proj.weight",
              r"layers.1.self_attn.o_proj.weight", r"shared_experts", r"experts.0.", r"layers.0.mlp.gate_proj"]
    picks = []
    for w in wanted:
        k = next((k for k in idx if w in k and k.endswith(".weight") and "visual" not in k), None)
        if k:
            picks.append(k)
    report = []
    for key in picks[:max_tensors]:
        with safe_open(str(Path(model_dir) / idx[key]), "pt") as f:
            raw = f.get_tensor(key)
            sk = key + "_scale_inv"
            scale = f.get_tensor(sk) if sk in f.keys() else None
        x = raw.float()
        if scale is not None:
            bs = 128
            s = scale.float().repeat_interleave(bs, 0)[: x.shape[0]].repeat_interleave(bs, 1)[:, : x.shape[1]]
            x = x * s
        ref = x.to(torch.bfloat16).float()
        p = params.get(key)
        if p is None:
            p = params.get(key.replace("model.", "", 1))
        status = "no-param-with-same-name"
        if p is not None:
            got = p.detach().float().cpu()
            if got.shape == ref.shape:
                err = (got - ref).abs().max().item()
                status = f"max_abs_err={err:.3g} (scale {ref.abs().max().item():.3g})"
                report.append({"key": key, "fp8": scale is not None, "match": err <= 1e-2 * max(ref.abs().max().item(), 1e-6),
                               "status": status})
                continue
            status = f"shape {tuple(got.shape)} vs {tuple(ref.shape)}"
        report.append({"key": key, "fp8": scale is not None, "match": None, "status": status})
    return report


def gpu_mem():
    import torch
    if not torch.cuda.is_available():
        return {}
    return {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())}


def lora_modules(pm):
    return [(n, m) for n, m in pm.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]


def step_loss(model, ex):
    import torch
    dev = model.get_input_embeddings().weight.device
    ids = torch.tensor([ex["input_ids"]], device=dev)
    lab = torch.tensor([ex["labels"]], device=dev)
    return model(input_ids=ids, labels=lab).loss


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-min", type=float, default=210)
    ap.add_argument("--tiny", action="store_true", help="CPU rehearsal with a random 4-layer GLM")
    ap.add_argument("--tiny-examples", type=int, default=3)
    ap.add_argument("--g6-cap", type=float, default=60,
                    help="training time cap in minutes (G6) and G4's projection limit")
    ap.add_argument("--skip-g7", action="store_true", help="never run G7 on the pod (evaluate at home)")
    ap.add_argument("--epochs", type=int, help="override glm_common.EPOCHS (e.g. 1 for large datasets)")
    ap.add_argument("--proj-factor", type=float, default=1.0, help="G4 projection x factor (first-step warm-up makes it pessimistic; we measured real = ~0.4 x projected)")
    ap.add_argument("--train-first", action="store_true",
                    help="lean run: skip G5 (evaluate the base model at home); "
                         "G7 only if the budget still allows, else skipped (adapter eval at home)")
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

    GATE_CAPS_MIN["G6"] = a.g6_cap
    R = Run(a.out, a.cap_min)
    R.start_watchdog()
    try:
        body(a, R)
        R.res["status"] = "complete"
    except KeyboardInterrupt:
        R.res["status"] = "stopped: wall-clock hard stop (watchdog) or manual interrupt"
    except SystemExit as e:
        R.res["status"] = f"stopped: {e}"
    except Exception as e:
        R.res["status"] = f"crashed: {type(e).__name__}: {str(e)[:300]}"
        import traceback; R.log(traceback.format_exc()[-2000:])
    finally:
        R.res["adapter_ready"] = (R.out / "adapter" / "SHA256SUMS").exists()
        R.save()
        R.log(f"EXIT status={R.res['status']} adapter_ready={R.res['adapter_ready']} - pull {R.out} to the notebook, verify, then wipe the pod")
    if R.res["status"] != "complete":
        sys.exit(1)


def body(a, R):
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed
    tok = G.load_tokenizer(a.tokenizer, Path(a.tokenizer) / "chat_template.jinja")
    processor = AutoProcessor.from_pretrained(a.tokenizer)
    processor.tokenizer.chat_template = tok.chat_template
    tkw = dict(G.TEMPLATE_KWARGS)
    system = Path(a.system).read_text(encoding="utf-8")
    train = TR.build(tok, a.train, system, G.MAX_LEN, "train", tkw, require_thinking=True)
    held = TR.build(tok, a.heldout, system, G.MAX_LEN, "heldout", tkw, require_thinking=True)
    held_rows = TR.load_rows(a.heldout)
    tools = next((r.get("tools") for r in TR.load_rows(a.train) if r.get("tools")), None)
    if a.tiny:
        train, held = train[: a.tiny_examples], held[:1]
        # 16 GB notebook: logits over GLM's 154k vocab for 4-9k tokens need 5+ GB. Trim for the CPU rehearsal only
        # (same code paths); keep only examples that still have learned tokens after the trim.
        cut = lambda e, n=2048: {"input_ids": e["input_ids"][:n], "labels": e["labels"][:n]}
        train = [x for x in map(cut, train) if any(l != -100 for l in x["labels"])] or train[:1]
        held = [x for x in map(cut, held) if any(l != -100 for l in x["labels"])] or held[:1]
        for k in GATE_CAPS_MIN:          # tiny: per-gate caps off, so the watchdog itself can be tested
            GATE_CAPS_MIN[k] = 0
        guard = contextlib.nullcontext
        gen_caps = dict(max_new_voice=8, max_new_tool=8, max_new_image=8)
    else:
        guard = G.attention_guard
        gen_caps = {}
    R.log(f"data: train {len(train)} ex, heldout {len(held)} ex; tiny={a.tiny}")

    # ---- G3 load + numeric check
    R.budget("G3")
    set_seed(G.SEED)
    if a.tiny:
        model = tiny_model(a.tokenizer)
        R.gate("G3", True, note="tiny model, no FP8 check", params=sum(p.numel() for p in model.parameters()))
    else:
        model = G.load_model(a.model)
        chk = fp8_numeric_check(model, a.model)
        compared = [c for c in chk if c["match"] is not None]
        R.gate("G3", len(compared) >= 3 and all(c["match"] for c in compared),
               tensors=chk, peak_gib=gpu_mem())

    # ---- G4 diagnostic adapter (disposable)
    R.budget("G4")
    if a.tiny:
        G.EXPECTED_LORA_MODULES, G.EXPECTED_TRAINABLE = 16, None
        _orig = G.attach_lora

        def attach_tiny(m, gradient_checkpointing=True):
            from peft import LoraConfig, get_peft_model
            if gradient_checkpointing:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                m.enable_input_require_grads()
            pm = get_peft_model(m, LoraConfig(r=G.RANK, lora_alpha=G.ALPHA, lora_dropout=G.DROPOUT,
                                              bias="none", target_modules=G.TARGETS, task_type="CAUSAL_LM"))
            assert len(lora_modules(pm)) == 16, len(lora_modules(pm))
            return pm
        G.attach_lora = attach_tiny
    diag = G.attach_lora(model)
    diag.train()
    short = min(train, key=lambda e: len(e["input_ids"]))
    longest = max(train, key=lambda e: len(e["input_ids"]))
    with guard():
        loss = step_loss(diag, short)
        a_ok = math.isfinite(loss.item()) and (a.tiny or 0.3 < loss.item() < 6.0)
        loss.backward()
        per_dev, zero = {}, 0
        for n, m in lora_modules(diag):
            g = m.lora_B["default"].weight.grad
            dev = str(m.lora_B["default"].weight.device)
            nz = g is not None and g.abs().sum().item() > 0
            per_dev.setdefault(dev, [0, 0])[0 if nz else 1] += 1
            zero += not nz
        b_ok = zero == 0 and all(v[0] > 0 for v in per_dev.values())
        opt = torch.optim.AdamW([p for p in diag.parameters() if p.requires_grad], lr=G.LR)
        before = {n: m.lora_B["default"].weight.detach().clone() for n, m in lora_modules(diag)}
        opt.step()
        opt.zero_grad()
        c_ok = all(not torch.equal(before[n], m.lora_B["default"].weight) for n, m in lora_modules(diag))
        diag.eval()
        probe = torch.tensor([short["input_ids"][:256]], device=diag.get_input_embeddings().weight.device)
        with torch.no_grad():
            on = diag(input_ids=probe).logits.float()
            with diag.disable_adapter():
                off = diag(input_ids=probe).logits.float()
        ddir = R.out / "diag_adapter"
        diag.save_pretrained(ddir)
        diag.load_adapter(str(ddir), adapter_name="reloaded")
        diag.set_adapter("reloaded")
        with torch.no_grad():
            re_on = diag(input_ids=probe).logits.float()
        diag.set_adapter("default")
        diag.delete_adapter("reloaded")
        d_ok = (on - off).abs().max().item() > 0 and (re_on - on).abs().max().item() < 1e-3 * max(on.abs().max().item(), 1)
        diag.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        L = step_loss(diag, longest)
        L.backward()
        opt.zero_grad()
        s_long = time.time() - t
        e_ok = math.isfinite(L.item())
    tokens_train = sum(len(e["input_ids"]) for e in train)
    proj_min = s_long / len(longest["input_ids"]) * tokens_train * (a.epochs or G.EPOCHS) / 60
    R.gate("G4", a_ok and b_ok and c_ok and d_ok and e_ok and (a.tiny or proj_min * a.proj_factor <= a.g6_cap),
           a_loss=round(loss.item(), 4), b_grads_per_device=per_dev, b_zero_grad_modules=zero, c_changed=c_ok,
           d_adapter_changes_logits=d_ok, e_longest_tokens=len(longest["input_ids"]),
           e_seconds=round(s_long, 1), projected_g6_min=round(proj_min, 1), peak_gib=gpu_mem())

    # ---- G4b learnability (same disposable adapter, own optimizer)
    R.budget("G4b")
    # Disposable adapter: a high lr is fine - this only proves the adapter CAN memorise one example.
    opt = torch.optim.AdamW([p for p in diag.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    with guard():
        for _ in range(8):
            l = step_loss(diag, short)
            l.backward()
            opt.step()
            opt.zero_grad()
            losses.append(round(l.item(), 4))
    if a.tiny:   # random weights start at ln(vocab) ~ 12; only require a steady decline
        ok = all(b < x for x, b in zip(losses, losses[1:]))
    else:
        ok = losses[-1] < 0.8 * losses[0]
    R.gate("G4b", ok, losses=losses, lr=1e-3)

    # ---- remove the diagnostic adapter completely
    del opt
    model = diag.unload()
    if hasattr(model, "peft_config"):
        del model.peft_config
    shutil.rmtree(ddir, ignore_errors=True)
    left = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))
    if left:
        R.gate("cleanup", False, lora_modules_left=left)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    R.log("diagnostic adapter removed (0 LoRA modules left)")

    # ---- G5 baseline (no adapter)
    if a.train_first:
        base = None
        R.res["gates"]["G5"] = {"skipped": "train-first: base evaluated at home"}
        R.save()
        R.log("G5 skipped (--train-first)")
    else:
        R.budget("G5")
        model.eval()
        with guard():
            base = E.run_eval(model, tok, processor, system, held_rows, held, tools, tkw, "base", R.out,
                              seed=G.SEED, log=R.log, **gen_caps)
        R.gate("G5", True, heldout_loss=round(base["heldout_loss"], 4),
               tools={t["id"]: t["pass"] for t in base["tools"]}, image=base["image"].get("mentions", base["image"]))

    # ---- G6 fresh adapter, fresh optimizer, 40 updates
    R.budget("G6")
    set_seed(G.SEED)
    pm = G.attach_lora(model)
    targs = TrainingArguments(
        output_dir=str(R.out / "trainer"), num_train_epochs=1 if a.tiny else (a.epochs or G.EPOCHS), learning_rate=G.LR,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=G.GRAD_ACCUM,
        lr_scheduler_type="cosine", warmup_steps=0, bf16=not a.tiny, use_cpu=a.tiny, logging_steps=1,
        eval_strategy="epoch", save_strategy="no" if a.tiny else "steps", save_steps=25, save_total_limit=1,
        save_only_model=True, report_to="none", remove_unused_columns=False, seed=G.SEED)
    trainer = Trainer(model=pm, args=targs, train_dataset=train, eval_dataset=held,
                      data_collator=TR.Collate(tok.pad_token_id))
    with guard():
        trainer.train()
    hist = [h for h in trainer.state.log_history if "loss" in h]
    first, last = (hist[0]["loss"], hist[-1]["loss"]) if hist else (None, None)
    adir = R.out / "adapter"
    pm.save_pretrained(adir)
    tok.save_pretrained(adir)
    import hashlib
    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in sorted(adir.iterdir()) if f.is_file()]
    (adir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    R.log("ADAPTER READY - pull it to the notebook now (pull_results.ps1), before G7")
    keys = E.adapter_key_report(adir)
    R.gate("G6", adir.joinpath("adapter_model.safetensors").exists() and not keys["unmapped"],
           updates=trainer.state.global_step, first_loss=first, last_loss=last,
           eval=[h for h in trainer.state.log_history if "eval_loss" in h], adapter_keys=keys)

    # ---- G7 adapter eval
    if a.skip_g7 or (a.train_first and time.time() + GATE_CAPS_MIN["G7"] * 60 > R.hard_stop):
        R.res["gates"]["G7"] = {"skipped": f"train-first: no budget left at {R.elapsed_min():.0f} min - eval at home"}
        R.save()
        R.log("G7 skipped (budget) - adapter is home, evaluate there")
        return
    R.budget("G7")
    pm.eval()
    with guard():
        ad = E.run_eval(pm, tok, processor, system, held_rows, held, tools, tkw, "adapter", R.out,
                        seed=G.SEED, log=R.log, **gen_caps)
    R.gate("G7", True, heldout_loss_base=round(base["heldout_loss"], 4) if base else None,
           heldout_loss_adapter=round(ad["heldout_loss"], 4),
           tools_base={t["id"]: t["pass"] for t in base["tools"]} if base else None,
           tools_adapter={t["id"]: t["pass"] for t in ad["tools"]},
           image_adapter=ad["image"].get("mentions", ad["image"]))
    R.log(f"DONE - download {R.out}, then delete the volume")


if __name__ == "__main__":
    main()