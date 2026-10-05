# GLM-5.3-Flash LoRA: a tested recipe

A plain Hugging Face + PEFT recipe for training an attention-only LoRA on **GLM-5.3-Flash** (320B MoE, ~18B active)
on one rented node, plus a way to **merge the adapter into an EXL3 checkpoint** so it can be served by engines that
have no LoRA support (e.g. TensorFold on two DGX Sparks).

It's built for people paying for GPUs by the hour: one model load, a check before every expensive step, a hard
budget limit, and a results file even if something goes wrong.

## What you need

- One node with **~600 GB of GPU memory and NVLink/SXM**: 8x H200 SXM worked. **8x H200 PCIe was 3-4x slower**
  (see Lessons). 4x H200 is not enough.
- ~500 GB disk, a Hugging Face token with access to the base weights you choose.
- Your data as JSONL (format below).

## Files

| File | What it does |
|---|---|
| `glm_common.py` | Single source of truth: pinned base/tokenizer revisions, official template md5, LoRA targets, loading |
| `train_lora.py` | Encoder (learned spans), dataset build, `--dry-run` for tokenising only |
| `token_manifest.py` | Token-level checks: truncation, template match, every learned span starts after `<think>` and ends on a stop token |
| `run_glm.py` | One-process runner: G3 load + exact checkpoint check, G4 diagnostic adapter (grads, memory, speed projection), G4b learnability, G6 training, G7 eval; watchdog + budget cap |
| `eval_glm.py` | Held-out loss, voice samples, a synthetic image prompt, 5 tool tasks |
| `setup_glm.sh` | Pod setup: pinned env (installs a torch with kernels for your GPUs if needed), pinned downloads, manifest re-check |
| `pull_results.ps1` | Pull results + adapter, verify SHA256 checksums, optionally wipe the pod |
| `merge_lora_into_exl3.py` | Merge the adapter into an EXL3 checkpoint's BF16 attention tensors, byte-exact, in place on copies |

## Data format

One conversation per line. Assistant turns with `"train": true` are learned (thinking + reply); everything else
is context. Optional per-row `system` and `tools`.

```json
{"id": "ex-001", "system": "You are ...", "messages": [
  {"role": "user", "content": "Can you check what we decided about the garden?"},
  {"role": "assistant", "content": "", "reasoning_content": "I don't remember the details. Look it up.",
   "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_memories", "arguments": {"query": "garden decision"}}}],
   "train": true},
  {"role": "tool", "tool_call_id": "c1", "content": "Memory: raised beds on the south side, start in March."},
  {"role": "assistant", "content": "Raised beds on the south side, starting in March.",
   "reasoning_content": "Found it. Keep it short.", "train": true}]}
```

## Run

```bash
# on the pod, from the folder you uploaded
export HF_TOKEN=...            # never paste it anywhere else
DATA=data bash setup_glm.sh    # G1 + G2 + manifest
python3 run_glm.py --model /workspace/glm53-fp8 --tokenizer /workspace/glm_tokenizer \
  --system system_prompt.txt --train data/train.jsonl --heldout data/heldout.jsonl \
  --out /workspace/runs/glm --cap-min <N> --train-first --skip-g7
```
Set `N` so that `N x hourly price / 60` fits your budget. Pull the adapter the moment the log says `ADAPTER READY`.

## Lessons (the expensive way)

- **Use the official chat template.** Abliterated / repackaged checkpoints often ship a stub or modified template.
  We pin the upstream template by md5 and refuse to run otherwise.
- **Load the FP8 checkpoint dequantised to BF16** (`FineGrainedFP8Config(dequantize=True)`) with
  `Glm5NextForConditionalGeneration` (AutoModelForCausalLM won't load it), balanced across GPUs, SDPA with the
  efficient backend forced. G3 checks loaded tensors against an independent FP8 x block-scale dequant (error 0).
- **Attention-only LoRA** (r16, alpha 32): KDA q/k/v/o + MLA q_a/q_b/kv_a_proj_with_mqa/o -> 180 modules, all
  45 layers, ~35M params. Experts, routers, indexer and vision untouched.
- **Train what you serve.** Learned spans must be byte-identical to the serving template's generation prompt; the
  manifest checks this per turn. That includes the template kwargs: `glm_template_kwargs.json` and
  `glm_common.TEMPLATE_KWARGS` set `reasoning_effort: "low"` because our thinking data was produced at low effort.
  Change both to match your data, and serve with the same value.
- **Ground your data.** If the assistant's replies in your data rely on context the model can't see in the example
  (retrieved memories, injected notes), the fine-tune learns to state things with confidence and no source.
- **NVLink matters.** Naive model parallel over 8 GPUs: ~20 s/update on H200 SXM vs ~70 s on H200 PCIe.
- **The G4 projection is pessimistic** (the first measured step includes warm-up): real training took ~0.4x the
  projection. `--proj-factor` lets you account for that.

## Numbers from our runs

- Setup (install + 328 GB download): 6-9 minutes at ~800 MB/s.
- Pilot: 46 examples (~8.4k tokens each), 2 epochs, 46 updates: ~15 min training on 8x H200 SXM; whole run ~57 min.
- Peak memory ~96 GiB per H200.

## Merging into EXL3

In the EXL3 checkpoint we used, all `self_attn` tensors are plain BF16 (only routed experts are EXL3-encoded), so an
attention-only adapter can be merged directly: `W' = bf16(fp32(W) + (alpha/r) * B @ A)`. The script copies only the
touched shards, overwrites the target tensors' bytes in place (headers and expert bytes stay identical), symlinks the
rest, and reports how much of each delta survives BF16 rounding. Run `--dry-run` first.

Note: light adapters sit close to BF16's resolution; expect ~85-90 % of the adapter's direction to survive the
rounding. Stronger training makes this a non-issue.

## Acknowledgements

This recipe stands on other people's work. Thank you:

- **[Z.ai / zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash)** for GLM-5.3-Flash and its official chat template (MIT).
- **[orcarouter](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8)** for the uncensored FP8 checkpoint we trained on (MIT).
- **[neko-legends](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)** for the EXL3 quantisation we merged into (see its own license).
- **[turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)** for the EXL3 format (MIT).
- **[Ash Hart / TensorFold](https://github.com/ashhart/TensorFold)** for the engine that serves GLM-5.3-Flash on two DGX Sparks (Apache-2.0).
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** and **[PEFT](https://github.com/huggingface/peft)** (Apache-2.0).
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** for the KDA kernels (MIT).
- Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* (2024), which explained what
  our first adapter did wrong.

No model weights are distributed here. You download them from their authors, under their licenses.

## License

MIT (this recipe's code and documentation).
