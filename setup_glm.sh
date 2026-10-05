#!/usr/bin/env bash
# G1 + G2 of README.md: pinned environment, pinned downloads, offline checks.
# Run from /workspace/recipe (this folder uploaded there). Needs HF_TOKEN in the environment
# (you type:  export HF_TOKEN=...  in the pod shell - never pasted into chat).
set -euo pipefail
cd /workspace/recipe
mkdir -p /workspace/logs
date +%s > /workspace/logs/run_start_epoch          # hard cap clock starts here 
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a /workspace/logs/setup.log; }

: "${HF_TOKEN:?HF_TOKEN not set - export it in the pod shell first}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/.hf

# ---- G1: pinned environment (cap 15 min) -----------------------------------
# B300 (Blackwell Ultra, sm_103) needs a PyTorch built with sm_103 kernels; RunPod's newest official
# template is 2.8. If the visible GPUs aren't covered by this torch build, upgrade torch+torchvision
# together from the official wheel index (cu129, then cu130) and re-check. Stop if neither works.
gpu_ok() { python3 - <<'PY'
import torch, sys
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
arch = torch.cuda.get_arch_list()
ok = all(f"sm_{a}{b}" in arch or f"compute_{a}{b}" in arch for a, b in caps)
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU caps", sorted(caps), "| build has them:", ok)
sys.exit(0 if ok else 1)
PY
}
if ! gpu_ok 2>&1 | tee -a /workspace/logs/setup.log; then
  for CU in cu129 cu130; do
    log "torch build lacks kernels for these GPUs - installing torch 2.9 / torchvision 0.24 from ${CU}"
    timeout 10m pip install -q --break-system-packages --upgrade "torch==2.9.*" "torchvision==0.24.*" \
      --index-url "https://download.pytorch.org/whl/${CU}" 2>&1 | tail -2 | tee -a /workspace/logs/setup.log || true
    if gpu_ok 2>&1 | tee -a /workspace/logs/setup.log; then break; fi
  done
  gpu_ok >/dev/null 2>&1 || { log "G1 FAIL: no torch build with kernels for these GPUs"; exit 1; }
fi
log "G1 install"
timeout 15m pip install -q --break-system-packages \
  "transformers==5.18.0" "peft==0.21.1" accelerate safetensors hf_transfer huggingface_hub \
  flash-linear-attention pillow 2>&1 | tail -3 | tee -a /workspace/logs/setup.log
# torchvision must match the image's torch exactly (GLM's processor needs it; a mismatched wheel breaks torch)
python3 -c "import torchvision" 2>/dev/null || { log "torchvision missing - installing the build matching torch";
  TV=$(python3 -c "import torch;v=torch.__version__.split('+')[0];m={'2.8':'0.23','2.9':'0.24','2.10':'0.25'};print(m['.'.join(v.split('.')[:2])])");
  CU=$(python3 -c "import torch;print('cu'+torch.version.cuda.replace('.',''))");
  timeout 5m pip install -q --break-system-packages --no-deps "torchvision==${TV}.*" --index-url "https://download.pytorch.org/whl/${CU}"; }
python3 - <<'PY' 2>&1 | tee -a /workspace/logs/setup.log
import torch, transformers, peft, torchvision, PIL
from transformers import Glm5NextForConditionalGeneration, FineGrainedFP8Config  # must resolve
print("torchvision", torchvision.__version__, "| pillow", PIL.__version__)
assert transformers.__version__ == "5.18.0", transformers.__version__
n = torch.cuda.device_count()
caps = {torch.cuda.get_device_capability(i) for i in range(n)}
arch = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| peft", peft.__version__)
print("GPUs", n, [torch.cuda.get_device_name(i) for i in range(n)][:1], "| capability", caps, "| torch arch", arch)
for c in caps:
    if f"sm_{c[0]}{c[1]}" not in arch and f"compute_{c[0]}{c[1]}" not in arch:
        raise SystemExit(f"torch build has no kernels for sm_{c[0]}{c[1]} - wrong image for these GPUs")
mem = min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
assert n * mem * 0.85 > 598.5 * 1.12, f"not enough GPU memory: {n} x {mem:.0f} GiB"
try:
    import fla; print("fla OK")
except Exception as e:
    print("fla NOT available (slow torch fallback for KDA):", type(e).__name__)
print("G1 PASS")
PY

# ---- G2: pinned downloads (cap 40 min) ---------------------------------------
log "G2 tokenizer + template (zai-org @ eb9eb208)"
timeout 5m hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --include "tokenizer*" --include "chat_template.jinja" --include "config.json" \
  --include "generation_config.json" --include "processor_config.json" \
  --local-dir /workspace/glm_tokenizer 2>&1 | tail -1
echo "16ffa9ab6061be83657ecb9f58069808  /workspace/glm_tokenizer/chat_template.jinja" | md5sum -c - \
  | tee -a /workspace/logs/setup.log

log "G2 weights (orcarouter FP8 @ 3cec42d6, ~328 GB)"
timeout 40m hf download orcarouter/GLM-5.3-Flash-Uncensored-FP8 --revision 3cec42d6 \
  --include "*.safetensors" --include "*.json" \
  --local-dir /workspace/glm53-fp8 2>&1 | tail -1
python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('/workspace/glm_tokenizer'); print('processor OK')" \
  2>&1 | tail -1 | tee -a /workspace/logs/setup.log
python3 - <<'PY' 2>&1 | tee -a /workspace/logs/setup.log
import json, pathlib
d = pathlib.Path("/workspace/glm53-fp8")
idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
want = set(idx.values()); have = {p.name for p in d.glob("*.safetensors")}
missing = want - have
print("shards in index", len(want), "| on disk", len(have), "| missing", len(missing))
if missing: raise SystemExit("G2 FAIL: missing shards")
print("visual tensors", sum(k.startswith("model.visual.") for k in idx))
print("G2 PASS")
PY

# ---- offline re-check on the pod (CPU, seconds) ------------------------------
log "manifest on pod (must equal notebook)"
python3 token_manifest.py --tok /workspace/glm_tokenizer --template /workspace/glm_tokenizer/chat_template.jinja \
  --kwargs glm_template_kwargs.json --system system_prompt.txt \
  --data ${DATA:-data}/train.jsonl ${DATA:-data}/heldout.jsonl --max-len 10240 --grad-accum 2 \
  2>&1 | tee -a /workspace/logs/setup.log
python3 train_lora.py --preset glm --model /workspace/glm53-fp8 --tokenizer /workspace/glm_tokenizer \
  --chat-template /workspace/glm_tokenizer/chat_template.jinja --train ${DATA:-data}/train.jsonl \
  --heldout ${DATA:-data}/heldout.jsonl --system system_prompt.txt --out /workspace/runs/dry --dry-run \
  2>&1 | tee -a /workspace/logs/setup.log
log "setup done; elapsed $(( $(date +%s) - $(cat /workspace/logs/run_start_epoch) ))s"
