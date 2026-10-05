"""Single source of truth for the GLM-5.3-Flash LoRA run .

train_lora.py (--preset glm), smoke_glm.py and eval_glm.py all import from here,
so pinned values, loading, LoRA attachment and the attention guard cannot drift apart.
torch / peft are imported lazily so the notebook dry-run works without a GPU.
"""
import contextlib
import hashlib
import math
from pathlib import Path

# ---- pinned  -------------------------------------------------------
BASE_REPO = "orcarouter/GLM-5.3-Flash-Uncensored-FP8"
BASE_REVISION = "3cec42d6"                      
TOKENIZER_REPO = "zai-org/GLM-5.3-Flash"
TOKENIZER_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
OFFICIAL_TEMPLATE_MD5 = "16ffa9ab6061be83657ecb9f58069808"
TEMPLATE_KWARGS = {"reasoning_effort": "low"}
TRANSFORMERS_VERSION = "5.18.0"

TARGETS = (r".*language_model\.layers\.\d+\.self_attn\."
           r"(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)$")
EXPECTED_LORA_MODULES = 180
EXPECTED_TRAINABLE = 35_299_328                 # r=16, measured on a meta-built model
RANK, ALPHA, DROPOUT = 16, 32, 0.05

MAX_LEN = 10_240
GRAD_ACCUM = 2
LR = 1e-4
EPOCHS = 2
SEED = 42
BF16_WEIGHTS_GIB = 598.5                        # official BF16 sibling size


def load_tokenizer(tok_dir, template_path):
    """Tokenizer from the pinned zai-org snapshot; template MUST be the official file."""
    from transformers import AutoTokenizer
    raw = Path(template_path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != OFFICIAL_TEMPLATE_MD5:
        raise SystemExit(f"chat template md5 {md5} != official {OFFICIAL_TEMPLATE_MD5} - refusing to run")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    tok.chat_template = raw.decode("utf-8")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def balanced_max_memory(headroom=1.12):
    """Even weight split across all visible GPUs (greedy 'auto' would fill GPU 0 first)."""
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("no CUDA devices")
    per_gpu = math.ceil(BF16_WEIGHTS_GIB / n * headroom)
    total = min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
    if per_gpu > total * 0.85:
        raise SystemExit(f"weights need ~{per_gpu} GiB/GPU but GPUs have {total:.0f} GiB - not enough GPUs")
    return {i: f"{per_gpu}GiB" for i in range(n)}, n, per_gpu, total


def load_model(model_dir):
    """FP8 checkpoint -> BF16 at load, image-text class, balanced over GPUs, SDPA attention."""
    import torch
    from transformers import FineGrainedFP8Config, Glm5NextForConditionalGeneration
    max_memory, n, per_gpu, total = balanced_max_memory()
    print(f"[load] {n} GPUs x {total:.0f} GiB; weight budget {per_gpu} GiB/GPU")
    model = Glm5NextForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        attn_implementation="sdpa",
    )
    return model


def attach_lora(model, gradient_checkpointing=True):
    """Attention-only LoRA; asserts the exact module/param counts verified offline."""
    from peft import LoraConfig, get_peft_model
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
                                          target_modules=TARGETS, task_type="CAUSAL_LM"))
    n_mod = sum(1 for _, m in pm.named_modules() if hasattr(m, "lora_A") and len(m.lora_A))
    n_par = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    bad = [n for n, m in pm.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)
           and any(s in n for s in ("visual", "mlp", "expert", "indexer"))]
    if n_mod != EXPECTED_LORA_MODULES or n_par != EXPECTED_TRAINABLE or bad:
        raise SystemExit(f"LoRA mismatch: modules {n_mod} (want {EXPECTED_LORA_MODULES}), "
                         f"trainable {n_par} (want {EXPECTED_TRAINABLE}), forbidden hits {len(bad)}")
    print(f"[lora] {n_mod} modules, {n_par:,} trainable params - as verified offline")
    return pm


@contextlib.contextmanager
def attention_guard():
    """Forbid SDPA's math backend: it would materialise 64 x S x S fp32 for the dense DSA mask."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        yield
