"""Merge a GLM-5.3-Flash LoRA adapter into the EXL3 checkpoint's BF16 attention tensors -> "<model>-merged".

The EXL3 checkpoint (e.g. neko-legends/GLM-5.3-Flash-Uncensored-EXL3 @ 07135ec0) keeps all self_attn tensors as plain BF16.
The adapter targets 180 of them (same names and shapes,
verified against the header dump). Per target: W' = bf16(fp32(W) + (alpha/r) * B @ A).

Because W' has exactly W's dtype and shape, each touched shard is COPIED and only the byte ranges of the
target tensors are overwritten in place: header, metadata, order, expert (EXL3) bytes and every other
tensor stay byte-identical. Untouched files are symlinked to the originals. The original snapshot is
never written to. RAM use: one tensor at a time (a few hundred MB at most).

The output folder contains your fine-tune: treat it like your adapter.

Needs only numpy (no torch). Run as a user who can read the snapshot:
  python3 merge_lora_into_exl3.py --snapshot <the EXL3 checkpoint snapshot dir> --adapter <adapter dir> --out <new dir>
  add --dry-run to check names/shapes and print the stats without writing.
Prints numbers only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

PREFIX = "base_model.model."


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def read_tensor(path, header, base, name):
    t = header[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if t["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(t["shape"])
    if t["dtype"] == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(t["shape"]).astype(np.float32)
    raise SystemExit(f"{name}: unsupported dtype {t['dtype']}")


def to_bf16_bytes(x):
    """float32 -> bf16 with round-to-nearest-even (same as torch's .to(torch.bfloat16)); NaN kept NaN."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    nan = np.isnan(x)
    rounded = np.where(nan, (u >> 16) | 0x40, rounded)
    return rounded.astype("<u2").tobytes()


def bf16_to_f32(b, shape):
    return (np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def load_adapter(adir):
    cfg = json.load(open(adir / "adapter_config.json"))
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("fan_in_fan_out"):
        raise SystemExit("adapter uses DoRA/rsLoRA/fan_in_fan_out - this merge formula does not apply")
    scale = cfg["lora_alpha"] / cfg["r"]
    path = adir / "adapter_model.safetensors"
    h, base = read_header(path)
    pairs = {}
    for k in h:
        if k == "__metadata__":
            continue
        m = re.match(r"(.*)\.lora_([AB])\.weight$", k)
        if not m or not k.startswith(PREFIX):
            raise SystemExit(f"unexpected adapter key {k}")
        pairs.setdefault(m.group(1)[len(PREFIX):] + ".weight", {})[m.group(2)] = k
    return cfg, scale, path, h, base, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    snap, adir, out = Path(a.snapshot), Path(a.adapter), Path(a.out)
    if out.resolve() == snap.resolve():
        raise SystemExit("--out must differ from --snapshot (the original is never written)")

    cfg, scale, apath, ah, abase, pairs = load_adapter(adir)
    print(f"adapter: r {cfg['r']} alpha {cfg['lora_alpha']} -> scale {scale} | targets {len(pairs)}")

    weight_map = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    by_file = {}
    for name in pairs:
        if name not in weight_map:
            raise SystemExit(f"target not in the EXL3 checkpoint: {name}")
        by_file.setdefault(weight_map[name], []).append(name)
    print(f"the EXL3 checkpoint files touched: {len(by_file)} of {len(set(weight_map.values()))}")

    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=False)
        for p in snap.iterdir():                       # everything untouched -> symlink to the real file
            if p.name not in by_file:
                os.symlink(os.path.realpath(p), out / p.name)

    stats, sums = [], []
    for fname, names in sorted(by_file.items()):
        src = snap / fname
        h, base = read_header(src)
        dst = out / fname
        if not a.dry_run:
            shutil.copyfile(os.path.realpath(src), dst)
        for name in sorted(names):
            t = h[name]
            if t["dtype"] != "BF16":
                raise SystemExit(f"{name}: the EXL3 checkpoint dtype {t['dtype']} (expected BF16)")
            W = read_tensor(src, h, base, name)
            A = read_tensor(apath, ah, abase, pairs[name]["A"])
            B = read_tensor(apath, ah, abase, pairs[name]["B"])
            if B.shape[0] != W.shape[0] or A.shape[1] != W.shape[1]:
                raise SystemExit(f"{name}: shape W {W.shape} vs B {B.shape} A {A.shape}")
            delta = scale * (B @ A)
            new = to_bf16_bytes(W + delta)
            got = bf16_to_f32(new, W.shape) - W        # the change that survives bf16 rounding
            dn = float(np.linalg.norm(delta))
            stats.append({
                "name": name,
                "rel_change": dn / max(float(np.linalg.norm(W)), 1e-12),
                "kept": float(np.linalg.norm(got)) / max(dn, 1e-12),
                "cos": float((got * delta).sum()) / max(float(np.linalg.norm(got)) * dn, 1e-12),
                "changed_frac": float((got != 0).mean()),
            })
            if not a.dry_run:
                with open(dst, "r+b") as f:
                    a0, b0 = t["data_offsets"]
                    if len(new) != b0 - a0:
                        raise SystemExit(f"{name}: byte size changed")
                    f.seek(base + a0)
                    f.write(new)
        if not a.dry_run:
            h2, base2 = read_header(dst)
            if h2 != h or base2 != base:
                raise SystemExit(f"{fname}: header changed - abort")
            sums.append(f"{hashlib.sha256(open(dst, 'rb').read()).hexdigest()}  {fname}")
        print(f"  {fname}: {len(names)} tensors {'checked' if a.dry_run else 'merged'}")

    def q(key, f):
        v = sorted(s[key] for s in stats)
        return " ".join(f"{p}%={f(v[min(int(len(v) * p / 100), len(v) - 1)])}" for p in (0, 50, 100))
    print(f"relative change |dW|/|W|: {q('rel_change', lambda x: f'{x:.2e}')}")
    print(f"change kept after bf16 rounding (|got|/|delta|): {q('kept', lambda x: f'{x:.3f}')}")
    print(f"direction kept (cosine got vs delta):            {q('cos', lambda x: f'{x:.3f}')}")
    print(f"fraction of weights that changed:                 {q('changed_frac', lambda x: f'{x:.3f}')}")
    report = {"adapter_config": cfg, "scale": scale, "files": sorted(by_file), "tensors": stats}
    if not a.dry_run:
        # reports sit NEXT TO the model folder, so the loader never sees unknown files inside it
        (out.parent / f"{out.name}_MERGE_REPORT.json").write_text(json.dumps(report, indent=1))
        (out.parent / f"{out.name}_MERGED_SHA256SUMS").write_text("\n".join(sums) + "\n")
        print(f"done -> {out} (untouched files symlinked; original snapshot unchanged)")


if __name__ == "__main__":
    sys.exit(main())
