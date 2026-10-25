#!/usr/bin/env python3
# File: <project>\moe_track\mtp_align\11_merge_blk40_and_export.py   (WSL: <project>/moe_track/mtp_align/11_merge_blk40_and_export.py)
"""Turn the trained blk.40 LoRA into a draft-model GGUF for the server, without touching the 72 GB merged model or any existing file.

What it does (all in Python, no torch needed, nothing is run for you):
  1. reads the 19 MTP tensors of the original checkpoint and the adapter from 10_train_blk40_mtp_lora.py,
  2. merges W' = W + (alpha/r) * B @ A into the 8 adapted matrices (computed in float32, stored as bfloat16 like the checkpoint),
  3. writes a small Hugging Face-style folder that contains ONLY what llama.cpp's converter needs for an MTP-only export: the 19 MTP tensors (8 of them merged),
     the embedding table, the final norm and lm_head (byte-for-byte copies), plus config / tokenizer files,
  4. re-reads what it wrote and proves: the 8 merged tensors equal bf16(W + s*B@A) and every other tensor is byte-identical to the original,
  5. prints the three commands to convert, quantize and serve (you run them).

    python 11_merge_blk40_and_export.py --adapter blk40_lora_adapter [--out-dir DIR] [--force]        (folder default: mtp_align/hf_mtp_stuart_<timestamp>)
    python 11_merge_blk40_and_export.py --combine-spine DIR --adapter blk40_lora_adapter              (optional, extra: one adapter folder holding both sets of matrices)

Disk: about 3.7 GB for the folder (embedding table 1.0 GB + lm_head 1.0 GB + MTP block 1.7 GB), checked before writing. Never overwrites: the output folder must not exist.
The adapter folder name ending in _NOT_BETTER (the trainer's flag for "did not beat the untouched head") is refused unless --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import mtp_block_numpy as MN  # noqa: E402
import stensors as ST  # noqa: E402
import paths  # noqa: E402

HF_CACHE = Path("<project>/hf_cache")
INV_NAMES = {v: k for k, v in MN.HF_NAME_MAP.items()}
SKIP_COPY = {"model.safetensors.index.json", "README.md", ".gitattributes"}


def log(msg: str) -> None:
    print(f"[merge-blk40] {msg}", flush=True)


def find_snapshot(hf_cache: Path) -> Path:
    base = hf_cache / "models--unsloth--Qwen3.6-35B-A3B" / "snapshots"
    snaps = [p for p in base.iterdir() if p.is_dir() and (p / "model.safetensors.index.json").exists()] if base.exists() else []
    if not snaps:
        raise SystemExit(f"no complete snapshot under {base}")
    return sorted(snaps, key=lambda p: p.stat().st_mtime)[-1]


def read_adapter(adapter_dir: Path) -> Tuple[Dict, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("format") != "pelias-blk40-lora-v1":
        raise SystemExit(f"{adapter_dir}: not an adapter written by 10_train_blk40_mtp_lora.py (format {cfg.get('format')!r})")
    pairs = {}
    with ST.SafeFile(adapter_dir / "adapter_model.safetensors") as f:
        for tgt, path in cfg["targets"].items():
            base = f"base_model.model.{path}"
            pairs[tgt] = (f.get(f"{base}.lora_A.weight"), f.get(f"{base}.lora_B.weight"))
    return cfg, pairs


def merge_weight(w: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """float32 W + scale * B @ A, as bfloat16 bits."""
    if a.shape[1] != w.shape[1] or b.shape[0] != w.shape[0] or a.shape[0] != b.shape[1]:
        raise SystemExit(f"adapter shapes A{a.shape} B{b.shape} do not fit the weight {w.shape}")
    return ST.f32_to_bf16(w.astype(np.float32) + np.float32(scale) * (b.astype(np.float32) @ a.astype(np.float32)))


def plan_sources(snapshot: Path):
    """Which shard holds which tensor we need. Returns dict name -> shard file name, for the 19 MTP tensors, the embedding, the final norm and lm_head."""
    wm = ST.read_index(snapshot)
    need = {"mtp." + k for k in MN.HF_NAME_MAP}
    missing = sorted(need - set(wm))
    if missing:
        raise SystemExit(f"checkpoint lacks MTP tensors: {missing}")
    out = {k: wm[k] for k in need}
    for key in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"):
        if key not in wm:
            raise SystemExit(f"checkpoint lacks {key}")
        out[key] = wm[key]
    return out


def export_hf_folder(snapshot: Path, adapter_dir: Path, out_dir: Path, cfg: Dict, pairs: Dict, force: bool) -> Dict:
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; choose another --out-dir (this script never overwrites)")
    src = plan_sources(snapshot)
    scale = float(cfg.get("scale") or cfg["lora_alpha"] / cfg["r"])
    targets = {"mtp." + INV_NAMES[t]: t for t in pairs}                       # HF tensor name -> parameter key
    unknown = [t for t in pairs if t not in INV_NAMES]
    if unknown:
        raise SystemExit(f"adapter has unknown targets {unknown}")

    shards: Dict[str, ST.SafeFile] = {}
    try:
        for shard in sorted(set(src.values())):
            shards[shard] = ST.SafeFile(snapshot / shard)
        entries, report = [], {"scale": scale, "r": cfg["r"], "alpha": cfg["lora_alpha"], "merged": {}, "copied": []}
        total = 0
        for name in sorted(src):
            sf = shards[src[name]]
            if name in targets:
                dt, shape = sf.info(name)
                if dt != "BF16":
                    raise SystemExit(f"{name} is {dt}, expected BF16; the merge writes bf16 and would silently change precision")
                w = sf.get(name)
                a, b = pairs[targets[name]]
                bits = merge_weight(w, a, b, scale)
                delta = float(np.linalg.norm(ST.bf16_to_f32(bits) - w) / max(1e-12, float(np.linalg.norm(w))))
                report["merged"][name] = {"shape": list(shape), "relative_change": delta}
                blob = bits.tobytes()
                entries.append((name, "BF16", shape, len(blob), (lambda bb=blob: bb)))
                total += len(blob)
            else:
                e = ST.raw_entry(sf, name)
                entries.append(e)
                total += e[3]
                report["copied"].append(name)
        probe = out_dir.parent
        while not probe.exists() and probe != probe.parent:                       # the target folder's parents may not exist yet
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        if free < total + (2 << 30):
            raise SystemExit(f"not enough free disk space at {probe}: need {total / 2**30:.1f} GiB + 2 GiB margin, have {free / 2**30:.1f} GiB. Use --out-dir on a drive with room.")
        out_dir.mkdir(parents=True, exist_ok=False)
        shard_name = "model-mtp.safetensors"
        log(f"writing {total / 2**30:.2f} GiB to {out_dir / shard_name} ...")
        ST.save_stream(out_dir / shard_name, entries, metadata={"format": "pt", "pelias": "blk40-merged"})
        (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": {e[0]: shard_name for e in entries}}, indent=1), encoding="utf-8")
        for f in sorted(snapshot.iterdir()):
            if f.is_file() and f.suffix != ".safetensors" and f.name not in SKIP_COPY:
                shutil.copyfile(f, out_dir / f.name)                        # follows the cache's symlinks
        report["source_snapshot"] = snapshot.name
        report["adapter"] = str(adapter_dir)
        report["adapter_sha256"] = hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
        (out_dir / "MERGE_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        # ---- verification: re-read the new shard against the sources -----------------------------------------------------------------------------------
        log("verifying what was written ...")
        bad: List[str] = []
        with ST.SafeFile(out_dir / shard_name) as new:
            if sorted(new.keys()) != sorted(src):
                bad.append("tensor name set differs")
            for name in sorted(src):
                sf = shards[src[name]]
                if name in targets:
                    a, b = pairs[targets[name]]
                    want = merge_weight(sf.get(name), a, b, scale).tobytes()
                    if bytes(new.get_raw(name)) != want:
                        bad.append(f"{name}: merged bytes differ from W + s*B@A")
                else:
                    if bytes(new.get_raw(name)) != bytes(sf.get_raw(name)) or new.info(name) != sf.info(name):
                        bad.append(f"{name}: not identical to the original")
        if bad:
            raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad))
        log(f"verified: {len(targets)} merged tensors equal bf16(W + s*B@A); the other {len(src) - len(targets)} tensors are byte-identical to the checkpoint")
        return report
    finally:
        for sf in shards.values():
            sf.close()


def combine_spine(spine_dir: Path, blk40_dir: Path, out_dir: Path) -> Path:
    """One adapter folder with the Phase-2 spine adapter's tensors and the blk.40 tensors side by side (optional; the draft GGUF does not need it)."""
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists")
    sp, bk = spine_dir / "adapter_model.safetensors", blk40_dir / "adapter_model.safetensors"
    for f in (sp, bk, spine_dir / "adapter_config.json"):
        if not f.exists():
            raise SystemExit(f"missing {f}")
    with ST.SafeFile(sp) as a, ST.SafeFile(bk) as b:
        clash = sorted(set(a.keys()) & set(b.keys()))
        if clash:
            raise SystemExit(f"tensor names present in both adapters: {clash[:5]}")
        entries = [ST.raw_entry(a, k) for k in a.keys()] + [ST.raw_entry(b, k) for k in b.keys()]
        out_dir.mkdir(parents=True, exist_ok=False)
        ST.save_stream(out_dir / "adapter_model.safetensors", entries, metadata={"format": "pt"})
        n = len({k.split(".lora_")[0] for k in list(a.keys()) + list(b.keys()) if ".lora_" in k})
    cfg = json.loads((spine_dir / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["pelias_note"] = f"{n} modules: spine adapter + blk.40 MTP adapter (blk.40 scale = alpha/r from its own adapter_config.json)"
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "blk40_adapter_config.json").write_text((blk40_dir / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    log(f"wrote the combined adapter ({n} modules) to {out_dir}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="folder written by 10_train_blk40_mtp_lora.py")
    ap.add_argument("--hf-cache", default=str(HF_CACHE))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--force", action="store_true", help="allow an adapter that did not beat the untouched head")
    ap.add_argument("--combine-spine", default="", help="also write one adapter folder with this spine adapter + the blk.40 adapter")
    a = ap.parse_args()
    adapter = Path(a.adapter)
    if adapter.name.endswith("_NOT_BETTER") and not a.force:
        raise SystemExit(f"{adapter.name}: the trainer flagged this adapter as NOT better than the untouched head. Merging it would not help. --force overrides.")
    cfg, pairs = read_adapter(adapter)
    chosen = (cfg.get("heldout_metrics") or {}).get("chosen")
    if chosen not in ("best", None) and not a.force:
        raise SystemExit(f"adapter 'chosen' = {chosen}; use --force to merge anyway")
    snap = find_snapshot(Path(a.hf_cache))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out_dir) if a.out_dir else HERE / f"hf_mtp_stuart_{stamp}"
    log(f"base {snap.name}; adapter {adapter} (r={cfg['r']}, alpha={cfg['lora_alpha']}, {len(pairs)} modules)")
    hm = cfg.get("heldout_metrics") or {}
    if hm.get("before") and hm.get("after"):
        b_, a_ = hm["before"], hm["after"]
        log(f"hold-out acceptance estimate (sampling): step 1 {b_.get('overlap1', 0):.3f} -> {a_.get('overlap1', 0):.3f}" +
            (f", step 2 {b_.get('overlap2', 0):.3f} -> {a_.get('overlap2', 0):.3f}" if "overlap2" in a_ else ""))
    rep = export_hf_folder(snap, adapter, out_dir, cfg, pairs, a.force)
    if a.combine_spine:
        combine_spine(Path(a.combine_spine), adapter, out_dir.with_name(f"adapter_spine_plus_blk40_{stamp}"))
    gguf_dir = paths.CREATOR / "gguf"
    bf16 = gguf_dir / f"stuart-creator-mtp-{stamp}-BF16.gguf"
    q4 = gguf_dir / f"stuart-creator-mtp-{stamp}-Q4_K_M.gguf"
    base_dir = snap
    base_bf16 = gguf_dir / "stuart-creator-mtp-original-BF16.gguf"
    base_q4 = gguf_dir / "stuart-creator-mtp-original-Q4_K_M.gguf"
    print(f"""
[merge-blk40] DONE. Merged folder: {out_dir}
[merge-blk40] Next, in WSL (each line is separate; none of them overwrites anything - they stop if the target file exists):

  # 1) convert to a draft-only GGUF (uses your pinned llama.cpp converter; run it with the same venv you use for the other conversions)
  test ! -e {bf16} && PYTHONPATH={paths.LLAMA_PROD}/gguf-py python3 {paths.LLAMA_PROD}/convert_hf_to_gguf.py {out_dir} --mtp --outtype bf16 --outfile {bf16}

  # 2) quantize it (Q4_K_M is about 0.5 GB for the block; the embedding/output copies it carries add roughly 0.7 GB)
  test ! -e {q4} && {paths.LLAMA_PROD_BIN}/llama-quantize {bf16} {q4} Q4_K_M

  # 3) serve with it (needs the upgraded launcher from 12_upgrade_llama_launcher.sh)
  llama --serve --stuart-creator --bg --mtp --mtp-gguf {q4}

Then measure it (the server must be up):  python fork/5e_mtp_acceptance.py run --label mtp_aligned

To compare with the UNTOUCHED head, build the same kind of draft file from the original checkpoint, serve it the same way, and measure with --label mtp_original:
  test ! -e {base_bf16} && PYTHONPATH={paths.LLAMA_PROD}/gguf-py python3 {paths.LLAMA_PROD}/convert_hf_to_gguf.py {base_dir} --mtp --outtype bf16 --outfile {base_bf16}
  test ! -e {base_q4} && {paths.LLAMA_PROD_BIN}/llama-quantize {base_bf16} {base_q4} Q4_K_M
  llama --serve --stuart-creator --bg --mtp --mtp-gguf {base_q4}
  python fork/5e_mtp_acceptance.py run --label mtp_original
(5e_mtp_acceptance.py compares against its saved baseline on its own; the commands above are only needed for a like-for-like draft file.)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
