#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_merge_blk40.py
"""Tests mtp_align/11_merge_blk40_and_export.py on a small fake Hugging Face snapshot: the merge arithmetic, byte-identical copies, the written file set,
the "never overwrite" rules, the disk check, the NOT_BETTER guard and the optional combined adapter. Needs only numpy.

    python tests/test_merge_blk40.py
"""
from __future__ import annotations

import collections
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "mtp_align"))
import mtp_block_numpy as MN  # noqa: E402
import stensors as ST  # noqa: E402

spec = importlib.util.spec_from_file_location("merge11", HERE.parent / "mtp_align" / "11_merge_blk40_and_export.py")
MG = importlib.util.module_from_spec(spec)
sys.modules["merge11"] = MG
spec.loader.exec_module(MG)

FAILS = []
DU = collections.namedtuple("usage", "total used free")
D, HQ, FF = 32, 48, 16                      # tiny stand-in sizes; the script does not care about the real ones
TARGET_PATHS = {"fc": "mtp.fc", "q_proj": "mtp.layers.0.self_attn.q_proj", "k_proj": "mtp.layers.0.self_attn.k_proj", "v_proj": "mtp.layers.0.self_attn.v_proj",
                "o_proj": "mtp.layers.0.self_attn.o_proj", "shared_gate": "mtp.layers.0.mlp.shared_expert.gate_proj",
                "shared_up": "mtp.layers.0.mlp.shared_expert.up_proj", "shared_down": "mtp.layers.0.mlp.shared_expert.down_proj"}


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def bf16_bits(x):
    """Independent round-to-nearest-even float32 -> bf16 (not the stensors implementation)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def shape_of(key):
    return {"fc": (D, 2 * D), "q_proj": (HQ, D), "k_proj": (HQ // 2, D), "v_proj": (HQ // 2, D), "o_proj": (D, HQ // 2),
            "shared_gate": (FF, D), "shared_up": (FF, D), "shared_down": (D, FF), "router": (8, D), "gate_up": (8, 2 * FF, D), "down": (8, D, FF)}.get(key, (D,))


def make_snapshot(root: Path, q_dtype: str = "BF16") -> Path:
    snap = root / "models--unsloth--Qwen3.6-35B-A3B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    rng = np.random.default_rng(5)
    tensors, dtypes = {}, {}
    for hf, key in MN.HF_NAME_MAP.items():
        name = "mtp." + hf
        tensors[name] = (rng.standard_normal(shape_of(key)) * 0.1).astype(np.float32)
        dtypes[name] = "BF16"
    tensors["model.language_model.embed_tokens.weight"] = rng.standard_normal((40, D)).astype(np.float32)
    tensors["model.language_model.norm.weight"] = rng.standard_normal((D,)).astype(np.float32)
    tensors["lm_head.weight"] = rng.standard_normal((40, D)).astype(np.float32)
    for k in list(tensors):
        dtypes[k] = "BF16"
    if q_dtype != "BF16":
        dtypes["mtp.layers.0.self_attn.q_proj.weight"] = q_dtype
    # the bf16 values are what the checkpoint "really" holds
    stored = {k: ST.bf16_to_f32(bf16_bits(v)) if dtypes[k] == "BF16" else v for k, v in tensors.items()}
    names = sorted(stored)
    half = len(names) // 2
    wm = {}
    for i, group in enumerate((names[:half], names[half:])):
        shard = f"model-0000{i + 1}-of-00003.safetensors"
        ST.save_tensors(snap / shard, {k: stored[k] for k in group}, dtypes={k: dtypes[k] for k in group}, metadata={"format": "pt"})
        wm.update({k: shard for k in group})
    # a third shard with a tensor nobody needs
    ST.save_tensors(snap / "model-00003-of-00003.safetensors", {"model.language_model.layers.0.junk": np.zeros((4, 4), np.float32)}, dtypes={"model.language_model.layers.0.junk": "BF16"})
    wm["model.language_model.layers.0.junk"] = "model-00003-of-00003.safetensors"
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    (snap / "config.json").write_text('{"model_type": "qwen3_5_moe"}', encoding="utf-8")
    (snap / "tokenizer.json").write_text('{"fake": true}', encoding="utf-8")
    (snap / "README.md").write_text("not copied", encoding="utf-8")
    return snap


def make_adapter(root: Path, name: str = "blk40_lora_adapter", r: int = 4, alpha: float = 8.0, seed: int = 9, fmt: str = "pelias-blk40-lora-v1") -> Path:
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    tensors = {}
    for key, path in TARGET_PATHS.items():
        o, i = shape_of(key)
        tensors[f"base_model.model.{path}.lora_A.weight"] = (rng.standard_normal((r, i)) * 0.3).astype(np.float32)
        tensors[f"base_model.model.{path}.lora_B.weight"] = (rng.standard_normal((o, r)) * 0.3).astype(np.float32)
    ST.save_tensors(d / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    cfg = {"format": fmt, "r": r, "lora_alpha": alpha, "scale": alpha / r, "targets": TARGET_PATHS, "heldout_metrics": {"chosen": "best"}}
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def run_main(argv):
    old = sys.argv
    sys.argv = ["11_merge_blk40_and_export.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = MG.main()
        return rc, buf.getvalue(), None
    except SystemExit as e:
        return None, buf.getvalue(), str(e)
    finally:
        sys.argv = old


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cache = td / "cache"
        snap = make_snapshot(cache)
        ad = make_adapter(td)
        out = td / "out" / "hf_mtp"
        (td / "out").mkdir()

        rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache), "--out-dir", str(out)])
        check("merge runs", rc == 0 and err is None, f"{err}\n{text[-400:]}")
        check("the folder holds the shard, index, report and config files, not the README or the original shards",
              sorted(p.name for p in out.iterdir()) == ["MERGE_REPORT.json", "config.json", "model-mtp.safetensors", "model.safetensors.index.json", "tokenizer.json"],
              str(sorted(p.name for p in out.iterdir())))
        check("it prints the convert / quantize / serve commands", "convert_hf_to_gguf.py" in text and "llama-quantize" in text and "--mtp-gguf" in text and "5e_mtp_acceptance.py run --label mtp_aligned" in text)
        check("the commands never write to an existing file (each starts with test ! -e)", text.count("test ! -e") >= 4)

        with ST.SafeFile(out / "model-mtp.safetensors") as new, ST.SafeFile(snap / "model-00001-of-00003.safetensors") as s1, ST.SafeFile(snap / "model-00002-of-00003.safetensors") as s2:
            def src(name):
                return s1 if name in s1.keys() else s2
            expect_names = {"mtp." + k for k in MN.HF_NAME_MAP} | {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"}
            check("exactly the 19 MTP tensors + embedding + final norm + lm_head (22) and nothing else", set(new.keys()) == expect_names and len(expect_names) == 22)
            scale = 8.0 / 4
            good_merge, good_copy = True, True
            for key, path in TARGET_PATHS.items():
                name = path + ".weight"
                with ST.SafeFile(ad / "adapter_model.safetensors") as af:
                    A = af.get(f"base_model.model.{path}.lora_A.weight")
                    B = af.get(f"base_model.model.{path}.lora_B.weight")
                W = src(name).get(name)
                want = bf16_bits(W + np.float32(scale) * (B @ A)).tobytes()
                good_merge &= bytes(new.get_raw(name)) == want
                good_merge &= new.info(name) == ("BF16", shape_of(key))
            for name in sorted(expect_names - {p + ".weight" for p in TARGET_PATHS.values()}):
                good_copy &= bytes(new.get_raw(name)) == bytes(src(name).get_raw(name)) and new.info(name) == src(name).info(name)
            check("the 8 adapted matrices equal bf16(W + (alpha/r) B@A) computed independently", good_merge)
            check("the other 14 tensors are byte-identical to the checkpoint (norms, router, experts, shared gate, embedding, lm_head)", good_copy)
        idx = json.loads((out / "model.safetensors.index.json").read_text())
        check("index lists every tensor in the single shard", set(idx["weight_map"]) == expect_names and set(idx["weight_map"].values()) == {"model-mtp.safetensors"})
        rep = json.loads((out / "MERGE_REPORT.json").read_text())
        check("report: 8 merged with a non-zero relative change, 14 copied, adapter hash recorded",
              len(rep["merged"]) == 8 and len(rep["copied"]) == 14 and all(v["relative_change"] > 0 for v in rep["merged"].values()) and len(rep["adapter_sha256"]) == 64)
        check("config / tokenizer files were copied byte for byte", (out / "config.json").read_bytes() == (snap / "config.json").read_bytes() and (out / "tokenizer.json").exists())

        # ---- never overwrites --------------------------------------------------------------------------------------------------------------------------
        before = (out / "model-mtp.safetensors").read_bytes()
        rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache), "--out-dir", str(out)])
        check("a second run into the same folder is refused and changes nothing", rc is None and "already exists" in (err or "") and (out / "model-mtp.safetensors").read_bytes() == before)

        # ---- guards ------------------------------------------------------------------------------------------------------------------------------------
        nb = make_adapter(td, "blk40_lora_adapter_NOT_BETTER", seed=3)
        rc, text, err = run_main(["--adapter", str(nb), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "nb")])
        check("an adapter flagged _NOT_BETTER is refused", rc is None and "NOT better" in (err or "") and not (td / "out" / "nb").exists())
        rc, text, err = run_main(["--adapter", str(nb), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "nb"), "--force"])
        check("--force lets it through", rc == 0 and (td / "out" / "nb" / "model-mtp.safetensors").exists())
        bad_fmt = make_adapter(td, "other_adapter", fmt="peft-lora")
        rc, text, err = run_main(["--adapter", str(bad_fmt), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "bf")])
        check("an adapter from another tool is refused", rc is None and "not an adapter written by" in (err or ""))
        wrongr = make_adapter(td, "wrong_shape", r=4)
        cfgp = wrongr / "adapter_config.json"
        c = json.loads(cfgp.read_text())
        c["targets"]["fc"] = "mtp.layers.0.self_attn.k_proj"            # points fc at tensors of another shape
        cfgp.write_text(json.dumps(c))
        rc, text, err = run_main(["--adapter", str(wrongr), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "ws")])
        check("adapter shapes that do not fit the weight are refused, nothing is left behind", rc is None and ("do not fit" in (err or "") or "differs" in (err or "")) and not (td / "out" / "ws" / "model-mtp.safetensors").exists(), str(err))

        # ---- a non-bf16 target ---------------------------------------------------------------------------------------------------------------------------
        cache2 = td / "cache2"
        make_snapshot(cache2, q_dtype="F32")
        rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache2), "--out-dir", str(td / "out" / "f32")])
        check("a checkpoint tensor that is not bf16 stops the merge (no silent precision change)", rc is None and "expected BF16" in (err or ""))

        # ---- disk check ----------------------------------------------------------------------------------------------------------------------------------
        real = shutil.disk_usage
        shutil.disk_usage = lambda p: DU(10 << 30, (10 << 30) - (1 << 20), 1 << 20)       # 1 MiB free
        try:
            rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "full")])
        finally:
            shutil.disk_usage = real
        check("too little free disk space is reported before anything is written", rc is None and "not enough free disk space" in (err or "") and not (td / "out" / "full").exists())

        # ---- combined adapter ------------------------------------------------------------------------------------------------------------------------------
        spine = td / "spine"
        spine.mkdir()
        sp = {f"base_model.model.model.layers.0.self_attn.q_proj.lora_{x}.weight": np.ones((4, 8), np.float32) * (1 if x == "A" else 2) for x in "AB"}
        ST.save_tensors(spine / "adapter_model.safetensors", sp, metadata={"format": "pt"})
        (spine / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "peft_type": "LORA"}), encoding="utf-8")
        rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "comb" / "hf"), "--combine-spine", str(spine)])
        check("--combine-spine runs", rc == 0, f"{err}")
        combos = list((td / "out" / "comb").glob("adapter_spine_plus_blk40_*"))
        ok = False
        if combos:
            with ST.SafeFile(combos[0] / "adapter_model.safetensors") as f:
                ks = set(f.keys())
                ok = len(ks) == 2 + 16 and all(bytes(f.get_raw(k)) == bytes(ST.SafeFile(spine / "adapter_model.safetensors").get_raw(k)) for k in sp)
        check("combined adapter holds the spine tensors and the 16 blk.40 tensors, spine bytes unchanged", ok)
        check("combined adapter keeps the blk.40 config next to it", bool(combos) and (combos[0] / "blk40_adapter_config.json").exists() and "9 modules" in json.loads((combos[0] / "adapter_config.json").read_text())["pelias_note"])
        clash = td / "clash"
        clash.mkdir()
        ST.save_tensors(clash / "adapter_model.safetensors", {f"base_model.model.{TARGET_PATHS['fc']}.lora_A.weight": np.ones((4, 64), np.float32)}, metadata={"format": "pt"})
        (clash / "adapter_config.json").write_text("{}", encoding="utf-8")
        rc, text, err = run_main(["--adapter", str(ad), "--hf-cache", str(cache), "--out-dir", str(td / "out" / "clash" / "hf"), "--combine-spine", str(clash)])
        check("clashing tensor names between the two adapters are refused", rc is None and "present in both" in (err or ""))

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
