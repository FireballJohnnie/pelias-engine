#!/usr/bin/env python3
# File: <project>\moe_track\tests\test_trainer_io.py
"""Tests the bookkeeping of mtp_align/10_train_blk40_mtp_lora.py on the numpy-backed torch stand-in: sample export, reading the hidden-state dump, the one-step /
two-step index shifts (the part most likely to be off by one), adapter files, and the "never overwrite" naming.

    python tests/test_trainer_io.py
"""
from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mtp_align"))
import fake_torch  # noqa: E402

torch = fake_torch.install()
import mtp_block_numpy as MN  # noqa: E402
import mtp_torch as MT  # noqa: E402

spec = importlib.util.spec_from_file_location("trainer10", HERE.parent / "mtp_align" / "10_train_blk40_mtp_lora.py")
TR = importlib.util.module_from_spec(spec)
sys.modules["trainer10"] = TR
spec.loader.exec_module(TR)

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main() -> int:
    tt = torch.from_numpy
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---- export-samples with a fake train.pt -------------------------------------------------------------------------------------------------------
        rng = np.random.default_rng(1)

        def mk(i, split, n):
            ids = [int(x) for x in rng.integers(10, 900, size=n)]
            lab = [-100] * (n // 3) + ids[n // 3:]
            return {"input_ids": ids, "labels": lab, "n_assistant": n - n // 3, "split": split, "idx": i}

        data = {"samples": [mk(i, "train", 40 + i) for i in range(30)] + [mk(100 + i, "eval", 50 + i) for i in range(3)] + [mk(200, "train", 5)]}
        torch.load = lambda path, **k: data
        TR.SAMPLES_IDS, TR.SAMPLES_META = td / "mtp_samples.ids", td / "mtp_samples_meta.json"
        a = types.SimpleNamespace(train_pt=str(td / "train.pt"), n_train=20, n_hold=5, max_len=48, seed=11)
        (td / "train.pt").write_bytes(b"x")
        check("export-samples runs", TR.cmd_export_samples(a) == 0)
        meta = __import__("json").loads(TR.SAMPLES_META.read_text())
        check("5 hold-out + 20 training samples", meta["split"].count("hold") == 5 and meta["split"].count("train") == 20)
        check("cropped to max_len", max(len(x) for x in meta["ids"]) <= 48)
        check("the too-short sample (5 tokens) was dropped", all(len(x) >= 16 for x in meta["ids"]))
        hold_ids = {tuple(meta["ids"][i]) for i, s in enumerate(meta["split"]) if s == "hold"}
        train_ids = {tuple(meta["ids"][i]) for i, s in enumerate(meta["split"]) if s == "train"}
        check("hold-out and training samples are disjoint", not (hold_ids & train_ids))
        eval_rows = {tuple(x["input_ids"][:48]) for x in data["samples"] if x["split"] == "eval"}
        check("the 3 eval rows are in the hold-out set (2 more were taken from the training rows)", eval_rows <= hold_ids and len(hold_ids) == 5)
        raw = TR.SAMPLES_IDS.read_bytes()
        magic, n = struct.unpack("<ii", raw[:8])
        pos, parsed = 8, []
        for _ in range(n):
            (m,) = struct.unpack("<i", raw[pos:pos + 4])
            parsed.append(np.frombuffer(raw[pos + 4:pos + 4 + 4 * m], dtype=np.int32).tolist())
            pos += 4 + 4 * m
        check("ids file: magic, count, contents equal the meta file", magic == TR.MAGIC_IDS and n == 25 and parsed == meta["ids"] and pos == len(raw))
        a2 = types.SimpleNamespace(**vars(a))
        TR.cmd_export_samples(a2)
        check("a second export does not overwrite", (td / "mtp_samples.ids").exists() and len(list(td.glob("mtp_samples_*.ids"))) == 1)

        # ---- hidden dump reader ------------------------------------------------------------------------------------------------------------------------
        D = 64
        lens = [7, 12, 9]
        Hs_f = [rng.standard_normal((n, D)).astype(np.float32) for n in lens]
        with open(td / "h.bin", "wb") as f:
            f.write(struct.pack("<3i", TR.MAGIC_HID, D, len(lens)))
            for n, H in zip(lens, Hs_f):
                f.write(struct.pack("<i", n))
                f.write(bf16_bits(H).tobytes())
        nemb, idx, mm = TR.read_hidden_index(td / "h.bin")
        check("hidden index: n_embd and sample lengths", nemb == D and [n for _, n in idx] == lens)
        ok = all(np.array_equal(TR.bf16_rows(mm, off, n, D), (bf16_bits(H).astype(np.uint32) << 16).view(np.float32)) for (off, n), H in zip(idx, Hs_f))
        check("hidden rows read back exactly (bf16 values)", ok)
        (td / "trunc.bin").write_bytes((td / "h.bin").read_bytes()[:-10])
        try:
            TR.read_hidden_index(td / "trunc.bin")
            check("a truncated dump is refused", False)
        except SystemExit:
            check("a truncated dump is refused", True)

        # ---- the index shifts of a sample, against a plain numpy computation --------------------------------------------------------------------------
        cfg = MN.Config(hidden=D, n_heads=4, n_kv_heads=2, head_dim=16, rot_dim=8, rope_base=1.0e4, n_experts=8, top_k=2, ff_exp=16, ff_shared=16)
        p = MN.random_params(cfg, seed=41, scale=0.08)
        head = MT.MTPHead({k: tt(v.astype(np.float32)) for k, v in p.items()}, cfg, r=4, alpha=8.0, seed=2)
        rg = np.random.default_rng(42)
        for nme, m in head.lin.items():
            m.lora_A.data[...] = rg.standard_normal(m.lora_A.shape).astype(np.float32) * 0.2
            m.lora_B.data[...] = rg.standard_normal(m.lora_B.shape).astype(np.float32) * 0.2
        V, T = 37, 14
        W = (rg.standard_normal((V, D)) * 0.3).astype(np.float32)
        H = rg.standard_normal((T, D)).astype(np.float32)
        emb = (rg.standard_normal((V, D)) * 0.5).astype(np.float32)
        ids = rg.integers(0, V, size=T)
        labels = [-100 if (i < 5 or i in (8, 9)) else int(ids[i]) for i in range(T)]
        args = types.SimpleNamespace(ce_chunk=3, ce_weight=0.1, steps=2, step2_weight=0.5)
        res = TR.sample_pass(torch, head, tt(W), tt(H), tt(emb[ids]), torch.tensor(ids.astype(np.int64)), labels, args, False)

        merged = {k: v.copy() for k, v in p.items()}
        for nme, m in head.lin.items():
            merged[nme] = merged[nme] + m.scale * (m.lora_B.data @ m.lora_A.data)
        P = T - 2
        E = emb[ids]
        O1, O2 = MN.mtp_forward_two_step(merged, cfg, H[:P], E[1:P + 1], E[2:P + 2], np.arange(1, P + 1))

        def lsm(z):
            z = z - z.max(-1, keepdims=True)
            return z - np.log(np.exp(z).sum(-1, keepdims=True))

        def stats(Oq, teacher_h, y):
            ls, lt = lsm(Oq @ W.T), lsm(teacher_h @ W.T)
            pt = np.exp(lt)
            return np.stack([(pt * (lt - ls)).sum(-1), -ls[np.arange(len(y)), y], ((Oq @ W.T).argmax(-1) == (teacher_h @ W.T).argmax(-1)).astype(float),
                             np.minimum(pt, np.exp(ls)).sum(-1)], -1)

        sel1 = [t for t in range(P) if labels[t + 2] != -100]
        ref1 = stats(O1[sel1], H[1:P + 1][sel1], ids[2:P + 2][sel1]).sum(0)
        Pm = P - 1
        sel2 = [t for t in range(Pm) if labels[t + 3] != -100]
        ref2 = stats(O2[:Pm][sel2], H[2:P + 1][sel2], ids[3:P + 2][sel2]).sum(0)
        got1, got2 = res["s1"].data, res["s2"].data
        check("step 1 counts and statistics match the plain computation (inputs H[t], token t+1; teacher H[t+1]; target token t+2)",
              res["n1"] == len(sel1) and np.allclose(got1, ref1, atol=2e-3), f"{got1} vs {ref1}")
        check("step 2 counts and statistics match (token t+2, own output, teacher H[t+2], target token t+3)",
              res["n2"] == len(sel2) and np.allclose(got2, ref2, atol=2e-3), f"{got2} vs {ref2}")
        check("loss is the weighted sum of both steps",
              abs(float(res["loss"].data) - ((got1[0] + 0.1 * got1[1]) / len(sel1) + 0.5 * (got2[0] + 0.1 * got2[1]) / len(sel2))) < 1e-4)
        args1 = types.SimpleNamespace(**{**vars(args), "steps": 1})
        res1 = TR.sample_pass(torch, head, tt(W), tt(H), tt(emb[ids]), torch.tensor(ids.astype(np.int64)), labels, args1, False)
        check("steps=1 skips the second step", res1["s2"] is None and res1["n2"] == 0)
        tiny = TR.sample_pass(torch, head, tt(W), tt(H[:4]), tt(emb[ids[:4]]), torch.tensor(ids[:4].astype(np.int64)), labels[:4], args, False)
        check("a sample with fewer than 5 positions is skipped", tiny is None)

        meter = TR.Meter()
        meter.add(res)
        sm = meter.summary()
        check("meter averages per token", abs(sm["overlap1"] - got1[3] / len(sel1)) < 1e-6 and abs(sm["kl2"] - got2[0] / len(sel2)) < 1e-6)

        # ---- adapter files ------------------------------------------------------------------------------------------------------------------------------
        class A:
            lora_r, lora_alpha, epochs, lr, steps, step2_weight, ce_weight, accum, seed = 4, 8.0, 1, 1e-3, 2, 0.5, 0.1, 1, 0
        out1 = TR.save_adapter(head, td / "adapter", A, Path("snap"), {"before": {"x": np.float32(1.0)}}, [])
        out2 = TR.save_adapter(head, td / "adapter", A, Path("snap"), {}, [])
        check("a second save goes to a new folder", out1 != out2 and out1.exists() and out2.exists())
        fresh = MT.MTPHead({k: tt(v.astype(np.float32)) for k, v in p.items()}, cfg, r=4, alpha=8.0, seed=77)
        fresh.load_lora_state(TR.load_adapter_state(out1, torch))
        a_, _, _ = head.step1(tt(H[:6]), tt(E[1:7]), torch.arange(6))
        b_, _, _ = fresh.step1(tt(H[:6]), tt(E[1:7]), torch.arange(6))
        check("adapter on disk reproduces the model", np.allclose(a_.data, b_.data, atol=1e-6))
        check("module paths match the checkpoint tensor names",
              [TR.adapter_module_path(t) for t in ("fc", "q_proj", "shared_down")] == ["mtp.fc", "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.mlp.shared_expert.down_proj"])

    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
