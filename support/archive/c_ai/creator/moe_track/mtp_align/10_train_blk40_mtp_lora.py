#!/usr/bin/env python3
# File: <project>\moe_track\mtp_align\10_train_blk40_mtp_lora.py   (WSL: <project>/moe_track/mtp_align/10_train_blk40_mtp_lora.py)
"""Teach the MTP draft head (blk.40) to guess what the Stuart model will say, so the server accepts more drafted tokens.

The spine (layers 0..39) stays exactly as it is. Only a small LoRA on the head's own attention / fc / shared-expert matrices is trained, on a 12 GB card, in minutes.

WHAT IT LEARNS: at every position t the head sees (the model's hidden state at t, the token that came next) and has to produce the same word distribution the model itself
produces one step later. That is exactly what decides whether a drafted token is accepted, so the loss is a KL between the head and the model (plus a small cross-entropy on the real text).
The model's own hidden states come from the finished, merged Stuart GGUF through dump_hidden.cpp (stage 10a), so the head trains on what the server will really feed it.

Four commands (run them in this order; nothing here touches the production server or your model files):

  python 10_train_blk40_mtp_lora.py selftest         checks this code against an independent numpy implementation on random weights (CPU, ~1 min; run it first)
  python 10_train_blk40_mtp_lora.py export-samples   picks training text from your Pelias train.pt  ->  mtp_samples.ids + mtp_samples_meta.json
  bash 10a_dump_hidden.sh                            (stage 10a; needs the server stopped) runs the Stuart GGUF over those samples -> mtp_hidden.bin
  python 10_train_blk40_mtp_lora.py train            trains the LoRA, prints held-out acceptance before/after, writes blk40_lora_adapter/
  python 10_train_blk40_mtp_lora.py eval [--adapter DIR]    held-out numbers for the untouched head (or for an adapter)

Then 11_merge_blk40_and_export.py turns the adapter into a draft GGUF for `llama --mtp --mtp-gguf FILE`.

Numbers it reports on the held-out samples (assistant tokens only), for draft step 1 and step 2:
  overlap   expected acceptance probability when the server samples (temperature 1.0): the sum over tokens of min(p_model, p_head). THIS is the number to watch.
  agree     how often the head's top token equals the model's top token (the acceptance rate for greedy decoding).
  kl, ce    the two losses.
Reads: the HF snapshot (only the 19 MTP tensors, the embedding table and lm_head), train.pt, the hidden-state dump. Writes only inside mtp_align/ (never overwrites).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))

import mtp_block_numpy as MN  # noqa: E402
import stensors as ST  # noqa: E402

HOME = Path.home()
ENGINE = Path(os.environ.get("STUART_ENGINE", str(HOME / "stuart_creator_engine")))
HF_CACHE = Path(os.environ.get("HF_CACHE_DIR", "<project>/hf_cache"))
TRAIN_PT = Path(os.environ.get("TRAIN_PT", str(ENGINE / "pelias_data" / "train.pt")))
SAMPLES_IDS = HERE / "mtp_samples.ids"
SAMPLES_META = HERE / "mtp_samples_meta.json"
HIDDEN_BIN = HERE / "mtp_hidden.bin"
ADAPTER_OUT = HERE / "blk40_lora_adapter"
MAGIC_IDS, MAGIC_HID = 0x50534944, 0x50484944
INV_NAMES = {v: k for k, v in MN.HF_NAME_MAP.items()}          # parameter key -> "fc.weight", "layers.0.self_attn.q_proj.weight", ...


def log(msg: str) -> None:
    print(f"[blk40] {msg}", flush=True)


def fresh_path(p: Path) -> Path:
    """p if it does not exist, else p with a timestamp (never overwrite)."""
    if not p.exists():
        return p
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.name}_{stamp}") if p.suffix == "" else p.with_name(f"{p.stem}_{stamp}{p.suffix}")


def find_snapshot(hf_cache: Path) -> Path:
    base = hf_cache / "models--unsloth--Qwen3.6-35B-A3B" / "snapshots"
    snaps = [p for p in base.iterdir() if p.is_dir() and (p / "model.safetensors.index.json").exists()] if base.exists() else []
    if not snaps:
        raise SystemExit(f"no complete snapshot under {base}")
    return sorted(snaps, key=lambda p: p.stat().st_mtime)[-1]


def adapter_module_path(target: str) -> str:
    return "mtp." + INV_NAMES[target][: -len(".weight")]


# ---- loading the head ---------------------------------------------------------------------------------------------------------------------------------------
def load_mtp_params_torch(snapshot: Path, torch, dtype):
    """Same, as torch tensors in `dtype` (bfloat16 on the checkpoint: loaded without a float32 detour)."""
    wm = ST.read_index(snapshot)
    want = {"mtp." + k: v for k, v in MN.HF_NAME_MAP.items()}
    missing = sorted(set(want) - set(wm))
    if missing:
        raise SystemExit(f"checkpoint lacks MTP tensors: {missing}")
    out, shapes = {}, {}
    by_shard: Dict[str, List[str]] = {}
    for k in want:
        by_shard.setdefault(wm[k], []).append(k)
    for shard, keys in sorted(by_shard.items()):
        with ST.SafeFile(snapshot / shard) as f:
            for k in sorted(keys):
                out[want[k]] = tensor_from_file(f, k, torch).to(dtype)
                shapes[want[k]] = tuple(out[want[k]].shape)
    return out, shapes


def tensor_from_file(f, name: str, torch):
    dt, shape = f.info(name)
    tdt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}.get(dt)
    if tdt is None:
        raise SystemExit(f"{name}: unsupported dtype {dt}")
    raw = bytearray(f.get_raw(name))
    return torch.frombuffer(raw, dtype=tdt).reshape(shape)       # shares the bytearray, which the tensor keeps alive


def load_single(snapshot: Path, key_suffix: str, torch, dtype=None):
    wm = ST.read_index(snapshot)
    keys = [k for k in wm if k.endswith(key_suffix) and "visual" not in k and not k.startswith("mtp.")]
    if not keys:
        return None
    k = sorted(keys, key=len)[0]
    with ST.SafeFile(snapshot / wm[k]) as f:
        t = tensor_from_file(f, k, torch)
    return t.to(dtype) if dtype is not None else t


# ---- sample / hidden-state files ----------------------------------------------------------------------------------------------------------------------------
def cmd_export_samples(a) -> int:
    import torch
    if not Path(a.train_pt).exists():
        raise SystemExit(f"train data not found: {a.train_pt}  (set --train-pt or STUART_ENGINE)")
    data = torch.load(str(a.train_pt), map_location="cpu", weights_only=False)      # your own file; holds plain Python lists and dicts
    samples = data["samples"]
    tr = [s for s in samples if s.get("split") == "train"]
    ev = [s for s in samples if s.get("split") == "eval"]
    rng = random.Random(a.seed)
    rng.shuffle(tr)
    rng.shuffle(ev)

    def usable(s):
        ids, lab = s["input_ids"][: a.max_len], s["labels"][: a.max_len]
        return len(ids) >= 16 and sum(1 for x in lab[2:] if x != -100) >= 8

    tr = [s for s in tr if usable(s)]
    ev = [s for s in ev if usable(s)]
    hold = ev[: a.n_hold]
    if len(hold) < a.n_hold:                                   # not enough eval rows: take the rest from the train rows (kept out of training)
        need = a.n_hold - len(hold)
        hold += tr[:need]
        tr = tr[need:]
    train = tr[: a.n_train]
    if not train or not hold:
        raise SystemExit(f"not enough usable samples (train {len(train)}, hold-out {len(hold)})")
    rows = [(s, "train") for s in train] + [(s, "hold") for s in hold]
    ids_l = [list(map(int, s["input_ids"][: a.max_len])) for s, _ in rows]
    lab_l = [list(map(int, s["labels"][: a.max_len])) for s, _ in rows]
    if SAMPLES_IDS.exists() or SAMPLES_META.exists():
        SAMPLES_IDS_NEW, SAMPLES_META_NEW = fresh_path(SAMPLES_IDS), fresh_path(SAMPLES_META)
        log(f"existing sample files found; writing new ones: {SAMPLES_IDS_NEW.name}, {SAMPLES_META_NEW.name}")
    else:
        SAMPLES_IDS_NEW, SAMPLES_META_NEW = SAMPLES_IDS, SAMPLES_META
    with open(SAMPLES_IDS_NEW, "wb") as f:
        f.write(struct.pack("<ii", MAGIC_IDS, len(ids_l)))
        for ids in ids_l:
            f.write(struct.pack("<i", len(ids)))
            f.write(np.asarray(ids, dtype=np.int32).tobytes())
    SAMPLES_META_NEW.write_text(json.dumps({"source": str(a.train_pt), "max_len": a.max_len, "split": [r[1] for r in rows], "ids": ids_l, "labels": lab_l,
                                            "ids_file": SAMPLES_IDS_NEW.name}), encoding="utf-8")
    n_pos = sum(len(x) for x in ids_l)
    n_asst = sum(1 for lab in lab_l for x in lab if x != -100)
    log(f"wrote {SAMPLES_IDS_NEW.name} and {SAMPLES_META_NEW.name}: {len(train)} training + {len(hold)} hold-out samples, {n_pos} positions ({n_asst} assistant tokens)")
    log(f"next (server must be stopped): bash {HERE / '10a_dump_hidden.sh'}      ~{n_pos / 1100 / 60:.0f}-{n_pos / 500 / 60:.0f} minutes at prompt speed")
    return 0


def read_hidden_index(path: Path):
    """Memory-map the hidden-state dump. Returns (n_embd, [(offset_in_uint16_units, n_tok)], memmap)."""
    with open(path, "rb") as f:
        magic, n_embd, n_samples = struct.unpack("<3i", f.read(12))
        if magic != MAGIC_HID:
            raise SystemExit(f"{path} is not a hidden-state dump (magic {magic:#x})")
    size = path.stat().st_size
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    pos = 6                                                                  # uint16 units: 12 bytes of header
    idx = []
    for _ in range(n_samples):
        n = int(mm[pos:pos + 2].view(np.int32)[0])
        pos += 2
        idx.append((pos, n))
        pos += n * n_embd
    if pos * 2 != size:
        raise SystemExit(f"{path}: size does not match its own header (truncated dump?)")
    return n_embd, idx, mm


def bf16_rows(mm, off: int, n: int, d: int) -> np.ndarray:
    u = np.asarray(mm[off: off + n * d]).astype(np.uint32) << 16
    return u.view(np.float32).reshape(n, d)


# ---- one sample ---------------------------------------------------------------------------------------------------------------------------------------------
def sample_pass(torch, head, w_head, Hs, Es, ids_t, labels: List[int], args, train: bool):
    """Both draft steps over one sample. Returns {'s1': stats[4], 'n1': int, 's2': ..., 'n2': ..., 'loss': tensor or None}; stats = sums of (kl, ce, agree, overlap)."""
    from mtp_torch import run_chunks
    T = Hs.shape[0]
    P = T - 2
    if P < 3:
        return None
    dev = Hs.device
    pos = torch.arange(1, P + 1, device=dev)
    O1, K1, V1 = head.step1(Hs[:P], Es[1:P + 1], pos)
    res = {"n1": 0, "n2": 0, "s1": None, "s2": None, "loss": None}
    loss = None
    idx1 = [i for i in range(P) if labels[i + 2] != -100]
    if idx1:
        it = torch.tensor(idx1, device=dev)
        s1 = run_chunks(O1[it], Hs[1:P + 1][it], ids_t[2:P + 2][it], w_head, args.ce_chunk, train)
        res["s1"], res["n1"] = s1, len(idx1)
        loss = (s1[0] + args.ce_weight * s1[1]) / len(idx1)
    if args.steps >= 2:
        Pm = P - 1
        O2 = head.step2(O1[:Pm], Es[2:P + 1], pos[:Pm], K1[:Pm], V1[:Pm])
        idx2 = [i for i in range(Pm) if labels[i + 3] != -100]
        if idx2:
            it2 = torch.tensor(idx2, device=dev)
            s2 = run_chunks(O2[it2], Hs[2:P + 1][it2], ids_t[3:P + 2][it2], w_head, args.ce_chunk, train)
            res["s2"], res["n2"] = s2, len(idx2)
            l2 = args.step2_weight * (s2[0] + args.ce_weight * s2[1]) / len(idx2)
            loss = l2 if loss is None else loss + l2
    res["loss"] = loss
    return res


class Meter:
    def __init__(self):
        self.s = {1: np.zeros(4), 2: np.zeros(4)}
        self.n = {1: 0, 2: 0}

    def add(self, res):
        for k in (1, 2):
            if res[f"s{k}"] is not None:
                self.s[k] += res[f"s{k}"].detach().float().cpu().numpy().astype(np.float64)
                self.n[k] += res[f"n{k}"]

    def summary(self) -> Dict[str, float]:
        out = {}
        for k in (1, 2):
            if self.n[k]:
                m = self.s[k] / self.n[k]
                out.update({f"kl{k}": m[0], f"ce{k}": m[1], f"agree{k}": m[2], f"overlap{k}": m[3], f"n{k}": self.n[k]})
        return out


def fmt(m: Dict[str, float]) -> str:
    parts = []
    for k in (1, 2):
        if f"overlap{k}" in m:
            parts.append(f"step{k}: overlap {m[f'overlap{k}']:.3f}  agree {m[f'agree{k}']:.3f}  kl {m[f'kl{k}']:.3f}  ce {m[f'ce{k}']:.3f}  (n={int(m[f'n{k}'])})")
    return " | ".join(parts)


def score_of(m: Dict[str, float]) -> float:
    return m.get("overlap1", 0.0) + 0.5 * m.get("overlap2", 0.0)


# ---- session setup shared by train / eval -------------------------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, a, need_hidden=True):
        import torch
        from mtp_torch import MTPHead, hf_cfg_to_config
        self.torch = torch
        if not torch.cuda.is_available() and not a.cpu:
            raise SystemExit("torch does not see a CUDA GPU here. Run this inside the WSL venv that has the CUDA build of torch (the one the Pelias trainer uses); "
                             "--cpu forces the CPU, which takes many hours.")
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() and not a.cpu else "cpu")
        self.dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
        self.snap = find_snapshot(Path(a.hf_cache))
        log(f"snapshot {self.snap.name}; device {self.dev}; compute dtype {str(self.dtype).replace('torch.', '')}")
        params, shapes = load_mtp_params_torch(self.snap, torch, self.dtype)
        hf = json.loads((self.snap / "config.json").read_text(encoding="utf-8"))
        self.cfg, src = hf_cfg_to_config(hf, shapes)
        defaulted = [k for k, v in src.items() if v == "default"]
        log(f"head: hidden {self.cfg.hidden}, heads {self.cfg.n_heads}/{self.cfg.n_kv_heads}, head_dim {self.cfg.head_dim}, rot {self.cfg.rot_dim}, rope_base {self.cfg.rope_base:g}, "
            f"experts {self.cfg.n_experts} top-{self.cfg.top_k}, ff {self.cfg.ff_exp}; values not found in config.json (defaults used): {defaulted or 'none'}")
        self.head = MTPHead(params, self.cfg, r=a.lora_r, alpha=a.lora_alpha, seed=a.seed).to(self.dev)
        del params
        emb = load_single(self.snap, "embed_tokens.weight", torch, self.dtype)
        if emb is None:
            raise SystemExit("embed_tokens.weight not found in the checkpoint")
        self.emb = emb                                                       # stays on the CPU: only a few rows per sample are needed
        w = load_single(self.snap, "lm_head.weight", torch, self.dtype)
        self.w_head = (w if w is not None else emb).to(self.dev)
        if w is None:
            log("lm_head.weight not in the checkpoint: using the embedding table as the output matrix (tied weights)")
        log(f"vocabulary {self.w_head.shape[0]}, trainable LoRA parameters {sum(int(p.numel()) for p in self.head.trainable_parameters()):,}")
        self.meta = json.loads(Path(a.samples_meta).read_text(encoding="utf-8")) if Path(a.samples_meta).exists() else None
        if need_hidden:
            if self.meta is None:
                raise SystemExit(f"{a.samples_meta} not found: run export-samples first")
            if not Path(a.hidden).exists():
                raise SystemExit(f"{a.hidden} not found: run  bash 10a_dump_hidden.sh  first")
            self.n_embd, self.hidx, self.hmm = read_hidden_index(Path(a.hidden))
            if self.n_embd != self.cfg.hidden:
                raise SystemExit(f"hidden dump has n_embd {self.n_embd}, the head needs {self.cfg.hidden}")
            if len(self.hidx) != len(self.meta["ids"]):
                raise SystemExit(f"hidden dump has {len(self.hidx)} samples, the sample list has {len(self.meta['ids'])}")
            self.train_ids = [i for i, s in enumerate(self.meta["split"]) if s == "train"]
            self.hold_ids = [i for i, s in enumerate(self.meta["split"]) if s == "hold"]
            log(f"{len(self.train_ids)} training samples, {len(self.hold_ids)} hold-out samples")

    def load_sample(self, i: int):
        torch = self.torch
        off, n = self.hidx[i]
        ids = self.meta["ids"][i][:n]
        lab = self.meta["labels"][i][:n]
        if len(ids) != n:
            raise SystemExit(f"sample {i}: dump has {n} positions but the sample list has {len(ids)}")
        Hs = torch.from_numpy(bf16_rows(self.hmm, off, n, self.n_embd)).to(self.dev).to(self.dtype)
        ids_t = torch.tensor(ids, dtype=torch.long)
        Es = self.emb[ids_t].to(self.dev)
        return Hs, Es, ids_t.to(self.dev), lab

    def evaluate(self, which: List[int], args) -> Dict[str, float]:
        torch = self.torch
        meter = Meter()
        with torch.no_grad():
            for i in which:
                Hs, Es, ids_t, lab = self.load_sample(i)
                r = sample_pass(torch, self.head, self.w_head, Hs, Es, ids_t, lab, args, False)
                if r is not None:
                    meter.add(r)
        return meter.summary()


# ---- adapter files ------------------------------------------------------------------------------------------------------------------------------------------
def save_adapter(head, out_dir: Path, args, snapshot: Path, metrics: Dict, history: List) -> Path:
    out = fresh_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for tgt, (a_, b_) in head.lora_state().items():
        base = f"base_model.model.{adapter_module_path(tgt)}"
        tensors[f"{base}.lora_A.weight"] = a_.numpy().astype(np.float32)
        tensors[f"{base}.lora_B.weight"] = b_.numpy().astype(np.float32)
    ST.save_tensors(out / "adapter_model.safetensors", tensors, metadata={"format": "pt", "pelias": "blk40-lora-v1"})
    cfg = {"format": "pelias-blk40-lora-v1", "r": args.lora_r, "lora_alpha": args.lora_alpha, "scale": args.lora_alpha / args.lora_r, "use_rslora": False,
           "targets": {t: adapter_module_path(t) for t in head.lora_state()}, "base_snapshot": snapshot.name,
           "settings": {k: getattr(args, k) for k in ("epochs", "lr", "steps", "step2_weight", "ce_weight", "accum", "seed")},
           "heldout_metrics": metrics, "history": history}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2, default=float), encoding="utf-8")
    return out


def load_adapter_state(adapter_dir: Path, torch) -> Dict[str, Tuple]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    state = {}
    with ST.SafeFile(adapter_dir / "adapter_model.safetensors") as f:
        for tgt, path in cfg["targets"].items():
            base = f"base_model.model.{path}"
            state[tgt] = (torch.from_numpy(f.get(f"{base}.lora_A.weight")), torch.from_numpy(f.get(f"{base}.lora_B.weight")))
    return state


# ---- train / eval -------------------------------------------------------------------------------------------------------------------------------------------
def cmd_train(a) -> int:
    S = Session(a, need_hidden=True)
    torch = S.torch
    head, dev = S.head, S.dev
    params = head.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    n_train = len(S.train_ids)
    updates_per_epoch = max(1, math.ceil(n_train / a.accum))
    total_updates = a.epochs * updates_per_epoch

    def lr_at(u: int) -> float:
        if u < a.warmup:
            return a.lr * (u + 1) / a.warmup
        prog = (u - a.warmup) / max(1, total_updates - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    base = S.evaluate(S.hold_ids, a)
    log(f"BEFORE training (untouched head), hold-out: {fmt(base)}")
    best_score, best_state, history = score_of(base), None, [{"epoch": 0, **base}]
    t0 = time.time()
    update = 0
    for epoch in range(1, a.epochs + 1):
        order = list(S.train_ids)
        random.Random(a.seed + epoch).shuffle(order)
        meter, skipped = Meter(), 0
        pending = 0
        opt.zero_grad(set_to_none=True)

        def do_update():
            nonlocal update, pending
            for g in opt.param_groups:
                g["lr"] = lr_at(update)
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            pending = 0

        for n_done, i in enumerate(order, 1):
            Hs, Es, ids_t, lab = S.load_sample(i)
            r = sample_pass(torch, head, S.w_head, Hs, Es, ids_t, lab, a, True)
            if r is None or r["loss"] is None:
                continue
            if not bool(torch.isfinite(r["loss"]).item()):
                skipped += 1
                continue
            (r["loss"] / a.accum).backward()
            meter.add(r)
            pending += 1
            if pending == a.accum:
                do_update()
            if n_done % 25 == 0:
                log(f"epoch {epoch}  {n_done}/{len(order)}  train {fmt(meter.summary())}")
        if pending:
            do_update()
        ev = S.evaluate(S.hold_ids, a)
        peak = (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else 0.0
        log(f"EPOCH {epoch}/{a.epochs}  {time.time() - t0:.0f}s  peak VRAM {peak:.2f} GiB  skipped(non-finite) {skipped}\n         train: {fmt(meter.summary())}\n         hold-out: {fmt(ev)}")
        history.append({"epoch": epoch, **ev})
        sc = score_of(ev)
        if sc > best_score:
            best_score = sc
            best_state = {k: (x.clone(), y.clone()) for k, (x, y) in head.lora_state().items()}
            log(f"         new best hold-out score {sc:.4f}")
    if best_state is None:
        log("the hold-out score never beat the untouched head. Nothing worth saving as 'best'; saving the last epoch anyway so you can look at it, but DO NOT merge it.")
        final_state, tag = head.lora_state(), "NOT_BETTER"
    else:
        head.load_lora_state(best_state)
        final_state, tag = best_state, "best"
    final = S.evaluate(S.hold_ids, a)
    target = Path(a.out) if a.out else ADAPTER_OUT
    if tag != "best":
        target = target.with_name(target.name + "_NOT_BETTER")
    out = save_adapter(head, target, a, S.snap, {"before": base, "after": final, "chosen": tag}, history)
    log(f"AFTER ({tag}), hold-out: {fmt(final)}")
    gain = score_of(final) - score_of(base)
    log(f"overlap step 1: {base.get('overlap1', 0):.3f} -> {final.get('overlap1', 0):.3f}" + (f"   step 2: {base.get('overlap2', 0):.3f} -> {final.get('overlap2', 0):.3f}" if "overlap2" in final else ""))
    log(f"saved {out}   (score change {gain:+.4f})")
    log("next: python 11_merge_blk40_and_export.py --adapter " + str(out))
    return 0


def cmd_eval(a) -> int:
    if a.adapter:
        ac = json.loads((Path(a.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        a.lora_r, a.lora_alpha = int(ac["r"]), float(ac["lora_alpha"])
    S = Session(a, need_hidden=True)
    if a.adapter:
        S.head.load_lora_state(load_adapter_state(Path(a.adapter), S.torch))
        log(f"adapter {a.adapter} loaded")
    which = S.hold_ids + (S.train_ids[: a.n_train_eval] if a.n_train_eval else [])
    log(f"{'adapter' if a.adapter else 'untouched head'} on {len(S.hold_ids)} hold-out samples" + (f" + {a.n_train_eval} training samples" if a.n_train_eval else "") + ":")
    log(fmt(S.evaluate(which, a)))
    return 0


# ---- selftest -----------------------------------------------------------------------------------------------------------------------------------------------
def cmd_selftest(a) -> int:
    import torch
    from mtp_torch import LORA_TARGETS, MTPHead, hf_cfg_to_config, run_chunks, token_stats
    fails = []

    def check(name, ok, detail=""):
        print(("ok   " if ok else "FAIL ") + name + (f"   {detail}" if (detail and not ok) else ""), flush=True)
        if not ok:
            fails.append(name)

    cfg = MN.Config(hidden=64, n_heads=4, n_kv_heads=2, head_dim=16, rot_dim=8, rope_base=1.0e4, n_experts=8, top_k=2, ff_exp=16, ff_shared=16)
    p = MN.random_params(cfg, seed=31, scale=0.08)
    r = np.random.default_rng(32)
    T = 9
    h = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e1 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    e2 = r.standard_normal((T, cfg.hidden)).astype(np.float32)
    pos = np.arange(T) + 1
    ref1, ref2 = MN.mtp_forward_two_step(p, cfg, h, e1, e2, pos)
    tp = {k: torch.from_numpy(v.astype(np.float32)) for k, v in p.items()}
    head = MTPHead(tp, cfg, r=4, alpha=8.0, seed=3)
    tt = torch.from_numpy
    posT = torch.arange(1, T + 1)
    with torch.no_grad():
        O1, K1, V1 = head.step1(tt(h), tt(e1), posT)
        O2 = head.step2(O1, tt(e2), posT, K1, V1)
    check("torch step 1 == numpy reference (float32, CPU)", np.allclose(O1.numpy(), ref1, atol=2e-4), f"{np.abs(O1.numpy() - ref1).max():.2e}")
    check("torch step 2 == numpy reference (float32, CPU)", np.allclose(O2.numpy(), ref2, atol=2e-4), f"{np.abs(O2.numpy() - ref2).max():.2e}")

    # LoRA: random A/B -> merged weights in the numpy model give the same output; this is the formula 11_merge_blk40_and_export.py uses
    g = torch.Generator().manual_seed(7)
    merged = {k: v.copy() for k, v in p.items()}
    with torch.no_grad():
        for name, m in head.lin.items():
            m.lora_A.copy_(torch.randn(m.lora_A.shape, generator=g) * 0.3)
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.3)
            merged[name] = m.merged_weight().numpy()
        O1l, K1l, V1l = head.step1(tt(h), tt(e1), posT)
        O2l = head.step2(O1l, tt(e2), posT, K1l, V1l)
    rm1, rm2 = MN.mtp_forward_two_step(merged, cfg, h, e1, e2, pos)
    check("LoRA head == numpy head with merged weights (step 1)", np.allclose(O1l.numpy(), rm1, atol=3e-4), f"{np.abs(O1l.numpy() - rm1).max():.2e}")
    check("LoRA head == numpy head with merged weights (step 2)", np.allclose(O2l.numpy(), rm2, atol=3e-4), f"{np.abs(O2l.numpy() - rm2).max():.2e}")

    # gradients: only the LoRA matrices get them; finite-difference check on a few entries
    V = 40
    W = torch.randn(V, cfg.hidden, generator=g) * 0.3
    teacher = torch.randn(T - 2, cfg.hidden, generator=g)
    y = torch.randint(0, V, (T - 2,), generator=g)

    def loss_fn():
        o1, k1, v1 = head.step1(tt(h)[: T - 2], tt(e1)[: T - 2], posT[: T - 2])
        o2 = head.step2(o1[:-1], tt(e2)[: T - 3], posT[: T - 3], k1[:-1], v1[:-1])
        s1 = run_chunks(o1, teacher, y, W, 4, True)
        s2 = run_chunks(o2, teacher[1:], y[1:], W, 4, True)
        return (s1[0] + 0.1 * s1[1]) / (T - 2) + 0.5 * (s2[0] + 0.1 * s2[1]) / (T - 3)

    for q in head.parameters():
        q.grad = None
    L = loss_fn()
    L.backward()
    lora_params = [(n, q) for n, q in head.named_parameters() if q.requires_grad]
    frozen = [(n, q) for n, q in head.named_parameters() if not q.requires_grad]
    check("16 trainable tensors (A and B of 8 modules)", len(lora_params) == 16, str([n for n, _ in lora_params]))
    check("every LoRA matrix received a finite, non-zero gradient", all(q.grad is not None and bool(torch.isfinite(q.grad).all()) and float(q.grad.abs().sum()) > 0 for _, q in lora_params))
    check("frozen tensors received no gradient", all(q.grad is None for _, q in frozen))
    ok_fd, worst = True, 0.0
    for name, q in lora_params[:: max(1, len(lora_params) // 5)]:
        flat = q.data.view(-1)
        j = int(torch.argmax(q.grad.view(-1).abs()))
        analytic = float(q.grad.view(-1)[j])
        eps = 5e-3
        old = float(flat[j])
        with torch.no_grad():
            flat[j] = old + eps
            lp = float(loss_fn())
            flat[j] = old - eps
            lm = float(loss_fn())
            flat[j] = old
        fd = (lp - lm) / (2 * eps)
        rel = abs(fd - analytic) / max(1e-6, abs(fd), abs(analytic))
        worst = max(worst, rel)
        ok_fd &= (rel < 0.08) or abs(fd - analytic) < 5e-4
    check("finite-difference check of the gradients (5 entries)", ok_fd, f"worst relative error {worst:.3f}")

    # adapter save/load round trip through the real files
    class A:  # a stand-in for the parsed arguments
        lora_r, lora_alpha, epochs, lr, steps, step2_weight, ce_weight, accum, seed = 4, 8.0, 1, 1e-3, 2, 0.5, 0.1, 1, 0
    with tempfile.TemporaryDirectory() as td:
        out = save_adapter(head, Path(td) / "ad", A, Path("snap"), {}, [])
        fresh = MTPHead(tp, cfg, r=4, alpha=8.0, seed=99)
        fresh.load_lora_state(load_adapter_state(out, torch))
        with torch.no_grad():
            P1, _, _ = fresh.step1(tt(h), tt(e1), posT)
        check("adapter file round trip reproduces the output", np.allclose(P1.numpy(), O1l.numpy(), atol=1e-6))
        with ST.SafeFile(out / "adapter_model.safetensors") as sf_:
            names = sf_.keys()
        check("adapter has 16 tensors named base_model.model.mtp....lora_A/B.weight", len(names) == 16 and all(n.startswith("base_model.model.mtp.") for n in names))
        check("adapter paths follow the checkpoint names", f"base_model.model.mtp.layers.0.mlp.shared_expert.up_proj.lora_B.weight" in names and "base_model.model.mtp.fc.lora_A.weight" in names)

    # statistics on a small vocabulary against numpy
    st = token_stats(torch.randn(5, cfg.hidden, generator=g), torch.randn(5, cfg.hidden, generator=g), torch.randint(0, V, (5,), generator=g), W)
    check("token_stats is finite and in range", bool(torch.isfinite(st).all()) and bool((st[:, 3] <= 1.0001).all()) and bool((st[:, 3] >= 0).all()) and bool((st[:, 0] >= -1e-5).all()))

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        hb = MTPHead({k: v.to(torch.bfloat16) for k, v in tp.items()}, cfg, r=4, alpha=8.0, seed=3).to(dev)
        hb.load_lora_state({k: (m.lora_A.detach().cpu(), m.lora_B.detach().cpu()) for k, m in head.lin.items() if m.r > 0})
        with torch.no_grad():
            Ob, Kb, Vb = hb.step1(tt(h).to(dev).bfloat16(), tt(e1).to(dev).bfloat16(), posT.to(dev))
            Ob2 = hb.step2(Ob, tt(e2).to(dev).bfloat16(), posT.to(dev), Kb, Vb)
        for nm, got, ref in (("step 1", Ob, O1l), ("step 2", Ob2, O2l)):
            a_, b_ = got.float().cpu().flatten(), ref.flatten()
            cos = float(torch.dot(a_, b_) / (a_.norm() * b_.norm()))
            check(f"GPU bfloat16 {nm} agrees with float32 CPU (cosine >= 0.98)", cos >= 0.98, f"{cos:.4f}")
    else:
        print("note CUDA not available: the GPU/bfloat16 check was skipped")
    print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    e = sub.add_parser("export-samples")
    e.add_argument("--train-pt", default=str(TRAIN_PT))
    e.add_argument("--n-train", type=int, default=224)
    e.add_argument("--n-hold", type=int, default=32)
    e.add_argument("--max-len", type=int, default=2048, help="crop every sample to this many tokens (causal model: the kept positions are unchanged)")
    e.add_argument("--seed", type=int, default=11)
    e.set_defaults(fn=cmd_export_samples)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval)):
        t = sub.add_parser(name)
        t.add_argument("--hf-cache", default=str(HF_CACHE))
        t.add_argument("--samples-meta", default=str(SAMPLES_META))
        t.add_argument("--hidden", default=str(HIDDEN_BIN))
        t.add_argument("--lora-r", type=int, default=16)
        t.add_argument("--lora-alpha", type=float, default=32.0)
        t.add_argument("--steps", type=int, choices=(1, 2), default=2, help="2 also trains the second draft step (the server drafts 2 tokens)")
        t.add_argument("--step2-weight", type=float, default=0.5)
        t.add_argument("--ce-weight", type=float, default=0.1, help="weight of the hard-label cross-entropy next to the KL")
        t.add_argument("--ce-chunk", type=int, default=128, help="rows per lm_head chunk (VRAM)")
        t.add_argument("--seed", type=int, default=3407)
        t.add_argument("--cpu", action="store_true", help="force CPU (very slow; for debugging)")
        t.set_defaults(fn=fn)
        if name == "train":
            t.add_argument("--epochs", type=int, default=4)
            t.add_argument("--lr", type=float, default=3e-4)
            t.add_argument("--accum", type=int, default=2, help="samples per optimizer update")
            t.add_argument("--warmup", type=int, default=10)
            t.add_argument("--clip", type=float, default=1.0)
            t.add_argument("--out", default="", help="adapter folder (default mtp_align/blk40_lora_adapter; never overwritten, a timestamp is added if it exists)")
        else:
            t.add_argument("--adapter", default="")
            t.add_argument("--n-train-eval", type=int, default=0, help="also score this many training samples")
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
