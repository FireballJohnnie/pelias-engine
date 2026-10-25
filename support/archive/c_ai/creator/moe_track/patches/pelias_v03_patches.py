#!/usr/bin/env python3
# File: <project>\moe_track\patches\pelias_v03_patches.py   (WSL: <project>/moe_track/patches/pelias_v03_patches.py)
"""Pelias v0.2 -> v0.3: small fixes applied IN MEMORY by the launcher shim (moe_track/stuart_pelias0.3.py). The tar peliasv0.2.tar and the unpacked copy stay byte-identical.

What is in here (and what is deliberately not):

  1. PEAK-VRAM GUARD (on by default).  pelias.train.run_training calls torch.cuda.reset_peak_memory_stats() before EVERY micro-batch, but it logs
     torch.cuda.max_memory_allocated() once per optimizer step (every grad_accum micro-batches). So the logged "peak" only covers the last micro-batch of each
     group. The guard lets the reset through only when the call comes from run_training AND the micro-batch counter starts a new accumulation group
     (micro % grad_accum == 0); every other caller, and every case it cannot read, passes straight through to the original function.

  2. KV-CACHED PROBE (on by default).  pelias.train.stage_probe re-runs the whole growing sequence for every new token with use_cache=False, so each token streams
     all 40 expert layers AND recomputes the prompt. The replacement prefills the prompt once and then feeds one token at a time with the model's cache
     (Qwen3.5's hybrid cache: attention K/V + DeltaNet state). Because that relies on the installed transformers handling the cache the way its source says,
     the first prompt is CHECKED: the second generated token from the cached path must equal the one from an uncached full pass. If it does not, or anything in
     the cached path raises, the probe falls back to the original uncached loop for the rest of the run and says so in the log. The streaming of the expert
     weights per token is unchanged, so the gain is the removed prompt recomputation, not a new speed limit. Measure it; the doc's 1.38 tok/s is an estimate.

  3. LAYER-INPUT OFFLOAD TO PINNED HOST RAM (OFF by default, opt-in: --v03-offload-xl or PELIAS_V03_OFFLOAD_XL=1).  With gradient checkpointing, the only large thing each
     decoder layer keeps for the backward pass is its input X_L ([1, T, 2048] bf16; 40 layers = 0.67 GiB at T=4096, 2.7 GiB at T=16384). This replaces the
     checkpoint function of the decoder layers by one that copies X_L to page-locked host memory after the forward and copies it back for the recompute.
     It does NOT keep Y_moe on the host: the backward of the expert function must stream W_gu / W_dn again anyway, so that would save no PCIe traffic.
     At the launcher's 4096-token length the saving is only about 0.5 GiB, so leave it off unless you raise the training length. It fails LOUD (an exception at the first step)
     rather than silently training wrong, and the smoke stage is the place to try it.

NOT DONE, ON PURPOSE:  the "zero-sync routing plan" of the roadmap draft.  It cannot work as written: (a) pelias.experts._plan returns four values and its callers unpack
four; the draft returned three, which would crash at the first layer; (b) the chunk loop needs the number of rows per chunk as a HOST integer to slice the sorted
activations and to skip empty chunks, so removing counts.tolist() would need the whole expert function rewritten around one grouped GEMM over all 256 experts with
the full per-layer weight (about 1.6 GiB of bf16 dequantised weight instead of 0.2 GiB per chunk), with a new VRAM peak that only a real run can measure.
Memoising the plan between the recompute and the backward is unsafe (the allocator reuses addresses, so a wrong hit would be silent). See UNFINISHED.md.

Everything here is import-safe without torch; torch is imported only inside the functions that need it.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

PATCH_VERSION = "0.3.0"
ENV_OFFLOAD = "PELIAS_V03_OFFLOAD_XL"


# ===================================================================================================================================================
# 1. peak-VRAM guard
# ===================================================================================================================================================
def make_peak_guard(orig_reset: Callable) -> Callable:
    """Wrap torch.cuda.reset_peak_memory_stats. Only the call inside pelias.train.run_training is filtered, and only when it is not the first micro-batch of a group."""

    def guarded(device: Any = None) -> Any:
        try:
            frame = sys._getframe(1)
            code = frame.f_code
            if code.co_name == "run_training" and code.co_filename.replace("\\", "/").endswith("pelias/train.py"):
                loc = frame.f_locals
                micro = loc.get("micro")
                accum = getattr(loc.get("args"), "grad_accum", None)
                if isinstance(micro, int) and isinstance(accum, int) and accum > 1 and micro % accum != 0:
                    return None                      # inside an accumulation group: keep the running peak
        except Exception:                            # noqa: BLE001  - never let the guard break a run; fall through to the real reset
            pass
        return orig_reset(device)

    guarded._pelias_v03 = True                       # type: ignore[attr-defined]
    guarded._orig = orig_reset                       # type: ignore[attr-defined]
    return guarded


def install_peak_guard(torch: Any) -> bool:
    cur = torch.cuda.reset_peak_memory_stats
    if getattr(cur, "_pelias_v03", False):
        return False
    torch.cuda.reset_peak_memory_stats = make_peak_guard(cur)
    return True


def remove_peak_guard(torch: Any) -> None:
    cur = torch.cuda.reset_peak_memory_stats
    if getattr(cur, "_pelias_v03", False):
        torch.cuda.reset_peak_memory_stats = cur._orig


# ===================================================================================================================================================
# 2. KV-cached probe
# ===================================================================================================================================================
def make_cached_probe(T: Any, torch: Any, log_prefix: str = "[v0.3 probe]") -> Callable:
    """Build the replacement for pelias.train.stage_probe. T is the pelias.train module (its helpers are looked up at call time, so launcher patches stay effective)."""
    orig_probe = T.stage_probe

    @torch.no_grad()
    def stage_probe_cached(args: Any, log: Callable) -> None:
        import json
        from pathlib import Path

        device = torch.device(args.device)
        model, streamer, rep = T.stage_load(args, log)
        from peft import PeftModel
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(T.resolve_snapshot(args.hf_cache)))
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        prompts = json.loads(Path(args.probe_file).read_text()) if args.probe_file else [
            [{"role": "user", "content": "Who are you, and who made you?"}],
            [{"role": "user", "content": "What is the boiling point of water at sea level?"}],
        ]
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        base = model.get_base_model()
        state = {"use_cache": True, "checked": False}

        def forward(ids_or_tok: Any, past: Any, cache: bool) -> Any:
            streamer.phase = +1
            emb = T.embed_on_host(model, ids_or_tok, device)
            if cache:
                return base.model(inputs_embeds=emb, past_key_values=past, use_cache=True)
            return base.model(inputs_embeds=emb, use_cache=False)

        def next_token(out: Any) -> Any:
            return base.lm_head(out.last_hidden_state[:, -1]).argmax(-1, keepdim=True).cpu()

        def uncached(ids: Any) -> Any:
            gen = ids
            for _ in range(args.max_new_tokens):
                nxt = next_token(forward(gen, None, False))
                gen = torch.cat([gen, nxt], dim=1)
                if int(nxt) == im_end:
                    break
            return gen

        def cached(ids: Any, verify: bool) -> Any:
            gen = ids
            out = forward(ids, None, True)
            past = out.past_key_values
            if past is None:
                raise RuntimeError("the model returned no cache with use_cache=True")
            nxt = next_token(out)
            for k in range(args.max_new_tokens):
                gen = torch.cat([gen, nxt], dim=1)
                if int(nxt) == im_end or k + 1 >= args.max_new_tokens:
                    break
                out = forward(nxt, past, True)
                past = out.past_key_values
                new = next_token(out)
                if verify and k == 0:
                    ref = next_token(forward(gen, None, False))           # full pass over prompt + first token, no cache
                    if int(ref) != int(new):
                        raise RuntimeError(f"cached second token {int(new)} != uncached {int(ref)}")
                    state["checked"] = True
                    log(f"{log_prefix} cached decode agrees with the uncached pass on the second token")
                nxt = new
            return gen

        for msgs in prompts:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            gen = None
            if state["use_cache"]:
                try:
                    gen = cached(ids, verify=not state["checked"])
                except Exception as exc:                                  # noqa: BLE001
                    state["use_cache"] = False
                    log(f"{log_prefix} KV-cache path unusable here ({type(exc).__name__}: {str(exc)[:200]}); using the original uncached loop from now on")
            if gen is None:
                gen = uncached(ids)
            log("PROMPT: " + msgs[-1]["content"])
            log("REPLY : " + tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=False))

    stage_probe_cached._pelias_v03 = True                   # type: ignore[attr-defined]
    stage_probe_cached._orig = orig_probe                   # type: ignore[attr-defined]
    return stage_probe_cached


# ===================================================================================================================================================
# 3. layer-input offload (opt-in)
# ===================================================================================================================================================
def make_offload_checkpoint(torch: Any, hidden_dim: int = 2048, pool_limit_bytes: int = 4 << 30) -> Callable:
    """Return `offload_checkpoint(function, *args, **ignored)`, a drop-in for the `_gradient_checkpointing_func` of transformers decoder layers.

    forward : runs the layer under no_grad; the layer input x (CUDA, bf16, last dim hidden_dim) is copied to a pinned host buffer on a side stream that first waits for the
              stream that produced x. Nothing of x is kept on the GPU by this function.
    backward: waits for that copy, brings x back, re-runs the layer with grad enabled and backpropagates through it (the same recompute a normal checkpoint does).
    Anything it does not recognise (not a CUDA bf16 hidden tensor, or the output is not a tensor / tuple of tensors) is handed to torch.utils.checkpoint unchanged, except an
    unrecognised OUTPUT which raises, because the forward has then already been run.
    """
    from torch.utils.checkpoint import checkpoint as std_checkpoint

    side = {"stream": None}
    pool: Dict[int, List[Any]] = {}                                   # bucket bytes -> list of (raw uint8 pinned buffer, event-or-None) ready for reuse
    pool_bytes = {"n": 0}
    BUCKET = 8 << 20

    def get_stream() -> Any:
        if side["stream"] is None:
            side["stream"] = torch.cuda.Stream()
        return side["stream"]

    def take(nbytes: int) -> Any:
        bucket = ((nbytes + BUCKET - 1) // BUCKET) * BUCKET
        lst = pool.get(bucket)
        if lst:
            raw, ev = lst.pop()
            if ev is not None:
                ev.synchronize()                                      # the last H2D read of this buffer must be finished before it is written again
            return raw
        pool_bytes["n"] += bucket
        return torch.empty(bucket, dtype=torch.uint8, device="cpu", pin_memory=True)

    def give(raw: Any, ev: Any) -> None:
        if pool_bytes["n"] > pool_limit_bytes:
            pool_bytes["n"] -= raw.numel()                            # too much pinned RAM held: drop this one (after its last read has finished)
            if ev is not None:
                ev.synchronize()
            return
        pool.setdefault(raw.numel(), []).append((raw, ev))

    class _Offloaded(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, run: Callable, x: Any, dummy: Any) -> Any:
            ctx.run, ctx.x_requires_grad, ctx.dev = run, x.requires_grad, x.device
            nbytes = x.numel() * x.element_size()
            raw = take(nbytes)
            buf = raw[:nbytes].view(x.dtype).view(x.shape)
            s = get_stream()
            ready = torch.cuda.current_stream().record_event()
            with torch.cuda.stream(s):
                s.wait_event(ready)                                   # x must be fully written before the copy reads it
                buf.copy_(x.detach(), non_blocking=True)
                x.record_stream(s)                                    # keep the allocator from reusing x's memory until this copy is done
                ctx.copied = s.record_event()
            ctx.raw, ctx.buf = raw, buf
            with torch.no_grad():
                out = run(x.detach())
            if not (torch.is_tensor(out) or (isinstance(out, (tuple, list)) and all(torch.is_tensor(o) for o in out))):
                raise RuntimeError(f"offload_checkpoint: the decoder layer returned {type(out).__name__}; only a tensor or a tuple of tensors is supported. Run without --v03-offload-xl.")
            ctx.is_tuple = isinstance(out, (tuple, list))
            return tuple(out) if ctx.is_tuple else out

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            cur = torch.cuda.current_stream()
            cur.wait_event(ctx.copied)
            x = ctx.buf.to(ctx.dev, non_blocking=True)
            give(ctx.raw, cur.record_event())
            ctx.raw = ctx.buf = None
            with torch.enable_grad():
                xr = x.detach().requires_grad_(ctx.x_requires_grad)
                out = ctx.run(xr)
            outs = list(out) if isinstance(out, (tuple, list)) else [out]
            pairs = [(o, g) for o, g in zip(outs, grads) if torch.is_tensor(o) and o.requires_grad and g is not None]
            if pairs:
                torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
            return None, (xr.grad if ctx.x_requires_grad else None), None

    def offload_checkpoint(function: Callable, *args: Any, **kwargs: Any) -> Any:
        x = args[0] if args else None
        ok = (len(args) >= 1 and torch.is_tensor(x) and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() >= 2 and x.shape[-1] == hidden_dim
              and torch.is_grad_enabled() and all(not torch.is_tensor(a) for a in args[1:]))
        if not ok:
            return std_checkpoint(function, *args, use_reentrant=False)
        run = (lambda xx: function(xx)) if len(args) == 1 else (lambda xx: function(xx, *args[1:]))
        dummy = torch.empty(0, device=x.device, requires_grad=True)   # makes the output require grad even when x does not (first layer)
        out = _Offloaded.apply(run, x, dummy)
        return out

    offload_checkpoint._pelias_v03 = True                             # type: ignore[attr-defined]
    offload_checkpoint._Offloaded = _Offloaded                        # type: ignore[attr-defined]  (the tests drive it directly)
    return offload_checkpoint


def install_offload(T: Any, torch: Any, log: Callable) -> bool:
    """Wrap pelias.train.attach_lora (called right after gradient_checkpointing_enable) so the decoder layers get the offloading checkpoint function."""
    orig_attach = T.attach_lora
    if getattr(orig_attach, "_pelias_v03", False):
        return False
    fn = make_offload_checkpoint(torch)

    def attach_lora_offload(model: Any, *a: Any, **k: Any) -> Any:
        n = 0
        for m in model.modules():
            if type(m).__name__.endswith("DecoderLayer") and getattr(m, "gradient_checkpointing", False) and hasattr(m, "_gradient_checkpointing_func"):
                m._gradient_checkpointing_func = fn
                n += 1
        log(f"[v0.3] layer-input offload to pinned RAM installed on {n} decoder layers" + ("" if n else "  (NONE found: gradient checkpointing was not enabled the way this patch expects; training continues normally)"))
        return orig_attach(model, *a, **k)

    attach_lora_offload._pelias_v03 = True                            # type: ignore[attr-defined]
    attach_lora_offload._orig = orig_attach                           # type: ignore[attr-defined]
    T.attach_lora = attach_lora_offload
    return True


# ===================================================================================================================================================
# entry point
# ===================================================================================================================================================
def install_v03_patches(T: Any, *, peak_guard: bool = True, cached_probe: bool = True, offload_xl: Optional[bool] = None,
                        log: Callable = print, torch_module: Any = None) -> Dict[str, Any]:
    """Apply the v0.3 adjustments to the imported pelias.train module T, in memory. Returns what was installed (also logged)."""
    try:
        torch = torch_module if torch_module is not None else __import__("torch")
    except ImportError:                                                # only the launcher's torch-free self-test gets here; a real run always has torch
        log("[v0.3] torch cannot be imported in this process, so the v0.3 adjustments were NOT installed")
        return {"version": PATCH_VERSION, "skipped": "no torch"}
    if offload_xl is None:
        offload_xl = os.environ.get(ENV_OFFLOAD, "") not in ("", "0", "false", "False")
    done: Dict[str, Any] = {"version": PATCH_VERSION}
    if peak_guard:
        done["peak_guard"] = install_peak_guard(torch)
    if cached_probe and hasattr(T, "stage_probe") and not getattr(T.stage_probe, "_pelias_v03", False):
        T.stage_probe = make_cached_probe(T, torch)
        done["cached_probe"] = True
    if offload_xl:
        done["offload_xl"] = install_offload(T, torch, log)
    log("[v0.3] patches installed in memory: " + ", ".join(f"{k}={v}" for k, v in done.items()))
    return done


def uninstall_v03_patches(T: Any, torch_module: Any = None) -> None:
    """Undo everything install_v03_patches did (used by the tests; the launcher never needs it)."""
    torch = torch_module if torch_module is not None else __import__("torch")
    remove_peak_guard(torch)
    for name in ("stage_probe", "attach_lora"):
        cur = getattr(T, name, None)
        if getattr(cur, "_pelias_v03", False):
            setattr(T, name, cur._orig)
