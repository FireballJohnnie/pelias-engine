#!/usr/bin/env bash
# File: <project>\moe_track\tests\run_all_tests.sh
# The exact set of checks the package was verified with, in one place. These are tests of the TOOLS (small fake models, CPU only); they are not measurements
# and need no GPU and no real model. Nothing here touches the production server, your model files or any setting. Not needed for normal use.
#
# Needs: python3 + numpy, g++, a llama.cpp source tree at the pinned commit, two builds of it as shared libraries (stock and with patches/pelias_tier_v1.patch),
#        and two tiny models:  python tests/make_tiny_moe.py  (see its header)  and  llama-quantize for the Q4_K_M one.
#
#   SRC=/path/llama.cpp STOCK=/path/stock/bin PATCHED=/path/patched/bin TINY=/path/tiny_models  bash tests/run_all_tests.sh
set -uo pipefail
: "${SRC:?set SRC to the llama.cpp source folder}" "${STOCK:?set STOCK to the stock build's bin folder (libllama.so ...)}"
: "${PATCHED:?set PATCHED to the patched build's bin folder}" "${TINY:?set TINY to the folder with tiny_f16.gguf and tiny_q4km.gguf}"
MT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d)"
export PYTHONDONTWRITEBYTECODE=1
res=()
run() { local name="$1"; shift; if "$@" > "$W/$name.log" 2>&1; then res+=("PASS $name"); else res+=("FAIL $name   (log: $W/$name.log)"); fi; }
cd "$MT"
g++ -O2 -std=c++17 -DMOE_TRACE_DEFAULT_BACKEND_DIR="\"$STOCK\"" trace/moe_trace.cpp -I"$SRC/include" -I"$SRC/ggml/include" -L"$STOCK" -lllama -lggml -lggml-base -Wl,-rpath,"$STOCK" -o "$W/moe_trace" \
    || { echo "could not build the tracer"; exit 1; }
run gates        python3 tests/test_gates.py
run gguf_min     python3 tests/test_gguf_min.py --gguf-py "$SRC/gguf-py" --model-dir "$TINY"
run tracer       python3 tests/test_tracer.py --tracer "$W/moe_trace" --model "$TINY/tiny_f16.gguf" --gguf-py "$SRC/gguf-py"
run kl_gate      python3 tests/test_kl_gate.py
run predictor    python3 tests/test_predictor.py
run sim          python3 tests/test_sim.py
run verify_g0    python3 tests/test_verify_g0.py
run fork_scripts python3 tests/test_fork_scripts.py
run phase6       python3 tests/test_phase6.py --llama-src "$SRC" --stock-bin "$STOCK" --tier-bin "$PATCHED" --model "$TINY/tiny_f16.gguf" --model "$TINY/tiny_q4km.gguf"
run dump_hidden  python3 tests/test_dump_hidden.py --llama-src "$SRC" --lib-dir "$STOCK" --model "$TINY/tiny_f16.gguf"
run mtp_numpy    python3 tests/test_mtp_numpy.py
run mtp_torch    python3 tests/test_mtp_torch_fake.py
run trainer_io   python3 tests/test_trainer_io.py
run merge_blk40  python3 tests/test_merge_blk40.py
run pelias_v03   python3 tests/test_pelias_v03.py
run launcher     python3 tests/test_launcher.py
run pelias_serve python3 tests/test_pelias_serve.py --llama-src "$SRC" --lib-dir "$STOCK" --model "$TINY/tiny_f16.gguf"
# the skipped-expert (-1) change lives in the PATCHED ggml, so this test must be linked against the patched libraries
if g++ -O2 -std=c++17 tests/test_mmid_skip.cpp -I"$SRC/ggml/include" -L"$PATCHED" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"$PATCHED" -o "$W/test_mmid_skip" > "$W/mmid_build.log" 2>&1; then
    run mmid_skip "$W/test_mmid_skip"
else
    res+=("FAIL mmid_skip (build)")
fi
printf '%s\n' "${res[@]}"
echo "logs in $W"
printf '%s\n' "${res[@]}" | grep -q '^FAIL' && exit 1 || exit 0
