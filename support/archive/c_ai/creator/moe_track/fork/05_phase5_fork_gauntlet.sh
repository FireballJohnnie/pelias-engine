#!/usr/bin/env bash
# File: <project>\moe_track\fork\05_phase5_fork_gauntlet.sh   (WSL: <project>/moe_track/fork/05_phase5_fork_gauntlet.sh)
# Phase 5 helper: build the fork binaries, make the Schwerz profile, and hand over to fork/gauntlet.py for the measuring.
# Each fork gets its OWN source folder and its OWN build folder under ~/llama-forks (WSL disk). Your production build is never touched.
# No sudo. Nothing starts a server except 'run', which calls gauntlet.py (it stops the production server first and can restore it with --restore).
#
#   bash fork/05_phase5_fork_gauntlet.sh plan                 the seven variants and what each costs (runs nothing)
#   bash fork/05_phase5_fork_gauntlet.sh builds               which fork builds exist
#   bash fork/05_phase5_fork_gauntlet.sh build KEY            clone + build one fork. KEY = master | schwerz | parmesan     (20-40 min each)
#   bash fork/05_phase5_fork_gauntlet.sh profile              Schwerz routing profile from YOUR prompts (needs the schwerz build and trace prompts)
#   bash fork/05_phase5_fork_gauntlet.sh run SPEC [SPEC...]   same as: python fork/gauntlet.py run ...   (see that file's --help)
#   bash fork/05_phase5_fork_gauntlet.sh report               table of everything measured so far
#
# Overrides:  MOE_FORKS=~/llama-forks  JOBS=8  CUDA_ARCH=89
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"

MODE="${1:-plan}"
[ "$#" -gt 0 ] && shift
BUILD_SCRIPT="$CREATOR/build_llama_cuda_v2.sh"
GAUNTLET=("$PY" "$HERE/gauntlet.py")
[ -x "$PY" ] || GAUNTLET=(python3 "$HERE/gauntlet.py")

case "$MODE" in
plan|builds|report)
    exec "${GAUNTLET[@]}" "$MODE" "$@"
    ;;
run)
    exec "${GAUNTLET[@]}" run "$@"
    ;;
build)
    KEY="${1:-}"
    case "$KEY" in master|schwerz|parmesan) ;; *) moe_die "usage: $0 build master|schwerz|parmesan   (the production build and the Phase 6 build have their own scripts)";; esac
    command -v git >/dev/null 2>&1 || moe_die "git not found"
    [ -f "$BUILD_SCRIPT" ] || moe_die "missing $BUILD_SCRIPT"
    IFS=$'\t' read -r REPO COMMIT SRC < <("${GAUNTLET[@]}" build-info "$KEY") || moe_die "could not read the build record for $KEY"
    [ -n "$REPO" ] && [ -n "$COMMIT" ] && [ -n "$SRC" ] || moe_die "incomplete build record for $KEY"
    moe_log "fork $KEY: $REPO @ ${COMMIT:0:9} -> $SRC"
    mkdir -p "$(dirname "$SRC")"
    if [ ! -d "$SRC/.git" ]; then
        moe_log "cloning (a few minutes) ..."
        git clone --quiet --no-checkout "$REPO" "$SRC" || moe_die "clone failed"
    fi
    cd "$SRC" || moe_die "cannot enter $SRC"
    git fetch --quiet origin "$COMMIT" 2>/dev/null || git fetch --quiet origin 2>/dev/null || true
    git cat-file -e "${COMMIT}^{commit}" 2>/dev/null || moe_die "commit ${COMMIT:0:9} is not available from $REPO"
    git checkout --quiet --detach "$COMMIT" || moe_die "checkout failed (uncommitted changes in $SRC?)"
    moe_log "building with $BUILD_SCRIPT (LLAMA_DIR=$SRC, BUILD_NAME=build-wsl) ..."
    LLAMA_DIR="$SRC" BUILD_NAME="build-wsl" JOBS="${JOBS:-8}" CUDA_ARCH="${CUDA_ARCH:-89}" bash "$BUILD_SCRIPT"
    rc=$?
    [ "$rc" -eq 0 ] || moe_die "the build script stopped with code $rc (its last lines say why)"
    # the profile collector is a test program; build it too for the Schwerz fork (only that fork documents it)
    if [ "$KEY" = "schwerz" ] && [ -d "$SRC/build-wsl" ]; then
        moe_log "building the profile collector (test-llama-archs) ..."
        cmake --build "$SRC/build-wsl" --target test-llama-archs -j "${JOBS:-8}" >/dev/null 2>&1 \
            && moe_log "built $SRC/build-wsl/bin/test-llama-archs" \
            || moe_log "WARNING: test-llama-archs did not build; 'profile' will not work until it does (cmake --build $SRC/build-wsl --target test-llama-archs)"
    fi
    moe_log "built: $SRC/build-wsl/bin/llama-server"
    ;;
profile)
    IFS=$'\t' read -r _ _ SRC < <("${GAUNTLET[@]}" build-info schwerz) || moe_die "could not read the build record for schwerz"
    COLLECTOR="$SRC/build-wsl/bin/test-llama-archs"
    [ -x "$COLLECTOR" ] || moe_die "$COLLECTOR not found. Build the Schwerz fork first: bash $0 build schwerz"
    [ -f "$GGUF" ] || moe_die "model not found: $GGUF"
    PROMPTS="$LOGS/moe_trace/prompts"
    [ -d "$PROMPTS" ] || moe_die "no prompts in $PROMPTS. Run: python trace/build_prompts.py"
    OUT_DIR="$LOGS/moe_trace/schwerz"
    "$(command -v python3)" "$HERE/make_schwerz_corpus.py" --prompts-dir "$PROMPTS" --out-dir "$OUT_DIR" || moe_die "could not make corpus.json"
    PROFILE="$OUT_DIR/calibration.gguf"
    if [ -e "$PROFILE" ] || [ -e "$PROFILE.calibration" ]; then
        moe_die "$PROFILE (or its .calibration folder) already exists; the collector needs a fresh output path. Move it away first."
    fi
    if server_healthy; then moe_die "the production server is running on port $PORT and holds the GPU. Stop it first:  llama --stop"; fi
    moe_log "collecting the profile (loads the model once, then decodes every corpus prompt; can take 30+ minutes)"
    # exactly the command documented in the fork's wiki page Generic-Hybrid-MoE-Reproduction; flags may differ at the pinned commit (see --help of the tool)
    env -u GGML_MOE_EXPERT_PROFILE LD_LIBRARY_PATH="$SRC/build-wsl/bin" "$COLLECTOR" --collect-moe-profile "$GGUF" \
        --profile-corpus "$OUT_DIR/corpus.json" --profile-output "$PROFILE" \
        --profile-threads "${PROFILE_THREADS:-16}" --replay-cache-slots 64 --replay-load-mode none \
        2>&1 | tee "$OUT_DIR/collect.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || moe_die "the collector stopped with code $rc (see $OUT_DIR/collect.log)"
    moe_log "profile written: $PROFILE"
    ;;
*)
    moe_die "unknown mode '$MODE' (plan, builds, build KEY, profile, run SPEC..., report)"
    ;;
esac
