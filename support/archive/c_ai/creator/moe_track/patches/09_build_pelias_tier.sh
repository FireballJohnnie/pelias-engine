#!/usr/bin/env bash
# File: <project>\moe_track\patches\09_build_pelias_tier.sh   (WSL: <project>/moe_track/patches/09_build_pelias_tier.sh)
# Phase 6 step 1: make a SEPARATE copy of your pinned llama.cpp, add the Pelias-Tier patch to the copy, and build it with your own CUDA build script.
#
# Your production build (<llama.cpp>, build-wsl) is never modified. The copy lives in WSL's own disk (fast builds):  ~/llama-pelias-tier
# No sudo. Writes only inside ~/llama-pelias-tier (and its build-wsl folder).
#
#   bash patches/09_build_pelias_tier.sh check       dry run: shows what it would do, changes nothing
#   bash patches/09_build_pelias_tier.sh prepare     clone + patch only (no build)
#   bash patches/09_build_pelias_tier.sh build       prepare, then build llama-server / llama-quantize / llama-perplexity with CUDA   (about 20-40 min)
#
# Overrides:  TIER_SRC=~/llama-pelias-tier   JOBS=8   CUDA_ARCH=89   LLAMA_PROD=<llama.cpp>   LLAMA_COMMIT=<full hash>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"

MODE="${1:-check}"
# a repository on the Windows drive can belong to another user as far as git can tell ("dubious ownership"); allow reading THIS one, for this script only
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="${LLAMA_PROD:-<llama.cpp>}"
TIER_SRC="${TIER_SRC:-$HOME/llama-pelias-tier}"
PATCH="$HERE/pelias_tier_v1.patch"
NEW_SRC_DIR="$HERE/src"
BUILD_SCRIPT="$CREATOR/build_llama_cuda_v2.sh"

case "$MODE" in check|prepare|build) ;; *) moe_die "unknown mode '$MODE' (use check, prepare or build)";; esac

[ -f "$PATCH" ]                          || moe_die "missing $PATCH"
[ -f "$NEW_SRC_DIR/llama-pelias-tier.cpp" ] || moe_die "missing $NEW_SRC_DIR/llama-pelias-tier.cpp"
[ -f "$NEW_SRC_DIR/llama-pelias-tier.h" ]   || moe_die "missing $NEW_SRC_DIR/llama-pelias-tier.h"
command -v git >/dev/null 2>&1           || moe_die "git not found"
[ -d "$LLAMA_PROD/.git" ]                || moe_die "no git repository at $LLAMA_PROD (set LLAMA_PROD)"

moe_log "mode: $MODE"
moe_log "production tree (read only): $LLAMA_PROD"
moe_log "pelias-tier copy            : $TIER_SRC"
moe_log "pinned commit               : $LLAMA_COMMIT"

prod_head="$(git -C "$LLAMA_PROD" rev-parse HEAD 2>/dev/null || true)"
if [ "$prod_head" != "$LLAMA_COMMIT" ]; then
    moe_log "NOTE: production tree is at ${prod_head:0:9}, not ${LLAMA_COMMIT:0:9}. The patch was written and tested against ${LLAMA_COMMIT:0:9}; the copy will be checked out there."
fi

if [ "$MODE" = "check" ]; then
    if [ -d "$TIER_SRC/.git" ]; then
        moe_log "copy exists; its HEAD is $(git -C "$TIER_SRC" rev-parse --short HEAD)"
        if git -C "$TIER_SRC" apply --check -R "$PATCH" >/dev/null 2>&1; then moe_log "patch already applied in the copy"; else moe_log "patch not applied in the copy yet"; fi
    else
        moe_log "copy does not exist yet; 'prepare' would clone $LLAMA_PROD into $TIER_SRC"
    fi
    [ -f "$BUILD_SCRIPT" ] && moe_log "build script found: $BUILD_SCRIPT" || moe_log "WARNING: $BUILD_SCRIPT not found; 'build' needs it"
    moe_log "nothing was changed"
    exit 0
fi

# ---- 1. the copy -----------------------------------------------------------------------------------------------------------
FRESH=0
if [ ! -d "$TIER_SRC/.git" ]; then
    moe_log "cloning $LLAMA_PROD -> $TIER_SRC (local clone, a few minutes on the first run)"
    git clone --quiet --no-checkout "$LLAMA_PROD" "$TIER_SRC" || moe_die "clone failed"
    FRESH=1        # a --no-checkout clone has an empty work tree, which git status would report as 'everything deleted'
fi
cd "$TIER_SRC" || moe_die "cannot enter $TIER_SRC"

if ! git cat-file -e "${LLAMA_COMMIT}^{commit}" 2>/dev/null; then
    moe_die "commit ${LLAMA_COMMIT:0:9} is not in the copy (is $LLAMA_PROD at or past it?)"
fi

if git apply --check -R "$PATCH" >/dev/null 2>&1; then
    moe_log "patch is already applied; leaving the sources as they are"
else
    # a clean checkout of the pinned commit, then the patch
    if [ "$FRESH" -eq 0 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        moe_die "the copy has uncommitted changes other than this patch. Move it away (mv $TIER_SRC ${TIER_SRC}.old) and run again."
    fi
    git checkout --quiet --detach "$LLAMA_COMMIT" || moe_die "checkout of ${LLAMA_COMMIT:0:9} failed"
    git apply --check "$PATCH"                     || moe_die "the patch does not apply cleanly to ${LLAMA_COMMIT:0:9}"
    git apply "$PATCH"                             || moe_die "applying the patch failed"
    moe_log "patch applied"
fi
cp -f "$NEW_SRC_DIR/llama-pelias-tier.cpp" "$NEW_SRC_DIR/llama-pelias-tier.h" "$TIER_SRC/src/" || moe_die "could not copy the two new source files"
moe_log "new source files copied into $TIER_SRC/src"

if [ "$MODE" = "prepare" ]; then
    moe_log "prepared. Build with:  bash $0 build"
    exit 0
fi

# ---- 2. the build, with your own script (finds your CUDA toolkit, switches NCCL off, arch 89) ------------------------------------
[ -f "$BUILD_SCRIPT" ] || moe_die "missing $BUILD_SCRIPT"
moe_log "building with $BUILD_SCRIPT (LLAMA_DIR=$TIER_SRC, BUILD_NAME=build-wsl)"
LLAMA_DIR="$TIER_SRC" BUILD_NAME="build-wsl" JOBS="${JOBS:-8}" CUDA_ARCH="${CUDA_ARCH:-89}" bash "$BUILD_SCRIPT"
rc=$?
if [ "$rc" -ne 0 ]; then moe_die "the build script stopped with code $rc (its last lines say why)"; fi

moe_log "built: $TIER_SRC/build-wsl/bin/llama-server"
moe_log "next:  python sim/export_hot_set.py   then   bash 13_phase6_run.sh check"
