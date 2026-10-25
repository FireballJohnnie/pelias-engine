#!/usr/bin/env bash
# File: <project>\moe_track\12_upgrade_llama_launcher.sh   (WSL: <project>/moe_track/12_upgrade_llama_launcher.sh)
# Installs the upgraded launcher (moe_track/lib/llama_launcher.sh) as ~/.local/bin/llama.
#
#   bash 12_upgrade_llama_launcher.sh            shows what would happen; writes nothing
#   bash 12_upgrade_llama_launcher.sh install    1) copies your current ~/.local/bin/llama to ~/.local/bin/llama.<date_time>.bak (never overwrites a backup)
#                                                2) puts the new launcher in place
#   bash 12_upgrade_llama_launcher.sh restore    puts the newest .bak back (the launcher being replaced is first saved as llama.<date_time>.replaced)
#
# It does not start, stop or touch any server. Only files in ~/.local/bin are written. Override the folder with  BIN_DIR=...  (used by the tests).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW="$HERE/lib/llama_launcher.sh"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
TARGET="$BIN_DIR/llama"
MODE="${1:-show}"
STAMP="$(date +%Y%m%d_%H%M%S)"

say() { printf '[upgrade] %s\n' "$*"; }
die() { printf '[upgrade] ERROR: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum "$1" | awk '{print $1}'; }

[ -f "$NEW" ] || die "missing $NEW"
bash -n "$NEW" || die "the new launcher has a syntax error ($NEW)"

case "$MODE" in
show)
    say "new launcher  : $NEW  ($(grep -m1 '^LAUNCHER_VERSION=' "$NEW" | cut -d'"' -f2))"
    if [ -f "$TARGET" ]; then
        say "current       : $TARGET  ($(wc -l < "$TARGET") lines)"
        if [ "$(sha "$NEW")" = "$(sha "$TARGET")" ]; then say "they are identical: nothing to do"; else
            say "install would first copy it to $TARGET.$STAMP.bak"
            say "differences (first 40 lines):"; diff "$TARGET" "$NEW" | head -40
        fi
    else
        say "current       : none ($TARGET does not exist); install would just create it"
    fi
    say "nothing was written. To install:  bash $0 install"
    ;;
install)
    mkdir -p "$BIN_DIR" || die "cannot create $BIN_DIR"
    if [ -f "$TARGET" ]; then
        if [ "$(sha "$NEW")" = "$(sha "$TARGET")" ]; then say "already up to date: $TARGET"; exit 0; fi
        BKP="$TARGET.$STAMP.bak"
        [ -e "$BKP" ] && die "$BKP already exists; run again in a second"
        cp -p "$TARGET" "$BKP" || die "could not back up $TARGET"
        [ "$(sha "$TARGET")" = "$(sha "$BKP")" ] || die "backup does not match the original; stopping before anything is replaced"
        say "backup: $BKP"
    fi
    TMP="$TARGET.new.$$"
    cp "$NEW" "$TMP" && chmod +x "$TMP" || die "could not stage the new launcher"
    [ "$(sha "$TMP")" = "$(sha "$NEW")" ] || { rm -f "$TMP"; die "staged copy does not match; nothing replaced"; }
    mv -f "$TMP" "$TARGET" || { rm -f "$TMP"; die "could not move the new launcher into place"; }
    say "installed: $TARGET   ($("$TARGET" --version))"
    say "try:  llama --serve --stuart-creator --print        (shows the command; starts nothing)"
    ;;
restore)
    LAST="$(ls -t "$TARGET".*.bak 2>/dev/null | head -1)"
    [ -n "$LAST" ] || die "no backup found next to $TARGET"
    if [ -f "$TARGET" ]; then
        SAVE="$TARGET.$STAMP.replaced"
        [ -e "$SAVE" ] && die "$SAVE already exists; run again in a second"
        cp -p "$TARGET" "$SAVE" || die "could not save the current launcher"
        say "current launcher saved as $SAVE"
    fi
    cp -p "$LAST" "$TARGET.restore.$$" && mv -f "$TARGET.restore.$$" "$TARGET" || die "restore failed"
    say "restored $LAST -> $TARGET"
    ;;
*)
    die "unknown mode '$MODE' (show | install | restore)"
    ;;
esac
