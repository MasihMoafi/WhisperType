#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APPLICATIONS_DIR/whispertype.desktop"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

install_link() {
    local source="$1"
    local target="$2"
    if [[ -L "$target" ]]; then
        [[ "$(readlink "$target")" == "$source" ]] || die "refusing to replace existing link: $target"
        return
    fi
    [[ ! -e "$target" ]] || die "refusing to replace existing file: $target"
    ln -s "$source" "$target"
}

mkdir -p "$BIN_DIR" "$APPLICATIONS_DIR"
install_link "$ROOT/whispertype" "$BIN_DIR/whispertype"
install_link "$ROOT/whispertype" "$BIN_DIR/vc"

if [[ -e "$DESKTOP_FILE" ]] && ! grep -Fq 'X-WhisperType-Managed=true' "$DESKTOP_FILE"; then
    die "refusing to replace unmanaged launcher: $DESKTOP_FILE"
fi

TEMP_DESKTOP="$(mktemp "$APPLICATIONS_DIR/.whispertype.desktop.XXXXXX")"
trap 'rm -f "$TEMP_DESKTOP"' EXIT
cat >"$TEMP_DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=WhisperType
Comment=Private GPU dictation at your cursor
Exec="$BIN_DIR/whispertype" start
Icon=audio-input-microphone
Terminal=true
Categories=Utility;
Keywords=voice;dictation;speech;transcription;whisper;
StartupNotify=false
X-WhisperType-Managed=true
EOF
chmod 0644 "$TEMP_DESKTOP"
mv "$TEMP_DESKTOP" "$DESKTOP_FILE"
trap - EXIT

printf 'Installed commands: %s and %s\n' "$BIN_DIR/whispertype" "$BIN_DIR/vc"
printf 'Installed application launcher: %s\n' "$DESKTOP_FILE"
