#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

export HOME="$TEST_ROOT/home"
mkdir -p "$HOME"

"$ROOT/install-launcher.sh"

test "$(readlink "$HOME/.local/bin/whispertype")" = "$ROOT/whispertype"
test "$(readlink "$HOME/.local/bin/vc")" = "$ROOT/whispertype"
grep -Fq "Exec=\"$HOME/.local/bin/whispertype\" start" "$HOME/.local/share/applications/whispertype.desktop"
grep -Fq "Terminal=true" "$HOME/.local/share/applications/whispertype.desktop"

"$HOME/.local/bin/whispertype" --help | grep -Fq "Usage: whispertype"
if "$HOME/.local/bin/whispertype" unknown-command >/dev/null 2>&1; then
    echo "unknown commands must fail" >&2
    exit 1
fi

# The installed symlink resolves back to this checkout, and installation is safe to repeat.
"$HOME/.local/bin/vc" install-launcher
test "$(readlink "$HOME/.local/bin/vc")" = "$ROOT/whispertype"

echo "launcher contract passed"
