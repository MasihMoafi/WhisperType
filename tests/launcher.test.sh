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

# One engine at a time: a second start must refuse while the first holds the lock.
export XDG_RUNTIME_DIR="$TEST_ROOT/run" WHISPERTYPE_ENGINE="sleep 30"
mkdir -p "$XDG_RUNTIME_DIR"
"$HOME/.local/bin/whispertype" status | grep -Fq "stopped"
"$HOME/.local/bin/whispertype" start --background >/dev/null
"$HOME/.local/bin/whispertype" status | grep -Fq "running"
"$HOME/.local/bin/whispertype" status --json | jq -e '.running == true and (.pid | type) == "number"' >/dev/null
if "$HOME/.local/bin/whispertype" start >/dev/null 2>&1; then
    echo "second engine must be refused" >&2
    exit 1
fi
"$HOME/.local/bin/whispertype" stop >/dev/null
"$HOME/.local/bin/whispertype" status --json | jq -e '.running == false' >/dev/null

# The menu maps a choice to a command; 8 quits without starting anything.
printf '8\n' | "$HOME/.local/bin/whispertype" menu 2>&1 | grep -Fq "OpenRouter"

echo "launcher contract passed"
