#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.12)"
else
    echo "ERROR: Python 3.12 not found. Run $ROOT/install.sh first." >&2
    exit 1
fi

if [[ -x "$ROOT/whisper.cpp/build-whispertype/bin/whisper-cli" ]]; then
    WHISPER_LIB_DIR="$ROOT/whisper.cpp/build-whispertype/bin"
else
    WHISPER_LIB_DIR="$ROOT/whisper.cpp/build/bin"
fi
export LD_LIBRARY_PATH="$WHISPER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "${1:-}" == "parakeet" ]]; then
    shift
    exec "$PYTHON" -u "$ROOT/parakeet-commander/portable_commander_parakeet.py" "$@"
fi
exec "$PYTHON" -u "$ROOT/Linux/voice_transcribe_gpu.py" "$@"
