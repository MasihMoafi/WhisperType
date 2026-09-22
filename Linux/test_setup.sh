#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="$BASE_DIR/whisper.cpp/models/ggml-medium.en.bin"
if [[ -x "$BASE_DIR/whisper.cpp/build-whispertype/bin/whisper-cli" ]]; then
    WHISPER_BUILD_DIR="$BASE_DIR/whisper.cpp/build-whispertype"
else
    WHISPER_BUILD_DIR="$BASE_DIR/whisper.cpp/build"
fi
WHISPER_CLI="$WHISPER_BUILD_DIR/bin/whisper-cli"
WHISPER_LIB_DIR="$WHISPER_BUILD_DIR/bin"
EXPECTED_MODEL_SHA256="cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356"
FAILURES=0

pass() { printf '   ✓ %s\n' "$1"; }
fail() { printf '   ✗ %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

if [[ -n "${WHISPERTYPE_PYTHON:-}" ]]; then
    PYTHON="$WHISPERTYPE_PYTHON"
elif [[ -x "$BASE_DIR/.venv/bin/python" ]]; then
    PYTHON="$BASE_DIR/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.12)"
else
    PYTHON="$(command -v python3 2>/dev/null || true)"
fi

echo "=== WhisperType setup check ==="
echo
echo "1. Runtime Python:"
if [[ -n "$PYTHON" && -x "$PYTHON" ]]; then
    "$PYTHON" --version
    pass "$PYTHON"
else
    fail "Python 3.12 was not found"
fi

echo
echo "2. Python dependencies (using the runtime above):"
for module in sounddevice scipy numpy pyperclip pynput dotenv google.genai; do
    if [[ -n "$PYTHON" ]] && "$PYTHON" -c "import $module" >/dev/null 2>&1; then
        pass "$module"
    else
        fail "$module"
    fi
done

echo
echo "3. NVIDIA GPU:"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    pass "NVIDIA GPU detected"
else
    fail "NVIDIA GPU / working nvidia-smi not detected"
fi

echo
echo "4. CUDA build:"
if [[ ! -x "$WHISPER_CLI" ]]; then
    fail "whisper-cli not found at $WHISPER_CLI"
else
    pass "whisper-cli found"
    LDD_OUTPUT="$(LD_LIBRARY_PATH="$WHISPER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$WHISPER_CLI" 2>&1 || true)"
    if grep -Eq 'libggml-cuda|libcudart|libcublas' <<<"$LDD_OUTPUT"; then
        pass "CUDA-linked whisper.cpp build detected"
    else
        fail "whisper-cli is not linked to the expected CUDA libraries"
    fi
    if grep -q 'not found' <<<"$LDD_OUTPUT"; then
        fail "whisper-cli has unresolved shared libraries"
        grep 'not found' <<<"$LDD_OUTPUT" | sed 's/^/     /'
    fi
fi

echo
echo "5. Whisper model:"
if [[ -f "$MODEL_PATH" ]]; then
    ACTUAL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
    if [[ "$ACTUAL_SHA256" == "$EXPECTED_MODEL_SHA256" ]]; then
        pass "ggml-medium.en.bin checksum verified"
    else
        fail "ggml-medium.en.bin checksum mismatch"
    fi
else
    fail "model not found at $MODEL_PATH"
fi

echo
echo "=== Result ==="
if (( FAILURES == 0 )); then
    echo "✓ Runtime ready: $BASE_DIR/run-whisper.sh"
    exit 0
fi

echo "✗ $FAILURES required check(s) failed. Run ./install.sh, then retry."
exit 1
