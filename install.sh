#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${WHISPERTYPE_PYTHON:-$(command -v python3.12 || true)}"
WHISPER_DIR="$ROOT/whisper.cpp"
WHISPER_BUILD_DIR="$WHISPER_DIR/build-whispertype"
WHISPER_REV="a630b35c6fc02c8879f751ec3f39a61327f01dc7"
MODEL="$WHISPER_DIR/models/ggml-medium.en.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
MODEL_SHA256="cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356"

die() { echo "ERROR: $*" >&2; exit 1; }
for command_name in git cmake sha256sum nvidia-smi; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done
[[ -n "$PYTHON" && -x "$PYTHON" ]] || die "Python 3.12 is required"
"$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || die "Python 3.12 is required"
nvidia-smi >/dev/null 2>&1 || die "a working NVIDIA driver/GPU is required"

if [[ -e "$WHISPER_DIR" && ! -d "$WHISPER_DIR/.git" ]]; then
    die "$WHISPER_DIR exists but is not a Git checkout"
fi
if [[ ! -d "$WHISPER_DIR/.git" ]]; then
    git clone --no-checkout https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    git -C "$WHISPER_DIR" checkout --detach "$WHISPER_REV"
fi
CURRENT_REV="$(git -C "$WHISPER_DIR" rev-parse HEAD)"
if [[ "$CURRENT_REV" != "$WHISPER_REV" ]]; then
    die "whisper.cpp is at $CURRENT_REV; expected pinned revision $WHISPER_REV"
fi

"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements.lock"

cmake -S "$WHISPER_DIR" -B "$WHISPER_BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$WHISPER_BUILD_DIR" --config Release -j"$(nproc)"

mkdir -p "$(dirname "$MODEL")"
if [[ ! -f "$MODEL" ]]; then
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --output "$MODEL" "$MODEL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$MODEL" "$MODEL_URL"
    else
        die "curl or wget is required to download the model"
    fi
fi
echo "$MODEL_SHA256  $MODEL" | sha256sum --check --status || die "model checksum mismatch"

WHISPERTYPE_PYTHON="$ROOT/.venv/bin/python" "$ROOT/Linux/test_setup.sh"
"$ROOT/install-launcher.sh"
echo
echo "Installation verified. Run: whispertype (or vc), or open WhisperType from Applications."
