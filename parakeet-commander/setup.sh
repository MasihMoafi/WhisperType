#!/usr/bin/env bash
# setup.sh — Bootstrap parakeet.cpp (CUDA-only) for Voice-Commander
# Run from the repo root: bash parakeet-commander/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PARAKEET_DIR="$REPO_ROOT/parakeet.cpp"
MODEL_DIR="$PARAKEET_DIR/models"
MODEL_FILE="$MODEL_DIR/tdt-0.6b-v2-f16.gguf"
HF_MODEL_URL="https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v2-f16.gguf"

echo "=== parakeet.cpp setup (CUDA) for Voice-Commander ==="

# Require CUDA
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null 2>&1; then
    echo "ERROR: No NVIDIA GPU / nvidia-smi not found. This build is CUDA-only."
    exit 1
fi
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ── 1. Clone ──────────────────────────────────────────────────────────────────
if [ -d "$PARAKEET_DIR/.git" ]; then
    echo "[1/3] parakeet.cpp already cloned — pulling..."
    git -C "$PARAKEET_DIR" pull --ff-only
else
    echo "[1/3] Cloning mudler/parakeet.cpp..."
    git clone --recursive https://github.com/mudler/parakeet.cpp "$PARAKEET_DIR"
fi

# ── 2. Build with CUDA ────────────────────────────────────────────────────────
echo "[2/3] Building parakeet-cli with PARAKEET_GGML_CUDA=ON..."
cmake -B "$PARAKEET_DIR/build" \
      -S "$PARAKEET_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPARAKEET_BUILD_CLI=ON \
      -DPARAKEET_GGML_CUDA=ON

cmake --build "$PARAKEET_DIR/build" --config Release -j"$(nproc)"

# Locate binary (cmake version-dependent path)
PARAKEET_CLI="$PARAKEET_DIR/build/bin/parakeet-cli"
[ -f "$PARAKEET_CLI" ] || PARAKEET_CLI="$PARAKEET_DIR/build/examples/cli/parakeet-cli"
[ -f "$PARAKEET_CLI" ] || { echo "ERROR: parakeet-cli binary not found after build."; exit 1; }
echo "      Binary: $PARAKEET_CLI"

# ── 3. Download model ─────────────────────────────────────────────────────────
echo "[3/3] Downloading tdt_ctc-110m-f16.gguf (~268 MB)..."
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_FILE" ]; then
    echo "      Already present, skipping."
else
    if command -v wget &>/dev/null; then
        wget -c -O "$MODEL_FILE" "$HF_MODEL_URL"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$MODEL_FILE" "$HF_MODEL_URL"
    else
        echo "ERROR: need wget or curl"; exit 1
    fi
fi

echo
echo "=== Done. Run with:"
echo "    python $SCRIPT_DIR/portable_commander_parakeet.py"
