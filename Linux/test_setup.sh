#!/bin/bash
# Quick test script to check Voice Commander setup

echo "=== Voice Commander Setup Check ==="
echo ""

# Check Python
echo "1. Python:"
if command -v python &> /dev/null; then
    python --version
    echo "   ✓ 'python' command available"
else
    echo "   ✗ 'python' command not found"
fi

if command -v python3 &> /dev/null; then
    python3 --version
    echo "   ✓ 'python3' command available"
else
    echo "   ✗ 'python3' command not found"
fi

echo ""
echo "2. Python Dependencies:"
python3 -c "import sounddevice" 2>/dev/null && echo "   ✓ sounddevice" || echo "   ✗ sounddevice"
python3 -c "import scipy" 2>/dev/null && echo "   ✓ scipy" || echo "   ✗ scipy"
python3 -c "import numpy" 2>/dev/null && echo "   ✓ numpy" || echo "   ✗ numpy"
python3 -c "import pyperclip" 2>/dev/null && echo "   ✓ pyperclip" || echo "   ✗ pyperclip"
python3 -c "import pynput" 2>/dev/null && echo "   ✓ pynput" || echo "   ✗ pynput"

echo ""
echo "3. NVIDIA GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "   ✓ NVIDIA GPU detected"
else
    echo "   ✗ NVIDIA GPU not detected"
fi

echo ""
echo "4. CUDA Toolkit:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "   ✓ CUDA toolkit installed"
else
    echo "   ✗ CUDA toolkit not found (optional for rebuild)"
fi

echo ""
echo "5. Whisper.cpp:"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
WHISPER_CLI="$BASE_DIR/whisper.cpp/build/bin/whisper-cli"

if [ -f "$WHISPER_CLI" ]; then
    echo "   ✓ whisper-cli found at: $WHISPER_CLI"
    
    # Check for CUDA support
    if ldd "$WHISPER_CLI" 2>/dev/null | grep -q cuda; then
        echo "   ✓ Built with CUDA support"
    else
        echo "   ⚠ Built WITHOUT CUDA support (CPU only)"
        echo "     Run ./Linux/setup_gpu.sh to rebuild with GPU support"
    fi
else
    echo "   ✗ whisper-cli not found"
fi

echo ""
echo "6. Whisper Model:"
MODEL_PATH="$BASE_DIR/whisper.cpp/models/ggml-medium.en.bin"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "   ✓ Model found: $SIZE"
else
    echo "   ✗ Model not found at: $MODEL_PATH"
fi

echo ""
echo "=== Summary ==="
if [ -f "$WHISPER_CLI" ] && [ -f "$MODEL_PATH" ]; then
    if ldd "$WHISPER_CLI" 2>/dev/null | grep -q cuda; then
        echo "✓ Ready to run with GPU acceleration!"
        echo "  Run: python Linux/portable_commander_gpu.py"
    else
        echo "⚠ Ready to run (CPU only)"
        echo "  For GPU acceleration, run: ./Linux/setup_gpu.sh"
        echo "  Then run: python Linux/portable_commander_gpu.py"
    fi
else
    echo "✗ Setup incomplete. Run: ./Linux/setup_gpu.sh"
fi
echo ""
