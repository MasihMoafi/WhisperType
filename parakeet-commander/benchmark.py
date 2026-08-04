#!/usr/bin/env python3
"""Sequential Benchmark: Whisper Medium vs Parakeet 0.6B on CUDA GPU"""
import os, sys, time, tempfile, uuid, subprocess
import numpy as np
from scipy.io.wavfile import write as write_wav

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Audio fixture: 3-second 16kHz audio with sample tone/noise
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), False)
audio = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)
wav_path = os.path.join(tempfile.gettempdir(), f"bench_{uuid.uuid4()}.wav")
write_wav(wav_path, sr, audio)

print(f"=== Voice Commander GPU Benchmark ({duration}s audio fixture) ===")

# 1. Whisper Medium (CUDA)
whisper_exe = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "whisper-cli")
whisper_model = os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-medium.en.bin")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

print("\n[1/2] Running Whisper Medium (CUDA)...")
t0 = time.perf_counter()
res_w = subprocess.run([whisper_exe, "-m", whisper_model, "-f", wav_path, "-nt"], capture_output=True, text=True, env=env)
t_whisper = (time.perf_counter() - t0) * 1000.0

# 2. Parakeet 0.6B (CUDA)
_cli_a = os.path.join(BASE_DIR, "parakeet.cpp", "build", "bin", "parakeet-cli")
_cli_b = os.path.join(BASE_DIR, "parakeet.cpp", "build", "examples", "cli", "parakeet-cli")
parakeet_exe = _cli_a if os.path.exists(_cli_a) else _cli_b
parakeet_model = os.path.join(BASE_DIR, "parakeet.cpp", "models", "tdt-0.6b-v2-f16.gguf")

env["PARAKEET_DEVICE"] = "CUDA0"

print("[2/2] Running Parakeet 0.6B (CUDA)...")
t0 = time.perf_counter()
res_p = subprocess.run([parakeet_exe, "transcribe", "--model", parakeet_model, "--input", wav_path], capture_output=True, text=True, env=env)
t_parakeet = (time.perf_counter() - t0) * 1000.0

os.remove(wav_path)

print("\n" + "="*45)
print(f"Whisper Medium (GPU)  : {t_whisper:.1f} ms")
print(f"Parakeet 0.6B  (GPU)  : {t_parakeet:.1f} ms")
print(f"Speedup Ratio         : {t_whisper / max(t_parakeet, 0.1):.2f}x faster")
print("="*45)
