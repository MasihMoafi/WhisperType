#!/usr/bin/env python3
"""Voice transcription with GPU acceleration - no external API required."""

import os
import subprocess
import tempfile
import uuid
import threading
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np
import pyperclip
from pynput import keyboard
import sys

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

WHISPER_CPP_DIR = os.path.join(BASE_DIR, "whisper.cpp")
WHISPER_EXECUTABLE = os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-cli")
WHISPER_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", "ggml-medium.en.bin")

# Library paths for CUDA
for lib_dir in ["build/src", "build/ggml/src", "build/ggml/src/ggml-cuda"]:
    path = os.path.join(WHISPER_CPP_DIR, lib_dir)
    if os.path.exists(path):
        os.environ['LD_LIBRARY_PATH'] = path + ":" + os.environ.get('LD_LIBRARY_PATH', '')

HOTKEYS = [keyboard.Key.f8, keyboard.Key.f9, keyboard.Key.ctrl_r]
SAMPLERATE = 16000


def verify_gpu():
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def get_paste_keys():
    """Return appropriate paste key combo based on focused window."""
    if not sys.platform.startswith("linux") or os.environ.get("XDG_SESSION_TYPE") != "x11":
        return keyboard.Key.ctrl, 'v'
    try:
        active = subprocess.check_output(["xprop", "-root", "_NET_ACTIVE_WINDOW"], text=True)
        win_id = active.strip().split()[-1]
        if win_id == "0x0":
            return keyboard.Key.ctrl, 'v'
        wm_class = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"], text=True).lower()
        terminals = {"gnome-terminal", "alacritty", "kitty", "konsole", "terminator", "xfce4-terminal", "tilix", "wezterm", "xterm", "urxvt"}
        if any(t in wm_class for t in terminals):
            return (keyboard.Key.ctrl, keyboard.Key.shift), 'v'
    except Exception:
        pass
    return keyboard.Key.ctrl, 'v'


class Recorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.recording:
                return
            print(">>> Recording... Press F8/F9/Right Ctrl to stop.")
            self.recording = True
            self.audio_data = []
            self.record_thread = threading.Thread(target=self._record, daemon=True)
            self.record_thread.start()

    def stop(self):
        with self.lock:
            if not self.recording:
                return
            print(">>> Processing...")
            self.recording = False
        
        def run_transcription():
            if hasattr(self, 'record_thread'):
                self.record_thread.join(timeout=1.0)
            self._transcribe()
            
        threading.Thread(target=run_transcription, daemon=True).start()

    def _record(self):
        chunk_size = 1600  # 100ms chunks at 16000Hz to reduce latency on stop
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16') as stream:
            while self.recording:
                chunk, _ = stream.read(chunk_size)
                with self.lock:
                    if self.recording:
                        self.audio_data.append(chunk)

    def _transcribe(self):
        with self.lock:
            if not self.audio_data:
                print("No audio.")
                return
            audio = np.concatenate(self.audio_data)

        wav_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        write_wav(wav_path, SAMPLERATE, audio)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        result = subprocess.run(
            [WHISPER_EXECUTABLE, "-m", WHISPER_MODEL_PATH, "-f", wav_path, "-nt"],
            capture_output=True, text=True, env=env
        )
        os.remove(wav_path)

        text = result.stdout.strip()
        if not text:
            if result.stderr:
                print(f"Error: {result.stderr}")
            return

        print(f"Transcribed: {text}")
        pyperclip.copy(text)
        
        ctrl = keyboard.Controller()
        mods, key = get_paste_keys()
        if isinstance(mods, tuple):
            with ctrl.pressed(*mods):
                ctrl.tap(key)
        else:
            with ctrl.pressed(mods):
                ctrl.tap(key)
        print("Pasted.")


recorder = Recorder()


def on_press(key):
    if key in HOTKEYS:
        if recorder.recording:
            recorder.stop()
        else:
            recorder.start()


def main():
    if not verify_gpu():
        print("ERROR: GPU not available. Ensure NVIDIA drivers and CUDA are installed.")
        sys.exit(1)

    print("Voice Transcribe (GPU) - Ready")
    print("  F8/F9/Right Ctrl: Start/Stop recording")
    print("  Transcribed text auto-pastes at cursor")
    print("  Ctrl+C to exit")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
