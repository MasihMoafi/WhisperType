#!/usr/bin/env python3
"""Voice Commander — parakeet.cpp backend (CUDA-only, no LLM).

Hotkeys: F8 / F9 / Right Ctrl — toggle recording.
Transcribed text is pasted directly at the cursor, no refinement step.

Setup: bash parakeet-commander/setup.sh  (from repo root)
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid

import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard
from scipy.io.wavfile import write as write_wav

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
PARAKEET_DIR = os.path.join(REPO_ROOT, "parakeet.cpp")

_CLI_A = os.path.join(PARAKEET_DIR, "build", "bin", "parakeet-cli")
_CLI_B = os.path.join(PARAKEET_DIR, "build", "examples", "cli", "parakeet-cli")
PARAKEET_EXECUTABLE = _CLI_A if os.path.exists(_CLI_A) else _CLI_B
PARAKEET_MODEL_PATH = os.path.join(PARAKEET_DIR, "models", "tdt-0.6b-v2-f16.gguf")

HOTKEYS    = [keyboard.Key.f8, keyboard.Key.f9, keyboard.Key.ctrl_r]
SAMPLERATE = 16000

# ── HELPERS ───────────────────────────────────────────────────────────────────

def verify_setup() -> bool:
    ok = True
    if not subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
        print("ERROR: No NVIDIA GPU detected. This script is CUDA-only.")
        ok = False
    if not os.path.exists(PARAKEET_EXECUTABLE):
        print(f"ERROR: parakeet-cli not found: {PARAKEET_EXECUTABLE}")
        print("       Run: bash parakeet-commander/setup.sh")
        ok = False
    if not os.path.exists(PARAKEET_MODEL_PATH):
        print(f"ERROR: Model not found: {PARAKEET_MODEL_PATH}")
        print("       Run: bash parakeet-commander/setup.sh")
        ok = False
    return ok


# Common spoken filler words / vocal hesitations (case-insensitive)
FILLER_WORDS_RE = re.compile(
    r"\b(um+s?|uh+s?|ah+s?|er+s?|hm+s?|hmm+s?|mhm+s?)\b",
    re.IGNORECASE,
)

def strip_filler_words(text: str) -> str:
    cleaned = FILLER_WORDS_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()

# Commonly mistranscribed words -> intended word (case-insensitive)
WORD_REPLACEMENTS = {}

def apply_word_replacements(text: str) -> str:
    for misheard, correct in WORD_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(misheard)}\b", re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text

_TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\]")

def _clean(raw: str) -> str:
    """Strip ggml log noise, timestamp prefixes, and spoken filler words."""
    lines = []
    for line in raw.splitlines():
        line_str = line.strip()
        if any(line_str.startswith(p) for p in ["ggml_", "[parakeet]", "pk::", "system_info:"]):
            continue
        if "CUDA graph warmup" in line_str or "VRAM:" in line_str:
            continue
        cleaned_line = _TIMESTAMP_RE.sub("", line_str).strip()
        if cleaned_line:
            lines.append(cleaned_line)
    cleaned_text = " ".join(lines).strip()
    cleaned_text = strip_filler_words(cleaned_text)
    return apply_word_replacements(cleaned_text)


def _get_active_wm_classes():
    try:
        win_id = subprocess.check_output(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"], text=True
        ).strip().split()[-1]
        if win_id == "0x0":
            return []
        line = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"], text=True)
        return [c.strip().strip('"').lower() for c in line.split("=", 1)[1].split(",")]
    except Exception:
        return []

_TERMINALS = {
    "gnome-terminal", "org.gnome.terminal", "gnome-terminal-server",
    "alacritty", "kitty", "konsole", "yakuake", "terminator",
    "xfce4-terminal", "mate-terminal", "tilix", "guake",
    "wezterm", "foot", "xterm", "urxvt", "st",
}

def _terminal_focused() -> bool:
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11":
        return False
    return any(c in _TERMINALS for c in _get_active_wm_classes())


def send_paste(ctrl: keyboard.Controller):
    mode = os.environ.get("VC_PASTE_MODE", "auto").strip().lower()
    if mode not in {"auto", "ctrl_v", "ctrl_shift_v"}:
        mode = "auto"
    use_shift = (
        mode == "ctrl_shift_v"
        or (mode == "auto" and sys.platform.startswith("linux") and _terminal_focused())
    )
    if use_shift:
        with ctrl.pressed(keyboard.Key.ctrl, keyboard.Key.shift):
            ctrl.press("v"); ctrl.release("v")
    else:
        with ctrl.pressed(keyboard.Key.ctrl):
            ctrl.press("v"); ctrl.release("v")


# ── RECORDER ──────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.recording  = False
        self.audio_data = []
        self.lock       = threading.Lock()

    def start(self):
        with self.lock:
            if self.recording:
                return
            print(">>> Recording... (F8/F9/Right Ctrl to stop)")
            self.recording  = True
            self.audio_data = []
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()

    def stop_and_process(self):
        with self.lock:
            if not self.recording:
                return
            print(">>> Stopped. Transcribing...")
            self.recording = False
        threading.Thread(target=self._after_stop, daemon=True).start()

    def _after_stop(self):
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self._process()

    def _record_loop(self):
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype="int16") as s:
            while self.recording:
                chunk, _ = s.read(1600)  # 100 ms
                with self.lock:
                    if self.recording:
                        self.audio_data.append(chunk)

    def _process(self):
        with self.lock:
            if not self.audio_data:
                print("No audio."); return
            audio = np.concatenate(self.audio_data)

        wav = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        write_wav(wav, SAMPLERATE, audio)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"   # first GPU, mirrors whisper variant
        env["PARAKEET_DEVICE"]      = "CUDA0"  # explicit CUDA selection

        result = subprocess.run(
            [PARAKEET_EXECUTABLE, "transcribe",
             "--model", PARAKEET_MODEL_PATH,
             "--input", wav],
            capture_output=True, text=True, env=env,
        )

        try:
            os.remove(wav)
        except OSError:
            pass

        if result.returncode != 0:
            print(f"parakeet-cli error (exit {result.returncode}):\n{result.stderr}")
            return

        text = _clean(result.stdout) or _clean(result.stderr)
        if not text:
            print("No transcription produced."); return

        print(f"Transcribed: {text}")
        pyperclip.copy(text)
        print("Pasting...")
        ctrl = keyboard.Controller()
        send_paste(ctrl)


# ── WIRING ────────────────────────────────────────────────────────────────────

recorder = Recorder()

def on_press(key):
    if key in HOTKEYS:
        if recorder.recording:
            recorder.stop_and_process()
        else:
            recorder.start()

def main():
    if not verify_setup():
        sys.exit(1)

    print("=" * 55)
    print("VoiceCommander — parakeet.cpp / CUDA")
    print(f"  binary : {PARAKEET_EXECUTABLE}")
    print(f"  model  : tdt-0.6b-v2-f16.gguf")
    print(f"  device : CUDA0")
    print()
    print("  F8 / F9 / Right Ctrl — record → transcribe → paste")
    print("  Ctrl+C to exit")
    print("=" * 55)

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()
