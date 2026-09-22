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
WHISPER_BUILD_DIR = next(
    (
        path
        for path in (
            os.path.join(WHISPER_CPP_DIR, "build-whispertype"),
            os.path.join(WHISPER_CPP_DIR, "build"),
        )
        if os.path.isfile(os.path.join(path, "bin", "whisper-cli"))
    ),
    os.path.join(WHISPER_CPP_DIR, "build-whispertype"),
)
WHISPER_EXECUTABLE = os.path.join(WHISPER_BUILD_DIR, "bin", "whisper-cli")
WHISPER_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", "ggml-medium.en.bin")

# Library paths for CUDA
for lib_dir in ["bin", "src", "ggml/src", "ggml/src/ggml-cuda"]:
    path = os.path.join(WHISPER_BUILD_DIR, lib_dir)
    if os.path.exists(path):
        os.environ['LD_LIBRARY_PATH'] = path + ":" + os.environ.get('LD_LIBRARY_PATH', '')

HOTKEYS = [keyboard.Key.f8, keyboard.Key.f9, keyboard.Key.ctrl_r]
SAMPLERATE = 16000


def verify_gpu():
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def send_paste(controller: keyboard.Controller = None):
    import time
    time.sleep(0.05)
    mode = os.environ.get("VC_PASTE_MODE", "auto").strip().lower()
    if mode not in {"auto","ctrl_v","ctrl_shift_v"}:
        mode = "auto"

    use_shift = False
    if mode == "ctrl_shift_v":
        use_shift = True
    elif mode == "auto" and sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "x11":
        try:
            active = subprocess.check_output(["xprop", "-root", "_NET_ACTIVE_WINDOW"], text=True)
            win_id = active.strip().split()[-1]
            if win_id != "0x0":
                wm_class = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"], text=True).lower()
                terminals = {"gnome-terminal", "alacritty", "kitty", "konsole", "terminator", "xfce4-terminal", "tilix", "wezterm", "xterm", "urxvt"}
                if any(t in wm_class for t in terminals):
                    use_shift = True
        except Exception:
            pass

    # 1. Try kernel-level uinput injection via evdev (works on Wayland and X11)
    try:
        from evdev import UInput, ecodes
        cap = {ecodes.EV_KEY: [ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFTSHIFT, ecodes.KEY_V]}
        with UInput(cap, name="whispertype-virtual-keyboard") as ui:
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1)
            if use_shift:
                ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1)
            ui.syn()
            time.sleep(0.01)

            ui.write(ecodes.EV_KEY, ecodes.KEY_V, 1)
            ui.syn()
            time.sleep(0.01)
            ui.write(ecodes.EV_KEY, ecodes.KEY_V, 0)
            ui.syn()
            time.sleep(0.01)

            if use_shift:
                ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 0)
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 0)
            ui.syn()
            return
    except Exception:
        pass

    # 2. Try ydotool if installed
    try:
        env = os.environ.copy()
        uid = os.getuid()
        for sock in [f"/run/user/{uid}/ydotoold.socket", "/tmp/.ydotool_socket"]:
            if os.path.exists(sock):
                env["YDOTOOL_SOCKET"] = sock
                break
        keys = ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"] if use_shift else ["29:1", "47:1", "47:0", "29:0"]
        if subprocess.run(["ydotool", "key"] + keys, capture_output=True, env=env).returncode == 0:
            return
    except Exception:
        pass

    # 3. Fallback to pynput
    if controller is None:
        controller = keyboard.Controller()
    if use_shift:
        with controller.pressed(keyboard.Key.ctrl, keyboard.Key.shift):
            controller.press('v')
            controller.release('v')
    else:
        with controller.pressed(keyboard.Key.ctrl):
            controller.press('v')
            controller.release('v')



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
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16', device=None) as stream:
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
        send_paste(ctrl)
        print("Pasted.")


recorder = Recorder()
_last_toggle_time = 0
_toggle_lock = threading.Lock()


def toggle_recording():
    global _last_toggle_time
    import time
    with _toggle_lock:
        now = time.time()
        if now - _last_toggle_time < 0.3:
            return
        _last_toggle_time = now
        if recorder.recording:
            recorder.stop()
        else:
            recorder.start()


def is_hotkey(key):
    """Accept only explicitly configured pynput keys.

    Avoid raw numeric virtual-key fallbacks: on Linux, mouse/media devices
    can expose keyboard-like key codes and accidentally trigger recording.
    """
    return key in HOTKEYS


def on_press(key):
    if is_hotkey(key):
        toggle_recording()


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
