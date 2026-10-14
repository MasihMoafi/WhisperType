import importlib.util
import os
from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Linux"))

import desktop_input
from evdev import ecodes
from pynput import keyboard


class DesktopInputTests(unittest.TestCase):
    def test_wayland_uses_key_down_events_and_ignores_repeats_and_mouse(self):
        events = [
            (ecodes.EV_KEY, ecodes.KEY_RIGHTCTRL, 1),
            (ecodes.EV_KEY, ecodes.KEY_RIGHTCTRL, 2),
            (ecodes.EV_KEY, ecodes.KEY_RIGHTCTRL, 0),
            (ecodes.EV_KEY, ecodes.BTN_LEFT, 1),
            (ecodes.EV_REL, ecodes.KEY_F8, 1),
            (ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1),
            (ecodes.EV_KEY, ecodes.KEY_F8, 1),
            (ecodes.EV_KEY, ecodes.KEY_F9, 1),
        ]
        device = Mock(path="keyboard", name="device")
        device.name = "Test keyboard"
        device.read.return_value = [SimpleNamespace(type=t, code=c, value=v) for t, c, v in events]
        callback = Mock()
        with patch.dict(os.environ, XDG_SESSION_TYPE="wayland"), \
                patch.object(desktop_input, "_keyboard_devices", return_value={device.path: device}), \
                patch.object(desktop_input.select, "select", side_effect=[([device], [], []), KeyboardInterrupt]), \
                patch.object(keyboard, "Listener") as x11_listener:
            with self.assertRaises(KeyboardInterrupt):
                desktop_input.listen_for_hotkeys(callback)
        self.assertEqual([c.args[0] for c in callback.call_args_list],
                         [keyboard.Key.ctrl_r, keyboard.Key.f8, keyboard.Key.f9])
        x11_listener.assert_not_called()
        device.close.assert_called_once()

    def test_mouse_and_own_virtual_keyboard_are_excluded(self):
        keys = [ecodes.KEY_A, ecodes.KEY_Z, ecodes.KEY_SPACE, ecodes.KEY_ENTER, ecodes.KEY_RIGHTCTRL]
        physical = Mock(path="physical")
        physical.name = "Keychron"
        physical.capabilities.return_value = {ecodes.EV_KEY: keys}
        mouse = Mock(path="mouse")
        mouse.name = "Mouse"
        mouse.capabilities.return_value = {ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT]}
        virtual = Mock(path="virtual")
        virtual.name = "whispertype-virtual-keyboard"
        virtual.capabilities.return_value = {ecodes.EV_KEY: keys}
        with patch("evdev.list_devices", return_value=["physical", "mouse", "virtual"]), \
                patch("evdev.InputDevice", side_effect=[physical, mouse, virtual]):
            self.assertEqual(desktop_input._keyboard_devices(), {"physical": physical})
        physical.close.assert_not_called()
        mouse.close.assert_called_once()
        virtual.close.assert_called_once()

    def test_wayland_permission_failure_is_visible(self):
        with patch.dict(os.environ, XDG_SESSION_TYPE="wayland"), \
                patch.object(desktop_input, "_keyboard_devices", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "No readable keyboard"):
                desktop_input.listen_for_hotkeys(Mock())

    def test_x11_keeps_existing_listener(self):
        callback = Mock()
        with patch.dict(os.environ, XDG_SESSION_TYPE="x11"), \
                patch.object(keyboard, "Listener") as listener:
            desktop_input.listen_for_hotkeys(callback)
        listener.assert_called_once_with(on_press=callback)
        listener.return_value.__enter__.return_value.join.assert_called_once()

    def test_all_linux_engines_route_hotkeys_and_native_paste(self):
        paths = ["Linux/voice_transcribe_gpu.py", "Linux/portable_commander_gpu.py",
                 "parakeet-commander/portable_commander_parakeet.py"]
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                spec = importlib.util.spec_from_file_location(f"engine_{index}", ROOT / path)
                module = importlib.util.module_from_spec(spec)
                with patch.dict(os.environ, WHISPERTYPE_BACKEND="local"):
                    spec.loader.exec_module(module)
                with patch.object(module, "listen_for_hotkeys") as listener, \
                        patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                        patch.object(module.os.path, "exists", return_value=True):
                    module.main()
                listener.assert_called_once_with(module.on_press)
                with patch.object(module, "send_wayland_paste", return_value=True) as paste, \
                        patch.dict(os.environ, VC_PASTE_MODE="ctrl_shift_v"), \
                        patch.object(module.subprocess, "run") as fallback:
                    module.send_paste()
                paste.assert_called_once_with(True)
                fallback.assert_not_called()


def load_module(path, **env):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("WHISPERTYPE_")}
    with patch.dict(os.environ, {**clean, **env}, clear=True):
        spec.loader.exec_module(module)
    return module


def load_whisper(**env):
    return load_module("Linux/voice_transcribe_gpu.py", **env)


class BackendTests(unittest.TestCase):
    def test_default_backend_is_openrouter(self):
        self.assertEqual(load_whisper().BACKEND, "openrouter")

    def test_backend_is_selectable(self):
        self.assertEqual(load_whisper(WHISPERTYPE_BACKEND="local").BACKEND, "local")

    def test_cloud_uses_company_key_only(self):
        cloud = load_module("Linux/cloud_transcribe.py")
        response = Mock(json=Mock(return_value={"text": " hi "}))
        with patch.dict(os.environ, OPENROUTER_API_KEY="other", OPENROUTER_API_KEY_COMPANY="company"), \
                patch.object(cloud.SESSION, "post", return_value=response) as post, \
                patch.object(cloud, "encode_audio", return_value=(b"flac", "flac")):
            self.assertEqual(cloud.transcribe("x.wav"), "hi")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer company")
        self.assertEqual(post.call_args.kwargs["json"]["input_audio"]["format"], "flac")

    def test_audio_falls_back_to_original_without_ffmpeg(self):
        cloud = load_module("Linux/cloud_transcribe.py")
        with patch.object(cloud.subprocess, "run", side_effect=FileNotFoundError), \
                patch("builtins.open", unittest.mock.mock_open(read_data=b"wav")):
            self.assertEqual(cloud.encode_audio("x.wav"), (b"wav", "wav"))

    def test_recording_start_warms_cloud_connection(self):
        module = load_whisper()
        with patch.object(module.cloud, "warm") as warm, patch.object(module.sd, "InputStream"):
            module.Recorder().start()
            time.sleep(0.05)
        warm.assert_called_once()

    def test_engine_reports_its_model_for_status(self):
        module = load_whisper()
        with tempfile.TemporaryDirectory() as state, \
                patch.dict(os.environ, WHISPERTYPE_STATE_DIR=state, OPENROUTER_API_KEY_COMPANY="k"), \
                patch.object(module, "listen_for_hotkeys"):
            module.main()
            with open(os.path.join(state, "engine.json")) as f:
                self.assertEqual(json.load(f), {"backend": "openrouter", "model": "openai/gpt-4o-mini-transcribe"})

if __name__ == "__main__":
    unittest.main()
