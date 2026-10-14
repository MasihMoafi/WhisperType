"""Global dictation hotkeys for X11 and Wayland."""

import json
import os
import select
import subprocess
import time

from pynput import keyboard


def send_wayland_paste(use_shift):
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland":
        return False
    if "gnome" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower():
        return False
    helper = os.path.join(os.path.dirname(__file__), "gnome_paste.py")
    result = subprocess.run(
        ["/usr/bin/python3", helper, "ctrl_shift_v" if use_shift else "ctrl_v"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"GNOME paste failed: {result.stderr.strip()}", flush=True)
        return False
    return True


def _keyboard_devices():
    from evdev import InputDevice, ecodes, list_devices

    devices = {}
    required = {ecodes.KEY_A, ecodes.KEY_Z, ecodes.KEY_SPACE, ecodes.KEY_ENTER}
    hotkeys = {ecodes.KEY_F8, ecodes.KEY_F9, ecodes.KEY_RIGHTCTRL}
    for path in list_devices():
        device = None
        try:
            device = InputDevice(path)
            keys = set(device.capabilities().get(ecodes.EV_KEY, []))
            if (required <= keys and keys & hotkeys
                    and ecodes.BTN_MOUSE not in keys
                    and not device.name.startswith("whispertype-")):
                devices[path] = device
                continue
        except OSError:
            pass
        if device is not None:
            device.close()
    return devices


def listen_for_hotkeys(on_press):
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland":
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        return

    from evdev import ecodes

    keymap = {
        ecodes.KEY_F8: keyboard.Key.f8,
        ecodes.KEY_F9: keyboard.Key.f9,
        ecodes.KEY_RIGHTCTRL: keyboard.Key.ctrl_r,
    }
    devices = _keyboard_devices()
    if not devices:
        raise RuntimeError(
            "No readable keyboard devices. Wayland hotkeys require read access "
            "to your keyboard in /dev/input/event*."
        )
    print("Wayland hotkeys: " + ", ".join(d.name for d in devices.values()), flush=True)
    refresh_at = time.monotonic() + 2
    try:
        while True:
            ready, _, _ = select.select(list(devices.values()), [], [], 1)
            for device in ready:
                try:
                    for event in device.read():
                        # Only key-down events. Releases, repeats and mouse events
                        # must never toggle the recorder.
                        if event.type == ecodes.EV_KEY and event.value == 1:
                            key = keymap.get(event.code)
                            if key is not None:
                                on_press(key)
                except OSError:
                    devices.pop(device.path, None)
                    device.close()
            if time.monotonic() >= refresh_at:
                discovered = _keyboard_devices()
                for path, device in discovered.items():
                    if path in devices:
                        device.close()
                    else:
                        devices[path] = device
                        print(f"Keyboard connected: {device.name}", flush=True)
                refresh_at = time.monotonic() + 2
    finally:
        for device in devices.values():
            device.close()


def write_engine_state(backend, model):
    """Record the running engine for `whispertype status`; the launcher sets the directory."""
    state_dir = os.environ.get("WHISPERTYPE_STATE_DIR")
    if state_dir:
        with open(os.path.join(state_dir, "engine.json"), "w") as f:
            json.dump({"backend": backend, "model": model}, f)
