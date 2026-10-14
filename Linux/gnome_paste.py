"""Send paste through GNOME's native keyboard API using system PyGObject."""

import sys
import time

from gi.repository import Gio, GLib


def paste(use_shift):
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    destination = "org.gnome.Mutter.RemoteDesktop"
    interface = destination + ".Session"

    def call(path, iface, method, parameters=None):
        return bus.call_sync(
            destination, path, iface, method, parameters, None,
            Gio.DBusCallFlags.NONE, 5000, None,
        )

    path = call("/org/gnome/Mutter/RemoteDesktop", destination, "CreateSession").unpack()[0]
    call(path, interface, "Start")
    pressed = []
    try:
        time.sleep(0.05)
        keys = [29, 42, 47] if use_shift else [29, 47]  # Ctrl, Shift, V (evdev)
        for key in keys:
            call(path, interface, "NotifyKeyboardKeycode", GLib.Variant("(ub)", (key, True)))
            pressed.append(key)
            time.sleep(0.02)
    finally:
        for key in reversed(pressed):
            call(path, interface, "NotifyKeyboardKeycode", GLib.Variant("(ub)", (key, False)))
            time.sleep(0.02)
        call(path, interface, "Stop")


if __name__ == "__main__":
    paste(sys.argv[1] == "ctrl_shift_v")
