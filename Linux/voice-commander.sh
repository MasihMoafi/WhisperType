#!/bin/bash
# Wrapper to fix conda libstdc++ conflict
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
exec python "$(dirname "$0")/portable_commander_gpu.py" "$@"
