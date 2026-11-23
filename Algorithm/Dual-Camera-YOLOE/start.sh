#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LIBCAMERA_LOG_LEVELS=ERROR

# Coordinate with the existing on-device installation when both are present.
lock=.camera.lock
if [ -d "$HOME/pi-dual-yolo" ]; then
    lock="$HOME/pi-dual-yolo/.camera.lock"
fi
exec flock -n "$lock" .venv/bin/python detect.py csi:0 csi:1 "$@"
