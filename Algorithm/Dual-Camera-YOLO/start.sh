#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LIBCAMERA_LOG_LEVELS=ERROR
exec flock -n .camera.lock .venv/bin/python detect.py csi:0 csi:1 "$@"
