#!/usr/bin/env python3
"""Run object detection on two independent Raspberry Pi cameras."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time


def source(value):
    kind, sep, device = value.partition(":")
    if not sep or kind not in {"csi", "usb"} or not device:
        raise argparse.ArgumentTypeError("use csi:0, usb:0, or usb:/dev/v4l/by-id/…")
    if kind == "csi" and not device.isdecimal():
        raise argparse.ArgumentTypeError("a CSI camera requires a numeric index")
    return kind, int(device) if device.isdecimal() else device


def open_camera(cv2, stack, spec):
    kind, device = spec
    if kind == "csi":
        from picamera2 import Picamera2

        camera = Picamera2(device)
        stack.callback(camera.close)
        # Picamera2 RGB888 arrays use the BGR byte order OpenCV expects.
        camera.configure(camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 15.0},
        ))
        camera.start()
        stack.callback(camera.stop)
        return camera.capture_array

    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    stack.callback(camera.release)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open USB camera {device}")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read():
        for _ in range(3):
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                return frame
        raise RuntimeError(f"USB camera {device} opened but did not return a frame")

    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", type=source, nargs=2, metavar="CAMERA")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolox_nano.onnx"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--frames", type=int, default=0, help="stop after this many pairs; 0 runs continuously")
    parser.add_argument("--headless", action="store_true", help="run without opening a preview")
    parser.add_argument("--camera-check", action="store_true", help="read one pair without loading YOLO")
    parser.add_argument("--save-check", type=Path, help="save a single annotated pair to this directory")
    args = parser.parse_args()
    if args.cameras[0] == args.cameras[1]:
        parser.error("specify two different cameras")
    if not 0 < args.conf <= 1 or args.frames < 0:
        parser.error("--conf must be in (0,1] and --frames must be non-negative")

    import cv2
    import numpy as np

    cv2.setNumThreads(3)
    detector = None
    if not args.camera_check:
        from yolox_backend import Detector

        detector = Detector(args.model, args.conf)

    root = None
    running = True
    if not args.headless and not args.camera_check:
        import tkinter as tk

        root = tk.Tk()
        root.title("Dual Camera YOLOX | Q to close")
        panel = tk.Label(root, borderwidth=0)
        panel.pack()

        def stop(*_):
            nonlocal running
            running = False

        root.protocol("WM_DELETE_WINDOW", stop)
        root.bind("<Escape>", stop)
        root.bind("<KeyPress-q>", stop)

    try:
        with ExitStack() as stack:
            cameras = [open_camera(cv2, stack, spec) for spec in args.cameras]
            pair = 0
            while running:
                start = time.monotonic()
                frames = [read() for read in cameras]
                if args.camera_check:
                    for spec, frame in zip(args.cameras, frames):
                        print(f"{spec}: {frame.shape[1]}x{frame.shape[0]}", flush=True)
                    break
                annotated = []
                counts = []
                for index, frame in enumerate(frames):
                    marked, classes = detector.annotate(frame)
                    counts.append(len(classes))
                    view = cv2.resize(marked, (640, 480))
                    kind, device = args.cameras[index]
                    cv2.putText(view, f"Camera {index + 1} ({kind}:{Path(str(device)).name})", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
                    annotated.append(view)
                pair += 1
                if pair == 1 or pair % 20 == 0 or pair == args.frames:
                    print(f"Pair {pair}: {counts} detections in {time.monotonic() - start:.2f}s", flush=True)
                if args.save_check and pair == 1:
                    args.save_check.mkdir(parents=True, exist_ok=True)
                    for index, view in enumerate(annotated, 1):
                        path = args.save_check / f"camera-{index}.jpg"
                        if not cv2.imwrite(str(path), view):
                            raise RuntimeError(f"Could not save {path}")
                if root is not None:
                    width = min(1280, root.winfo_screenwidth() - 48)
                    height = width * 3 // 8
                    preview = cv2.resize(np.hstack(annotated), (width, height))
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    data = f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes()
                    photo = tk.PhotoImage(data=data, format="PPM")
                    panel.configure(image=photo)
                    root.update()
                if args.frames and pair >= args.frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if root is not None:
            root.destroy()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError, OSError) as error:
        sys.exit(f"Detection stopped: {error}")
