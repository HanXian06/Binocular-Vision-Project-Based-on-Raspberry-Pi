"""Render the latest paired camera images without waiting for the detector."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from multiprocessing import get_context
import os
from threading import Event, Thread
import time

import cv2
import numpy as np

from stereo_fusion import Detection, StereoFusion, clip_box, draw


def init_worker(model, conf, threads, independent, cores):
    global _detector, _fusion, _eligible
    from ncnn_backend import Detector

    if cores:
        os.sched_setaffinity(0, cores)
    cv2.setNumThreads(1)
    _detector = Detector(model, conf, threads)
    _fusion = None if independent else StereoFusion()
    _eligible = 0


def identify(frames, stamps):
    global _eligible
    results = [[Detection(*row) for row in _detector.detect(frame)] for frame in frames]
    if (_fusion is not None and all(stamp is not None for stamp in stamps)
            and abs(stamps[0] - stamps[1]) <= 45_000_000):
        _eligible += 1
        results, _ = _fusion.merge(frames, results, _detector if _eligible % 3 == 1 else None)
    return results


def thumbnail(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                      (160, 120), interpolation=cv2.INTER_AREA)


def track_box(item, previous, current, shape):
    """Shift a display box only if its texture is still present nearby."""
    scale = np.array([shape[1] / 160, shape[0] / 120])
    x, y, w, h = np.rint(clip_box(item.box / np.tile(scale, 2), previous.shape)).astype(int)
    if min(w, h) < 5:
        return None
    patch = previous[y:y+h, x:x+w]
    if patch.size == 0:
        return None
    if float(patch.std()) < 7:
        if np.mean(cv2.absdiff(patch, current[y:y+h, x:x+w])) > 4:
            return None
        return item
    margin = 7
    x0, y0 = max(0, x-margin), max(0, y-margin)
    region = current[y0:min(120, y+h+margin), x0:min(160, x+w+margin)]
    response = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, (dx, dy) = cv2.minMaxLoc(response)
    if score < .83:
        return None
    displacement = np.r_[(np.array((x0+dx-x, y0+dy-y)) * scale), 0, 0]
    return replace(item, box=clip_box(item.box + displacement, shape))


def run_preview(args, open_camera):
    import tkinter as tk

    root = tk.Tk()
    root.title("Dual Camera YOLOE Stereo | Q to close")
    panel = tk.Label(root, borderwidth=0)
    panel.pack()
    width = min(800, root.winfo_screenwidth() - 48) & ~1
    height = width * 3 // 8
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    header = f"P6\n{width} {height}\n255\n".encode()
    photo = tk.PhotoImage(width=width, height=height)
    panel.configure(image=photo)

    latest = None
    failure = None
    stop = Event()
    pending = None
    accepted = None
    completed = 0
    last_display_sequence = -1
    previous = None
    last_submit = -1
    intervals = []
    total_frames = 0
    started = time.monotonic()

    def capture():
        nonlocal latest, failure
        try:
            with ExitStack() as stack:
                cameras = [open_camera(cv2, stack, spec) for spec in args.cameras]
                sequence = 0
                while not stop.is_set():
                    frames, stamps = zip(*(read() for read in cameras))
                    sequence += 1
                    latest = sequence, time.monotonic(), frames, stamps
        except Exception as exc:
            failure = exc

    capture_thread = Thread(target=capture, name="preview-capture", daemon=True)
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    reserved = cores[:-1] if len(cores) == 4 else []
    worker_threads = min(args.threads, len(reserved)) if reserved else args.threads
    pool = ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=init_worker,
                               initargs=(args.model, args.conf, worker_threads, args.independent, reserved))

    def render(snapshot):
        nonlocal last_display_sequence, previous, total_frames
        frame_id, timestamp, frames, _ = snapshot
        for side, source in enumerate(frames):
            frame = source.copy()
            if accepted is not None:
                seq, then, thumbnails, results = accepted
                if frame_id == seq:
                    draw(frame, results[side])
                elif 0 <= timestamp - then <= .60:
                    gray = thumbnail(frame)
                    matches = [track_box(d, thumbnails[side], gray, frame.shape) for d in results[side]]
                    draw(frame, [match for match in matches if match is not None])
            kind, device = args.cameras[side]
            cv2.putText(frame, f"Camera {side + 1} ({kind}:{device})", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            half = width // 2
            preview = cv2.resize(frame, (half, height))
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB, dst=rgb[:, side*half:(side+1)*half])
        photo.configure(data=header + rgb.tobytes(), format="PPM")
        last_display_sequence = frame_id
        now = time.monotonic()
        if previous is not None:
            intervals.append(now - previous)
            if len(intervals) > 200:
                del intervals[0]
        previous = now
        total_frames += 1

    def tick():
        nonlocal pending, accepted, completed, last_submit, failure
        try:
            if failure is not None or stop.is_set():
                root.quit()
                return
            if pending is not None and pending[0].done():
                future, (seq, timestamp, frames, _) = pending
                results = future.result()
                accepted = seq, timestamp, tuple(map(thumbnail, frames)), results
                pending = None
                completed += 1
                if completed % 30 == 0 or completed == args.frames:
                    total_time = time.monotonic() - started
                    p95 = np.percentile(intervals, 95) * 1000 if intervals else 0
                    print(f"Analysis {completed}: image={total_frames/total_time:.2f} FPS/camera "
                          f"analysis={completed/total_time:.2f} FPS/camera image_p95={p95:.1f}ms",
                          flush=True)
                if args.frames and completed >= args.frames:
                    root.quit()
                    return
            snapshot = latest
            if snapshot is not None:
                seq, _, frames, stamps = snapshot
                if pending is None and seq != last_submit:
                    pending = pool.submit(identify, frames, stamps), snapshot
                    last_submit = seq
                if seq != last_display_sequence and (previous is None or time.monotonic() - previous >= .072):
                    render(snapshot)
            root.after(12, tick)
        except Exception as exc:
            failure = exc
            root.quit()

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.bind("<Escape>", lambda _: root.quit())
    root.bind("<KeyPress-q>", lambda _: root.quit())
    capture_thread.start()
    root.after(0, tick)
    try:
        root.mainloop()
    finally:
        stop.set()
        capture_thread.join(timeout=2)
        pool.shutdown(wait=True, cancel_futures=True)
        root.destroy()
    if failure is not None:
        raise RuntimeError(str(failure)) from failure
