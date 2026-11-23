"""YOLO26 NCNN CPU inference with resize and channel conversion in NCNN."""

import cv2
import ncnn
import numpy as np

from coco_labels import CLASSES


class Detector:
    def __init__(self, model, threshold, threads=4, fp16=True, *, reference=False):
        self.threshold = threshold
        labels = model / "labels.txt"
        self.open_vocabulary = labels.exists()
        self.classes = labels.read_text(encoding="utf-8").splitlines() if self.open_vocabulary else CLASSES
        if not self.classes or any(not name.strip() for name in self.classes):
            raise ValueError(f"Invalid labels file: {labels}")
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        self.net.opt.use_packing_layout = True
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_arithmetic = fp16
        params = model / ("reference.param" if reference else "model.ncnn.param")
        if self.net.load_param(str(params)) != 0:
            raise ValueError(f"Cannot read NCNN graph: {params}")
        if self.net.load_model(str(model / "model.ncnn.bin")) != 0:
            raise ValueError(f"Cannot read NCNN weights: {model}")
        # The fixed shape and the optional logits output are set by export_model.py.
        shape, mode = (model / "input.txt").read_text().split()
        self.h, self.w = map(int, shape.split("x"))
        self.logits = mode == "logits" and not reference
        p = min(threshold, 1 - 1e-7)
        self.cutoff = float(np.log(p / (1 - p))) if self.logits else threshold
        self.input_name, = self.net.input_names()
        self.output_name, = self.net.output_names()

    def infer(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.w / w, self.h / h)
        rw, rh = round(w * scale), round(h * scale)
        pixels = ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(frame), ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, rw, rh
        )
        top, left = (self.h - rh) // 2, (self.w - rw) // 2
        if rw != self.w or rh != self.h:
            pixels = ncnn.copy_make_border(
                pixels, top, self.h - rh - top, left, self.w - rw - left,
                ncnn.BorderType.BORDER_CONSTANT, 114.0,
            )
        pixels.substract_mean_normalize([], [1 / 255] * 3)
        with self.net.create_extractor() as ex:
            ex.set_light_mode(True)
            status = ex.input(self.input_name, pixels)
            if status != 0:
                raise RuntimeError(f"NCNN input failed: {status}")
            status, output = ex.extract(self.output_name)
            if status != 0:
                raise RuntimeError(f"NCNN inference failed: {status}")
        return np.asarray(output).squeeze(), scale, (left, top)

    def detect(self, frame):
        h, w = frame.shape[:2]
        raw, scale, origin = self.infer(frame)
        if raw.ndim != 2 or raw.shape[0] != 4 + len(self.classes):
            raise ValueError(f"Unexpected YOLO26 output shape: {raw.shape}")

        # Reduce channels first, then select candidates and do coordinate math
        # only for boxes that can pass the confidence threshold.
        class_matrix = raw[4:]
        best = class_matrix.max(axis=0)
        ids = np.flatnonzero(best > self.cutoff)
        if not ids.size:
            return ()
        labels = class_matrix[:, ids].argmax(axis=0)
        scores = best[ids]
        if self.logits:
            scores = 1 / (1 + np.exp(-scores))
        centers = raw[:2, ids].T
        half_sizes = raw[2:4, ids].T * 0.5
        low = np.maximum((centers - half_sizes - origin) / scale, 0)
        high = np.minimum((centers + half_sizes - origin) / scale, (w - 1, h - 1))
        size = high - low
        good = np.all(size > 1, axis=1)
        boxes = np.concatenate((low, size), axis=1)[good]
        scores, labels = scores[good], labels[good]
        if self.open_vocabulary:
            # Expanded vocabularies often give overlapping names to one object.
            keep = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), self.threshold, 0.45, top_k=300
            )
        else:
            keep = cv2.dnn.NMSBoxesBatched(
                boxes.tolist(), scores.tolist(), labels.tolist(), self.threshold, 0.45, top_k=300
            )
        return tuple((boxes[i], float(scores[i]), self.classes[labels[i]]) for i in keep)

    def annotate(self, frame):
        names = []
        for box, score, name in self.detect(frame):
            x, y, w, h = box.astype(int)
            names.append(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 235, 100), 2)
            cv2.putText(frame, f"{name} {score:.0%}", (x, max(16, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (50, 235, 100), 2, cv2.LINE_AA)
        return frame, names
