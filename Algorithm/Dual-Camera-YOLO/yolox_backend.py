"""Inference for the official 416px YOLOX-Nano ONNX export."""

import cv2
import numpy as np

CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,"
    "tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,"
    "orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,"
    "dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,"
    "toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"
).split(",")


class Detector:
    size = 416

    def __init__(self, path, threshold):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.threshold = threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = self.size // stride
            x, y = np.meshgrid(np.arange(side), np.arange(side))
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((side * side, 1), stride))
        self.grid = np.concatenate(grids)
        self.stride = np.concatenate(strides)

    def annotate(self, frame):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas))
        pred = self.net.forward().reshape(-1, 5 + len(CLASSES))
        if len(pred) != len(self.grid):
            raise RuntimeError(f"Unexpected YOLOX output shape: {pred.shape}")
        scores = pred[:, 4:5] * pred[:, 5:]
        classes = scores.argmax(axis=1)
        confidence = scores.max(axis=1)
        selected = confidence >= self.threshold
        centers = (pred[selected, :2] + self.grid[selected]) * self.stride[selected]
        sizes = np.exp(pred[selected, 2:4]) * self.stride[selected]
        low = np.maximum((centers - sizes / 2) / scale, (0, 0))
        high = np.minimum((centers + sizes / 2) / scale, (w - 1, h - 1))
        boxes = np.concatenate((low, high - low), axis=1)
        valid = np.all(boxes[:, 2:] > 1, axis=1)
        boxes = boxes[valid]
        labels = classes[selected][valid]
        probs = confidence[selected][valid]
        out = frame.copy()
        names = []
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label)
            keep = cv2.dnn.NMSBoxes(boxes[ids].tolist(), probs[ids].tolist(), self.threshold, .45)
            for idx in np.asarray(keep, dtype=int).ravel():
                i = ids[idx]
                x, y, bw, bh = boxes[i].astype(int)
                name = CLASSES[label]
                names.append(name)
                cv2.rectangle(out, (x, y), (x + bw, y + bh), (50, 235, 100), 2)
                cv2.putText(out, f"{name} {probs[i]:.0%}", (x, max(16, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (50, 235, 100), 2, cv2.LINE_AA)
        return out, names
