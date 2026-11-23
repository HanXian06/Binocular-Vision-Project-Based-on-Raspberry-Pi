"""Associate detections through local image correspondences, with optional crop rechecks."""

from dataclasses import dataclass

import cv2
import numpy as np


PERSON_PARTS = {"jacket", "shirt", "pants", "tie", "hat", "shoe", "socks", "gloves",
                "face mask", "belt", "eyeglasses", "smartwatch", "wristwatch", "watch band"}


@dataclass(frozen=True)
class Detection:
    box: np.ndarray  # x, y, width, height
    score: float
    label: str
    pair: int = 0
    status: str = "single"


def overlap(a, b):
    area = np.maximum(np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return float(area / max(a[2:].prod() + b[2:].prod() - area, 1))


def inside(points, box):
    return np.all((points >= box[:2]) & (points <= box[:2] + box[2:]), axis=1)


def clip_box(box, shape):
    h, w = shape[:2]
    lo = np.clip(box[:2], (0, 0), (w, h))
    hi = np.clip(box[:2] + box[2:], (0, 0), (w, h))
    return np.concatenate((lo, hi - lo))


class StereoFusion:
    def __init__(self):
        self.features = cv2.ORB_create(nfeatures=600, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def correspondences(self, frames):
        keypoints, descriptors = zip(*(self.features.detectAndCompute(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None) for frame in frames))
        if any(d is None or len(d) < 2 for d in descriptors):
            return np.empty((2, 0, 2))

        def neighbors(src, dst):
            return {m.queryIdx: m.trainIdx for m, n in self.matcher.knnMatch(src, dst, k=2)
                    if m.distance < .75 * n.distance}

        forward = neighbors(*descriptors)
        reverse = neighbors(*reversed(descriptors))
        pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
        if not pairs:
            return np.empty((2, 0, 2))
        return np.array([[keypoints[side][pair[side]].pt for pair in pairs] for side in (0, 1)])

    @staticmethod
    def project(box, points, shape, other=None):
        select = inside(points[0], box)
        if other is not None:
            select &= inside(points[1], other)
        a, b = points[:, select]
        if len(a) < 5:
            return None
        displacement = b - a
        median = np.median(displacement, axis=0)
        inliers = np.linalg.norm(displacement - median, axis=1) <= 6
        count = np.count_nonzero(inliers)
        if count < 5 or count / len(a) < .6:
            return None
        # Reject a small cluster of shared texture in otherwise unrelated boxes.
        spread = np.ptp(a[inliers], axis=0) / np.maximum(box[2:], 1)
        if spread.max() < .35 or spread.min() < .14:
            return None
        projected = clip_box(box + np.r_[np.median(displacement[inliers], axis=0), 0, 0], shape)
        if np.any(projected[2:] < 12) or projected[2:].prod() < .6 * box[2:].prod():
            return None
        return projected, count

    @staticmethod
    def recheck(detector, frame, expected_box, labels):
        padded = expected_box + np.r_[-.18 * expected_box[2:], .36 * expected_box[2:]]
        x, y, w, h = np.rint(clip_box(padded, frame.shape)).astype(int)
        found = []
        for box, score, label in detector.detect(frame[y:y+h, x:x+w]):
            box = box + [x, y, 0, 0]
            if label in labels and overlap(box, expected_box) >= .35:
                found.append(Detection(box, score, label))
        return max(found, key=lambda item: item.score, default=None)

    def merge(self, frames, detections, detector=None):
        """Return per-view boxes with pair-local IDs. A crop budget is one call at most."""
        result = [list(view) for view in detections]
        stats = {"matched": 0, "review": 0, "rescued": 0, "conflict": 0}
        if not any(result):
            return result, stats
        points = self.correspondences(frames)
        candidates = {}
        for i, a in enumerate(result[0]):
            for j, b in enumerate(result[1]):
                # Shared keypoints on clothing do not make the person and the
                # garment the same object.
                if (a.label == "person" and b.label in PERSON_PARTS
                        or b.label == "person" and a.label in PERSON_PARTS):
                    continue
                projected = self.project(a.box, points, frames[1].shape, b.box)
                if projected is not None:
                    box, count = projected
                    iou = overlap(box, b.box)
                    if iou >= .32:
                        candidates[i, j] = count * iou
        used = [set(), set()]
        for (i, j), weight in sorted(candidates.items(), key=lambda item: -item[1]):
            alternatives = [v for (a, b), v in candidates.items() if (a, b) != (i, j)
                            and (a == i or b == j)]
            if i in used[0] or j in used[1] or weight <= 1.25 * max(alternatives, default=0):
                continue
            a, b = result[0][i], result[1][j]
            stats["matched"] += 1
            pair = stats["matched"]
            if a.label == b.label:
                status = "both"
            else:
                status = "conflict"
                if detector is not None:
                    # Use the lower-score view at higher effective resolution.
                    side = int(a.score >= b.score)
                    other = (a, b)[1-side]
                    checked = self.recheck(detector, frames[side], (a, b)[side].box, {other.label})
                    detector = None
                    stats["review"] += 1
                    if checked is not None:
                        a = Detection(a.box, a.score, other.label)
                        b = Detection(b.box, b.score, other.label)
                        status = "rechecked"
                stats["conflict"] += status == "conflict"
            result[0][i] = Detection(a.box, a.score, a.label, pair, status)
            result[1][j] = Detection(b.box, b.score, b.label, pair, status)
            used[0].add(i)
            used[1].add(j)
        if detector is None:
            return result, stats
        proposals = []
        for side in (0, 1):
            for index, item in enumerate(result[side]):
                if index in used[side] or item.score < .4:
                    continue
                estimate = self.project(item.box, points if side == 0 else points[::-1],
                                        frames[1 - side].shape)
                if estimate is None:
                    continue
                box, count = estimate
                if max((overlap(box, d.box) for d in result[1-side]), default=0) < .3:
                    proposals.append((item.score * count, side, index, item, box))
        if proposals:
            _, side, index, item, box = max(proposals, key=lambda proposal: proposal[0])
            checked = self.recheck(detector, frames[1-side], box, {item.label})
            stats["review"] += 1
            if checked is not None:
                pair = stats["matched"] + 1
                result[side][index] = Detection(item.box, item.score, item.label, pair, "rechecked")
                result[1-side].append(Detection(checked.box, checked.score, checked.label, pair, "rechecked"))
                stats["rescued"] += 1
        return result, stats


def draw(frame, detections):
    colors = {"single": (50, 235, 100), "both": (240, 220, 45),
              "rechecked": (50, 205, 250), "conflict": (50, 125, 250)}
    for item in detections:
        x, y, w, h = item.box.astype(int)
        color = colors[item.status]
        if item.status == "conflict":
            title = f"#{item.pair} ? {item.label}"
        elif item.pair:
            title = f"#{item.pair} {item.label}"
        else:
            title = f"{item.label} {item.score:.0%}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, title, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 2, cv2.LINE_AA)
