"""Motion-gated object detection driven by the codec's own motion vectors.

Per decoded frame, `Engine.step` decides what (if anything) to send to the detector:

- "every":  full frame, every frame. The baseline.
- "frame":  full frame only when something moved; otherwise reuse cached boxes.
- "region": only the regions that moved. Cached boxes with no motion are kept, boxes whose
            blocks all move together are shifted by the median vector, the rest are re-detected.

Keyframes carry no vectors, so they always run the full frame and resync the cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

MB = 16  # H.264 macroblock size


@dataclass
class Grid:
    """Per-macroblock motion for one P-frame."""
    dirty: np.ndarray  # bool (mbh, mbw): moved, or intra-coded (new content)
    moved: np.ndarray  # bool (mbh, mbw): has a non-zero vector
    vec: np.ndarray  # float (mbh, mbw, 2): mean motion (dx, dy) in pixels


def grid(frame, width: int, height: int, min_motion: float = 2.5) -> Grid | None:
    """Motion grid from a PyAV frame decoded with flags2=+export_mvs, or None (keyframe / no vectors).

    min_motion: pixels per frame below which a vector counts as noise. On flat, noisy areas the
    encoder matches sensor noise with random sub-2px vectors; real movers are usually faster.
    """
    sd = frame.side_data.get("MOTION_VECTORS")
    if sd is None:
        return None
    a = sd.to_ndarray()
    mbw, mbh = (width + MB - 1) // MB, (height + MB - 1) // MB
    x = np.clip(a["dst_x"] // MB, 0, mbw - 1)
    y = np.clip(a["dst_y"] // MB, 0, mbh - 1)
    seen = np.zeros((mbh, mbw), bool)
    seen[y, x] = True

    # Past references only: dst - src is how far the block moved since the reference frame.
    past = a["source"] < 0
    # Sub-pixel precision; the sign matches dst - src (how far the block moved).
    dx = (-a["motion_x"] / a["motion_scale"]).astype(np.float32)
    dy = (-a["motion_y"] / a["motion_scale"]).astype(np.float32)
    sx = np.zeros((mbh, mbw), np.float32)
    sy = np.zeros((mbh, mbw), np.float32)
    n = np.zeros((mbh, mbw), np.float32)
    np.add.at(sx, (y[past], x[past]), dx[past])
    np.add.at(sy, (y[past], x[past]), dy[past])
    np.add.at(n, (y[past], x[past]), 1)
    vec = np.stack([sx, sy], -1) / np.maximum(n, 1)[..., None]

    moved = np.zeros((mbh, mbw), bool)
    nz = np.hypot(dx, dy) >= min_motion
    moved[y[nz], x[nz]] = True
    return Grid(dirty=moved | ~seen, moved=moved, vec=vec)


@dataclass
class Det:
    box: np.ndarray  # float [x1, y1, x2, y2] in frame pixels
    cls: int
    conf: float
    state: str = "fresh"  # fresh | cached | shifted
    shifted: int = 0  # consecutive frames moved by vectors alone


@dataclass
class Step:
    dets: list[Det]
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)  # what we sent to the detector
    active: np.ndarray | None = None  # mask of motion clusters that counted
    full: bool = False
    calls: int = 0
    pixels: int = 0  # pixels sent to the detector
    ms: float = 0.0


class Detector:
    """Ultralytics YOLO over a batch of images. Returns boxes in each image's own pixels."""

    def __init__(self, weights: str = "yolo11n.pt", device: str | None = None, conf: float = 0.35):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(weights)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.names = self.model.names
        self.model.predict(np.zeros((64, 64, 3), np.uint8), device=self.device, verbose=False)  # warm up

    def __call__(self, imgs: list[np.ndarray], imgsz: int) -> list[list[tuple]]:
        # Letterbox into a fixed square ourselves: every new input shape recompiles the GPU graph.
        import cv2

        squares, scales = [], []
        for img in imgs:
            h, w = img.shape[:2]
            k = imgsz / max(h, w)
            sq = np.full((imgsz, imgsz, 3), 114, np.uint8)
            sq[: round(h * k), : round(w * k)] = cv2.resize(img, (round(w * k), round(h * k)))
            squares.append(sq)
            scales.append(k)
        results = self.model.predict(squares, imgsz=imgsz, device=self.device, conf=self.conf, verbose=False)
        out = []
        for r, k in zip(results, scales):
            b = r.boxes
            out.append([(xyxy / k, int(c), float(p)) for xyxy, c, p in zip(b.xyxy.cpu().numpy(), b.cls.tolist(), b.conf.tolist())])
        return out


class Engine:
    def __init__(self, detector: Detector, mode: str = "region", min_cluster: int = 8, max_gap: int = 300,
                 max_shift: int = 30, pad: float = 0.25, full_frac: float = 0.5):
        assert mode in ("every", "frame", "region"), mode
        self.det, self.mode = detector, mode
        self.min_cluster = min_cluster  # blocks in one changed region to count as motion (noise is scattered singles)
        self.max_gap = max_gap  # frames between forced full-frame runs, for cameras with very long GOPs
        self.max_shift = max_shift  # frames a box may be moved by vectors alone before re-detecting it
        self.pad = pad  # context added around each region, as a fraction of its size
        self.full_frac = full_frac  # if regions cover more than this, just run the full frame
        self.cache: list[Det] = []
        self.since_full = 0

    def step(self, img: np.ndarray, g: Grid | None) -> Step:
        h, w = img.shape[:2]
        self.since_full += 1
        if self.mode == "every" or g is None or self.since_full >= self.max_gap:
            return self._full(img)

        active = self._active(g.dirty)
        if self.mode == "frame":
            if active.any():
                return self._full(img, active)
            for d in self.cache:
                d.state = "cached"
            return Step(dets=list(self.cache), active=active)

        # region mode: classify cached boxes against the motion under them.
        explained = np.zeros_like(active)
        regions = []
        keep = []
        for d in self.cache:
            x0, y0, x1, y1 = self._blocks(d.box, active.shape)
            under = active[y0:y1, x0:x1]
            if not under.any():
                d.state = "cached"
                keep.append(d)
                continue
            v = g.vec[y0:y1, x0:x1][under & g.moved[y0:y1, x0:x1]]
            if len(v) >= 3 and d.shifted < self.max_shift:
                med = np.median(v, axis=0)
                if np.median(np.abs(v - med)) <= 2.0:  # blocks move together: rigid shift
                    d.box = d.box + np.array([med[0], med[1], med[0], med[1]])
                    d.state, d.shifted = "shifted", d.shifted + 1
                    explained[y0:y1, x0:x1] |= under
                    keep.append(d)
                    continue
            regions.append(d.box.copy())  # stale: re-detect where it was

        # Motion no shifted box accounts for: something new, or something that changed shape.
        labels, n = ndimage.label(active & ~explained)
        for sl in ndimage.find_objects(labels):
            if sl is not None:
                regions.append(np.array([sl[1].start * MB, sl[0].start * MB, sl[1].stop * MB, sl[0].stop * MB], np.float32))

        self.cache = keep
        if not regions:
            return Step(dets=list(self.cache), active=active)

        boxes = self._merge([self._padded(r, w, h) for r in regions])
        if sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes) > self.full_frac * w * h:
            return self._full(img, active)

        t = time.perf_counter()
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
        calls, found = 0, [None] * len(boxes)
        small = [max(c.shape[:2]) <= 400 for c in crops]
        for size in (320, 640):  # two fixed input sizes, one batched call each
            idx = [i for i in range(len(crops)) if small[i] == (size == 320)]
            if idx:
                for i, r in zip(idx, self.det([crops[i] for i in idx], size)):
                    found[i] = r
                calls += 1
        ms = (time.perf_counter() - t) * 1000

        # Re-detected regions replace whatever the cache had there.
        def inside(d):
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            return any(x0 <= cx < x1 and y0 <= cy < y1 for x0, y0, x1, y1 in boxes)

        self.cache = [d for d in self.cache if not inside(d)]
        for (x0, y0, _, _), res in zip(boxes, found):
            for xyxy, c, p in res:
                self.cache.append(Det(box=xyxy + np.array([x0, y0, x0, y0], np.float32), cls=c, conf=p))
        return Step(dets=list(self.cache), regions=boxes, active=active, calls=calls,
                    pixels=sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes), ms=ms)

    def _full(self, img, active=None) -> Step:
        t = time.perf_counter()
        res = self.det([img], 640)[0]
        ms = (time.perf_counter() - t) * 1000
        self.cache = [Det(box=xyxy, cls=c, conf=p) for xyxy, c, p in res]
        self.since_full = 0
        h, w = img.shape[:2]
        return Step(dets=list(self.cache), regions=[(0, 0, w, h)], active=active, full=True, calls=1, pixels=w * h, ms=ms)

    def _active(self, dirty):
        labels, n = ndimage.label(dirty)
        if not n:
            return np.zeros_like(dirty)
        sizes = np.bincount(labels.ravel())
        big = np.flatnonzero(sizes >= self.min_cluster)
        big = big[big != 0]
        return ndimage.binary_dilation(np.isin(labels, big))

    @staticmethod
    def _blocks(box, shape):
        mbh, mbw = shape
        x0, y0 = int(np.clip(box[0] // MB, 0, mbw - 1)), int(np.clip(box[1] // MB, 0, mbh - 1))
        x1, y1 = int(np.clip(-(-box[2] // MB), x0 + 1, mbw)), int(np.clip(-(-box[3] // MB), y0 + 1, mbh))
        return x0, y0, x1, y1

    def _padded(self, r, w, h):
        bw, bh = r[2] - r[0], r[3] - r[1]
        px, py = max(32, bw * self.pad), max(32, bh * self.pad)
        return [int(max(0, r[0] - px)), int(max(0, r[1] - py)), int(min(w, r[2] + px)), int(min(h, r[3] + py))]

    @staticmethod
    def _merge(boxes):
        boxes = [list(b) for b in boxes]
        merged = True
        while merged:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        return [tuple(b) for b in boxes]
