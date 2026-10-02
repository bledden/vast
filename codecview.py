"""Render what the codec saw into a picture: the camera frame dimmed, every block the encoder moved
(faint), the motion that counted (bright, with its vector), and the YOLO boxes. Published as its
own video broadcast, so it's in sync by construction and plays in any MoQ player.
"""
from __future__ import annotations

import cv2
import numpy as np

from engine import MB, Engine, Grid, Step

W, H = 960, 540  # output size; the encoder is configured once with it
COLORS = {"fresh": (92, 59, 255), "cached": (90, 200, 90), "shifted": (0, 200, 255)}  # BGR


def render(img: np.ndarray, g: Grid | None, step: Step, names) -> bytes:
    """RGBA bytes of the codec view for one frame."""
    h, w = img.shape[:2]
    sx, sy = W / w, H / h
    out = (cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)

    if g is not None:
        # Every block the encoder gave a vector, noise included: a faint tint.
        tint = cv2.resize(g.raw.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[tint] = (out[tint] * 0.6 + np.array([255, 180, 60]) * 0.4).astype(np.uint8)
        # The motion that counted (what gates YOLO): bright, with the vector stretched 3x.
        if step.active is not None:
            ys, xs = np.nonzero(step.active & g.moved)
            hot = cv2.resize((step.active & g.moved).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            out[hot] = (out[hot] * 0.4 + np.array([80, 40, 255]) * 0.6).astype(np.uint8)
            for y, x in zip(ys, xs):
                cx, cy = (x * MB + MB / 2) * sx, (y * MB + MB / 2) * sy
                dx, dy = g.vec[y, x]
                cv2.line(out, (int(cx), int(cy)), (int(cx - dx * 3 * sx), int(cy - dy * 3 * sy)), (255, 255, 255), 1, cv2.LINE_AA)

    for d in step.dets:
        x0, y0, x1, y1 = (d.box * [sx, sy, sx, sy]).astype(int)
        c = COLORS[d.state]
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        cv2.putText(out, f"{names[d.cls]} {d.state}", (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    if g is None:
        label = "keyframe: no motion vectors"
    else:
        label = f"P-frame: {int(g.raw.sum())} blocks with vectors, {int((step.active & g.moved).sum()) if step.active is not None else 0} counted"
    label += " | YOLO ran" if step.calls else " | YOLO skipped"
    cv2.rectangle(out, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(out, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGBA).tobytes()
