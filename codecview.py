"""Render what the codec saw into a picture: the camera frame dimmed, every block the encoder moved
(faint), the motion that counted (bright, with its vector), and the YOLO boxes. Published as its
own video broadcast, so it's in sync by construction and plays in any MoQ player.
"""
from __future__ import annotations

import cv2
import numpy as np

from engine import MB, Engine, Grid, Step

W, H = 960, 540  # output size; the encoder is configured once with it
ARROW = 3  # vectors are a few pixels per frame; stretch them so they're visible
COLORS = {"fresh": (92, 59, 255), "cached": (90, 200, 90), "shifted": (0, 200, 255)}  # BGR


def render(img: np.ndarray, g: Grid | None, step: Step, names) -> bytes:
    """RGBA bytes of the codec view for one frame."""
    h, w = img.shape[:2]
    sx, sy = W / w, H / h
    out = (cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)

    if g is not None:
        # Every block the encoder gave a vector, noise included: a faint tint and a thin arrow.
        tint = cv2.resize(g.raw.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[tint] = (out[tint] * 0.82 + np.array([255, 180, 60]) * 0.18).astype(np.uint8)
        counted = (step.active & g.moved) if step.active is not None else np.zeros_like(g.raw)
        hot = cv2.resize(counted.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[hot] = (out[hot] * 0.4 + np.array([80, 40, 255]) * 0.6).astype(np.uint8)

        def arrows(mask, color, thickness, limit):
            sparse = np.zeros_like(mask)
            sparse[::2, ::2] = mask[::2, ::2]  # every other block, so neighboring arrows don't merge
            ys, xs = np.nonzero(sparse)
            for y, x in list(zip(ys, xs))[:limit]:
                dx, dy = g.vec[y, x]
                if not (dx or dy):
                    continue
                end = ((x * MB + MB / 2) * sx, (y * MB + MB / 2) * sy)
                start = (end[0] - dx * ARROW * sx, end[1] - dy * ARROW * sy)  # tail where it came from
                cv2.arrowedLine(out, tuple(map(int, start)), tuple(map(int, end)), color, thickness, cv2.LINE_AA, tipLength=0.35)

        arrows(g.raw & ~counted, (170, 140, 100), 1, 1500)  # faint: what the encoder saw
        arrows(counted, (255, 255, 255), 1, 1500)  # bright: the motion that gates YOLO

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
