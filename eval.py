#!/usr/bin/env python3
"""Compare every-frame detection against motion-gated detection on a recorded clip.

    python eval.py clip.mp4 [--render out.mp4] [--wandb]

Runs the three Engine modes side by side on the same decoded frames. "every" is the reference:
recall is the share of its boxes (same class, IoU >= 0.5) that a gated mode also reports.
"""
import argparse, json
import av
import cv2
import numpy as np

from engine import Detector, Engine, grid

COLORS = {"fresh": (60, 60, 255), "cached": (90, 200, 90), "shifted": (0, 200, 255)}  # BGR


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def matches(ref, got):
    """Greedy one-to-one matches of `got` against `ref` by class and IoU >= 0.5."""
    used, hit = set(), 0
    for r in ref:
        for j, g in enumerate(got):
            if j not in used and g.cls == r.cls and iou(r.box, g.box) >= 0.5:
                used.add(j)
                hit += 1
                break
    return hit


def draw(img, step, stats, names):
    out = img.copy()
    if step.active is not None:
        m = cv2.resize(step.active.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (out[m] * 0.5 + np.array([80, 40, 255]) * 0.5).astype(np.uint8)
    if not step.full:
        for x0, y0, x1, y1 in step.regions:
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 2)
    for d in step.dets:
        x0, y0, x1, y1 = d.box.astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), COLORS[d.state], 2)
        cv2.putText(out, f"{names[d.cls]} {d.state}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[d.state], 1)
    label = "FULL FRAME" if step.full else (f"{len(step.regions)} region(s)" if step.regions else "no inference")
    cv2.rectangle(out, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(out, f"{label} | pixels {100 * step.pixels / (img.shape[0] * img.shape[1]):3.0f}% | "
                     f"total GPU {stats['ms']:6.0f} ms vs every-frame {stats['ref_ms']:6.0f} ms",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--render", help="write the region-mode overlay to this mp4")
    p.add_argument("--wandb", action="store_true", help="log per-mode results to Weights & Biases")
    args = p.parse_args()

    det = Detector(args.weights)
    modes = {m: Engine(det, mode=m) for m in ("every", "frame", "region")}
    totals = {m: dict(frames=0, inferred=0, calls=0, ms=0.0, pixels=0, ref=0, hit=0, got=0, ref_hit=0) for m in modes}

    c = av.open(args.video)
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}
    w, h = s.codec_context.width, s.codec_context.height
    if args.render:
        out = av.open(args.render, "w")
        enc = out.add_stream("libx264", rate=s.average_rate or 30)
        enc.width, enc.height, enc.pix_fmt, enc.options = w, h, "yuv420p", {"crf": "22"}

    for frame in c.decode(s):
        img = frame.to_ndarray(format="bgr24")
        g = grid(frame, w, h)
        steps = {m: e.step(img, g) for m, e in modes.items()}
        ref = steps["every"].dets
        for m, st in steps.items():
            t = totals[m]
            t["frames"] += 1
            t["inferred"] += st.calls > 0
            t["calls"] += st.calls
            t["ms"] += st.ms
            t["pixels"] += st.pixels
            t["ref"] += len(ref)
            t["got"] += len(st.dets)
            hit = matches(ref, st.dets)
            t["hit"] += hit
        if args.render:
            vis = draw(img, steps["region"], {"ms": totals["region"]["ms"], "ref_ms": totals["every"]["ms"]}, det.names)
            for pkt in enc.encode(av.VideoFrame.from_ndarray(vis, format="bgr24")):
                out.mux(pkt)
    if args.render:
        for pkt in enc.encode():
            out.mux(pkt)
        out.close()

    base = totals["every"]
    rows = {}
    for m, t in totals.items():
        rows[m] = {
            "frames_inferred_pct": round(100 * t["inferred"] / t["frames"], 1),
            "detector_calls": t["calls"],
            "pixels_pct": round(100 * t["pixels"] / (t["frames"] * w * h), 1),
            "detector_ms": round(t["ms"]),
            "ms_vs_every_pct": round(100 * t["ms"] / base["ms"], 1),
            "recall_pct": round(100 * t["hit"] / max(1, t["ref"]), 1),
            "precision_pct": round(100 * t["hit"] / max(1, t["got"]), 1),
        }
    print(json.dumps({"video": args.video, "frames": base["frames"], "device": det.device, "modes": rows}, indent=2))

    if args.wandb:
        import wandb
        for m, r in rows.items():
            with wandb.init(project="vast", name=f"{m}", config={"video": args.video, "mode": m, "device": det.device}, reinit=True) as run:
                run.log(r)


if __name__ == "__main__":
    main()
