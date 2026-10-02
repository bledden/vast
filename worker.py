#!/usr/bin/env python3
"""Live worker: subscribe to a camera over MoQ, decide per frame what to send to YOLO using the
codec's motion vectors, and publish the results as a `detections` track on `<camera>-ai`.

    python worker.py --broadcast cam [--mode frame|region|every] [--url http://localhost:4443]

Each frame of the `detections` track is JSON, timestamped with the source video frame:
    {"t": us, "w": W, "h": H, "full": bool, "regions": [[x0,y0,x1,y1]],
     "dets": [[x0,y0,x1,y1,label,conf,state]], "mb": 16, "motion": [[bx,by,dx,dy]],
     "stats": {"frames", "inferred", "ms", "full_ms"}}
"""
import argparse, asyncio, json, time

import av
import moq
import numpy as np

from engine import Detector, Engine, grid


class Worker:
    def __init__(self, engine: Engine, names):
        self.engine, self.names = engine, names
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.options = {"flags2": "+export_mvs"}
        self.frames = self.inferred = 0
        self.ms = 0.0
        self.full_ms = []  # measured full-frame latencies, to price the every-frame baseline honestly

    def process(self, payload: bytes, ts: int) -> list[bytes]:
        out = []
        if not payload:  # an empty packet would flush the decoder into EOF
            return out
        for frame in self.codec.decode(av.Packet(payload)):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            g = grid(frame, w, h)
            step = self.engine.step(img, g)
            self.frames += 1
            self.inferred += step.calls > 0
            self.ms += step.ms
            if step.full:
                self.full_ms.append(step.ms)
            motion = []
            if g is not None and step.active is not None:
                ys, xs = np.nonzero(step.active & g.moved)
                motion = [[int(x), int(y), round(float(g.vec[y, x, 0]), 1), round(float(g.vec[y, x, 1]), 1)] for y, x in zip(ys, xs)]
            out.append(json.dumps({
                "t": ts, "w": w, "h": h, "full": step.full,
                "regions": [] if step.full else [list(map(int, r)) for r in step.regions],
                "dets": [[*map(lambda v: round(float(v), 1), d.box), self.names[d.cls], round(d.conf, 2), d.state] for d in step.dets],
                "mb": 16, "motion": motion,
                "stats": {"frames": self.frames, "inferred": self.inferred, "ms": round(self.ms),
                          "full_ms": round(float(np.median(self.full_ms)), 1) if self.full_ms else None},
            }, separators=(",", ":")).encode())
        return out


async def run(args):
    det = Detector(args.weights)
    async with moq.connect(args.url) as client:
        out = client.create_broadcast(f"{args.broadcast}-ai")
        track = out.publish_track("detections")
        out.announce()
        print(f"publishing {args.broadcast}-ai/detections (mode={args.mode}, device={det.device})")

        while True:
            src = await client.request_broadcast(args.broadcast)
            cat = await src.catalog()
            name = next(iter(cat.video))
            worker = Worker(Engine(det, mode=args.mode, min_cluster=args.min_cluster), det.names)
            media = await src.subscribe_media(name, cat.video[name])
            started, lag = time.time(), 0.0
            async with media:
                async for f in media:
                    t = time.perf_counter()
                    for msg in await asyncio.to_thread(worker.process, bytes(f.payload), f.timestamp_us):
                        track.write_frame(msg, f.timestamp_us)
                    lag = time.perf_counter() - t
                    if worker.frames % 150 == 0:
                        print(f"{worker.frames} frames, inferred {100 * worker.inferred / worker.frames:.0f}%, "
                              f"detector {worker.ms:.0f} ms, last step {lag * 1000:.0f} ms")
            print("source ended; waiting for it to come back")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:4443")
    p.add_argument("--broadcast", default="cam")
    p.add_argument("--mode", default="frame", choices=["region", "frame", "every"])
    p.add_argument("--min-cluster", type=int, default=8)
    p.add_argument("--weights", default="yolo11n.pt")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
