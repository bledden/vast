#!/usr/bin/env python3
"""Live worker: subscribe to cameras over MoQ, decide per frame what to send to YOLO using the
codec's motion vectors, and publish `detections` and `events` tracks on `<camera>-ai`.

    python worker.py --broadcast driveway warehouse ... [--mode frame|region|every] [--ask-port 8078]

One process serves every camera and shares one detector, the way a server would. A camera's AI only
runs while someone subscribes to its `detections` track (MoQ tells the publisher); otherwise the
worker doesn't even pull its video. Each frame of
`detections` is JSON, timestamped with the source video frame:
    {"t": us, "w": W, "h": H, "full": bool, "regions": [[x0,y0,x1,y1]],
     "dets": [[x0,y0,x1,y1,label,conf,state]], "mb": 16, "motion": [[bx,by,dx,dy]],
     "stats": {"frames", "inferred", "ms", "full_ms"}}
`events` frames are a snapshot of the camera's last 10 events.
"""
import argparse, asyncio, json, os, time

import av
import moq
import numpy as np

from engine import Detector, Engine, grid

# Credentials live in a git-ignored .env next to this file; never printed.
if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
    for line in open(os.path.join(os.path.dirname(__file__), ".env")):
        k, _, v = line.strip().partition("=")
        if k and v and not k.startswith("#"):
            os.environ.setdefault(k, v.strip().strip('"'))

from agent import EventAgent  # noqa: E402  (reads the environment at import)
import ask  # noqa: E402


class Worker:
    def __init__(self, engine: Engine, names, agent: EventAgent):
        self.engine, self.names, self.agent = engine, names, agent
        self.started = []  # event-start messages produced by the last call
        self.frames = self.inferred = 0
        self.ms = 0.0
        self.full_ms = []  # measured full-frame latencies, to price the every-frame baseline honestly
        self.restart()

    def restart(self):
        """A new subscription to the camera starts at a keyframe with a fresh decoder."""
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.options = {"flags2": "+export_mvs"}
        self.clock = None  # (wall, media) at the first frame, to tell when we fall behind real time

    def process(self, payload: bytes, ts: int) -> list[bytes]:
        out = []
        if not payload:  # an empty packet would flush the decoder into EOF
            return out
        for frame in self.codec.decode(av.Packet(payload)):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            g = grid(frame, w, h)
            now = time.monotonic()
            if self.clock is None or ts < self.clock[1]:  # first frame, or the looping source restarted
                self.clock = (now, ts)
            behind = (now - self.clock[0]) - (ts - self.clock[1]) / 1e6
            if behind < -1:  # we got ahead (paused source): re-anchor
                self.clock = (now, ts)
            step = self.engine.step(img, g, hold=behind > 0.3)
            # An event is motion on a detected object; swaying trees and flicker don't count.
            movers = []
            if step.active is not None and step.active.any():
                for d in step.dets:
                    x0, y0, x1, y1 = Engine._blocks(d.box, step.active.shape)
                    if step.active[y0:y1, x0:x1].mean() > 0.2:
                        movers.append(self.names[d.cls])
            ev = self.agent.feed(img, ts, bool(movers), movers)
            if ev:
                self.started.append(ev)
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
                "stats": self.stats(),
            }, separators=(",", ":")).encode())
        return out

    def stats(self) -> dict:
        return {"frames": self.frames, "inferred": self.inferred, "ms": round(self.ms),
                "full_ms": round(float(np.median(self.full_ms)), 1) if self.full_ms else None}


async def camera(client, args, det: Detector, name: str):
    out = client.create_broadcast(f"{name}-ai")
    track = out.publish_track("detections")
    events_track = out.publish_track("events")
    stats_track = out.publish_track("stats")
    demand = track.demand()
    out.announce()
    agent = EventAgent(camera=name, save=args.save_events)
    events = {}  # id -> latest state; published as a snapshot so late viewers see history
    print(f"{name}: publishing {name}-ai (mode={args.mode}, cosmos={agent.model or 'disabled'})")

    worker = Worker(Engine(det, mode=args.mode, min_cluster=args.min_cluster, stride=args.stride), det.names, agent)

    def publish_stats(ts, watched):
        stats_track.write_frame(json.dumps({**worker.stats(), "watched": watched}).encode(), ts)

    while True:
        # Nobody subscribed to detections: run no model and don't even pull the camera's video.
        publish_stats(0, False)
        await demand.used()
        print(f"{name}: watched; starting AI")
        try:
            src = await client.announced_broadcast(name)
            cat = await src.catalog()
            video = next(iter(cat.video))
            worker.restart()
            media = await src.subscribe_media(video, cat.video[video])
            async with media:
                async for f in media:
                    if not demand.is_used():  # last viewer left: stop the models, unsubscribe from the camera
                        print(f"{name}: unwatched; stopping AI")
                        break
                    for msg in await asyncio.to_thread(worker.process, bytes(f.payload), f.timestamp_us):
                        track.write_frame(msg, f.timestamp_us)
                    changed = bool(worker.started)
                    for ev in worker.started:
                        events[ev["id"]] = ev
                    worker.started.clear()
                    while not agent.done.empty():
                        ev = agent.done.get()
                        events[ev["id"]] = ev
                        changed = True
                        if ev["state"] != "quiet":
                            ask.record(name, ev)
                        print(f"{name}: event {ev['id']} [{ev['state']}] {ev['labels']}: {ev['summary']} ({ev['ms']} ms)")
                    if changed:
                        recent = sorted(events.values(), key=lambda e: e["id"])[-10:]
                        events_track.write_frame(json.dumps(recent, separators=(",", ":")).encode(), f.timestamp_us)
                    if worker.frames % 30 == 0:
                        publish_stats(f.timestamp_us, True)
                    if worker.frames % 300 == 0:
                        print(f"{name}: {worker.frames} frames, inferred {100 * worker.inferred / worker.frames:.0f}%, "
                              f"detector {worker.ms:.0f} ms")
        except Exception as err:  # camera went away (or is restarting): wait for it to come back
            print(f"{name}: source error {err!r}; waiting")
            await asyncio.sleep(1)

async def run(args):
    det = Detector(args.weights)
    ask.serve(args.ask_port)
    async with moq.connect(args.url) as client:
        await asyncio.gather(*(camera(client, args, det, name) for name in args.broadcast))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:4443")
    p.add_argument("--broadcast", nargs="+", default=["cam"])
    p.add_argument("--mode", default="frame", choices=["region", "frame", "every"])
    p.add_argument("--min-cluster", type=int, default=8)
    p.add_argument("--stride", type=int, default=3, help="while motion continues, detect at most every Nth frame")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--save-events", action="store_true", help="write described event clips to events/ for upload.py")
    p.add_argument("--ask-port", type=int, default=8078, help="HTTP port for natural-language questions over the event log")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
