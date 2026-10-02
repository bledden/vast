"""Turn motion into events and have NVIDIA Cosmos describe them.

An event is a run of frames with codec motion. While it lasts we keep a few frames per second;
when motion stops (or the event gets long) the clip goes to Cosmos3-Reason on a background
thread, so the live worker never waits on it. Static scenes cost nothing: no motion, no event.

Needs GPU_BEARER_TOKEN (and optionally COSMOS_URL). With WANDB_API_KEY set, every Cosmos call
is traced in W&B Weave.
"""
from __future__ import annotations

import base64, io, json, os, queue, threading, time, urllib.request

import av
import cv2
import numpy as np

COSMOS_URL = os.environ.get("COSMOS_URL", "http://166.19.38.112:8001")
PROMPT = (
    "You are watching a fixed security camera. This clip is a moment where motion was detected; "
    "an object detector saw: {labels}. In one or two short sentences, say what happened: who or what, "
    "doing what, going where. Then on its own line write 'ALERT: yes' if a homeowner would want to "
    "know right now (someone approaching the house, a vehicle stopping, anything unusual), else 'ALERT: no'."
)

try:
    if os.environ.get("WANDB_API_KEY"):
        import weave

        weave.init("vast")
        op = weave.op
    else:
        raise ImportError
except ImportError:
    def op(f):
        return f


@op
def describe(clip_mp4: bytes, labels: list[str], model: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(labels=", ".join(labels) or "nothing")},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip_mp4).decode()}},
        ]}],
        "max_tokens": 200, "temperature": 0.2,
    }
    req = urllib.request.Request(f"{COSMOS_URL}/v1/chat/completions", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GPU_BEARER_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def cosmos_model() -> str:
    req = urllib.request.Request(f"{COSMOS_URL}/v1/models", headers={"Authorization": f"Bearer {os.environ['GPU_BEARER_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["data"][0]["id"]


def encode(frames: list[np.ndarray], fps: int) -> bytes:
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    s = out.add_stream("libx264", rate=fps)
    s.height, s.width = frames[0].shape[:2]
    s.pix_fmt, s.options = "yuv420p", {"crf": "28", "preset": "veryfast"}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(f, format="bgr24")):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()
    return buf.getvalue()


class EventAgent:
    def __init__(self, fps: float = 30, sample_fps: int = 4, quiet_s: float = 1.5, max_s: float = 10, min_s: float = 1.0,
                 width: int = 640):
        self.enabled = bool(os.environ.get("GPU_BEARER_TOKEN"))
        self.model = cosmos_model() if self.enabled else None
        self.every = max(1, round(fps / sample_fps))
        self.sample_fps, self.quiet, self.max = sample_fps, round(quiet_s * fps), round(max_s * fps)
        self.min = round(min_s * fps)  # frames of motion before it counts; leaves and flicker stay quiet
        self.width = width
        self.done: queue.Queue[dict] = queue.Queue()  # finished descriptions, drained by the worker
        self.event = None
        self.n = 0

    def feed(self, img: np.ndarray, ts: int, moving: bool, labels: list[str]) -> dict | None:
        """Call once per frame. Returns an event-start message to publish, if one began."""
        self.n += 1
        started = None
        if moving and self.event is None:
            self.event = {"id": self.n, "start": ts, "frames": [], "labels": set(), "quiet": 0, "len": 0, "moving": 0}
        e = self.event
        if e is None:
            return None
        e["len"] += 1
        e["moving"] += moving
        e["quiet"] = 0 if moving else e["quiet"] + 1
        if e["moving"] == self.min:
            started = {"id": e["id"], "start": e["start"], "state": "analyzing" if self.enabled else "no-agent"}
        e["labels"].update(labels)
        if e["len"] % self.every == 1:
            h, w = img.shape[:2]
            e["frames"].append(cv2.resize(img, (self.width, round(h * self.width / w / 2) * 2)))
        if e["quiet"] >= self.quiet or e["len"] >= self.max:
            self.event = None
            e["end"] = ts
            if self.enabled and e["frames"] and e["moving"] >= self.min:
                threading.Thread(target=self._describe, args=(e,), daemon=True).start()
        return started

    def _describe(self, e):
        t = time.time()
        try:
            text = describe(encode(e["frames"], self.sample_fps), sorted(e["labels"]), self.model)
            state = "alert" if "alert: yes" in text.lower() else "done"
        except Exception as err:  # surface the failure in the UI instead of dropping the event
            text, state = f"Cosmos call failed: {err}", "error"
        summary = "\n".join(l for l in text.splitlines() if not l.strip().lower().startswith("alert:")).strip()
        self.done.put({"id": e["id"], "start": e["start"], "end": e["end"], "labels": sorted(e["labels"]),
                       "state": state, "summary": summary, "ms": round((time.time() - t) * 1000)})
