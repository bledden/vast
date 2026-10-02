"""Ask the cameras: answer questions over the event log with W&B serverless inference.

Only events reach the log (motion on an object, described by Cosmos), so the language model
reads a few lines per minute of footage instead of the footage itself. Served on
http://localhost:<port>/ask?q=... for the viewer; calls are traced in W&B Weave.
"""
from __future__ import annotations

import json, os, socket, threading, time, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("ASK_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")  # non-reasoning: answers in about a second
LOG: list[dict] = []
LOCK = threading.Lock()

try:
    import weave

    op = weave.op if os.environ.get("WANDB_API_KEY") else (lambda f: f)
except ImportError:
    def op(f):
        return f


def record(camera: str, ev: dict):
    with LOCK:
        LOG.append({"time": time.strftime("%H:%M:%S"), "camera": camera, "alert": ev["state"] == "alert",
                    "objects": ev["labels"], "what": ev["summary"]})
        del LOG[:-500]


def project() -> str:
    if os.environ.get("WANDB_PROJECT_PATH"):
        return os.environ["WANDB_PROJECT_PATH"]
    import wandb

    return f"{wandb.Api().default_entity}/vast"


@op
def answer(question: str, log: list[dict]) -> str:
    lines = "\n".join(f"{e['time']} [{e['camera']}]{' ALERT' if e['alert'] else ''} {e['what']}" for e in log) or "(no events yet)"
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You answer questions about security cameras using only this event log. "
                                          "Each line is one moment where the camera saw motion, described by a vision model. "
                                          "Be brief: one to three sentences, cite times and cameras. If the log doesn't say, say so."},
            {"role": "user", "content": f"Event log:\n{lines}\n\nQuestion: {question}"},
        ],
        "max_tokens": 300, "temperature": 0.2,
    }
    req = urllib.request.Request("https://api.inference.wandb.ai/v1/chat/completions", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['WANDB_API_KEY']}",
        "OpenAI-Project": PROJECT, "User-Agent": "codec-vision/1.0"})  # Cloudflare blocks the urllib default
    with urllib.request.urlopen(req, timeout=60) as r:
        msg = json.load(r)["choices"][0]["message"]
    return (msg.get("content") or "").strip()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query).get("q", [""])[0].strip()
        if url.path != "/ask" or not q:
            self.send_error(404)
            return
        with LOCK:
            log = list(LOG)
        try:
            body, code = {"answer": answer(q, log), "events": len(log)}, 200
        except Exception as err:
            body, code = {"error": str(err)}, 502
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


PROJECT = ""


class DualStackServer(ThreadingHTTPServer):
    """Listen on IPv6 and IPv4, so http://localhost works whichever the browser tries first."""
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def serve(port: int):
    global PROJECT
    if not os.environ.get("WANDB_API_KEY") or not port:
        print("ask: disabled (no WANDB_API_KEY)")
        return
    PROJECT = project()
    server = DualStackServer(("::", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"ask: http://localhost:{port}/ask?q=... ({MODEL})")
