#!/usr/bin/env python3
"""Download a few indexed segments per camera from the VSS backend, plus the YOLO detections
their pipeline already stored for each one (the every-frame baseline we compare against).

    export INGRESS_URL=https://...  VSS_USERNAME=team-47  VSS_PASSWORD=...
    python fetch.py [--per-camera 10] [--out footage] [--list]

Writes footage/<camera>/<segment>.mp4 and .detections.json, then footage/<camera>.mp4: the
segments joined in order, ready for ./camera.sh. Credentials come from the environment only.
"""
import argparse, json, os, re, subprocess, sys, urllib.parse, urllib.request
from collections import defaultdict

# Cloudflare in front of VSS rejects Python's default User-Agent (error 1010).
UA = "codec-vision/1.0"


def call(url, token=None, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def sources(obj):
    """Every segment S3 URI anywhere in an explore item."""
    if isinstance(obj, str):
        return [obj] if obj.startswith("s3://") and "segments" in obj.split("/")[2] else []
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in sources(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in sources(v)]
    return []


def camera_of(item):
    for k in ("camera_id", "camera", "location"):
        for d in (item, item.get("metadata") or {}):
            if isinstance(d, dict) and d.get(k):
                return str(d[k])
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-camera", type=int, default=10)
    p.add_argument("--out", default="footage")
    p.add_argument("--list", action="store_true", help="print cameras and segment counts, download nothing")
    args = p.parse_args()

    backend = os.environ["INGRESS_URL"].rstrip("/")
    user = os.environ.get("VSS_USERNAME") or os.environ["USERNAME"]
    password = os.environ.get("VSS_PASSWORD") or os.environ["PASSWORD"]
    token = json.loads(call(f"{backend}/api/v1/auth/login", body={"username": user, "password": password}))["access_token"]

    cams = defaultdict(list)
    offset = 0
    while True:
        page = json.loads(call(f"{backend}/api/v1/videos/explore?scope=all&limit=48&offset={offset}", token))
        items = page if isinstance(page, list) else next((v for v in page.values() if isinstance(v, list)), [])
        if not items:
            break
        if offset == 0 and not any(sources(i) for i in items):
            print("explore returned no segment URIs; first item:", json.dumps(items[0])[:1500], file=sys.stderr)
            sys.exit(1)
        for item in items:
            cams[camera_of(item)].extend(sorted(set(sources(item))))
        offset += len(items)

    for cam, segs in sorted(cams.items()):
        print(f"{cam}\t{len(segs)} segments")
    if args.list:
        return

    for cam, segs in sorted(cams.items()):
        folder = os.path.join(args.out, re.sub(r"[^\w.-]", "_", cam))
        os.makedirs(folder, exist_ok=True)
        files = []
        for src in segs[: args.per_camera]:
            base = os.path.join(folder, os.path.basename(src).rsplit(".", 1)[0])
            q = urllib.parse.urlencode({"source": src, "token": token})
            if not os.path.exists(base + ".mp4"):
                with open(base + ".mp4", "wb") as f:
                    f.write(call(f"{backend}/api/v1/videos/stream?{q}"))
            try:
                det = call(f"{backend}/api/v1/videos/detections?{urllib.parse.urlencode({'source': src})}", token)
                with open(base + ".detections.json", "wb") as f:
                    f.write(det)
            except urllib.error.HTTPError as e:
                if e.code != 404:  # 404 = no YOLO sidecar for this segment
                    raise
            files.append(os.path.abspath(base + ".mp4"))
        with open(os.path.join(folder, "concat.txt"), "w") as f:
            f.writelines(f"file '{x}'\n" for x in files)
        joined = folder + ".mp4"
        # Re-encode once: segments may carry B-frames, and the demo wants one continuous H.264 stream without them.
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", os.path.join(folder, "concat.txt"), "-an", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "23", "-bf", "0", "-g", "60", joined], check=True)
        print(f"{cam}: {len(files)} segments -> {joined}")


if __name__ == "__main__":
    main()
