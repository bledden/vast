#!/usr/bin/env python3
"""Build long camera clips from a VSS instance by joining consecutive chunks of one source video.

    VSS_USERNAME=team-47 python fetch.py --list                       # source videos and their chunks
    VSS_USERNAME=team-47 python fetch.py neighborhood_20260901 --chunks 0-11

Chunks are the uploaded pieces of a long recording (name_chunk_0000, _0001, ...). This downloads
the requested range through the backend's stream endpoint (falling back to each chunk's segments),
joins them in order, and re-encodes once the way a simple camera sends video (H.264, no B-frames,
2s GOP) into footage/<name>.mp4. Prompts for the password; credentials are never stored.
"""
import argparse, getpass, json, os, re, subprocess, sys, tempfile, urllib.error, urllib.parse, urllib.request
from collections import defaultdict

# Cloudflare in front of VSS rejects Python's default User-Agent (error 1010).
UA = "codec-vision/1.0"
CHUNK = re.compile(r"^(?:\d{8}_\d{6}_)?(?P<video>.+?)_chunk_(?P<n>\d+)")


def call(url, token=None, body=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def uris(obj):
    """Every S3 URI anywhere in an explore item."""
    if isinstance(obj, str):
        return [obj] if obj.startswith("s3://") else []
    if isinstance(obj, dict):
        return [u for v in obj.values() for u in uris(v)]
    if isinstance(obj, list):
        return [u for v in obj for u in uris(v)]
    return []


def catalog(backend, token):
    """{video: {chunk number: {"chunk": uri or None, "segments": [uri]}}} over every explore page."""
    videos = defaultdict(lambda: defaultdict(lambda: {"chunk": None, "segments": set()}))
    offset = 0
    while True:
        page = json.loads(call(f"{backend}/api/v1/videos/explore?scope=all&limit=48&offset={offset}", token))
        items = page if isinstance(page, list) else next((v for v in page.values() if isinstance(v, list)), [])
        if not items:
            break
        for item in items:
            for uri in uris(item):
                m = CHUNK.match(os.path.basename(uri))
                if not m:
                    continue
                entry = videos[m["video"]][int(m["n"])]
                if "segments" in uri.split("/")[2]:
                    entry["segments"].add(uri)
                else:
                    entry["chunk"] = uri
        offset += len(items)
    return videos


def download(backend, token, uri, path):
    q = urllib.parse.urlencode({"source": uri, "token": token})
    data = call(f"{backend}/api/v1/videos/stream?{q}", timeout=300)
    with open(path, "wb") as f:
        f.write(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", help="source video name, as printed by --list")
    p.add_argument("--list", action="store_true")
    p.add_argument("--chunks", help="range like 0-11 (default: all)")
    p.add_argument("--out", default="footage")
    args = p.parse_args()

    backend = os.environ.get("VSS_URL", "https://team-47-vss.thecosmoslabs.com").rstrip("/")
    user = os.environ.get("VSS_USERNAME") or input("VSS username: ")
    password = os.environ.get("VSS_PASSWORD") or getpass.getpass(f"VSS password for {user}: ")
    token = json.loads(call(f"{backend}/api/v1/auth/login", body={"username": user, "password": password}))["access_token"]
    videos = catalog(backend, token)

    if args.list or not args.video:
        for name, chunks in sorted(videos.items()):
            nums = sorted(chunks)
            whole = sum(1 for c in chunks.values() if c["chunk"])
            print(f"{name:60} chunks {nums[0]:>4}-{nums[-1]:<4} ({len(nums)} indexed, {whole} with a chunk URI)")
        return

    chunks = videos.get(args.video)
    if not chunks:
        sys.exit(f"no chunks for {args.video!r}; run --list")
    nums = sorted(chunks)
    if args.chunks:
        lo, hi = map(int, args.chunks.split("-"))
        nums = [n for n in nums if lo <= n <= hi]
    gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in chunks]
    if gaps:
        print(f"warning: chunks {gaps} aren't indexed; the joined clip jumps there", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for n in nums:
            c = chunks[n]
            parts = [c["chunk"]] if c["chunk"] else sorted(c["segments"])
            for i, uri in enumerate(parts):
                path = os.path.join(tmp, f"{n:05d}_{i:03d}.mp4")
                try:
                    download(backend, token, uri, path)
                except urllib.error.HTTPError as err:
                    if uri != c["chunk"] or not c["segments"]:
                        raise
                    print(f"chunk {n}: stream refused ({err.code}); using its {len(c['segments'])} segments")
                    for j, seg in enumerate(sorted(c["segments"])):
                        segpath = os.path.join(tmp, f"{n:05d}_{j:03d}.mp4")
                        download(backend, token, seg, segpath)
                        files.append(segpath)
                    break
                files.append(path)
            print(f"chunk {n}: {len(parts)} file(s)")
        with open(os.path.join(tmp, "list.txt"), "w") as f:
            f.writelines(f"file '{x}'\n" for x in files)
        out = os.path.join(args.out, f"{args.video}.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", os.path.join(tmp, "list.txt"), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-bf", "0", "-g", "60", "-vf", "fps=30", out], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                         capture_output=True, text=True).stdout.strip()
    print(f"{out}: {len(nums)} chunks, {float(dur or 0) / 60:.1f} min")


if __name__ == "__main__":
    main()
