#!/usr/bin/env python3
"""Score video segments for motion using only codec data, and report how many are static.

    pip install av numpy scipy boto3
    python score.py ./clips                      # local folder of .mp4
    python score.py s3://team-x-vss-chunks-segments/some/prefix --limit 200

S3 uses S3_ENDPOINT / ACCESS_KEY / SECRET_KEY from the environment (the VAST VM sets them).
Prints one JSON line per segment, then a per-camera summary.

Two signals per P-frame, no pixel analysis:
  moved: fraction of 16x16 blocks with a non-zero motion vector or no vector at all (intra,
         i.e. new content). Needs H.264; FFmpeg exports no vectors for HEVC.
  bytes: compressed frame size, available for any codec without decoding.
A segment is "active" if any P-frame has a connected cluster of at least --cluster changed blocks.
Clusters, not totals: noisy sensors make the encoder intra-code scattered single blocks.
"""
import argparse, json, os, sys, tempfile
from collections import defaultdict
import av
import numpy as np
from scipy import ndimage


def score(path):
    c = av.open(path)
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}
    s.thread_type = "AUTO"
    w, h = s.codec_context.width, s.codec_context.height
    mbw, mbh = (w + 15) // 16, (h + 15) // 16
    moved, cluster, sizes, nframes = [], [], [], 0
    for pkt in c.demux(s):
        for f in pkt.decode():
            nframes += 1
            if f.key_frame:
                continue
            sizes.append(pkt.size)
            sd = f.side_data.get("MOTION_VECTORS")
            if sd is None:
                continue
            a = sd.to_ndarray()
            x = np.clip(a["dst_x"] // 16, 0, mbw - 1)
            y = np.clip(a["dst_y"] // 16, 0, mbh - 1)
            seen = np.zeros((mbh, mbw), bool)
            seen[y, x] = True
            dirty = ~seen  # intra blocks: new content with no reference
            m = (a["dst_x"] != a["src_x"]) | (a["dst_y"] != a["src_y"])
            dirty[y[m], x[m]] = True
            moved.append(float(dirty.mean()))
            labels, n = ndimage.label(dirty)
            cluster.append(int(np.bincount(labels.ravel())[1:].max()) if n else 0)
    c.close()
    return {
        "codec": s.codec_context.name, "size": f"{w}x{h}", "frames": nframes,
        "has_mvs": bool(moved),
        "cluster_max": max(cluster) if cluster else None,
        "moved_max": round(max(moved), 4) if moved else None,
        "moved_mean": round(float(np.mean(moved)), 4) if moved else None,
        "bytes_median": int(np.median(sizes)) if sizes else None,
        "bytes_max": int(max(sizes)) if sizes else None,
    }


def sources(target, limit):
    if not target.startswith("s3://"):
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.lower().endswith((".mp4", ".mov", ".mkv", ".ts")):
                    yield os.path.join(root, f), os.path.join(root, f)
        return
    import boto3
    bucket, _, prefix = target[5:].partition("/")
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"],
                      aws_access_key_id=os.environ["ACCESS_KEY"],
                      aws_secret_access_key=os.environ["SECRET_KEY"])
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if not o["Key"].lower().endswith(".mp4"):
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                s3.download_fileobj(bucket, o["Key"], tmp)
                tmp.flush()
                yield f"s3://{bucket}/{o['Key']}", tmp.name
            n += 1
            if limit and n >= limit:
                return


def camera(name):
    # Group by parent folder, which is how the corpus lays out cameras; good enough for a summary.
    return os.path.dirname(name.removeprefix("s3://")) or "."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("--cluster", type=int, default=8, help="blocks (16x16) in one changed region to count as motion")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    cams = defaultdict(lambda: [0, 0, 0])  # segments, active, without vectors
    for name, path in sources(args.target, args.limit):
        try:
            r = score(path)
        except Exception as e:
            print(json.dumps({"segment": name, "error": str(e)}), flush=True)
            continue
        r["active"] = r["cluster_max"] >= args.cluster if r["has_mvs"] else None  # unknown without vectors
        r["segment"] = name
        print(json.dumps(r), flush=True)
        cam = cams[camera(name)]
        cam[0] += 1
        cam[1] += bool(r["active"])
        cam[2] += not r["has_mvs"]

    print("\ncamera\tsegments\tstatic\tno_vectors", file=sys.stderr)
    for k, (n, active, nomv) in sorted(cams.items()):
        scored = n - nomv
        static = f"{100 * (scored - active) / scored:.0f}%" if scored else "n/a"
        print(f"{k}\t{n}\t{static}\t{nomv}", file=sys.stderr)


if __name__ == "__main__":
    main()
