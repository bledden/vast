#!/usr/bin/env python3
"""Overlay the codec's own motion data on the video, the way a browser canvas would:
tint every block with a motion vector, draw the vector scaled up, and chart compressed
frame size along the bottom.

    python extract.py video.mp4 > codec.jsonl
    python render.py video.mp4 codec.jsonl overlay.mp4
"""
import json, sys
import av
from PIL import Image, ImageDraw

SCALE = 3  # vectors are a few pixels per frame; stretch them so they're visible
CHART = 90  # height of the frame-size strip


def main():
    src, data, dst = sys.argv[1:4]
    rows = [json.loads(line) for line in open(data)]
    inp = av.open(src)
    w, h = inp.streams.video[0].codec_context.width, inp.streams.video[0].codec_context.height
    out = av.open(dst, "w")
    o = out.add_stream("libx264", rate=inp.streams.video[0].average_rate or 30)
    o.width, o.height, o.pix_fmt, o.options = w, h, "yuv420p", {"crf": "20"}
    pmax = max((r["bytes"] for r in rows if not r["key"]), default=1)
    bw = w / len(rows)

    for i, f in enumerate(inp.decode(video=0)):
        r = rows[i]
        img = f.to_image().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle([0, 0, w, h], fill=(0, 0, 0, 110))  # dim the frame so codec data pops
        for x, y, dx, dy, bw_, bh_ in r["mvs"]:
            d.rectangle([x - bw_ // 2, y - bh_ // 2, x + bw_ // 2 - 1, y + bh_ // 2 - 1], fill=(255, 40, 80, 90))
        for x, y, dx, dy, _, _ in r["mvs"]:
            ex, ey = x - dx * SCALE, y - dy * SCALE  # trail back toward where the block came from
            d.line([x, y, ex, ey], fill=(255, 255, 255, 230), width=2)
            d.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=(255, 255, 255, 230))

        d.rectangle([0, h - CHART, w, h], fill=(10, 10, 20, 200))
        for j, rr in enumerate(rows[: i + 1]):
            x0 = j * bw
            if rr["key"]:
                d.rectangle([x0, h - CHART + 8, x0 + bw, h - 8], fill=(90, 90, 90, 200))
            else:
                bar = max(1, (CHART - 16) * rr["bytes"] / pmax)
                col = (255, 60, 90, 255) if rr["mvs"] else (80, 200, 120, 255)
                d.rectangle([x0, h - 8 - bar, x0 + bw, h - 8], fill=col)

        kind = "I-frame (keyframe)" if r["key"] else "P-frame"
        d.text((16, 14), f"{kind}  {r['bytes']:>6} bytes  {len(r['mvs']):>4} motion vectors  (no pixels analyzed)", fill="white")
        d.text((16, h - CHART - 18), "compressed frame size (from the bitstream, no decode)", fill=(220, 220, 220, 255))
        for p in o.encode(av.VideoFrame.from_image(Image.alpha_composite(img, ov).convert("RGB"))):
            out.mux(p)

    for p in o.encode():
        out.mux(p)
    out.close()


if __name__ == "__main__":
    main()
