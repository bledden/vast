#!/usr/bin/env python3
"""Dump per-frame codec data as JSON lines: frame type, compressed size, and every
non-zero motion vector (x, y, dx, dy, w, h) the encoder already computed.

    python extract.py video.mp4 > codec.jsonl

H.264 only: FFmpeg exports no motion vectors for HEVC.
"""
import json, sys
import av


def main():
    c = av.open(sys.argv[1])
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}
    for pkt in c.demux(s):
        for f in pkt.decode():
            mvs = []
            sd = f.side_data.get("MOTION_VECTORS")
            if sd is not None:
                for m in sd.to_ndarray():
                    dx, dy = int(m["dst_x"] - m["src_x"]), int(m["dst_y"] - m["src_y"])
                    if dx or dy:
                        mvs.append([int(m["dst_x"]), int(m["dst_y"]), dx, dy, int(m["w"]), int(m["h"])])
            print(json.dumps({
                "t": float(f.time), "key": f.key_frame, "bytes": pkt.size, "mvs": mvs,
            }))


if __name__ == "__main__":
    main()
