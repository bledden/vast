# vast

Built at the [Real-Time Video Agents Hack](https://tokensand.com/vastsf) (VAST, NVIDIA, CoreWeave, Weights & Biases).

Most cameras are static and most frames show nothing new. The video encoder already knows
which parts changed: it computed motion vectors and frame sizes to compress the stream. We read
that data straight from the bitstream and only spend GPU time on segments and regions that moved.

![Motion vectors and frame sizes read from the H.264 stream](docs/overlay.gif)

## Scripts

```sh
pip install -r requirements.txt
```

| Script | What it does |
|---|---|
| `score.py <dir or s3://bucket/prefix>` | Per-segment motion score and a per-camera "% static" summary |
| `extract.py video.mp4 > codec.jsonl` | Frame type, compressed size, and motion vectors per frame |
| `render.py video.mp4 codec.jsonl out.mp4` | Draws that codec data over the video |
| `testclip.sh [out.mp4]` | Synthetic static camera with one object crossing |

`score.py` reads S3 credentials from `S3_ENDPOINT`, `ACCESS_KEY` and `SECRET_KEY`.

## How motion is scored

For each P-frame, a 16x16 block counts as changed if its motion vector is non-zero, or if it
has no vector at all (intra-coded, meaning new content). A segment is active when one connected
cluster of changed blocks reaches `--cluster` blocks (default 8). Clusters rather than totals,
because noisy sensors make the encoder intra-code scattered single blocks.

Limits:

- H.264 only. FFmpeg exports no motion vectors for HEVC; those segments report `n/a`.
- Vectors are what compresses best, not true optical flow. Flat, untextured surfaces show
  motion only at their edges.
- Use streams without B-frames, which most security cameras already do.
