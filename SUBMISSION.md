# team-47

## Project

**Codec Vision** runs AI on security cameras only when something changes. It reads motion
straight from the H.264 bitstream over MoQ, so YOLO and Cosmos skip static footage: 75% less
detector time at 100% recall on a quiet street camera.

**Stack:** MoQ (relay, CLI, Python and browser SDKs); FFmpeg/PyAV motion-vector export; YOLO11n;
NVIDIA Cosmos3-Reason on CoreWeave (event descriptions); VAST VSS (indexing only motion events for
search); Weights & Biases (experiment runs, Weave traces); Cursor.

**Code:** https://github.com/kixelated/vast

**Live app:** none (runs locally; see README)

**Supplementary:** https://wandb.ai/kixel-corp/vast (eval runs and Weave traces); demo video: NOT PROVIDED

## Feedback

NOT PROVIDED
