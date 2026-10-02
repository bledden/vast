# team-47

## Project

**Codec Motion** runs AI on security cameras only when someone is watching and the camera's own
video encoder saw motion. Over MoQ, YOLO and NVIDIA Cosmos skip static frames: 75% less detector
time at 100% recall on a quiet street camera.

**Stack:** MoQ (relay, CLI, Python and browser SDKs; subscriptions start and stop the models);
FFmpeg/PyAV H.264 motion-vector export; YOLO11n; NVIDIA Cosmos3-Reason on CoreWeave (event
descriptions and alerts); VAST VSS (corpus footage, indexing only motion events); Weights & Biases
(eval runs, Weave traces of every Cosmos call); Cursor.

**Code:** https://github.com/kixelated/vast

**Live app:** none (runs locally with `./demo.sh`; see README)

**Supplementary:** https://wandb.ai/kixel-corp/vast (eval runs and Weave traces); demo video: NOT PROVIDED

## Feedback

NOT PROVIDED
