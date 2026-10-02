#!/usr/bin/env bash
# Pretend a recorded clip is a live camera: loop it in real time into MoQ.
#   ./camera.sh clip.mp4 [broadcast] [relay]
# The clip should be H.264 without B-frames; re-encode once with -bf 0 if needed.
set -euo pipefail
clip=$1
name=${2:-cam}
relay=${3:-http://localhost:4443}
ffmpeg -hide_banner -loglevel error -stream_loop -1 -re -i "$clip" -an -c copy \
  -f mpegts -pes_payload_size 0 -muxdelay 0 - \
  | "${MOQ:-moq}" --connect "$relay" --broadcast "$name" import ts
