#!/usr/bin/env bash
# Synthetic static camera with real objects, from the sample images Ultralytics ships:
# a parked bus and bystanders on the left that never move (detections should be cached),
# and a person walking in from the right between 3s and 9s. H.264, no B-frames, 2s GOP.
set -euo pipefail
out=${1:-scene.mp4}
assets=$("${PYTHON:-python}" -c 'import ultralytics, os; print(os.path.join(os.path.dirname(ultralytics.__file__), "assets"))')
ffmpeg -hide_banner -loglevel error -y \
  -loop 1 -framerate 30 -i "$assets/bus.jpg" \
  -loop 1 -framerate 30 -i "$assets/zidane.jpg" \
  -filter_complex "\
color=c=0x5a6070:s=1280x720:r=30,noise=alls=6:allf=t[bg];\
[0]scale=-2:720[bus];\
[1]crop=400:670:748:41,scale=200:336[walker];\
[bg][bus]overlay=0:0[s];\
[s][walker]overlay=x='if(between(t,3,9),1280-(t-3)*120,2000)':y=330:shortest=1,format=yuv420p" \
  -t 12 -c:v libx264 -preset fast -crf 24 -g 60 -bf 0 -an "$out"
