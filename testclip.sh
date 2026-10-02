#!/usr/bin/env bash
# Synthesize a "static security camera": fixed background, sensor noise, one textured
# object crossing between 3s and 7s. H.264, no B-frames, 2s GOP.
set -euo pipefail
out=${1:-cam.mp4}
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=s=1280x720:r=30,trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/30/TB,noise=alls=4:allf=t" \
  -f lavfi -i "mandelbrot=s=140x240:r=30" \
  -filter_complex "[0][1]overlay=x='if(between(t,3,7),(t-3)*290-140,-500)':y='360+20*sin(t*6)':shortest=1,format=yuv420p" \
  -t 10 -c:v libx264 -preset fast -crf 26 -g 60 -bf 0 -an "$out"
