#!/usr/bin/env bash
# Re-encode clips the way a simple camera would send them: H.264, no B-frames, 2s GOP, no audio.
#   ./prep.sh ~/Movies/hack footage
set -euo pipefail
src=$1
out=${2:-footage}
mkdir -p "$out"
for f in "$src"/*.mp4; do
  # Short names: drop the upload timestamp prefix and chunk suffix noise.
  name=$(basename "$f" .mp4 | sed -E 's/^[0-9]{8}_[0-9]{6}_//; s/_chunk_/_/; s/_segment_.*//; s/[0-9a-f]{20}_[0-9a-f]{20}_//')
  [ -f "$out/$name.mp4" ] && continue
  ffmpeg -hide_banner -loglevel error -y -i "$f" -an -c:v libx264 -preset fast -crf 23 -bf 0 -g 60 \
    -vf "fps=30" "$out/$name.mp4"
  echo "$out/$name.mp4"
done
