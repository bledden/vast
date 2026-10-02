#!/usr/bin/env bash
# The whole demo in one command: local relay, four looping "cameras", the AI worker, the viewer.
# Open http://localhost:8077 in Chrome. Ctrl-C stops everything. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
trap 'kill 0' EXIT

# Prefer the long clips fetch.py joins from consecutive chunks; fall back to single chunks.
pick() { for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done; echo "$1"; }
cams=(
  "driveway:$(pick footage/neighborhood_20260901.mp4 footage/neighborhood_20260901_0007.mp4)"
  "night:$(pick footage/neighborhood_20260902.mp4 footage/neighborhood_20260902_0021.mp4)"
  "warehouse:$(pick footage/2025_test_Warehouse_017_Camera_01.mp4 footage/2025_test_Warehouse_017_Camera_01_0007.mp4)"
  "street:$(pick footage/sf4.mp4 footage/sf4_0009.mp4)"
)

for cam in "${cams[@]}"; do
  [ -f "${cam#*:}" ] || { echo "missing ${cam#*:}: see Footage in README.md" >&2; exit 1; }
done

# relay.toml is written for moq-relay 0.16+; MOQ_RELAY picks the binary if an older one is first on PATH.
"${MOQ_RELAY:-moq-relay}" relay.toml >logs/relay.log 2>&1 &
for _ in $(seq 50); do curl -sf http://localhost:4443/certificate.sha256 >/dev/null && break; sleep 0.2; done
curl -sf http://localhost:4443/certificate.sha256 >/dev/null || { echo "relay failed to start; see logs/relay.log" >&2; tail -5 logs/relay.log >&2; exit 1; }

names=()
for cam in "${cams[@]}"; do
  name=${cam%%:*}
  ./camera.sh "${cam#*:}" "$name" >"logs/$name.log" 2>&1 &
  names+=("$name")
done

python3 -m http.server -d web 8077 --bind :: >logs/web.log 2>&1 &
echo "viewer: http://localhost:8077"

.venv/bin/python -u worker.py --broadcast "${names[@]}" "$@" 2>&1 | grep --line-buffered -v '^objc'
