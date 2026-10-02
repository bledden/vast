#!/usr/bin/env bash
# The whole demo in one command: local relay, four looping "cameras", the AI worker, the viewer.
# Open http://localhost:8077 in Chrome. Ctrl-C stops everything. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
trap 'kill 0' EXIT

cams=(
  "driveway:footage/neighborhood_20260901_0007.mp4"
  "night:footage/neighborhood_20260902_0021.mp4"
  "warehouse:footage/2025_test_Warehouse_017_Camera_01_0007.mp4"
  "street:footage/sf4_0009.mp4"
)

moq-relay relay.toml >logs/relay.log 2>&1 &
until curl -sf http://localhost:4443/certificate.sha256 >/dev/null; do sleep 0.2; done

names=()
for cam in "${cams[@]}"; do
  name=${cam%%:*}
  ./camera.sh "${cam#*:}" "$name" >"logs/$name.log" 2>&1 &
  names+=("$name")
done

python3 -m http.server -d web 8077 --bind :: >logs/web.log 2>&1 &
echo "viewer: http://localhost:8077"

.venv/bin/python -u worker.py --broadcast "${names[@]}" "$@" 2>&1 | grep --line-buffered -v '^objc'
