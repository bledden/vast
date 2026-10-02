#!/usr/bin/env bash
# The whole demo in one command: four looping "cameras", the AI worker, and (locally) a relay and
# the viewer. Ctrl-C stops everything. Logs go to logs/.
#
#   ./demo.sh                                    local relay; open http://localhost:8077 in Chrome
#   RELAY_URL='https://cdn.moq.pro/<project>?jwt=<token>' PREFIX=codec-motion/ ./demo.sh
#                                                publish to a remote relay instead (see deploy.sh)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
trap 'kill 0' EXIT
fail() { echo "demo: $*" >&2; exit 1; }

# Prefer the long clips fetch.py joins from consecutive chunks; fall back to single chunks.
pick() { for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done; echo "$1"; }
cams=(
  "driveway:$(pick footage/neighborhood_20260901.mp4 footage/neighborhood_20260901_0007.mp4)"
  "night:$(pick footage/neighborhood_20260902.mp4 footage/neighborhood_20260902_0021.mp4)"
  "warehouse:$(pick footage/2025_test_Warehouse_017_Camera_01.mp4 footage/2025_test_Warehouse_017_Camera_01_0007.mp4)"
  "street:$(pick footage/sf4.mp4 footage/sf4_0009.mp4)"
)

[ -x .venv/bin/python ] || fail "no .venv: run 'uv venv --python 3.12 && uv pip install -r requirements.txt'"
command -v ffmpeg >/dev/null || fail "ffmpeg not found (brew install ffmpeg)"

# Without the corpus footage (see README), every camera plays a synthetic scene instead.
for i in "${!cams[@]}"; do
  clip=${cams[$i]#*:}
  [ -f "$clip" ] && continue
  if [ ! -f footage/scene.mp4 ]; then
    echo "demo: $clip not found; generating footage/scene.mp4 (see Footage in README.md for the real clips)"
    mkdir -p footage && PYTHON=.venv/bin/python ./scene.sh footage/scene.mp4
  fi
  cams[$i]="${cams[$i]%%:*}:footage/scene.mp4"
done

# MOQ / MOQ_RELAY pick the binaries when the right ones aren't first on PATH.
export MOQ=${MOQ:-moq}
command -v "$MOQ" >/dev/null || fail "$MOQ not found: cargo install moq-cli@0.13.0 (or set MOQ)"
prefix=${PREFIX:-}

if [ -n "${RELAY_URL:-}" ]; then
  relay=$RELAY_URL
  echo "demo: publishing to ${relay%%\?*} as ${prefix}<camera>"
else
  relay=http://localhost:4443
  MOQ_RELAY=${MOQ_RELAY:-moq-relay}
  command -v "$MOQ_RELAY" >/dev/null || fail "$MOQ_RELAY not found: cargo install moq-relay@0.16.0 (or set MOQ_RELAY)"
  relay_version=$("$MOQ_RELAY" --version | awk '{print $2}')
  [ "$(printf '0.16.0\n%s\n' "$relay_version" | sort -V | head -1)" = "0.16.0" ] ||
    fail "moq-relay $relay_version is too old for relay.toml (needs 0.16+); set MOQ_RELAY to a newer one"
  curl -s -o /dev/null http://localhost:4443/ && fail "port 4443 is in use (another relay or demo running?)"
  curl -s -o /dev/null http://localhost:8077/ && fail "port 8077 is in use (another demo running?)"

  "$MOQ_RELAY" relay.toml >logs/relay.log 2>&1 &
  for _ in $(seq 50); do curl -sf http://localhost:4443/certificate.sha256 >/dev/null && break; sleep 0.2; done
  curl -sf http://localhost:4443/certificate.sha256 >/dev/null || { tail -5 logs/relay.log >&2; fail "relay failed to start; see logs/relay.log"; }
  .venv/bin/python -m http.server -d web 8077 --bind :: >logs/web.log 2>&1 &
  echo "viewer: http://localhost:8077"
fi

names=()
for cam in "${cams[@]}"; do
  name=${cam%%:*}
  ./camera.sh "${cam#*:}" "$prefix$name" "$relay" >"logs/$name.log" 2>&1 &
  names+=("$prefix$name")
done

sub=()
[ -n "${RELAY_URL:-}" ] && sub=(--sub-url "${RELAY_URL%%\?*}")  # read cameras on the public URL; the token may be publish-only
.venv/bin/python -u worker.py --url "$relay" "${sub[@]}" --broadcast "${names[@]}" "$@" 2>&1 | grep --line-buffered -v '^objc'
