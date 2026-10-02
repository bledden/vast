#!/usr/bin/env bash
# Host the viewer on Cloudflare (a static-assets Worker) pointed at a remote relay. Viewers
# subscribe anonymously, so the project needs public subscribe access on the prefix.
#   RELAY_URL=https://cdn.moq.pro/<project> PREFIX=codec-motion/ ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${RELAY_URL:?set RELAY_URL to the relay URL viewers dial (no token)}"
case "$RELAY_URL" in *jwt=*) echo "deploy: RELAY_URL must not carry a token; it ships to every viewer" >&2; exit 1 ;; esac
rm -rf dist && cp -r web dist
printf 'export default { url: "%s", prefix: "%s" };\n' "$RELAY_URL" "${PREFIX:-}" >dist/config.js
npx --yes wrangler@4 deploy
