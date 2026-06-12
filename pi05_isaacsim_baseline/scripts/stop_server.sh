#!/usr/bin/env bash
# Stop a policy server started via start_mock_server.sh / serve_pi05_checkpoint.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pidf in "$PROJ/logs/mock_server.pid" "$PROJ/logs/pi05_server.pid"; do
  if [ -f "$pidf" ]; then
    PID="$(cat "$pidf")"
    if kill "$PID" 2>/dev/null; then echo "stopped $PID ($pidf)"; fi
    rm -f "$pidf"
  fi
done
