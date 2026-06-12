#!/usr/bin/env bash
# Stop any running policy server started by stage 4/5.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
bash "$PROJ/scripts/stop_server.sh"
