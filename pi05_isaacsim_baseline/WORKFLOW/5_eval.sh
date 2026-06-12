#!/usr/bin/env bash
# STAGE 5 — Evaluate: closed-loop rollout in IsaacLab driven by the policy server.
# Starts a server (newest checkpoint, else mock), runs NUM_ROLLOUTS, writes a summary.
#   bash WORKFLOW/5_eval.sh                          # pipeline.env defaults
#   bash WORKFLOW/5_eval.sh --rollouts 20 --max-steps 300
#   bash WORKFLOW/5_eval.sh --mock                   # smoke the loop with mock policy
# Output: logs/eval_policy_*.json (success rate, latency, control freq, safety clips).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

CK="$(latest_ckpt)"; N="$NUM_ROLLOUTS"; MS="$MAX_STEPS"; P="$PORT"; IMG="$IMAGE_MODE"; MOCK=0
while [ $# -gt 0 ]; do case "$1" in
  --rollouts) N="$2"; shift 2;;
  --max-steps) MS="$2"; shift 2;;
  --port) P="$2"; shift 2;;
  --image-mode) IMG="$2"; shift 2;;
  --mock) MOCK=1; shift;;
  *) shift;;
esac; done

_log "STAGE 5 eval: task=$TASK rollouts=$N max_steps=$MS port=$P img=$IMG mock=$MOCK"
if [ "$MOCK" = "0" ] && [ -n "$CK" ]; then
  bash "$PROJ/scripts/eval_pi05_in_isaaclab.sh" --task "$TASK" \
    --num_rollouts "$N" --max_steps "$MS" --port "$P" --image_mode "$IMG" \
    --config "$CONFIG_NAME" --ckpt "$CK"
else
  bash "$PROJ/scripts/eval_pi05_in_isaaclab.sh" --task "$TASK" \
    --num_rollouts "$N" --max_steps "$MS" --port "$P" --image_mode "$IMG"
fi
_log "done. See logs/eval_policy_*.json"
