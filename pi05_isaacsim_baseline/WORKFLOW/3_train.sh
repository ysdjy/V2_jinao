#!/usr/bin/env bash
# STAGE 3 — Train: fine-tune pi0.5 (LoRA low-mem) on data/lerobot/$REPO_ID.
#   bash WORKFLOW/3_train.sh                      # smoke: TRAIN_STEPS from pipeline.env
#   bash WORKFLOW/3_train.sh --steps 3000         # real run
#   bash WORKFLOW/3_train.sh --repo-id franka_real --steps 5000 --batch 1
#
# Notes: config is LoRA (fits one 48 GB GPU at batch=1). Full fine-tune OOMs on 48 GB —
# for that you need more GPU memory and FSDP with an even batch (see README §Train).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
require_venv

RID="$REPO_ID"; STEPS="$TRAIN_STEPS"; BATCH="$BATCH_SIZE"; FS="$FSDP"
while [ $# -gt 0 ]; do case "$1" in
  --repo-id) RID="$2"; shift 2;;
  --steps) STEPS="$2"; shift 2;;
  --batch) BATCH="$2"; shift 2;;
  --fsdp) FS="$2"; shift 2;;
  *) shift;;
esac; done

if [ ! -d "$PROJ/data/lerobot/$RID" ]; then
  echo "ERROR: dataset data/lerobot/$RID not found. Run WORKFLOW/2_convert.sh --repo-id $RID first." >&2
  exit 1
fi
_log "STAGE 3 train: repo-id=$RID steps=$STEPS batch=$BATCH fsdp=$FS"
bash "$PROJ/scripts/train_pi05.sh" --repo-id "$RID" --steps "$STEPS" --batch-size "$BATCH" --fsdp "$FS"
CK="$(latest_ckpt)"
_log "done. latest checkpoint: ${CK:-<none — check the train log>}"
_log "Next: bash WORKFLOW/4_serve.sh   then   bash WORKFLOW/5_eval.sh"
