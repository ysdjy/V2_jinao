#!/usr/bin/env bash
# Fine-tune pi0.5 on an IsaacLab-derived LeRobot dataset (smoke test by default).
#
# Usage:
#   bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 \
#       [--lerobot-root data/lerobot/franka_stack_cube_pi05] \
#       [--steps 10] [--batch-size 1] [--name pi05_isaaclab_franka] [--full]
#
# Smoke-test defaults: steps=10, batch_size=1. Requires the OpenPI venv.
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
VENV_PY="$PROJ/.venv_openpi/bin/python"
OPENPI="$PROJ/third_party/openpi"

REPO_ID="franka_stack_cube_pi05"
NAME="pi05_isaaclab_franka"
STEPS=10
BATCH=1
LEROBOT_ROOT=""
FSDP=1   # shard the model across N GPUs (FSDP). >1 needed: full pi0.5 (~3B) train step
         # needs ~50GB and does NOT fit one 48GB GPU; FSDP=2 halves per-GPU memory.
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo-id) REPO_ID="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --lerobot-root) LEROBOT_ROOT="$2"; shift 2;;
    --fsdp) FSDP="$2"; shift 2;;
    --full) STEPS=30000; BATCH=8; shift;;
    *) EXTRA+=("$1"); shift;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$PROJ/logs/train_pi05_${TS}.log"
CKPT_DIR="$PROJ/policies/checkpoints/pi05_isaaclab_${TS}"
mkdir -p logs "$PROJ/policies/checkpoints"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: OpenPI venv not found ($VENV_PY). Run scripts/setup_openpi_env.sh first." | tee "$LOG"
  exit 2
fi

# LeRobot datasets are resolved from HF_LEROBOT_HOME/<repo_id>. Point it at our data
# dir (or the provided root) so openpi can find the local dataset.
export HF_LEROBOT_HOME="${LEROBOT_ROOT:-$PROJ/data/lerobot}"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME" | tee "$LOG"

{
  echo "==== train_pi05 $TS ===="
  echo "repo_id=$REPO_ID name=$NAME steps=$STEPS batch=$BATCH"
  echo "checkpoint base=$CKPT_DIR"

  # 1. register our finetune config into openpi (reversible). Unregister first so the
  #    repo_id ALWAYS reflects the current dataset (registration is otherwise idempotent
  #    and would keep a stale repo_id from a previous run).
  "$VENV_PY" scripts/register_openpi_config.py --unregister >/dev/null 2>&1 || true
  "$VENV_PY" scripts/register_openpi_config.py --name "$NAME" --repo-id "$REPO_ID" || exit 3

  cd "$OPENPI"
  # pi0.5/OpenPI runs data-parallel across ALL visible GPUs and requires
  # batch_size % num_devices == 0. For a small smoke batch on a multi-GPU box, pin to
  # the first BATCH GPU(s) so e.g. batch-size=1 is valid (1 device). Real runs (--full,
  # larger even batch) keep all GPUs.
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
  if [ "$FSDP" -gt 1 ]; then
    # FSDP shards the model across $FSDP GPUs. Keep all GPUs visible. The data-parallel
    # axis is (NGPU / FSDP); batch must be divisible by that (=1 when FSDP==NGPU, so
    # batch-size=1 stays valid). Full pi0.5 needs FSDP>=2 to fit on 48GB cards.
    DATA_AXIS=$(( NGPU / FSDP )); [ "$DATA_AXIS" -lt 1 ] && DATA_AXIS=1
    echo "[train] FSDP=$FSDP across $NGPU GPUs (data axis=$DATA_AXIS); batch=$BATCH"
    EXTRA+=(--fsdp-devices "$FSDP")
  elif [ "$NGPU" -gt 1 ] && [ $((BATCH % NGPU)) -ne 0 ]; then
    USE=$BATCH; [ "$USE" -lt 1 ] && USE=1
    [ "$USE" -gt "$NGPU" ] && USE=$NGPU
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((USE-1)))"
    echo "[train] batch=$BATCH not divisible by $NGPU GPUs -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi

  # 2. compute norm stats for the dataset (required by training). tyro CLI -> --config-name.
  echo "---- compute_norm_stats ----"
  "$VENV_PY" scripts/compute_norm_stats.py --config-name "$NAME" || echo "[warn] compute_norm_stats failed (continuing to surface the real error in train)"

  # 3. train (smoke). XLA mem fraction kept modest; bf16 default in openpi.
  echo "---- train ----"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "$VENV_PY" scripts/train.py "$NAME" \
      --exp-name "isaaclab_${TS}" \
      --batch-size "$BATCH" \
      --num-train-steps "$STEPS" \
      --checkpoint-base-dir "$CKPT_DIR" \
      --no-wandb-enabled \
      --overwrite \
      "${EXTRA[@]}"
} 2>&1 | tee -a "$LOG"
echo "train exit: ${PIPESTATUS[0]}  log=$LOG  ckpt=$CKPT_DIR"
