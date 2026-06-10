#!/usr/bin/env bash
# Download all USD assets required by the open-drawer V0 task into Connection/assets,
# mirroring the Nucleus relative directory structure so that the USD files' internal
# relative references (e.g. ./Props/, ./configurations/) resolve offline.
#
# Run this only if Connection/assets is missing (e.g. assets were not committed to git).
# Once downloaded, the project is fully self-contained and needs no network access.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/assets"
BASE="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

FILES=(
  # --- Franka Emika Panda (relative refs are ./Props/ and ./Materials/) ---
  "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Materials/Materials.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/instanceable_collision_meshes.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_hand.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_leftfinger.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link0.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link1.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link2.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link3.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link4.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link5.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link6.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_link7.usd"
  "Isaac/IsaacLab/Robots/FrankaEmika/Props/panda_rightfinger.usd"
  # --- Sektion Cabinet ---
  "Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
  "Isaac/Props/Sektion_Cabinet/configurations/cabinet_default_physics.usd"
  "Isaac/Props/Sektion_Cabinet/configurations/cabinet_lab.usd"
  "Isaac/Props/Sektion_Cabinet/sektion_cabinet_collisions.usd"
  "Isaac/Props/Sektion_Cabinet/sektion_cabinet_visuals.usd"
)

echo "[download_assets] target: ${ASSETS_DIR}"
for rel in "${FILES[@]}"; do
  dest="${ASSETS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  echo "  -> ${rel}"
  curl -sfL "${BASE}/${rel}" -o "${dest}"
done

echo "[download_assets] done. ${#FILES[@]} files saved under ${ASSETS_DIR}"
