# Initial masks

FoundationPose (model-based) needs an **initial binary mask** of the target
object in the first frame to seed `estimate`. After that, `track` propagates the
pose without a mask.

A mask here is a single-channel PNG where **white (255) = object**, black = background,
same H x W as the RGB image.

Ways to obtain the initial mask (see `franka_d435_foundationpose/foundationpose/mask_provider.py`):

1. **File** — drop a hand-painted / pre-computed mask here, e.g.
   `target_object_mask.png`, and reference it in `configs/object_assets.yaml`.
2. **Bounding box** — `MaskProvider.from_bbox(H, W, [x0,y0,x1,y1])` for a quick
   rectangular placeholder.
3. **Sim segmentation** — in IsaacLab, `MaskProvider.from_sim_segmentation(seg, ids)`
   turns a semantic/instance segmentation map into a mask automatically.

**Future automation (not required now):** plug in SAM2 (point/box prompt) or
Grounded-SAM (text prompt) to generate this initial mask automatically; then let
FoundationPose handle `estimate` + `track`.
