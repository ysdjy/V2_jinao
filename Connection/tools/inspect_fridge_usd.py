# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the converted fridge USD: prim tree, joints, and key bounding boxes.

Writes the result to /tmp/fridge_inspect.txt (avoids stdout buffering issues).
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os  # noqa: E402

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "Props", "Fridge_12252", "fridge.usd")
)

lines = []
stage = Usd.Stage.Open(USD_PATH)
lines.append(f"DEFAULT_PRIM :: {stage.GetDefaultPrim().GetPath()}")

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

joints = []
xforms = []
meshes = []
art_roots = []

xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    path = str(prim.GetPath())
    tname = prim.GetTypeName()

    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_roots.append(path)

    if "Joint" in str(tname):
        info = {"path": path, "type": str(tname)}
        if prim.HasAPI(UsdPhysics.DriveAPI) or True:
            axis_attr = prim.GetAttribute("physics:axis")
            low = prim.GetAttribute("physics:lowerLimit")
            high = prim.GetAttribute("physics:upperLimit")
            if axis_attr and axis_attr.HasValue():
                info["axis"] = str(axis_attr.Get())
            if low and low.HasValue():
                info["lower"] = float(low.Get())
            if high and high.HasValue():
                info["upper"] = float(high.Get())
        b0 = prim.GetRelationship("physics:body0")
        b1 = prim.GetRelationship("physics:body1")
        if b0:
            info["body0"] = [str(t) for t in b0.GetTargets()]
        if b1:
            info["body1"] = [str(t) for t in b1.GetTargets()]
        joints.append(info)

    if tname == "Xform":
        xforms.append(path)

    if tname == "Mesh":
        meshes.append(path)

lines.append(f"ARTICULATION_ROOTS :: {art_roots}")
lines.append("")
lines.append("=== JOINTS ===")
for j in joints:
    lines.append(str(j))

lines.append("")
lines.append("=== XFORM PRIMS (depth<=3) ===")
for x in xforms:
    if x.count("/") <= 3:
        lines.append(x)

lines.append("")
lines.append("=== MESH PRIMS ===")
for m in meshes:
    lines.append(m)


def world_bbox(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    bound = bbox_cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    ctr = (mn + mx) * 0.5
    return mn, mx, ctr


lines.append("")
lines.append("=== KEY BOUNDING BOXES (world, door closed / default pose) ===")
default_path = str(stage.GetDefaultPrim().GetPath())
root = default_path
for label, path in [
    ("whole_fridge", default_path),
    ("link_1_body", f"{root}/link_1"),
    ("link_0_door", f"{root}/link_0"),
]:
    bb = world_bbox(path)
    if bb:
        lines.append(f"{label} :: center={tuple(round(v,4) for v in bb[2])} min={tuple(round(v,4) for v in bb[0])} max={tuple(round(v,4) for v in bb[1])}")

# world origin (translation) of door link_0
door_prim = stage.GetPrimAtPath(f"{root}/link_0")
if door_prim and door_prim.IsValid():
    m = xform_cache.GetLocalToWorldTransform(door_prim)
    t = m.ExtractTranslation()
    lines.append(f"link_0_world_origin :: {tuple(round(v,4) for v in t)}")

# handle visual meshes -> combined world bbox
lines.append("")
lines.append("=== HANDLE VISUAL MESHES (world bbox) ===")
import math  # noqa: E402

hmin = [math.inf, math.inf, math.inf]
hmax = [-math.inf, -math.inf, -math.inf]
found = False
for mp in meshes:
    if "visuals/handle" in mp.lower():
        bb = world_bbox(mp)
        if bb:
            lines.append(f"{mp} :: center={tuple(round(v,4) for v in bb[2])}")
            for i in range(3):
                hmin[i] = min(hmin[i], bb[0][i])
                hmax[i] = max(hmax[i], bb[1][i])
            found = True
if found:
    hctr = [(hmin[i] + hmax[i]) * 0.5 for i in range(3)]
    lines.append(f"HANDLE_COMBINED :: center={tuple(round(v,4) for v in hctr)} min={tuple(round(v,4) for v in hmin)} max={tuple(round(v,4) for v in hmax)}")

    # handle position in link_0 (door) local frame  -> use directly as FrameTransformer offset
    from pxr import Gf  # noqa: E402

    door_prim = stage.GetPrimAtPath(f"{root}/link_0")
    if door_prim and door_prim.IsValid():
        l2w = xform_cache.GetLocalToWorldTransform(door_prim)
        w2l = l2w.GetInverse()
        handle_world = Gf.Vec3d(hctr[0], hctr[1], hctr[2])
        handle_local = w2l.Transform(handle_world)
        lines.append(f"HANDLE_IN_LINK0_LOCAL :: {tuple(round(v,5) for v in handle_local)}")
        # also report link_0 rotation (as quaternion) for reference
        rot = l2w.ExtractRotationQuat()
        lines.append(f"LINK0_WORLD_QUAT(wxyz) :: ({round(rot.GetReal(),4)}, {tuple(round(v,4) for v in rot.GetImaginary())})")

with open("/tmp/fridge_inspect.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

simulation_app.close()
