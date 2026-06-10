"""Small USD pose-frame visualizer used by the skill test entry."""

from __future__ import annotations

import re

import torch


class DebugVisualizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._items = {}
        self._stage = None
        if enabled:
            try:
                import omni.usd

                self._stage = omni.usd.get_context().get_stage()
            except Exception:
                self.enabled = False

    def update_pose(self, name: str, pose_tensor: torch.Tensor | None):
        if not self.enabled or self._stage is None or pose_tensor is None:
            return
        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return
        path = "/Visuals/SkillRuntime/" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        xform = self._items.get(path)
        if xform is None:
            xform = UsdGeom.Xform.Define(self._stage, path)
            self._items[path] = xform
        pos = pose_tensor[:3].detach().cpu().tolist()
        quat = pose_tensor[3:7].detach().cpu().tolist()
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
