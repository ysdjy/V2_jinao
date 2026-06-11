"""Shared geometry helpers for the IK-based open/close drawer skills.

These let the open/close drawer skills work for ANY drawer / cabinet pose: the state machine only
provides the target drawer (its handle pose + drawer joint are read live from the scene), and the
open/pull direction and grasp orientation are derived from the live handle vs cabinet geometry.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

_UP = (0.0, 0.0, 1.0)


def open_direction_world(handle_pos: torch.Tensor, cabinet_root_pos: torch.Tensor) -> torch.Tensor:
    """Unit world vector pointing the way the drawer opens.

    The handle sits on the drawer front face, so the horizontal vector from the cabinet body toward
    the handle is the outward (opening) direction. Pulling the handle along this opens the drawer;
    pushing along its negative closes it. Works regardless of cabinet yaw.
    """
    v = (handle_pos - cabinet_root_pos).clone()
    v[2] = 0.0  # horizontal only
    n = torch.linalg.norm(v)
    if float(n) < 1e-6:
        return torch.tensor([0.0, -1.0, 0.0], device=handle_pos.device)
    return v / n


def grasp_quat_from_open_dir(open_dir: torch.Tensor, device) -> torch.Tensor:
    """TCP quaternion (w,x,y,z) for a front grasp of a horizontal handle bar.

    approach axis (TCP +Z) = -open_dir (gripper moves inward onto the handle);
    finger-open axis (TCP +Y) = world up (fingers straddle the bar top/bottom);
    the remaining axis runs along the (horizontal) handle bar.
    """
    up = torch.tensor(_UP, device=device)
    z = -open_dir / torch.linalg.norm(open_dir)
    x = torch.linalg.cross(up, z)
    nx = torch.linalg.norm(x)
    if float(nx) < 1e-6:  # approach nearly vertical: fall back to world X for the bar axis
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        x = x / nx
    y = torch.linalg.cross(z, x)
    R = torch.stack((x, y, z), dim=1)  # columns = hand axes in world
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]
