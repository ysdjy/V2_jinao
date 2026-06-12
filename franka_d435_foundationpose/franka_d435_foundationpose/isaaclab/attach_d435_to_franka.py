"""Build a D435-like ``CameraCfg`` attached to the Franka end-effector.

Imports ``isaaclab`` lazily, so this file is safe to *import* anywhere; the
isaaclab symbols are only touched when you call the builder functions (which
only happens after AppLauncher has started the simulator).

D435 color intrinsics note
--------------------------
The real D435 color stream at 640x480 has roughly fx=fy~=615 px, cx~=320,
cy~=240. IsaacLab derives intrinsics from focal_length / horizontal_aperture /
resolution. With horizontal_aperture=20.955 mm (the IsaacLab default sensor
width) and width=640, a focal_length of ~20.1 mm yields fx ~= 615 px, matching
the D435 reasonably. We expose these as parameters so they can be tuned to a
specific device's factory intrinsics.
"""

from __future__ import annotations


def _hand_eye_offset(hand_eye_yaml: str | None):
    """Return (pos_xyz, quat_wxyz) for CameraCfg.OffsetCfg from hand_eye.yaml.

    The offset is the camera pose in the parent (end-effector) link frame, i.e.
    the translation/rotation of ``T_ee_camera``. IsaacLab quaternions are
    (w, x, y, z); hand_eye.yaml stores (x, y, z, w).
    """
    pos = (0.05, 0.0, 0.08)
    quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    if hand_eye_yaml is None:
        return pos, quat_wxyz
    try:
        import yaml

        with open(hand_eye_yaml, "r") as f:
            cfg = yaml.safe_load(f) or {}
        te = cfg.get("T_ee_camera", {})
        t = te.get("translation_m")
        if t and len(t) == 3:
            pos = tuple(float(v) for v in t)
        q = te.get("quaternion_xyzw")
        if q and len(q) == 4:
            x, y, z, w = (float(v) for v in q)
            quat_wxyz = (w, x, y, z)
    except Exception:
        pass
    return pos, quat_wxyz


def build_d435_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/panda_hand/d435_color",
    width: int = 640,
    height: int = 480,
    focal_length: float = 20.1,
    horizontal_aperture: float = 20.955,
    clipping_range=(0.05, 6.0),
    update_period: float = 0.0,
    hand_eye_yaml: str | None = None,
):
    """Create a ``CameraCfg`` for a D435-like RGB-D camera on the Franka hand.

    The camera outputs ``rgb`` and ``distance_to_image_plane`` (metric depth in
    meters along the optical axis — exactly what FoundationPose wants).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    pos, quat_wxyz = _hand_eye_offset(hand_eye_yaml)

    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="ros"),
    )


def _quat_wxyz_from_T(T):
    """Extract a (w, x, y, z) quaternion from a 4x4 transform's rotation block."""
    from ..transforms.se3 import matrix_to_quat

    x, y, z, w = matrix_to_quat(T[:3, :3])
    return (float(w), float(x), float(y), float(z))


def build_camera_cfg_for_mount(mode, prim_path: str):
    """Build a ``CameraCfg`` from a :class:`MountMode` (end_effector or fixed_scene).

    - end_effector: the offset is the hand-eye extrinsic ``T_ee_camera`` relative
      to the parent (hand) link, with convention "ros".
    - fixed_scene: the offset is the world placement (eye + look-at) with
      convention "world".

    Returns ``(camera_cfg, extrinsic)`` where ``extrinsic`` is the 4x4 matrix
    used (T_ee_camera for EE mode, T_world_camera for fixed mode).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    from ..camera.mount_config import CAMERA_MODE_EE, look_at_to_T_world_camera

    optics = mode.sim_optics
    focal_length = float(optics.get("focal_length_mm", 20.1))
    horizontal_aperture = float(optics.get("horizontal_aperture_mm", 20.955))
    clipping_range = tuple(optics.get("clipping_range_m", [0.05, 6.0]))

    data_types = ["rgb", "distance_to_image_plane"]
    if mode.want_segmentation:
        # semantic (by class label) + instance (by prim path, unambiguous).
        data_types.append("semantic_segmentation")
        data_types.append("instance_segmentation_fast")

    if mode.name == CAMERA_MODE_EE:
        extrinsic = mode.T_ee_camera()  # camera -> ee
        offset = CameraCfg.OffsetCfg(
            pos=tuple(extrinsic[:3, 3].tolist()),
            rot=_quat_wxyz_from_T(extrinsic),
            convention="ros",
        )
    else:  # fixed_scene
        pose = mode.pose()
        eye = pose.get("translation_m", [1.2, -0.8, 1.0])
        target = pose.get("look_at_m", [0.4, 0.0, 0.4])
        extrinsic = look_at_to_T_world_camera(eye, target)  # camera -> world
        offset = CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in eye),
            rot=_quat_wxyz_from_T(extrinsic),
            convention="world",
        )

    camera_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=mode.height,
        width=mode.width,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
        ),
        offset=offset,
    )
    return camera_cfg, extrinsic
