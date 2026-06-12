"""Frame conventions and the base<-object pose chain.

==================================  T_target_source  ==========================

Every transform in this project is named ``T_<target>_<source>`` and maps a
point from the *source* frame into the *target* frame::

    p_target = T_target_source @ p_source

Frames
------
* ``camera``  : the D435 color optical frame (``d435_color_optical_frame``).
* ``ee``      : the Franka end-effector frame (``franka_hand`` / ``panda_hand``).
* ``base``    : the Franka base frame (``panda_link0``).
* ``object``  : the target object's own model/CAD frame.

The transforms we work with
---------------------------
* ``T_camera_object`` : object -> camera. Output of FoundationPose.
      p_camera = T_camera_object @ p_object
* ``T_ee_camera``     : camera -> ee. The hand-eye extrinsic (calibration).
      p_ee = T_ee_camera @ p_camera
* ``T_base_ee``       : ee -> base. From the robot forward kinematics.
      p_base = T_base_ee @ p_ee

Final goal: the object pose in the robot base frame::

    T_base_object = T_base_ee @ T_ee_camera @ T_camera_object

(That is just ``compose_T(T_base_ee, T_ee_camera, T_camera_object)``.)

Optical-frame note
------------------
The D435 *color optical frame* uses the standard camera-optical convention:
+X right, +Y down, +Z forward (into the scene). FoundationPose returns the
object pose in exactly this optical frame, so ``T_camera_object`` is directly
``p_camera = T_camera_object @ p_object`` with no extra axis swap needed here.
The IsaacLab ``Camera`` is configured with ``convention="ros"``; the optical
frame relationship is handled when we read the camera pose, not in this chain.
"""

from __future__ import annotations

import numpy as np

from .se3 import compose_T, validate_T

# Canonical frame names (kept consistent with configs/hand_eye.yaml).
CAMERA_FRAME = "d435_color_optical_frame"
EE_FRAME = "franka_hand"
BASE_FRAME = "panda_link0"

CONVENTION_DOC = (
    "T_target_source means transforming a point from the source frame to the "
    "target frame: p_target = T_target_source @ p_source"
)


def base_object_from_chain(T_base_ee, T_ee_camera, T_camera_object, validate: bool = True):
    """Compute ``T_base_object = T_base_ee @ T_ee_camera @ T_camera_object``.

    Parameters
    ----------
    T_base_ee : (4, 4) ee -> base, from robot forward kinematics.
    T_ee_camera : (4, 4) camera -> ee, the hand-eye extrinsic.
    T_camera_object : (4, 4) object -> camera, from FoundationPose.
    validate : if True, validate each input is a proper SE(3) matrix.

    Returns
    -------
    (4, 4) ``T_base_object`` : object -> base.
    """
    T_base_ee = np.asarray(T_base_ee, dtype=np.float64)
    T_ee_camera = np.asarray(T_ee_camera, dtype=np.float64)
    T_camera_object = np.asarray(T_camera_object, dtype=np.float64)
    if validate:
        validate_T(T_base_ee)
        validate_T(T_ee_camera)
        validate_T(T_camera_object)
    return compose_T(T_base_ee, T_ee_camera, T_camera_object)


def base_camera_from_chain(T_base_ee, T_ee_camera, validate: bool = True):
    """Compute ``T_base_camera = T_base_ee @ T_ee_camera`` (camera -> base)."""
    if validate:
        validate_T(T_base_ee)
        validate_T(T_ee_camera)
    return compose_T(T_base_ee, T_ee_camera)
