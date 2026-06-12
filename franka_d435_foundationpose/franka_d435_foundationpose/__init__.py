"""franka_d435_foundationpose.

Standalone project to bridge an Intel RealSense D435 (mounted on a Franka
end-effector) and FoundationPose for 6D object pose estimation.

Two-environment architecture (see README):
  * IsaacLab side  -> camera providers, transforms, file/pose client.
                      Light deps only. MUST NOT import FoundationPose.
  * FoundationPose side -> the heavy estimator wrapper + pose server.

This top-level package only imports light, dependency-free submodules so that
it is safe to ``import franka_d435_foundationpose`` inside ``env_isaaclab``.
"""

__version__ = "0.1.0"
