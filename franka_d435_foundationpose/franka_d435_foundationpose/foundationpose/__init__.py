"""FoundationPose wrapper, mock estimator, mesh & mask loaders.

NOTE: importing this subpackage does not import FoundationPose itself. The real
estimator is only imported lazily inside FoundationPoseEstimator when needed, so
this package stays importable on the IsaacLab side for the mock / type defs.
"""

from .foundationpose_wrapper import FoundationPoseEstimator, PoseResult
from .mock_foundationpose import MockFoundationPoseEstimator

__all__ = [
    "FoundationPoseEstimator",
    "PoseResult",
    "MockFoundationPoseEstimator",
]
