"""Wrapper around NVlabs FoundationPose with a clean, stable interface.

Design goals
------------
* The FoundationPose source is NEVER copied into this project. The repo path is
  read from ``configs/foundationpose.yaml`` and added to ``sys.path`` lazily.
* If FoundationPose cannot be imported (repo/weights/CUDA missing), we fall back
  to :class:`MockFoundationPoseEstimator` unless ``use_mock_if_unavailable``
  is False, in which case a clear error is raised.
* All poses use the project convention ``T_target_source``. FoundationPose
  natively returns ``ob_in_cam`` = object-in-camera = ``T_camera_object``
  (``p_camera = T_camera_object @ p_object``), which is exactly our convention,
  so no extra inversion is needed. This is made explicit in code below.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

from ..camera.frame_types import RGBDFrame
from ..utils import image_io
from ..utils.config import load_yaml, resolve_path
from ..utils.logging_utils import get_logger

logger = get_logger("foundationpose_wrapper")


@dataclass
class PoseResult:
    """Result of a pose estimate / track call.

    T_camera_object : (4, 4) object -> camera (p_camera = T @ p_object).
    score : optional confidence score (None when unavailable).
    success : whether a pose was produced.
    mode : "estimate" or "track".
    debug : free-form dict (timings, refine iters, backend name, ...).
    """

    T_camera_object: np.ndarray
    score: float | None = None
    success: bool = True
    mode: str = "estimate"
    debug: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T_camera_object = np.asarray(self.T_camera_object, dtype=np.float64).reshape(4, 4)

    def to_dict(self, extra: dict | None = None) -> dict:
        from ..transforms.frame_conventions import CONVENTION_DOC

        d = {
            "success": bool(self.success),
            "mode": self.mode,
            "score": self.score,
            "convention": CONVENTION_DOC,
            "T_camera_object": self.T_camera_object.tolist(),
            "debug": self.debug,
        }
        if extra:
            d.update(extra)
        return d

    def save_json(self, path: str, extra: dict | None = None) -> None:
        image_io.save_json(path, self.to_dict(extra))


class FoundationPoseEstimator:
    """Model-based FoundationPose estimator with mock fallback.

    Parameters
    ----------
    config_path : path to ``configs/foundationpose.yaml``.
    force_mock : if True, always use the mock estimator (ignores the repo).
    """

    def __init__(self, config_path: str, force_mock: bool = False):
        self.config_path = config_path
        self.cfg = load_yaml(config_path)
        self.device = self.cfg.get("device", "cuda")
        self.est_refine_iter = int(self.cfg.get("est_refine_iter", 5))
        self.track_refine_iter = int(self.cfg.get("track_refine_iter", 2))
        self.debug = bool(self.cfg.get("debug", False))
        self.debug_dir = resolve_path(self.cfg.get("debug_dir", "outputs/debug_foundationpose"))
        self._use_mock_if_unavailable = bool(self.cfg.get("use_mock_if_unavailable", True))

        self._impl = None  # real FoundationPose estimator object
        self._mock = None  # MockFoundationPoseEstimator
        self._mesh = None  # cached loaded mesh
        self._backend = None  # "foundationpose" or "mock"

        if force_mock:
            self._init_mock("force_mock=True")
        else:
            self._try_init_real()

    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_mock(self) -> bool:
        return self._backend == "mock"

    # ------------------------------------------------------------------ #
    def _init_mock(self, reason: str):
        from .mock_foundationpose import MockFoundationPoseEstimator

        logger.warning("Using MockFoundationPoseEstimator (%s).", reason)
        logger.warning(
            "Mock poses are NOT real estimates — they only validate the data "
            "flow, transforms, saving and visualization."
        )
        self._mock = MockFoundationPoseEstimator(self.cfg)
        self._backend = "mock"

    def _try_init_real(self):
        repo = resolve_path(self.cfg.get("foundationpose_repo"))
        weights = resolve_path(self.cfg.get("weights_dir"))
        problems = []
        if not repo or not os.path.isdir(repo):
            problems.append(f"FoundationPose repo not found: {repo}")
        if weights and not os.path.isdir(weights):
            problems.append(f"weights_dir not found: {weights}")

        if problems:
            self._report_unavailable(problems, repo, weights)
            return

        try:
            if repo not in sys.path:
                sys.path.insert(0, repo)
            # FoundationPose's main estimator class lives in estimater.py.
            from estimater import FoundationPose  # type: ignore  # noqa: F401

            self._FoundationPose = FoundationPose
            self._backend = "foundationpose"
            logger.info("FoundationPose imported from %s", repo)
        except Exception as e:  # pragma: no cover - depends on external repo
            self._report_unavailable(
                [f"failed to import FoundationPose from {repo}: {e}"], repo, weights
            )

    def _report_unavailable(self, problems, repo, weights):
        msg = (
            "FoundationPose is unavailable:\n  - "
            + "\n  - ".join(problems)
            + "\n\nTo enable the real estimator:\n"
            "  1. git clone https://github.com/NVlabs/FoundationPose "
            f"{repo or '/home1/banghai/Documents/FoundationPose'}\n"
            f"  2. download the weights into {weights or repo + '/weights'}\n"
            "  3. build the CUDA extensions per the official README, or use the\n"
            "     official Docker image (recommended; set use_docker: true).\n"
        )
        if self._use_mock_if_unavailable:
            self._init_mock(msg)
        else:
            raise RuntimeError(
                msg + "\nuse_mock_if_unavailable=false, so not falling back to mock."
            )

    # ------------------------------------------------------------------ #
    def _ensure_real_registered(self, mesh_path: str, K: np.ndarray):
        """Instantiate the real FoundationPose object for ``mesh_path`` (cached)."""
        from .mesh_loader import load_mesh

        if self._impl is not None and self._mesh_path == mesh_path:
            return
        mesh = load_mesh(mesh_path)
        self._mesh = mesh
        self._mesh_path = mesh_path

        # Lazy imports from the FoundationPose repo (already on sys.path).
        import trimesh  # noqa: F401
        try:
            from estimater import ScorePredictor, PoseRefinePredictor  # type: ignore
            import nvdiffrast.torch as dr  # type: ignore

            glctx = dr.RasterizeCudaContext()
            self._impl = self._FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                glctx=glctx,
                debug=1 if self.debug else 0,
                debug_dir=self.debug_dir,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"failed to initialize FoundationPose for mesh {mesh_path}: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    def estimate(self, frame: RGBDFrame, mesh_path: str, mask: np.ndarray) -> PoseResult:
        """Estimate the object pose from a single RGB-D frame + initial mask.

        Returns a :class:`PoseResult` whose ``T_camera_object`` follows the
        ``p_camera = T_camera_object @ p_object`` convention.
        """
        frame.validate(raise_on_error=False)
        if self.is_mock:
            return self._mock.estimate(frame, mesh_path, mask)

        self._ensure_real_registered(mesh_path, frame.K)
        mask_bool = np.asarray(mask).astype(bool)
        # FoundationPose.register returns ob_in_cam == T_camera_object directly.
        ob_in_cam = self._impl.register(
            K=frame.K.astype(np.float64),
            rgb=frame.rgb,
            depth=frame.depth.astype(np.float32),
            ob_mask=mask_bool,
            iteration=self.est_refine_iter,
        )
        T_camera_object = np.asarray(ob_in_cam, dtype=np.float64).reshape(4, 4)
        return PoseResult(
            T_camera_object=T_camera_object,
            score=None,
            success=True,
            mode="estimate",
            debug={"backend": "foundationpose", "refine_iter": self.est_refine_iter},
        )

    def track(self, frame: RGBDFrame, previous_pose: np.ndarray) -> PoseResult:
        """Track the object given the previous ``T_camera_object``.

        ``previous_pose`` is the prior frame's ``T_camera_object``.
        """
        frame.validate(raise_on_error=False)
        if self.is_mock:
            return self._mock.track(frame, previous_pose)

        if self._impl is None:
            raise RuntimeError("call estimate() before track() to register the mesh")
        ob_in_cam = self._impl.track_one(
            rgb=frame.rgb,
            depth=frame.depth.astype(np.float32),
            K=frame.K.astype(np.float64),
            iteration=self.track_refine_iter,
        )
        T_camera_object = np.asarray(ob_in_cam, dtype=np.float64).reshape(4, 4)
        return PoseResult(
            T_camera_object=T_camera_object,
            score=None,
            success=True,
            mode="track",
            debug={"backend": "foundationpose", "refine_iter": self.track_refine_iter},
        )
