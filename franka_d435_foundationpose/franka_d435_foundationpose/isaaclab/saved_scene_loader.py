"""Load a pre-saved scene USD and discover its prims (IsaacLab side).

Reads the saved scene **as data** via USD (pxr) — it does NOT import
SceneLayoutModule's Python code. Imports of ``isaaclab`` / ``pxr`` are lazy so
this module is importable anywhere; the USD calls only run inside the Isaac Sim
runtime (launched by isaaclab.sh).

Key facts about the example scene (scene_v0_*.usd), confirmed from its report:
  * Robot articulation : /World/envs/env_0/Robot   (hand: .../panda_hand)
  * Cabinet            : /World/envs/env_0/Cabinet
  * Knife              : /World/envs/env_0/Knife
  * Coffee machine     : /coffeemachine            <-- NOT under /World/envs/env_0
So prim discovery must NOT assume everything lives under /World/envs/env_0.
"""

from __future__ import annotations

import numpy as np

from ..transforms.se3 import make_T

# Names we treat as Franka end-effector links (first match wins).
DEFAULT_EE_CANDIDATES = ("franka_hand", "panda_hand", "panda_link8", "hand", "gripper")

# Marker substring used to name scenes that already have the D435 sensor baked in.
INSTRUMENTED_MARKER = "with_ee_d435"


def scene_layout_saved_dir() -> str:
    """Absolute path of SceneLayoutModule/saved_scenes (IsaacLab root sibling)."""
    import os

    from ..utils.config import PROJECT_ROOT

    isaaclab_root = os.path.dirname(PROJECT_ROOT)
    return os.path.join(isaaclab_root, "SceneLayoutModule", "saved_scenes")


def _list_usds(saved_dir: str):
    import glob
    import os

    return sorted(glob.glob(os.path.join(saved_dir, "*.usd")), key=os.path.getmtime)


def find_latest_scene(saved_dir: str | None = None, include_instrumented: bool = False) -> str:
    """Return the newest *.usd in saved_scenes.

    By default the previously-instrumented scenes (``*with_ee_d435*``) are
    excluded, so this returns the latest scene authored by SceneLayoutModule.
    """
    import os

    saved_dir = saved_dir or scene_layout_saved_dir()
    if not os.path.isdir(saved_dir):
        raise FileNotFoundError(f"saved_scenes dir not found: {saved_dir}")
    usds = _list_usds(saved_dir)
    if not include_instrumented:
        usds = [u for u in usds if INSTRUMENTED_MARKER not in os.path.basename(u)]
    if not usds:
        raise FileNotFoundError(f"no matching *.usd in {saved_dir}")
    return usds[-1]


def find_latest_instrumented_scene(saved_dir: str | None = None) -> str:
    """Return the newest instrumented (*with_ee_d435*.usd) scene."""
    import os

    saved_dir = saved_dir or scene_layout_saved_dir()
    usds = [u for u in _list_usds(saved_dir) if INSTRUMENTED_MARKER in os.path.basename(u)]
    if not usds:
        raise FileNotFoundError(
            f"no instrumented (*{INSTRUMENTED_MARKER}*.usd) scene in {saved_dir}; "
            "run instrument_latest_scene_with_sensors.py first."
        )
    return usds[-1]


def open_saved_scene(usd_path: str):
    """Open the saved USD as the current stage and return the pxr stage.

    Uses isaaclab.sim.open_stage (which replaces the current stage), then returns
    the live stage via isaaclab.sim.get_current_stage.
    """
    import os

    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"scene USD not found: {usd_path}")
    import isaaclab.sim as sim_utils

    ok = sim_utils.open_stage(usd_path)
    if not ok:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    return sim_utils.get_current_stage()


def _has_articulation_root(prim) -> bool:
    try:
        from pxr import UsdPhysics

        return prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    except Exception:
        return False


def discover_prims(stage, ee_candidates=DEFAULT_EE_CANDIDATES, verbose: bool = True) -> dict:
    """Traverse the stage and categorize interesting prims.

    Returns a dict with keys: ``articulation_roots``, ``franka_candidates``,
    ``ee_link_candidates``, ``coffee_machine``, ``cabinet``, ``knife``,
    ``sensors``, ``cameras``. Values are lists of prim path strings.
    """
    from pxr import UsdGeom

    if not ee_candidates:
        ee_candidates = DEFAULT_EE_CANDIDATES

    found = {
        "articulation_roots": [],
        "franka_candidates": [],
        "ee_link_candidates": [],
        "cubes": [],
        "coffee_machine": [],
        "cabinet": [],
        "knife": [],
        "sensors": [],
        "cameras": [],
    }
    ee_set = {c.lower() for c in ee_candidates}
    cube_raw = []

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        name = prim.GetName()
        lname = name.lower()
        lpath = path.lower()

        if _has_articulation_root(prim):
            found["articulation_roots"].append(path)
        if lname in ("robot", "franka", "panda") or lpath.endswith("/robot") or "franka" in lname or "panda_link0" == lname:
            found["franka_candidates"].append(path)
        if lname in ee_set:
            found["ee_link_candidates"].append(path)
        if "cube" in lname:
            cube_raw.append(path)
        if "coffee" in lname:
            found["coffee_machine"].append(path)
        if "cabinet" in lname:
            found["cabinet"].append(path)
        if "knife" in lname:
            found["knife"].append(path)
        if "sensor" in lpath:
            found["sensors"].append(path)
        if prim.IsA(UsdGeom.Camera):
            found["cameras"].append(path)

    # Keep only outermost cube prims (object-level), dropping nested cube meshes.
    found["cubes"] = _outermost_paths(cube_raw)

    if verbose:
        print("=" * 70)
        print("[saved_scene_loader] prim discovery")
        print("=" * 70)
        for key, vals in found.items():
            print(f"  {key:20s}: {vals if vals else '(none)'}")
        print("=" * 70)
    return found


def find_franka_root(stage, discovered: dict | None = None) -> str:
    """Return the Franka articulation root prim path.

    Preference: an articulation root that contains a panda_link0 / EE child;
    then a path ending in /Robot; then the first franka candidate.
    """
    discovered = discovered or discover_prims(stage, verbose=False)

    # Prefer an articulation root that looks like the Franka.
    for root in discovered["articulation_roots"]:
        prim = stage.GetPrimAtPath(root)
        child_names = {c.GetName().lower() for c in prim.GetChildren()}
        if "panda_link0" in child_names or child_names & {
            c.lower() for c in DEFAULT_EE_CANDIDATES
        }:
            return root

    for cand in discovered["franka_candidates"]:
        if cand.lower().endswith("/robot"):
            return cand

    if discovered["articulation_roots"]:
        return discovered["articulation_roots"][0]
    if discovered["franka_candidates"]:
        return discovered["franka_candidates"][0]

    raise RuntimeError(
        "Could not find a Franka robot prim. Discovered: "
        f"articulation_roots={discovered['articulation_roots']}, "
        f"franka_candidates={discovered['franka_candidates']}"
    )


def find_ee_link(stage, robot_path: str, ee_candidates=DEFAULT_EE_CANDIDATES) -> str:
    """Return the end-effector link prim path under ``robot_path``.

    Tries each candidate name in order; on failure prints the descendant link
    names to aid debugging and raises.
    """
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"robot prim not found at {robot_path}")

    # Map of lowercase child-link name -> full path (direct + nested descendants).
    descendants = {}
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.startswith(robot_path + "/"):
            descendants.setdefault(prim.GetName().lower(), p)

    for cand in ee_candidates:
        if cand.lower() in descendants:
            return descendants[cand.lower()]

    print(f"[saved_scene_loader] EE link not found under {robot_path}.")
    print("  candidates tried:", list(ee_candidates))
    print("  available descendant prim names:", sorted(descendants.keys()))
    raise RuntimeError(
        f"No end-effector link among {ee_candidates} found under {robot_path}."
    )


def get_world_pose_pxr(stage, prim_path: str):
    """Return (pos_xyz, quat_xyzw) world pose of a prim via USD xform compose.

    Note: this reads the authored/composed USD transform (UsdGeom), which is
    accurate for a freshly loaded, non-actuated scene. For physics-accurate
    poses after stepping, prefer the IsaacLab Articulation wrapper.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # Gf.Matrix4d, row-major
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()  # Gf.Quatd, real + imaginary
    pos = np.array([t[0], t[1], t[2]], dtype=np.float64)
    im = q.GetImaginary()
    quat_xyzw = np.array([im[0], im[1], im[2], q.GetReal()], dtype=np.float64)
    return pos, quat_xyzw


def world_pose_to_T(pos_xyz, quat_xyzw) -> np.ndarray:
    """Build T_world_body (4x4) from a position + quaternion [x,y,z,w]."""
    return make_T(R=np.asarray(quat_xyzw, dtype=np.float64), t=np.asarray(pos_xyz, dtype=np.float64))


def _rotx180_matrix():
    # ROS optical (+Z fwd, +Y down) -> USD camera (-Z fwd, +Y up): 180 deg about X.
    return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def author_usd_camera(stage, prim_path, T_ee_camera, optics, resolution, frame, mode):
    """Author a UsdGeom.Camera prim (D435-like) into the stage via pxr.

    The camera's local transform encodes the ROS optical extrinsic ``T_ee_camera``
    relative to its parent link, converted to the USD camera convention. D435
    pinhole params (focal length / aperture / clipping in mm) are written so an
    IsaacLab Camera can later read intrinsics from this prim. Custom ``fp:`` attrs
    record the frame / mode / resolution / extrinsic for downstream discovery.
    """
    from pxr import Gf, Sdf, UsdGeom

    from ..transforms.se3 import matrix_to_quat

    cam = UsdGeom.Camera.Define(stage, Sdf.Path(prim_path))
    prim = cam.GetPrim()

    fmm = float(optics.get("focal_length_mm", 20.1))
    hap = float(optics.get("horizontal_aperture_mm", 20.955))
    w = int(resolution.get("width", 640))
    h = int(resolution.get("height", 480))
    vap = hap * (h / float(w))
    clip = optics.get("clipping_range_m", [0.05, 6.0])

    cam.CreateFocalLengthAttr(fmm)
    cam.CreateHorizontalApertureAttr(hap)
    cam.CreateVerticalApertureAttr(vap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clip[0]), float(clip[1])))
    cam.CreateProjectionAttr("perspective")

    # Local transform: parent <- camera (USD camera convention).
    T = np.asarray(T_ee_camera, dtype=np.float64)
    R_usd = T[:3, :3] @ _rotx180_matrix()
    qx, qy, qz, qw = matrix_to_quat(R_usd)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    # IsaacLab expects the canonical op order [translate, orient, scale].
    xf.AddTranslateOp().Set(Gf.Vec3d(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])))
    xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Custom metadata for downstream discovery.
    prim.CreateAttribute("fp:camera_frame", Sdf.ValueTypeNames.String).Set(frame)
    prim.CreateAttribute("fp:camera_mode", Sdf.ValueTypeNames.String).Set(mode)
    prim.CreateAttribute("fp:resolution_width", Sdf.ValueTypeNames.Int).Set(w)
    prim.CreateAttribute("fp:resolution_height", Sdf.ValueTypeNames.Int).Set(h)
    prim.CreateAttribute("fp:T_ee_camera", Sdf.ValueTypeNames.String).Set(
        str(np.asarray(T_ee_camera).tolist())
    )
    return prim_path


def add_semantic_label(stage, prim_path, label, label_type="class") -> bool:
    """Best-effort: add a semantic label to a prim for segmentation.

    Tries the isaacsim semantics util first, then a raw Semantics schema.
    Returns True if a label was authored.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    # Path 1: isaacsim core util (name varies across versions).
    for modname, fn in (
        ("isaacsim.core.utils.semantics", "add_update_semantics"),
        ("omni.isaac.core.utils.semantics", "add_update_semantics"),
    ):
        try:
            mod = __import__(modname, fromlist=[fn])
            getattr(mod, fn)(prim, semantic_label=label, type_label=label_type)
            return True
        except Exception:
            pass
    # Path 2: raw USD Semantics schema.
    try:
        from pxr import Semantics

        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set(label_type)
        sem.CreateSemanticDataAttr().Set(label)
        return True
    except Exception:
        return False


def export_stage(stage, out_path) -> str:
    """Export the current stage to a new USD file (does not touch the source)."""
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = stage.Export(out_path)
    if ok is False:  # Export returns bool on some USD builds, None on others
        raise RuntimeError(f"stage.Export failed for {out_path}")
    return out_path


def _outermost_paths(paths):
    """Given prim paths, keep only those with no ancestor also in the list."""
    uniq = sorted(set(paths))
    out = []
    for p in uniq:
        if not any(p != q and p.startswith(q + "/") for q in uniq):
            out.append(p)
    return out


def find_cubes(stage, discovered: dict | None = None) -> list:
    """Return object-level cube prim paths, sorted (e.g. Cube_1, Cube_2, ...)."""
    discovered = discovered or discover_prims(stage, verbose=False)
    return list(discovered.get("cubes", []))


def select_target_cube(stage, object_name=None, object_prim_path=None, discovered=None) -> str:
    """Pick the target cube prim path.

    Priority: explicit ``object_prim_path`` -> a cube whose name matches an index
    in ``object_name`` (e.g. cube_0 -> first cube) -> the first discovered cube.
    """
    if object_prim_path:
        prim = stage.GetPrimAtPath(object_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"object_prim_path not found in stage: {object_prim_path}")
        return object_prim_path

    cubes = find_cubes(stage, discovered)
    if not cubes:
        raise RuntimeError("no cube prims discovered in the scene")

    # Map object_name like 'cube_0' / 'cube_2' to an index into the sorted cubes.
    if object_name:
        import re

        m = re.search(r"(\d+)", object_name)
        if m:
            idx = int(m.group(1))
            # accept either 0-based (cube_0) or the scene's own 1-based names.
            if 0 <= idx < len(cubes):
                return cubes[idx]
            for c in cubes:
                if c.rsplit("/", 1)[-1].lower() == object_name.lower():
                    return c
    return cubes[0]


def get_object_gt_pose(stage, prim_path, robot_base_path=None, sim_articulation=None):
    """Return ground-truth poses of an object prim.

    Returns a dict with ``T_world_object`` and, if ``robot_base_path`` is given,
    ``T_base_object`` (= inv(T_world_base) @ T_world_object). Poses come from USD
    xforms (pxr); pass ``sim_articulation`` only to document the source.
    """
    from ..transforms.se3 import invert_T

    pos_o, quat_o = get_world_pose_pxr(stage, prim_path)
    T_world_object = world_pose_to_T(pos_o, quat_o)
    out = {
        "object_prim_path": prim_path,
        "T_world_object": T_world_object,
        "source": "usd_xform",
    }
    if robot_base_path:
        pos_b, quat_b = get_world_pose_pxr(stage, robot_base_path)
        T_world_base = world_pose_to_T(pos_b, quat_b)
        out["T_base_object"] = invert_T(T_world_base) @ T_world_object
        out["base_frame"] = robot_base_path.rsplit("/", 1)[-1]
    return out


INSTRUMENT_MANIFEST_FIELDS = (
    "schema_version",
    "source_scene_usd",
    "instrumented_scene_usd",
    "camera_prim_path",
    "camera_mode",
    "mounted_link",
    "camera_frame",
    "resolution",
    "sensor_types",
    "discovered_cubes",
    "default_target_cube",
    "default_target_cube_prim",
    "semantic_labels_added",
    "did_not_modify_isaaclab_official_files",
    "did_not_import_scenelayoutmodule_python",
)


def build_instrument_manifest(
    source_scene_usd,
    instrumented_scene_usd,
    camera_prim_path,
    camera_mode,
    mounted_link,
    camera_frame,
    resolution,
    sensor_types,
    discovered_cubes,
    default_target_cube,
    default_target_cube_prim,
    semantic_labels_added,
    extra=None,
) -> dict:
    """Build the JSON manifest dict for an instrumented scene (all required fields)."""
    m = {
        "schema_version": "fp_instrument_v1",
        "source_scene_usd": source_scene_usd,
        "instrumented_scene_usd": instrumented_scene_usd,
        "camera_prim_path": camera_prim_path,
        "camera_mode": camera_mode,
        "mounted_link": mounted_link,
        "camera_frame": camera_frame,
        "resolution": resolution,
        "sensor_types": sensor_types,
        "discovered_cubes": list(discovered_cubes),
        "default_target_cube": default_target_cube,
        "default_target_cube_prim": default_target_cube_prim,
        "semantic_labels_added": list(semantic_labels_added),
        "did_not_modify_isaaclab_official_files": True,
        "did_not_import_scenelayoutmodule_python": True,
    }
    if extra:
        m.update(extra)
    return m


def write_instrument_report_md(manifest: dict, report_path: str) -> str:
    """Write the markdown report for an instrumented scene from its manifest."""
    import os

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    m = manifest
    res = m.get("resolution", {})
    st = m.get("sensor_types", {})
    L = [
        "# Instrumented scene report",
        "",
        f"- source scene USD: `{m['source_scene_usd']}`",
        f"- instrumented scene USD: `{m['instrumented_scene_usd']}`",
        "",
        "## Sensor",
        "",
        f"- camera prim path: `{m['camera_prim_path']}`",
        f"- camera mode: `{m['camera_mode']}`",
        f"- mounted on Franka link: `{m['mounted_link']}`",
        f"- camera frame: `{m['camera_frame']}`",
        f"- resolution: {res.get('width')} x {res.get('height')}",
        f"- rgb: {st.get('rgb')}, depth: {st.get('depth')}, segmentation: {st.get('segmentation')}",
        "",
        "## Objects",
        "",
        f"- discovered cubes: {m['discovered_cubes'] or '(none)'}",
        f"- default test target: `{m['default_target_cube']}` -> `{m['default_target_cube_prim']}`",
        f"- semantic labels added: {m['semantic_labels_added'] or '(none)'}",
        "",
        "## Guarantees",
        "",
        f"- did NOT modify IsaacLab official files: {m['did_not_modify_isaaclab_official_files']}",
        f"- did NOT import SceneLayoutModule Python code: {m['did_not_import_scenelayoutmodule_python']}",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    return report_path


def write_discovery_report(discovered: dict, scene_usd: str, report_path: str) -> str:
    """Write a markdown discovery report and return its path."""
    import os

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = [
        "# Scene discovery report",
        "",
        f"- source scene USD: `{scene_usd}`",
        "",
        "## Discovered prims",
        "",
    ]
    for key, vals in discovered.items():
        lines.append(f"### {key}")
        if vals:
            for v in vals:
                lines.append(f"- `{v}`")
        else:
            lines.append("- (none)")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Objects are NOT assumed to all live under `/World/envs/env_0` "
        "(e.g. some assets may be at global paths like `/coffeemachine`).",
        "- This report is generated by reading the USD as data; SceneLayoutModule "
        "Python code is not imported.",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path
