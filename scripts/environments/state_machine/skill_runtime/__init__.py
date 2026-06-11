"""Backward-compatibility shim.

The skill-runtime / state-machine code moved to
``projects/franka_skill_state_machine/`` (runtime/, skills/, state_machine/, learned_drawer/).
This shim re-exposes the moved modules under the old ``skill_runtime.<name>`` import path so
existing scripts (e.g. skill_test_ui_joint_gello.py, tests) keep working without edits.

New code should import from the project packages directly (runtime.*, skills.*, state_machine.*,
learned_drawer.*) after putting the project root on sys.path.
"""

import importlib
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "projects", "franka_skill_state_machine")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# old skill_runtime module name -> new dotted path
_MODULE_MAP = {
    # shared runtime
    "base_skill": "runtime.base_skill",
    "skill_types": "runtime.skill_types",
    "skill_request": "runtime.skill_request",
    "skill_result": "runtime.skill_result",
    "scene_state_provider": "runtime.scene_state_provider",
    "ik_joint_adapter": "runtime.ik_joint_adapter",
    "target_registry": "runtime.target_registry",
    "drawer_target_config": "runtime.drawer_target_config",
    "drawer_obs_adapter": "runtime.drawer_obs_adapter",
    "drawer_ik_common": "runtime.drawer_ik_common",
    "debug_visualizer": "runtime.debug_visualizer",
    "layout_debug_visualizer": "runtime.layout_debug_visualizer",
    "scene_layout": "runtime.scene_layout",
    "simple_scene_layout": "runtime.simple_scene_layout",
    "ui_controller": "runtime.ui_controller",
    "joint_debug_logger": "runtime.joint_debug_logger",
    "grasp_skill": "runtime.grasp_skill",
    "place_skill": "runtime.place_skill",
    "drawer_skill": "runtime.drawer_skill",
    # state machine
    "skill_executor": "state_machine.skill_executor",
    # the 4 deployable skills (renamed)
    "grasp_joint_skill": "skills.grasp_skill",
    "place_joint_skill": "skills.place_skill",
    "open_drawer_skill": "skills.open_drawer_skill",
    "close_drawer_skill": "skills.close_drawer_skill",
    # learned-drawer (stored aside)
    "official_drawer_policy": "learned_drawer.official_drawer_policy",
    "official_drawer_joint_skill": "learned_drawer.official_drawer_joint_skill",
    "custom_drawer_joint_skill": "learned_drawer.custom_drawer_joint_skill",
    "scripted_drawer_joint_skill": "learned_drawer.scripted_drawer_joint_skill",
}

for _old, _new in _MODULE_MAP.items():
    try:
        sys.modules[f"{__name__}.{_old}"] = importlib.import_module(_new)
    except Exception:
        # a missing optional module should not break the whole shim
        pass
