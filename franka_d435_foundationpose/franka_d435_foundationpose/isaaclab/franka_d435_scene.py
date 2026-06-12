"""InteractiveScene config: Franka + D435-like camera + a target cube.

Import only inside env_isaaclab (after AppLauncher has started). No
FoundationPose dependency.
"""

from __future__ import annotations


def build_scene_cfg(
    num_envs: int = 1,
    env_spacing: float = 3.0,
    hand_eye_yaml: str | None = None,
    cube_pos=(0.5, 0.0, 0.055),
    cube_size=(0.05, 0.05, 0.05),
):
    """Return an InteractiveSceneCfg instance with ground, light, Franka, camera, cube.

    The cube is a simple, segmentable target placed in front of the robot so the
    end-effector camera can see it. Replace it later with your own USD/mesh.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab_assets import FRANKA_PANDA_CFG

    from .attach_d435_to_franka import build_d435_camera_cfg

    camera_cfg = build_d435_camera_cfg(hand_eye_yaml=hand_eye_yaml)

    @configclass
    class FrankaD435SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        target_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetObject",
            spawn=sim_utils.CuboidCfg(
                size=tuple(cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                semantic_tags=[("class", "target_object")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cube_pos)),
        )

        camera = camera_cfg

    return FrankaD435SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
