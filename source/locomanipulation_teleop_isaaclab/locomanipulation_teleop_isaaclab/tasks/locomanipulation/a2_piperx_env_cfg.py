from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from locomanipulation_teleop_isaaclab.assets.a2_piper_x_asset import (
    A2_PIPERX_CFG,
)

from .a2_piper_env_cfg import A2FlatEnvCfg


@configclass
class A2PiperXFlatEnvCfg(A2FlatEnvCfg):
    """A2 + PiperX + gripper locomotion training."""

    robot: ArticulationCfg = A2_PIPERX_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )