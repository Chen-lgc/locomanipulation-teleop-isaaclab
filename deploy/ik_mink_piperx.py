"""PiperX Mink IK: independent arm motion controller (A2 RL policy unchanged).

Targets are in the A2 base_link coordinate frame. The reduced model was
built directly from robot_model/a2_piper_x.xml to ensure consistent FK.

First-stage constraint: joint limits only. PiperL-specific collision geometry
names are intentionally *not* used. Add verified PiperX collision pairs later.
"""

from pathlib import Path
import numpy as np
import mujoco
import mink


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "robot_model" / "scene_floating_piperx.xml"


class IKMink:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.joint_names = ["basepitch", "basez"] + [f"joint{i}" for i in range(1, 7)]
        self.target_site_id = self.model.site("attachment_site").id
        self.target_body_id = self.model.body("target").id
        self.target_mocap_id = int(self.model.body_mocapid[self.target_body_id])
        if self.target_mocap_id < 0:
            raise ValueError("target body must have mocap=true")
        for name in self.joint_names + ["joint7", "joint8"]:
            self.model.joint(name)
        if self.model.nq != 10 or self.model.nv != 10:
            raise ValueError(f"IK model expected nq=nv=10, got {self.model.nq}/{self.model.nv}")

        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site", frame_type="site",
            position_cost=1.0, orientation_cost=0.0, lm_damping=0.1,
        )
        posture_cost = np.zeros(self.model.nv)
        posture_cost[2:8] = 1e-3
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.solver = "daqp"
        self.dt = 0.005
        self.max_iters = 80
        self.pos_threshold = 0.02  # 2 cm, test-stage criterion

    def compute(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        initial_joints_position: np.ndarray,
        initial_base_pose: np.ndarray,
        optimize_height: bool = False,
        optimize_pitch: bool = False,
        visualize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (base [pitch,z], six arm joint targets, success)."""
        target = np.asarray(target_pos, dtype=float).reshape(3)
        q0 = np.asarray(initial_joints_position, dtype=float).reshape(6)
        base0 = np.asarray(initial_base_pose, dtype=float).reshape(2)
        quat = np.array(target_quat, dtype=float, copy=True).reshape(4)  # wxyz
        if not all(np.isfinite(x).all() for x in (target, q0, base0, quat)):
            raise ValueError("Nonfinite IK target or initial state")
        if np.linalg.norm(quat) < 1e-10:
            raise ValueError("Zero-norm target quaternion")
        quat /= np.linalg.norm(quat)

        self.data.qpos[:] = self.model.qpos0
        for name, value in zip(self.joint_names, np.r_[base0, q0]):
            self.data.qpos[int(self.model.jnt_qposadr[self.model.joint(name).id])] = float(value)
        # Fingers stay at their default for positional IK; the real gripper
        # is controlled separately in the full simulation.
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.data.mocap_pos[self.target_mocap_id] = target
        self.data.mocap_quat[self.target_mocap_id] = quat
        self.end_effector_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "target"))

        freeze = []
        if not optimize_pitch:
            freeze.append(0)
        if not optimize_height:
            freeze.append(1)
        constraints = ([mink.DofFreezingTask(model=self.model, dof_indices=freeze)]
                       if freeze else [])
        success = False
        for _ in range(self.max_iters):
            velocity = mink.solve_ik(
                self.configuration, [self.end_effector_task, self.posture_task],
                dt=self.dt, solver=self.solver, damping=1e-3,
                limits=self.limits, constraints=constraints,
            )
            if not np.isfinite(velocity).all():
                break
            self.configuration.integrate_inplace(velocity, self.dt)
            err = self.end_effector_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= self.pos_threshold:
                success = True
                break

        result = np.array(self.configuration.q, copy=True)
        # This test phase is not rendering IK in a second GLFW window.
        # `visualize` is retained for compatibility with existing callers.
        return result[:2], result[2:8], success
