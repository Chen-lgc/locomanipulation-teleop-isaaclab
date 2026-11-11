#!/usr/bin/env python3
"""Check A2+PiperX+gripper MJCF model and compatibility with existing controllers.

Run inside your usual MuJoCo Docker/Conda environment. This script does not
simulate a robot or start a window.
"""
from pathlib import Path
import argparse
import numpy as np
import mujoco


def check(path: Path):
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    data = mujoco.MjData(model)
    assert (model.nq, model.nv, model.nu) == (27, 26, 19), (
        f"Expected (nq,nv,nu)=(27,26,19), got {(model.nq,model.nv,model.nu)}"
    )
    leg_names = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR")
                 for j in ("hip", "thigh", "calf")]
    for name in leg_names + [f"joint{i}" for i in range(1, 9)]:
        assert model.joint(name).id >= 0, name
    for name in ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"):
        assert model.body(name).id >= 0, name
    assert model.site("attachment_site").id >= 0
    leg_q = [int(model.jnt_qposadr[model.joint(n).id]) for n in leg_names]
    arm_q = [int(model.jnt_qposadr[model.joint(f"joint{i}").id]) for i in range(1, 7)]
    finger_q = [int(model.jnt_qposadr[model.joint(f"joint{i}").id]) for i in (7, 8)]
    print("Leg qpos IDs:", leg_q)
    print("Arm qpos IDs:", arm_q)
    print("Finger qpos IDs:", finger_q)
    assert sorted(leg_q) == list(range(7, 19)), "A2 leg qpos indices moved"
    assert arm_q == list(range(19, 25)), "PiperX joint indices moved"
    assert finger_q == [25, 26], "PiperX finger indices moved"
    arm_u = [model.actuator(f"joint{i}").id for i in range(1, 7)]
    print("Arm actuator IDs:", arm_u)
    assert arm_u == list(range(12, 18)), "Arm motor positions differ from deploy/play_mujoco.py"
    assert model.actuator("gripper").id == 18, "Gripper actuator should be index 18"
    assert int(model.actuator_trnid[18, 0]) == model.joint("joint7").id
    # Check joint7/8 equality from MuJoCo's compiled constraints (MJCF must define this).
    found_eq = False
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            if (int(model.eq_obj1id[i]) == model.joint("joint8").id
                and int(model.eq_obj2id[i]) == model.joint("joint7").id
                and np.allclose(model.eq_data[i, :2], [0.0, -1.0])):
                found_eq = True
    assert found_eq, "Expected symmetric finger equality: joint8=-joint7"
    masses = float(model.body_mass.sum())
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    else:
        data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    base_id = model.body("base").id
    print("Total compiled body mass (kg):", f"{masses:.6f}")
    print("Combined COM world (m):", np.array(data.subtree_com[base_id]))
    print("Keyframe count:", model.nkey)
    print("Compiled model: PASS")
    print("NOTE: This does NOT validate stable standing, motor gains, grasp contact or IK.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("robot_model/a2_piper_x_scene_flat.xml"))
    args = ap.parse_args()
    check(args.model)
