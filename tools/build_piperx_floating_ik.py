#!/usr/bin/env python3
"""Generate PiperX reduced IK MJCF *from the working A2+PiperX MJCF*.

The generated model retains the arm subtree byte-for-byte (as XML subtrees),
including its actual mounting transform, attachment_site and finger joints.
Its robot base lives at (0,0,0) with 2 virtual joints (pitch and z), matching
LocoManipulation's original reduced IK interface.

Use this on the user's working copy, via a terminal; create/move files in VS Code.
No training, Isaac Sim, URDF or network access is required.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "robot_model" / "a2_piper_x.xml"
DEFAULT_DEST = ROOT / "robot_model" / "piper_x_floating.xml"


def build(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(f"找不到组合模型: {source}")
    original = ET.parse(source).getroot()
    if original.tag != "mujoco":
        raise ValueError("输入必须是 MJCF 根元素 <mujoco>")

    root_body = original.find("./worldbody/body[@name='base']")
    if root_body is None:
        raise ValueError("原模型中缺少 worldbody 下 name='base' 的 A2 机身")
    arm_mount = root_body.find("./body[@name='piperx_mount']")
    if arm_mount is None:
        raise ValueError("缺少 piperx_mount。请确认使用先前由 Piper L 母版生成的 a2_piper_x.xml")

    if arm_mount.find(".//site[@name='attachment_site']") is None:
        raise ValueError("PiperX 子树没有 attachment_site，无法定义 IK 末端")

    joint_names = {j.get("name") for j in arm_mount.findall(".//joint")}
    expected = {f"joint{i}" for i in range(1, 9)}
    missing = sorted(expected - joint_names)
    if missing:
        raise ValueError(f"PiperX 机械臂/夹爪缺少关节: {missing}")

    reduced = ET.Element("mujoco", {"model": "piper_x_floating"})
    for tag in ("compiler", "option", "size", "default", "asset"):
        el = original.find(tag)
        if el is not None:
            reduced.append(deepcopy(el))

    world = ET.SubElement(reduced, "worldbody")
    base = ET.SubElement(world, "body", {"name": "base", "pos": "0 0 0"})
    original_inertial = root_body.find("inertial")
    if original_inertial is None:
        raise ValueError("原 A2 base 没有显式 inertial")
    base.append(deepcopy(original_inertial))
    ET.SubElement(base, "joint", {
        "name": "basepitch", "type": "hinge", "axis": "0 1 0",
        "pos": "0 0 0", "range": "-0.2 0.2", "limited": "true",
    })
    ET.SubElement(base, "joint", {
        "name": "basez", "type": "slide", "axis": "0 0 1",
        "pos": "0 0 0", "range": "-0.1 0.1", "limited": "true",
    })

    # Keep base colliders and arm geometry identical to the complete MJCF.
    for child in root_body:
        if child.tag == "geom":
            base.append(deepcopy(child))
    base.append(deepcopy(arm_mount))

    # Preserve only the finger coupling equality; other A2 constraints
    # (if present) may reference omitted legs and cannot be used here.
    original_eq = original.find("equality")
    if original_eq is not None:
        selected = []
        for eq in original_eq:
            if eq.tag == "joint" and {eq.get("joint1"), eq.get("joint2")} == {"joint7", "joint8"}:
                selected.append(deepcopy(eq))
        if selected:
            eq_root = ET.SubElement(reduced, "equality")
            eq_root.extend(selected)

    mocap = ET.SubElement(world, "body", {
        "name": "target", "mocap": "true", "pos": "0.3 0 0.3",
        "quat": "1 0 0 0",
    })
    ET.SubElement(mocap, "site", {"name": "ik_target_site", "size": "0.01", "rgba": "0 0 1 1"})

    # Deliberately omit all actuators and keyframes: this is a pure IK model,
    # *not* the physics simulation. With 2 base + 6 arm + 2 fingers: nq=nv=10.
    ET.indent(reduced, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(reduced).write(destination, encoding="utf-8", xml_declaration=True)
    print(f"PiperX IK MJCF generated: {destination}")
    print("Expected order: basepitch, basez, joint1 ... joint8")
    print("Next: run tools/validate_piperx_floating_ik.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    build(args.source.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
