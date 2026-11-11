#!/usr/bin/env python3
"""Convert the *official project's* robot_model/a2_piper_l.xml into A2+PiperX+gripper MJCF.

Inputs: an existing checkout of iit-DLSLab/locomanipulation-teleop-isaaclab and
an existing checkout of agilexrobotics/agx_arm_urdf.

Preserves all A2/feet configuration, copies official PiperX geometry/inertia and
retains legacy link1..link8 / joint1..joint8 names and 19 actuator channels.
No network requests are made by this program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

LEG = ("FL", "FR", "RL", "RR")
ARM = tuple(f"joint{i}" for i in range(1, 9))
LINK_NAMES = {
    "base_link": "piperx_mount",
    **{f"link{i}": f"link{i}" for i in range(1, 7)},
    "flange_link": "piperx_flange",
    "gripper_base": "piperx_gripper_base",
    "gripper_link1": "link7",
    "gripper_link2": "link8",
}
JOINT_NAMES = {"gripper_joint1": "joint7", "gripper_joint2": "joint8"}
MESH_MAP = {
    "base_link": ("piper_x/meshes/base_link.stl", "px_base_link"),
    **{f"link{i}": (f"piper_x/meshes/link{i}.stl", f"px_link{i}") for i in range(1, 7)},
    "flange_link": ("piper_x/meshes/flange.stl", "px_flange"),
    "gripper_base": ("agx_gripper/meshes/gripper_base.stl", "px_gripper_base"),
    "gripper_link1": ("agx_gripper/meshes/gripper_link1.stl", "px_gripper_link1"),
    "gripper_link2": ("agx_gripper/meshes/gripper_link2.stl", "px_gripper_link2"),
}
JOINT_GRAPH = (
    ("base_link", "link1", "joint1"),
    ("link1", "link2", "joint2"),
    ("link2", "link3", "joint3"),
    ("link3", "link4", "joint4"),
    ("link4", "link5", "joint5"),
    ("link5", "link6", "joint6"),
    ("link6", "flange_link", "flange_joint"),
    ("flange_link", "gripper_base", "gripper_base_joint"),
    ("gripper_base", "gripper_link1", "gripper_joint1"),
    ("gripper_base", "gripper_link2", "gripper_joint2"),
)


def fmt(v):
    return " ".join(f"{float(x):.13g}" for x in v)


def vec(s: str, expected: int = 3):
    v = np.array([float(x) for x in s.split()], dtype=np.float64)
    if len(v) != expected:
        raise ValueError(f"Expected {expected} elements in {s!r}")
    return v


def origin_xyz_rpy(e):
    if e is None:
        return np.zeros(3), np.zeros(3)
    return vec(e.get("xyz", "0 0 0")), vec(e.get("rpy", "0 0 0"))


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_quat(R):
    """Convert a proper 3x3 rotation matrix to normalized quaternion wxyz."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array([(R[2, 1] - R[1, 2]) / s, s / 4,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s, s / 4,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, s / 4])
    return q / np.linalg.norm(q)


def inertial_from_urdf(link, body):
    e = link.find("inertial")
    if e is None:
        raise ValueError(f"Official link {link.get('name')} has no inertial element")
    pos, rpy = origin_xyz_rpy(e.find("origin"))
    mtag, itag = e.find("mass"), e.find("inertia")
    if mtag is None or itag is None:
        raise ValueError(f"Incomplete inertia for {link.get('name')}")
    mass = float(mtag.get("value"))
    if not mass > 0:
        raise ValueError(f"Nonpositive mass {mass} for {link.get('name')}")
    Ixx, Iyy, Izz = (float(itag.get(a)) for a in ("ixx", "iyy", "izz"))
    Ixy, Ixz, Iyz = (float(itag.get(a)) for a in ("ixy", "ixz", "iyz"))
    I_inertial = np.array([[Ixx, Ixy, Ixz],
                           [Ixy, Iyy, Iyz],
                           [Ixz, Iyz, Izz]], dtype=float)
    rot = rpy_matrix(rpy)
    I_body = rot @ I_inertial @ rot.T
    principal, basis = np.linalg.eigh(I_body)
    if np.any(principal <= 0):
        raise ValueError(f"Invalid inertia {principal} for {link.get('name')}")
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    quat = matrix_to_quat(basis)
    ET.SubElement(body, "inertial", {
        "pos": fmt(pos), "quat": fmt(quat),
        "mass": f"{mass:.14g}", "diaginertia": fmt(principal),
    })
    return mass


def joint_origin(joint):
    return origin_xyz_rpy(joint.find("origin"))


def build_joint_into_body(joint, body):
    name = JOINT_NAMES.get(joint.get("name"), joint.get("name"))
    kind = joint.get("type")
    if kind == "fixed":
        return
    if kind not in ("revolute", "prismatic"):
        raise ValueError(f"Unsupported joint {name}: {kind}")
    lim = joint.find("limit")
    if lim is None:
        raise ValueError(f"Missing limit for {name}")
    axis = joint.find("axis")
    attrs = {
        "name": name, "type": "hinge" if kind == "revolute" else "slide",
        "axis": (axis.get("xyz") if axis is not None else "0 0 1"),
        "range": f"{lim.get('lower')} {lim.get('upper')}",
        "limited": "true", "damping": "0.1" if kind == "revolute" else "0.02",
        "armature": "0.01" if kind == "revolute" else "0.001",
    }
    ET.SubElement(body, "joint", attrs)


def read_urdf(agx_root):
    urdf = ET.parse(agx_root / "piper_x/urdf/piper_x_description.urdf").getroot()
    gripper = ET.parse(agx_root / "piper_x/urdf/piper_x_with_gripper_description.xacro").getroot()
    links = {e.get("name"): e for e in list(urdf.findall("link")) + list(gripper.findall("link"))}
    joints = {e.get("name"): e for e in list(urdf.findall("joint")) + list(gripper.findall("joint"))}
    expected_links = set(LINK_NAMES)
    expected_joints = {j for _, _, j in JOINT_GRAPH}
    if not expected_links <= links.keys() or not expected_joints <= joints.keys():
        raise ValueError(f"Official PiperX model missing links {expected_links - links.keys()} "
                         f"or joints {expected_joints - joints.keys()}")
    for parent, child, joint_name in JOINT_GRAPH:
        joint = joints[joint_name]
        p = joint.find("parent")
        c = joint.find("child")
        if p is None or c is None or p.get("link") != parent or c.get("link") != child:
            raise ValueError(f"Mismatch in official PiperX joint chain at {joint_name}")
    # The Xacro uses a virtual aperture joint + two mimic joints. We deliberately
    # represent the 2 fingers directly and keep joint8=-joint7 as in Piper L.
    for which, mult in (("gripper_joint1", 0.5), ("gripper_joint2", -0.5)):
        mimic = joints[which].find("mimic")
        if mimic is None or mimic.get("joint") != "gripper" or not np.isclose(float(mimic.get("multiplier")), mult):
            raise ValueError(f"Unexpected {which} mimic relation; model conversion needs review")
    return links, joints


def convert(project, agx_root, mount_xyz, mount_rpy, overwrite=False):
    original_path = project / "robot_model" / "a2_piper_l.xml"
    out_path = project / "robot_model" / "a2_piper_x.xml"
    if not original_path.is_file():
        raise FileNotFoundError(f"Original A2 + PiperL model missing: {original_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}; use --overwrite to replace generated file")
    for suffix in ("scene_flat", "scene_rough"):
        src = project / "robot_model" / f"a2_piper_l_{suffix}.xml"
        dst = project / "robot_model" / f"a2_piper_x_{suffix}.xml"
        if src.exists() and dst.exists() and not overwrite:
            raise FileExistsError(f"Scene output already exists: {dst}; use --overwrite")
    links, joints = read_urdf(agx_root)
    for rel, _ in MESH_MAP.values():
        if not (agx_root / rel).is_file():
            raise FileNotFoundError(f"PiperX mesh missing: {agx_root / rel}")

    a2 = ET.parse(original_path)
    root = a2.getroot()
    if root.tag != "mujoco":
        raise ValueError("Expected source MJCF <mujoco>")
    root.set("model", "a2_piper_x")
    worldbody = root.find("worldbody")
    base = worldbody.find("./body[@name='base']") if worldbody is not None else None
    assets, actuators, sensors, equality = (
        root.find("asset"), root.find("actuator"), root.find("sensor"), root.find("equality"))
    if any(e is None for e in (base, assets, actuators, sensors)):
        raise ValueError("Source A2+PiperL must have base, asset, actuator and sensor")
    old_arm = base.find("./body[@name='link1']")
    if old_arm is None:
        raise ValueError("Cannot find original PiperL link1 under A2 body='base'")

    # Keep A2 main body, A2 legs, all motor/foot sensors, collider and materials.
    base.remove(old_arm)
    for geom in list(base.findall("geom")):
        if geom.get("mesh") == "piper_base_link":
            base.remove(geom)
    for mesh in list(assets.findall("mesh")):
        if "piper_l/" in mesh.get("file", ""):
            assets.remove(mesh)
    for actuator in list(actuators):
        if actuator.get("joint", "") in ARM or actuator.get("name") == "gripper":
            actuators.remove(actuator)
    for sensor in list(sensors):
        if sensor.get("joint", "") in ARM:
            sensors.remove(sensor)
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for eq in list(equality):
        if eq.get("joint1", "") in ("joint7", "joint8") or eq.get("joint2", "") in ("joint7", "joint8"):
            equality.remove(eq)

    asset_dir = project / "robot_model" / "piper_x" / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for original_name, (rel, mesh_name) in MESH_MAP.items():
        src = agx_root / rel
        dst = asset_dir / Path(rel).name
        # gripper_base.stl / link*.stl are distinct; names are verified below.
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Asset filename collision: {dst}; use --overwrite")
        shutil.copy2(src, dst)
        ET.SubElement(assets, "mesh", {
            "name": mesh_name,
            "file": f"piper_x/assets/{dst.name}",
        })

    mount = ET.SubElement(base, "body", {
        "name": "piperx_mount", "pos": fmt(mount_xyz),
        "quat": fmt(matrix_to_quat(rpy_matrix(mount_rpy))),
    })
    total_mass = inertial_from_urdf(links["base_link"], mount)

    ET.SubElement(mount, "geom", {
        "name": "px_visual_base_link", "type": "mesh", "mesh": MESH_MAP["base_link"][1],
        "contype": "0", "conaffinity": "0", "group": "2", "density": "0",
        "rgba": "0.85 0.87 0.95 1",
    })
    bodies = {"base_link": mount}
    for parent, child, joint_name in JOINT_GRAPH:
        joint = joints[joint_name]
        pos, rpy = joint_origin(joint)
        b = ET.SubElement(bodies[parent], "body", {
            "name": LINK_NAMES[child], "pos": fmt(pos),
            "quat": fmt(matrix_to_quat(rpy_matrix(rpy))),
        })
        total_mass += inertial_from_urdf(links[child], b)
        build_joint_into_body(joint, b)
        ET.SubElement(b, "geom", {
            "name": f"px_visual_{child}", "type": "mesh", "mesh": MESH_MAP[child][1],
            "contype": "0", "conaffinity": "0", "group": "2", "density": "0",
            "rgba": "0.75 0.82 0.92 1",
        })
        if child == "gripper_base":
            ET.SubElement(b, "site", {
                "name": "attachment_site", "pos": "0 0 0.138", "size": "0.01",
                "rgba": "1 0 0 1", "group": "1",
            })
        bodies[child] = b

    ET.SubElement(equality, "joint", {
        "joint1": "joint8", "joint2": "joint7",
        "polycoef": "0 -1 0 0 0",
    })
    # Retain 19 actuators and the same order: legs 12, arm 6, gripper 1.
    for n in range(1, 7):
        effort = 32 if n <= 3 else 8
        ET.SubElement(actuators, "motor", {
            "name": f"joint{n}", "joint": f"joint{n}",
            "ctrllimited": "true", "ctrlrange": f"{-effort} {effort}",
        })
    ET.SubElement(actuators, "motor", {
        "name": "gripper", "joint": "joint7",
        "ctrllimited": "true", "ctrlrange": "-10 10",
    })
    for n in range(1, 7):
        ET.SubElement(sensors, "jointpos", {"name": f"joint{n}_pos", "joint": f"joint{n}"})
    ET.SubElement(sensors, "jointpos", {"name": "gripper_pos", "joint": "joint7"})

    # The 27 qpos entries from A2+PiperL remain valid: 7 root + 12 legs +
    # 6 arm + 2 fingers. No index shifts are introduced by the conversion.
    keys = root.find("keyframe")
    if keys is not None:
        for key in keys.findall("key"):
            q = key.get("qpos")
            if q is not None and len(q.split()) != 27:
                raise ValueError(f"Source keyframe {key.get('name')} has {len(q.split())} entries, not 27")
    ET.indent(a2, space="  ")
    a2.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote: {out_path}")
    print(f"PiperX (incl. fingers & base) URDF mass: {total_mass:.6f} kg")
    print(f"Mount in A2 base coordinates: pos {fmt(mount_xyz)}, rpy {fmt(mount_rpy)}")

    for suffix in ("scene_flat", "scene_rough"):
        src = project / "robot_model" / f"a2_piper_l_{suffix}.xml"
        dst = project / "robot_model" / f"a2_piper_x_{suffix}.xml"
        if not src.is_file():
            continue
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Scene output exists: {dst}")
        # Only change include reference. Preserve all terrain and rendering content.
        scene = ET.parse(src)
        includes = scene.getroot().findall("include")
        replaced = 0
        for include in includes:
            if include.get("file") == "a2_piper_l.xml":
                include.set("file", "a2_piper_x.xml")
                replaced += 1
        if replaced != 1:
            raise ValueError(f"Expected exactly one A2+PiperL include in {src}; got {replaced}")
        ET.indent(scene, space="  ")
        scene.write(dst, encoding="utf-8", xml_declaration=True)
        print(f"Wrote: {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."), help="Existing locomanipulation training repo")
    ap.add_argument("--agx", type=Path, default=Path("~/agx_arm_urdf"), help="agilexrobotics/agx_arm_urdf checkout")
    ap.add_argument("--mount-x", type=float, default=0.0)
    ap.add_argument("--mount-y", type=float, default=0.0)
    ap.add_argument("--mount-z", type=float, default=0.10,
                    help="Official PiperL mounting offset, in A2 base frame, metres")
    ap.add_argument("--mount-roll", type=float, default=0.0)
    ap.add_argument("--mount-pitch", type=float, default=0.0)
    ap.add_argument("--mount-yaw", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing generated PiperX output")
    a = ap.parse_args()
    try:
        convert(a.repo.expanduser().resolve(), a.agx.expanduser().resolve(),
                (a.mount_x, a.mount_y, a.mount_z),
                (a.mount_roll, a.mount_pitch, a.mount_yaw), a.overwrite)
    except (FileNotFoundError, ValueError, FileExistsError, ET.ParseError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
