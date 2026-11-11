#!/usr/bin/env python3
"""Offline structural test for generator using fixtures (NOT a MuJoCo compile test)."""
import importlib.util
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('build_a2_piperx_from_l.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def write_mock_urdf(path, gripper=False):
    root=ET.Element('robot',{'name':'piper_x'})
    names = ['flange_link','gripper_base','gripper_link1','gripper_link2'] if gripper else ['base_link'] + [f'link{i}' for i in range(1,7)]
    for name in names:
        li=ET.SubElement(root,'link',{'name':name})
        inert=ET.SubElement(li,'inertial')
        ET.SubElement(inert,'origin',{'xyz':'0 0 0.01','rpy':'0 0 0'})
        ET.SubElement(inert,'mass',{'value':'0.5'})
        ET.SubElement(inert,'inertia',{'ixx':'.002','iyy':'.003','izz':'.004','ixy':'0','ixz':'0','iyz':'0'})
    joints=mod.JOINT_GRAPH[6:] if gripper else mod.JOINT_GRAPH[:6]
    for p,c,n in joints:
        ty='fixed' if n in ('flange_joint','gripper_base_joint') else 'prismatic' if 'gripper' in n else 'revolute'
        j=ET.SubElement(root,'joint',{'name':n,'type':ty})
        ET.SubElement(j,'parent',{'link':p});ET.SubElement(j,'child',{'link':c})
        ET.SubElement(j,'origin',{'xyz':'0 0 0.1','rpy':'0 0 0'})
        ET.SubElement(j,'axis',{'xyz':'0 0 1'})
        if ty!='fixed':
            ET.SubElement(j,'limit',{'lower':'-0.05' if n=='gripper_joint2' else '0' if ty=='prismatic' else '-3',
                                     'upper':'0' if n=='gripper_joint2' else '0.05' if ty=='prismatic' else '3'})
        if 'gripper' in n and n not in ('gripper_base_joint',):
            ET.SubElement(j,'mimic',{'joint':'gripper','multiplier':'0.5' if n=='gripper_joint1' else '-0.5'})
    ET.indent(root)
    path.write_text(ET.tostring(root,encoding='unicode'))

with tempfile.TemporaryDirectory() as tmp:
    parent=Path(tmp);project=parent/'test'; agx=parent/'agx'
    xml_dir=project/'robot_model';xml_dir.mkdir(parents=True)
    arm_urdf_dir=agx/'piper_x/urdf';arm_urdf_dir.mkdir(parents=True)
    write_mock_urdf(arm_urdf_dir/'piper_x_description.urdf')
    write_mock_urdf(arm_urdf_dir/'piper_x_with_gripper_description.xacro',True)
    for rel,name in mod.MESH_MAP.values():
        p=agx/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fake STL data')
    root=ET.Element('mujoco',{'model':'a2_piper_l'})
    assets=ET.SubElement(root,'asset')
    ET.SubElement(assets,'mesh',{'name':'piper_base_link','file':'piper_l/assets/base_link.stl'})
    ET.SubElement(assets,'mesh',{'name':'base_link','file':'a2/meshes/base_link.stl'})
    for n in range(1,9):
        ET.SubElement(assets,'mesh',{'name':f'link{n}','file':f'piper_l/assets/link{n}.stl'})
    world=ET.SubElement(root,'worldbody');base=ET.SubElement(world,'body',{'name':'base'})
    ET.SubElement(base,'joint',{'name':'floating_base_joint','type':'free'})
    ET.SubElement(base,'geom',{'mesh':'piper_base_link'})
    for leg in mod.LEG:
        b=ET.SubElement(base,'body',{'name':leg+'_hip'})
        for suffix in ('hip','thigh','calf'):
            if suffix!='hip':b=ET.SubElement(b,'body',{'name':leg+'_'+suffix})
            ET.SubElement(b,'joint',{'name':f'{leg}_{suffix}_joint'})
    arm=ET.SubElement(base,'body',{'name':'link1'})
    for i in range(1,9):
        ET.SubElement(arm,'joint',{'name':f'joint{i}'})
    eq=ET.SubElement(root,'equality');ET.SubElement(eq,'joint',{'joint1':'joint8','joint2':'joint7'})
    actuator=ET.SubElement(root,'actuator')
    for leg in mod.LEG:
        for suffix in ('hip','thigh','calf'):
            ET.SubElement(actuator,'motor',{'name':f'{leg}_{suffix}', 'joint':f'{leg}_{suffix}_joint'})
    for i in range(1,7):ET.SubElement(actuator,'motor',{'name':f'joint{i}','joint':f'joint{i}'})
    ET.SubElement(actuator,'motor',{'name':'gripper','joint':'joint7'})
    sensor=ET.SubElement(root,'sensor')
    for i in range(1,7):ET.SubElement(sensor,'jointpos',{'name':f'joint{i}_pos','joint':f'joint{i}'})
    ET.SubElement(sensor,'jointpos',{'name':'gripper_pos','joint':'joint7'})
    k=ET.SubElement(root,'keyframe');ET.SubElement(k,'key',{'name':'home','qpos':' '.join(['0']*27)})
    ET.indent(root)
    (xml_dir/'a2_piper_l.xml').write_text(ET.tostring(root,encoding='unicode'))
    (xml_dir/'a2_piper_l_scene_flat.xml').write_text('<mujoco><include file="a2_piper_l.xml"/></mujoco>')
    mod.convert(project,agx,(0,0,.10),(0,0,0))
    px=ET.parse(xml_dir/'a2_piper_x.xml').getroot()
    assert [e.get('name') for e in px.find('actuator')] == [f'{leg}_{joint}' for leg in mod.LEG for joint in ('hip','thigh','calf')] + [f'joint{i}' for i in range(1,7)] + ['gripper']
    assert [x.get('name') for x in px.findall('./worldbody//joint') if x.get('name','').startswith('joint')] == [f'joint{i}' for i in range(1,9)]
    assert px.find("./worldbody/body/body[@name='link1']") is None
    assert px.find("./worldbody/body/body[@name='piperx_mount']/body[@name='link1']") is not None
    assert px.find('.//site[@name="attachment_site"]') is not None
    assert px.find('equality/joint').get('polycoef') == '0 -1 0 0 0'
    assert px.find('asset/mesh[@name="px_gripper_link1"]') is not None
    scene=ET.parse(xml_dir/'a2_piper_x_scene_flat.xml').getroot()
    assert scene.find('include').get('file')=='a2_piper_x.xml'
    print('OFFLINE STRUCTURAL SELF-TEST: PASS')
