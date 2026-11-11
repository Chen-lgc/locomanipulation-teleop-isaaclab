# 将 locomanipulation-teleop-isaaclab 的 A2 + Piper L 改为 A2 + Piper X（含双指夹爪）

## 目标和边界

使用 **iit-DLSLab/locomanipulation-teleop-isaaclab** 的 `robot_model/a2_piper_l.xml` 作为母版。保留 A2 机身、腿、惯量、足端碰撞、腿部执行器和传感器；仅替换机械臂及夹爪的几何、关节关系、质量和惯量。**不改 PPO 的 12 维动作空间**；训练时 PiperX 六关节仍由随机轨迹和关节 PD 独立控制，夹爪独立处理。

从 AgileX 官方 `agx_arm_urdf` 读取 `piper_x_description.urdf` 与 `piper_x_with_gripper_description.xacro`，使用官方 STL、质量、惯量、关节轴与限位。保留原项目的关键接口命名：机械臂 `joint1`–`joint6`，两手指 `joint7` / `joint8`，刚体 `link1`–`link8`，夹爪执行器 `gripper`，末端 `attachment_site`。

官方 Xacro 使用一个虚拟 `gripper` 关节 + 两个 mimic 手指。为了兼容原项目，生成的 MJCF *省略虚拟关节*，保留两个手指滑动关节并加 `joint8 = -joint7` 耦合。于是预计 **nq=27，nv=26，nu=19**；两个手指共享一个夹爪指令，关节7范围 0～0.05 m，关节8范围 -0.05～0 m。总开度理论对应 0～0.1 m。机械臂、夹爪质量都会进入整体动力学。

当前生成脚本**只产生可视化的 PiperX/夹爪 mesh（无碰撞）**，与原项目训练时禁用机械臂碰撞的策略一致。能开合，不等于已经能可靠抓取物体。抓取训练前需额外配置手指及物体碰撞、摩擦、接触参数与夹持力限幅。

## 1. 准备项目和官方模型

在**宿主机训练仓库**中操作，不要覆盖此前运行成功的 LocoPilot Docker 子模块：

```bash
# 此路径为示例，替换为你已经完成 A2+PiperL 训练的实际仓库
cd ~/locomanipulation_a2_train

# 确认母版文件存在
ls robot_model/a2_piper_l.xml
ls robot_model/a2_piper_l_scene_flat.xml

# 若还未下载官方模型：
cd ~
git clone https://github.com/agilexrobotics/agx_arm_urdf.git
```

将本包脚本解压到训练项目中，或直接使用本包目录下的绝对路径调用。

## 2. 生成 A2 + PiperX + 夹爪 MJCF

```bash
cd ~/locomanipulation_a2_train

python tools/build_a2_piperx_from_l.py \
  --repo . \
  --agx ~/agx_arm_urdf
```

默认 `mount=(0,0,0.10)` 为沿用母版 A2 + Piper L 的安装高度，**不是 PiperX 实机支架尺寸的确认值**。机身、连接板和机械臂底座的实际尺寸、姿态仍须检查。若实际测量不同，可指定：

```bash
python tools/build_a2_piperx_from_l.py --repo . --agx ~/agx_arm_urdf \
  --mount-x 0 --mount-y 0 --mount-z 0.10 --mount-yaw 0
```

生成的文件：

```text
robot_model/
├── a2_piper_l.xml                 # 保留，未改
├── a2_piper_x.xml                 # 新生成
├── a2_piper_x_scene_flat.xml      # 新生成
├── a2_piper_x_scene_rough.xml     # 新生成（如果母版存在）
└── piper_x/assets/*.stl           # 从官方模型复制
```

若已经生成过，先 `git diff --stat` 检查，再增加 `--overwrite` 重新生成。

## 3. 无图形 MuJoCo 结构验收

在已安装 MuJoCo 的环境（例如工作站相应 Conda 环境）运行：

```bash
cd ~/locomanipulation_a2_train
python tools/validate_a2_piperx.py --model robot_model/a2_piper_x_scene_flat.xml
```

验证内容：

- `nq=27`、`nv=26`、`nu=19`，而不是无夹爪模型的 25/24/18；
- A2 腿部 qpos 索引仍为 `[7..18]`（内部顺序保持母版，控制器若需 FL/FR/RL/RR 仍应按名称映射）；
- PiperX 六关节 qpos 索引 `[19..24]`，双指 `[25,26]`；
- PiperX 的执行器编号 `[12..17]`，夹爪编号 18；
- `joint8 = -joint7` 耦合约束存在，且 `attachment_site` 存在；
- 输出新模型的总质量与整机质心（注意必须比较相同初始姿态）。

若无法编译，直接反馈完整 MuJoCo `ValueError`，不要跳过结构验收。

## 4. 转换为 Isaac Lab 3.0 的 USD

`pixi shell -e train` 是原项目推荐的 Isaac Lab 3.0 训练环境。转换需要 Isaac Lab 的 MJCF importer；不同安装方式可能需要额外的 `isaaclab[importers]`。**先完成第 3 步，再转换 USD**。

```bash
cd ~/locomanipulation_a2_train
pixi shell -e train

python tools/convert_a2_piperx_usd.py --repo . --force
```

目标路径：

```text
robot_model/generated_usd/a2_piper_x/a2_piper_x.usda
```

该脚本使用官方 `MjcfConverterCfg` / `MjcfConverter`，已语法检查，但尚未在你的 Isaac Lab 3.0 环境执行。转换成功之后仍需检查 USD 中的实际刚体层级、关节轴、限位、关节驱动和夹爪约束；尤其是 MJCF equality 的转换结果，**不能假设夹爪 mimic 一定会自动保留并有效工作**。若转换器不支持该 equality，需要在 USD 中另外建立 mimic/驱动关系。

Isaac Lab 3.0 文档：https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/how-to/import_new_asset.html

## 5. 后续训练接入（不直接覆盖已验证的 A2 + Piper L）

### 资产

复制 `source/locomanipulation_teleop_isaaclab/locomanipulation_teleop_isaaclab/assets/a2_piper_l_asset.py` 为 `a2_piper_x_asset.py`，将 `usd_path` 指向新 USD；另外检查 PiperX 的六关节力矩、PD、关节名称和范围。原文件的 `A2_CFG` 仍可沿用变量名，但不应直接认为 Piper L 的驱动参数也适用于 PiperX。

### 环境

复制 `tasks/locomanipulation/a2_piper_env_cfg.py` 为 `a2_piperx_env_cfg.py`（或者定义继承的 A2PiperX 配置），将 `robot` 替换为新资产。保留原来的 `action_space=12`、观测历史、随机机械臂轨迹与 PPO 参数。

原项目 `locomanipulation_env.py` 的 `_apply_action()` 在当前母版中写死 `torch.zeros(self.num_envs,20)`，这是 12 腿 + 6 臂 + 2 指关节。**新模型恰好保留了两手指关节，因此暂时不必修改这个长度**。注意原项目默认把两个指关节的目标设为 0，即默认闭爪；如需随机开合，新增独立的夹爪目标生成器，绝不要把夹爪添加到 PPO 12 维动作中。

原 `ARM_BODY_NAMES = (link1,...,link8)` 也可以保留。但因 USD 导入器可能合并固定连接和改变刚体层级，务必在训练前验证真实的 8 个刚体名称能查到。

注册新环境（建议名称 `LocoManipulation-A2-PiperX-Flat`），先做 128 环境、2 次迭代的测试，再用 4096 环境正式训练。

## 6. MuJoCo 部署注意

新模型的 `nq/nv/nu` 和 `joint1..joint8`/`gripper` 索引保持与原 A2+PiperL 一致，所以 `deploy/play_mujoco.py` 的关节切片仍有机会直接复用；但是需要在独立实验分支中配置新模型路径，检查动力学、关节正方向和原有的手柄操作逻辑。

**IK 不是自动兼容的。** 当前原项目的 `deploy/ik_mink.py` 始终加载 `robot_model/scene_floating.xml`，其中还是 Piper L 的 `piper_l_floating.xml`。如继续调用它，求得的是 Piper L 的关节目标，可能导致 PiperX 末端轨迹错误。必须另建 PiperX 的浮动/降维 IK 模型，再修改 IK 加载路径及碰撞体名称。

## 7. 设计及验证状态

- 已核对官方 PiperX 基础 URDF、带双指夹爪 Xacro 和原项目 A2+PiperL 的命名结构；
- 构建器通过 Python 语法检查和**模拟 XML 离线结构测试**；
- **尚未在你的完整 A2/PiperX 资源上编译 MuJoCo，也未验证 USD、稳定站立、闭爪力和 IK**；
- 原项目 A2 + Piper L 的 checkpoint / 模型 / 运行脚本不会被覆盖。
