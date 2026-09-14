# 环境搭建指导书：LLM 零样本控制机械臂仿真平台

> 目标：从零搭出一个"LLM 仅凭相机画面控制 YAM 机械臂抓方块入碗"的仿真评测环境。
> 复现对象：RoboCurve《GPT-6 Astra on robotic manipulation》（2026-09-04）。
> 适用系统：macOS（Apple Silicon 已验证）/ Linux；无需 GPU、无需真机。
> 预计耗时：首次搭建 1~2 小时（不含调试）。

---

## 0. 总体架构

```
┌─────────────┐   相机渲染图 + 状态坐标    ┌──────────────────┐
│  MuJoCo 仿真 │  ─────────────────────►  │  LLM (agent 策略) │
│  YAM 臂+方块 │  ◄─────────────────────  │  GPT-6 Astra / K3 │
│  +碗+相机    │   move_to(x,y,z,gripper) │  零样本、无微调    │
└─────────────┘                          └──────────────────┘
        ▲                                         │
        └──── inspect-robots 评测框架（任务/计分/日志）─┘
```

关键设计：LLM **看不到关节、不知道 IK**，每步只看 2~3 路相机图和末端坐标，
输出绝对末端目标——这是"零样本控制"的复刻，也是官方实验的核心设定。

## 1. 依赖安装

### 1.1 基础环境

```bash
# 需要 Python 3.12 + uv（https://docs.astral.sh/uv/）
uv venv .venv --python 3.12
source .venv/bin/activate
```

### 1.2 安装评测框架与仿真器

```bash
uv pip install "inspect-robots[rerun]"   # 核心评测框架（MIT，RoboCurve 开源）
uv pip install inspect-robots-agent      # LLM agent 策略插件（工具调用协议）
uv pip install "inspect-robots-yam[collision]"  # YAM 插件（附带碰撞模型，供参考）
uv pip install kitchenbench              # 官方任务集（可选，本任务不需要）
uv pip install mujoco pillow numpy       # MuJoCo 3.13+ 与图像处理
```

### 1.3 自检

```bash
inspect-robots list   # 应列出 cubepick embodiment、agent/scripted 策略等
# 零成本冒烟：内置 2D 玩具世界 + 脚本策略
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick --no-rerun
# 期望：run status: completed，4 succeeded
```

## 2. 获取 YAM 机械臂官方模型

Inspect Robots 官方用的 I2RT YAM 臂，其 MuJoCo 模型在 MuJoCo Menagerie（MIT）：

```bash
mkdir -p sim/assets && cd sim/assets
curl -sL -o m.zip "https://github.com/google-deepmind/mujoco_menagerie/archive/71f066ad0be9cd271f7ed58c030243ef157af9f4.zip"
unzip -q -o m.zip "mujoco_menagerie-*/i2rt_yam/*"
mv mujoco_menagerie-*/i2rt_yam . && rm -rf mujoco_menagerie-* m.zip
# 得到 sim/assets/i2rt_yam/{yam.xml, scene.xml, assets/*.stl, LICENSE}
```

- `yam.xml` 含完整运动学、惯性、**位置执行器**（6 关节 + 夹爪）和 home 关键帧
- 夹爪左右指通过 equality 联动，只需驱动 `left_finger`

## 3. 搭建任务场景（block_bowl_scene.xml）

在 `sim/assets/i2rt_yam/` 内新建 `block_bowl_scene.xml`（必须与 yam.xml 同目录，
否则 mesh 相对路径会解析失败）：

```xml
<mujoco model="yam block-into-bowl">
  <include file="yam.xml"/>
  <option timestep="0.002" iterations="50" tolerance="1e-8"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <texture type="2d" name="tabletex" builtin="checker" rgb1="0.45 0.33 0.22" rgb2="0.38 0.27 0.17" width="300" height="300"/>
    <material name="tablemat" texture="tabletex" texuniform="true" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="table" type="plane" size="0 0 0.05" material="tablemat" pos="0 0 -0.001"/>

    <!-- 红色方块：5cm 立方体，自由体；每回合按种子随机摆放 -->
    <body name="block" pos="0.28 0.10 0.026">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="0.025 0.025 0.025" mass="0.08" condim="4"
            rgba="0.85 0.1 0.1 1" friction="1.2 0.05 0.01"/>
    </body>

    <!-- 碗：底面圆盘 + 10 段盒体围墙（内径约 11cm，碗沿高 5cm），静态 -->
    <body name="bowl" pos="0.30 -0.14 0">
      <geom name="bowl_bottom" type="cylinder" size="0.068 0.004" pos="0 0 0.004" rgba="0.92 0.92 0.92 1" mass="0"/>
      <!-- 10 段围墙：半径 0.061，每段 euler z 旋转 i*36°，见完整文件 -->
      ...
    </body>

    <!-- 相机：顶视 + 斜侧视 + 腕部（位姿每帧由代码覆写跟踪夹爪） -->
    <camera name="top" pos="0.28 0 0.95" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="side" pos="0.28 -0.85 0.55" xyaxes="1 0 0 0 0.55 0.835" fovy="45"/>
    <camera name="wrist" pos="0 0 0.5" fovy="70"/>
  </worldbody>

  <!-- 焊接抓取约束（运行时按需激活，见 §5.4） -->
  <equality>
    <weld name="grasp_weld" body1="link_6" body2="block" active="false"/>
  </equality>
</mujoco>
```

碗围墙 10 段坐标速算：第 i 段位置 `(0.061cosθ, 0.061sinθ, 0.025)`，`euler="0 0 θ+π/2"`，
`θ = i·36° (i=0..9)`，尺寸 `0.021 0.007 0.025`。

**验证**：用 Python `mujoco.MjModel.from_xml_path()` 加载并用 `mujoco.Renderer`
渲染 top/side 两路，肉眼确认臂、方块、碗都在画面中。

## 4. 编写仿真 Embodiment（yam_sim.py）

实现 Inspect Robots 的 Embodiment 协议（`reset/step/close` + `info`）。
本项目的完整实现见 `sim/yam_sim.py`（约 300 行，可直接使用/参照），要点：

### 4.1 动作空间（给 LLM 的接口）

```python
Box(shape=(4,), low=[0.15,-0.25,0.005,0], high=[0.45,0.25,0.35,1],
    semantics=ActionSemantics(
        control_mode="eef_abs_pose",     # 绝对末端位姿
        frame="world", rotation_repr="none",
        dim_labels=("x","y","z","gripper")))
```

- 观测必须暴露**恰好一个**同形状状态字段 `eef_state`（agent 插件靠它定位反馈）
- 相机：`top`、`side`、`wrist` 三路 448×448

### 4.2 控制管线（每个 10Hz 控制步）

```
LLM 目标 (x,y,z,gripper)
  → 加上 _pinch_offset（爪尖垫片参考点补偿，启动时实测）
  → DLS 逆运动学（grasp_site 位置 + link_6 固定朝下姿态，6×6 雅可比）
  → 关节限位 + 每步 0.2rad 速率限制
  → 位置伺服驱动 mj_step（0.1s = 50 物理步）
  → 收敛补偿：残差 >3mm 时短脉冲修正
  → 焊接抓取状态机 + 成功判定
```

### 4.3 三个必踩的坑（先写进代码，省两天调试）

1. **IK 必须保存/恢复物理状态**。IK 迭代会改写 `data.qpos`，不恢复的话
   机械臂"瞬移"、打飞方块、一切结果作废：
   ```python
   saved = self.data.qpos.copy()
   ... # IK 迭代
   self.data.qpos[:] = saved; mujoco.mj_forward(self.model, self.data)
   ```
2. **MuJoCo position 执行器的 kp 是成对的**：`force = gainprm[0]·ctrl + biasprm[1]·qpos + biasprm[2]·qvel`，
   官方模型 `gainprm[0]=+kp`、`biasprm[1]=-kp`。加强伺服（原版 kp=10~40 太软，
   重力下垂 2~5cm）必须**两个同倍率改**，只改一个会得到错误的力甚至炸机。
3. **末端参考点不是 grasp_site**：YAM 手指是 L 形，真实捏取点在爪尖垫片，
   与 grasp_site 有固定偏移（实测约 (0.020, 0, -0.018) m）。启动时在就绪位形
   下测量两侧垫片中点作为 `_pinch_offset`，指令坐标全部按"捏取点"语义换算。

### 4.4 抓取保持：焊接抽象（必须的妥协）

MuJoCo 刚性点接触**无法**复现真实橡胶垫摩擦——实测纯物理接触下 4~5cm 方块
要么被挤出、要么缓慢滑脱，参数扫遍（夹持力/摩擦/接触刚度）也只有 ~60% 可靠。
官方是真机，没有这个问题。解决方案（业界常用，如 ManiSkill 的 sticky grasp）：

- 夹爪闭合（目标 <0.4）且方块在钳口窗口内（水平 <3.3cm、竖直 -0.01~0.04）：
  - 偏移 >1.2cm：给方块施加指向钳口中心的归中速度（模拟垫片把方块推正）
  - 居中后：激活 `grasp_weld`（weld 约束），把当前腕→块相对位姿写入 `eq_data`
- 夹爪张开（目标 >0.6）：`eq_active=0` 释放

这样保留了 LLM 的全部难点（视觉定位、逼近、闭合时机、搬运、释放时机），
只抽象掉接触摩擦。**报告中必须声明这一近似。**

### 4.5 腕部相机

YAM 官方真机有 2 路腕部相机（D405），对最后 5cm 对中至关重要。仿真里
Menagerie 模型不带相机，自行添加：场景里放一路世界刚体相机占位，
每帧渲染前用代码把相机世界位姿设为"捏取点 + 固定偏移、看向捏取点"
（lookat 计算四元数），实现跟踪夹爪的效果。

### 4.6 成功判定（仿真特权）

```python
inside = 水平距碗心 < 0.052 且 方块 z < 0.045 且 方块速度 < 0.05
连续 5 个控制步满足 → terminated=True, info["success"]=True
```

## 5. 验证（必须全部通过才能跑 LLM）

按顺序执行，任何一步失败都先修再继续：

```bash
cd sim && source ../.venv/bin/activate

# ① 场景编译渲染：肉眼检查 test_top.png / test_side.png
# ② 保位稳定：复位后保持 20 步，末端漂移应 < 5mm
# ③ 硬编码抓取（smoke_test.py）：路径点驱动完成 抓→举→运→放，success=True
python smoke_test.py
# ④ 全知对照（oracle_validate.py）：20 个种子全部成功
python oracle_validate.py    # 期望：Oracle policy: 20/20 = 100%
```

**对照实验是有效性的根基**：全知策略 20/20 意味着"仿真可解且不捣乱"，
之后 LLM 的任何失败都可归因于模型决策而非环境缺陷。每次改代码后重跑 ④。

## 6. 接入 LLM 跑评测

### 6.1 评测入口（run_eval.py 已封装）

```bash
# K3（Kimi 网关，环境变量 KIMI_BASE_URL/KIMI_API_KEY 已有则零配置）
nohup ../.venv/bin/python -u run_eval.py --scenes 1 --seed-start 0 \
  --model k3-agent --effort high --max-steps 2500 --log-dir logs_k3 > k3_0.out 2>&1 &

# GPT-6 Astra（需 OPENAI_API_KEY，20 次约 $19）
nohup ../.venv/bin/python -u run_eval.py --scenes 1 --seed-start 0 \
  --model gpt-6-astra --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY --effort high \
  --max-steps 2500 --log-dir logs_astra > astra_0.out 2>&1 &
```

### 6.2 关键参数约定

| 参数 | 取值 | 说明 |
|---|---|---|
| 任务指令 | "Pick up the red block from the table and place it inside the bowl." | 与官方原文一致 |
| 种子 | 0..19，每 trial 一个 | 方块位置由种子决定，模型间可复现对比 |
| max_steps | 2500（10Hz ≈ 250s 仿真时） | 对齐官方 2.5~8 分钟/次的量级 |
| max_llm_calls | 60 | 防死循环；约 10/20 的弱模型 trial 会耗尽 |
| scorer | success_at_end | 仿真特权判定 |
| 并行 | 5 进程 × 4 波 | 单 trial 10~25 分钟；务必 nohup 后台 |

### 6.3 结果统计

```bash
grep -hoE "success_at_end': [01]" logs_*/../<prefix>_*.out | sort | uniq -c
```

## 7. 已知局限（写报告时照抄）

1. 仿真物理 ≠ 真机：确定性、无传感噪声、无摩擦真实感
2. 抓取保持为焊接抽象（§4.4）；官方为真实橡胶垫摩擦
3. 方块 5cm、碗尺寸按视频估计；官方未公布
4. 单臂、腕部姿态固定（无 yaw/pitch/roll 控制）；官方双臂 6-DoF
5. PD 伺服加强过（官方真机伺服参数不可知）
6. 仿真分数 ≠ 真机分数；同仿真内跨模型横向对比有效

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `sim/assets/i2rt_yam/` | 官方 YAM 模型（Menagerie, MIT） |
| `sim/assets/i2rt_yam/block_bowl_scene.xml` | 任务场景 |
| `sim/yam_sim.py` | 仿真 embodiment（§4 全部要点） |
| `sim/run_eval.py` | 评测入口 |
| `sim/oracle_validate.py` | 全知对照实验 |
| `sim/smoke_test.py` | 硬编码冒烟测试 |
| `sim/batch_k3.sh` | K3 批量 20 次 |
| `HANDOFF.md` | 项目交接文档（含结果与更多调试记录） |
