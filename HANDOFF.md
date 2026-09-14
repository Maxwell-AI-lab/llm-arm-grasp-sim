# 仿真复现交接文档：LLM 零样本控制机械臂抓方块入碗

> 交接目的：让另一个编码 agent（或人）能独立继续/重跑本实验。
> 最近更新：2026-09-14 07:10（K3 正式 20 次跑已完成；Astra 未跑，缺 API key）

---

## 1. 项目背景

复现对象：RoboCurve 2026-09-04 报告 *"GPT-6 Astra on robotic manipulation"*。
原实验：用开源框架 **Inspect Robots** 的 agent 策略，让前沿 LLM（零样本、无微调）
通过**多视角相机 + 绝对末端位姿工具调用**控制 I2RT YAM 机械臂，任务：

> "Pick up the red block from the table and place it inside the bowl."

官方真机结果（n=20）：

| 模型 | 成功率 | 平均分(0-4) | 输出token/次 | 成本/次 | 时间/次 |
|---|---|---|---|---|---|
| GPT-6 Astra | **19/20 (95%)** | 3.95 | 2.1k | $0.94 | 2.5 min |
| Claude Fable 5.1 | 8/20 (40%) | 2.40 | 12.9k | $2.12 | 6.8 min |
| Claude Fable 5 | 1/20 (5%) | 1.30 | 19.2k | $2.69 | 8.2 min |

参考视频（量子位Daily）：BV14wbK6TEaS。

## 2. 本复现的方法

真机不可用 → 用 **MuJoCo 仿真**重建同一任务，**方法论与官方完全一致**：

- 同一评测框架：`inspect-robots`（PyPI，MIT）
- 同一 agent 策略插件：`inspect-robots-agent`（LLM 看相机渲染图 + 带标签状态，输出 `move_to` 工具调用，绝对末端目标 x,y,z + 夹爪 0~1）
- 官方 YAM 机械臂模型：MuJoCo Menagerie `i2rt_yam`（commit 71f066a，MIT）
- 每 trial 独立种子（方块位置随机），成功判据：方块静止在碗内（仿真特权判定）
- n=20、记录完整决策过程（transcript_echo）

### 与官方的偏差（报告时必须声明）

1. 仿真物理 vs 真实硬件：确定性物理、无传感噪声
2. **抓取保持用了"焊接"抽象**：MuJoCo 刚性点接触复现不了橡胶垫摩擦。实现：夹爪闭合且方块在钳口窗口内时，方块先被"推向钳口中心"（模拟垫片归中），居中后 weld 到手腕；夹爪张开 >0.6 释放
3. 方块 5cm（官方未公布尺寸，按画面估）；碗为 10 段盒体拼成的圆碗（内径 ~11cm）
4. 单臂（原实验双臂但实际只用一只完成任务）；腕部姿态固定朝下，无 yaw/pitch/roll 控制
5. PD 伺服增益加强（原版 kp=10~40 在仿真里重力下垂严重）

## 3. 目录与文件

工作区根：`/Users/max/Documents/kimi/tasks/2026-09-13/13-12-57-5fe96821/`

```
.venv/                          # Python 3.12 虚拟环境（uv 创建），所有依赖已装好
video/astra_arm.mp4             # 原始 B 站视频（360p）
sim/
  assets/i2rt_yam/              # 官方 YAM MuJoCo 模型（含 meshes）
    block_bowl_scene.xml        # ★ 场景：臂+桌面+碗+红方块+2 相机
  yam_sim.py                    # ★ 仿真 embodiment（IK、伺服、焊接抓取、成功判定）
  run_eval.py                   # ★ 评测入口（参数化模型/端点/effort/种子）
  oracle_validate.py            # 全知策略对照实验（应为 20/20）
  smoke_test.py                 # 硬编码抓取冒烟测试
  batch_k3.sh                   # K3 批量 20 次跑（4 波 × 5 并行）
  astra_launch.sh               # Astra 批量跑（含等 key 逻辑，可精简）
  logs_k3/*.json                # K3 正式 20 次 eval log
  k3_final_*.out                # K3 每 trial 完整决策记录（transcript）
  logs_astra/                   # （空，Astra 未跑）
```

## 4. 环境

```bash
cd /Users/max/Documents/kimi/tasks/2026-09-13/13-12-57-5fe96821
source .venv/bin/activate        # Python 3.12；已装 inspect-robots[rerun]、
                                 # inspect-robots-agent、inspect-robots-yam、
                                 # kitchenbench、mujoco 3.13、numpy、Pillow
```

K3 模型走 Kimi 网关（环境变量已存在，无需配置）：
`KIMI_BASE_URL=https://agent-gw.kimi.com/coding/v1`，`KIMI_API_KEY=<已设置>`，
模型 id `k3-agent`（另有 `k2d6-agent` = K2.6）。

## 5. 怎么跑

### 5.1 冒烟自检（免费，2 分钟）

```bash
cd sim && source ../.venv/bin/activate
python oracle_validate.py        # 期望输出：Oracle policy: 20/20 = 100%
```

**任何代码改动后必须先跑这个**。它用精确坐标驱动抓取，验证仿真本身可解。
如果 < 20/20，说明仿真被改坏了，LLM 结果无效。

### 5.2 跑 K3（免费，单 trial 示例）

```bash
cd sim && source ../.venv/bin/activate
nohup ../.venv/bin/python -u run_eval.py --scenes 1 --seed-start 0 \
  --effort high --max-steps 2500 --log-dir logs_k3 > k3_x.out 2>&1 &
```

批量 20 次：`./batch_k3.sh`（4 波 × 5 并行，约 1.5~2 小时）。

### 5.3 跑 GPT-6 Astra（需要 OpenAI API key，20 次约 $19）

```bash
cd sim
echo 'OPENAI_API_KEY=sk-...' > .env     # ← 唯一需要用户提供的
source ../.venv/bin/activate && set -a && source .env && set +a
nohup ../.venv/bin/python -u run_eval.py --scenes 1 --seed-start 0 \
  --model gpt-6-astra --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY --effort high \
  --max-steps 2500 --log-dir logs_astra > astra_0.out 2>&1 &
```

批量：参考 `astra_launch.sh` 里的 wave 循环（把等 key 部分删掉即可）。
建议先单跑 1~2 次确认链路和单价，再放 20 次。

### 5.4 换其他模型

任何 OpenAI 兼容端点：`--model <名> --base-url <url> --api-key-env <环境变量名>`。
例如 K2.6：`--model k2d6-agent`（其余同 K3，走 Kimi 网关，免费）。
Anthropic：`--model anthropic/claude-fable-5-1` + `ANTHROPIC_API_KEY`（wire 自动适配）。

### 5.5 看结果

```bash
grep -hoE "success_at_end': [01]" <prefix>_*.out | sort | uniq -c   # 成功率
tail -30 <某个>.out                                                 # 单 trial 决策过程
ls logs_*/  # eval log JSON（含 policy_config、results、samples）
```

## 6. 已有结果

| 模型 | 仿真 n=20 成功率 | 备注 |
|---|---|---|
| Kimi K3 (k3-agent, effort=high) | **0/20 (0%)**，CI [0,17%] | 15/20 曾夹住方块但后续失败 |
| GPT-6 Astra | 未跑（缺 key） | — |

K3 失败模式（从 transcript 归纳）：① 视觉定位误差 1~3cm，闭合时把方块挤出；
② 碰飞方块后陷入追逐循环烧光预算（10/20 耗尽 60 次 LLM 调用）；
③ 夹住后运输/放置掉落。

## 7. 关键实现细节与坑（改了会出事）

1. **IK 不能污染物理状态**：`_ik()` 内部会临时改写 `data.qpos` 做迭代，
   返回前必须保存/恢复完整 qpos 并 `mj_forward`——否则机械臂"瞬移"爆炸。
2. **MuJoCo position 执行器的增益是成对的**：`gainprm[0]=+kp` 且 `biasprm[1]=-kp`，
   只改一个会得到错误力（`force = gain·ctrl + bias1·qpos + bias2·qvel`）。
   加强伺服时两个都要按同倍率改（见 `yam_sim.py __init__`）。
3. **末端参考点不是 grasp_site**：YAM 手指是 L 形，真实捏取点在爪尖垫片，
   相对 grasp_site 有固定偏移（约 (0.020, 0, -0.018) m），启动时自动测量为
   `_pinch_offset`，勿删。
4. **爪尖垫片会在桌面上"触底"**：完全下压时垫片抵住桌面摩擦导致夹爪无法闭合，
   抬 2~3mm 即可。这是保留的真实物理特性，LLM 需要自己适应（K3 曾自行发现）。
5. **焊接抓取的触发窗口**：水平 <3.3cm、竖直 (-0.01, 0.04)，闭合目标 <0.4；
   偏移 >1.2cm 时先施加归中速度（模拟垫片推方块），居中后 weld；
   张开目标 >0.6 时释放。这套逻辑在 `_update_grasp()`。
6. **eef_state.gripper 是实测值**：夹到方块时读数会停在 ~0.6（被方块挡住），
   不是故障。LLM 可能困惑，属于任务的一部分。
7. **长任务用 nohup 后台跑**：前台 shell 300s 超时；每个 trial 需 10~25 分钟。
   注意：一条命令里串多个 `nohup ... &` 时要逐个确认存活（曾丢过一个进程）。
8. **`inspect_robots` 的 `store_frames=True` 很吃内存**（每步 2 张 448² PNG），
   批量跑时关掉。
9. K3 是推理模型：`reasoning_effort` 用 `low`/`high`；effort=low 实测明显更弱。

## 8. 建议的后续步骤

1. 拿到 OpenAI key 后跑 Astra（先 2 次试跑再放量 20 次）
2. （可选）跑 `k2d6-agent` 作免费对比组
3. 汇总三模型对比表（本仿真 + 官方真机数字并列，标注偏差）
4. 报告里必须包含：本文件 §2 的偏差声明、oracle 对照结果、
   "仿真分数 ≠ 真机分数" 的醒目提示

## 9. 复现 checklist

- [x] `python oracle_validate.py` → 20/20（2026-09-14 qwen3.8-max 会话复跑 3 次）
- [x] 单 trial 试跑成功结束（`status: success` + `success_at_end` 有值；qwen3.8-max seed 0，give_up 收尾）
- [x] 批量跑日志目录分开（logs_k3 / logs_qwen38max；logs_astra 仍空），不混
- [x] 统计命令输出 20 条结果（K3：20 × `success_at_end': 0`）
- [x] 报告附偏差声明（REPORT_qwen38max.md §5，含新增的传输层/wrist 相机/weld 释放闸三条）

> 2026-09-14 增补（qwen3.8-max 会话）：无 OpenAI key，改用 file-handshake 策略
> （sim/file_agent_policy.py，策略脑=本会话模型）。run 1（原版 motion 层）0/1，
> 定位 weld 释放闸（REPORT_qwen38max.md §4：夹爪插值从实测 0.66 起步 > 释放阈值 0.6）；
> run 2 用夹爪设定点常数化修复（与 Astra 管线同思路）后同 seed **1/1 成功**（2.9 min）。
> K3 的 0/20 跑在原版层上，被该闸污染，重跑前不作能力证据。
> run 4（seed 2）起信息集收紧为 C 档：事中仅 top+side 视图+官方本体感知，方块真值
> 隔离在 live_frames/meta_priv.json（赛后审计专用），wrist 视图被策略拒收；run 4 1/1。
> 成绩分档与视觉瞄准审计见 REPORT_qwen38max.md §3.5。
> 实时画面：http://127.0.0.1:8765（yam_sim 的 YAM_LIVE_DIR + live_frames/index.html）。
