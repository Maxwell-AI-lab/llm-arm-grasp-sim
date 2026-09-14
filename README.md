# LLM 零样本控制机械臂仿真评测（Block-into-Bowl）

用 MuJoCo 仿真复现 RoboCurve《GPT-6 Astra on robotic manipulation》实验：
前沿 LLM（GPT-6 Astra / Kimi K3 / …）零样本、无微调，仅凭相机画面通过
绝对末端位姿工具调用，控制 I2RT YAM 机械臂把红色方块抓进碗里。

- 评测框架：[Inspect Robots](https://github.com/robocurve/inspect-robots)（MIT）
- 机械臂模型：MuJoCo Menagerie `i2rt_yam`（MIT）
- 仿真器：MuJoCo 3.13（CPU 即可，无需 GPU）

## 文档

- **[SETUP.md](SETUP.md)** —— 从零搭建环境的完整指导书（推荐阅读顺序 ①）
- **[HANDOFF.md](HANDOFF.md)** —— 项目交接文档：已有结果、调试记录、复现 checklist（②）

## 30 秒上手

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install "inspect-robots[rerun]" inspect-robots-agent \
  "inspect-robots-yam[collision]" kitchenbench mujoco pillow numpy

cd sim
python oracle_validate.py     # 全知对照：应为 20/20
python run_eval.py --scenes 1 --seed-start 0 --model k3-agent \
  --effort high --max-steps 2500 --log-dir logs_k3
```

## 当前结果（仿真 n=20）

| 模型 | 成功率 | 备注 |
|---|---|---|
| Kimi K3 (effort=high) | 0/20 | 15/20 曾夹起，败于定位/放置 |
| GPT-6 Astra | 待测 | 需 OPENAI_API_KEY |

官方真机参考值：Astra 95% / Fable 5.1 40% / Fable 5 5%。
仿真分数 ≠ 真机分数；偏差声明见 SETUP.md §7。

## 许可

代码与文档 MIT；`sim/assets/i2rt_yam/` 为 i2rt robotics 的 MIT 模型（含其 LICENSE）。
