"""One trial with NO coordinate readout: stitched top+side camera image only.

Same simulator (YamBlockBowlEmbodiment) and move_to action space as the
inspect-robots-agent protocol, but the policy receives no measured pose,
no joint state, and no labeled coordinates — only one stitched image
(left half = top view, right half = side view) plus its own action history.
"""
import argparse
import base64
import io
import json
import time
import traceback
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

SIM = Path(__file__).resolve().parent
LIVE = Path("/private/tmp/robot-live-view-dsv4")


def live_status(message: str, decision: int = 0, ticks: int = 0) -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    p = LIVE / "status.tmp"
    p.write_text(json.dumps({"message": message, "decision": decision, "ticks": ticks, "updated": time.time()}, ensure_ascii=False))
    p.replace(LIVE / "status.json")


def live_images(obs) -> None:
    for camera in ["top", "side"]:
        p = LIVE / f"{camera}.tmp"
        Image.fromarray(obs.images[camera]).save(p, format="JPEG", quality=88)
        p.replace(LIVE / f"{camera}.jpg")

PROMPT = """You are a robot control policy. Pick up the red block from the table and place it inside the bowl.
You receive ONE current camera image per decision: the LEFT half is the top-down view and the RIGHT half is a side view.
No measured pose, joint state, or coordinate readout is provided. You must estimate positions visually.
Coordinates are in metres in the arm base frame: +x is forward from the arm base, +y is to its left, +z is upward, and table height is z=0. The robot has a single gripper pointing downward with fixed orientation. The commanded reference is the midpoint between the fingertips. gripper=1 opens it and gripper=0 closes it.
The red cube has side length 0.05 m. The top camera looks down; the side camera looks obliquely from the front (-y side).
Return exactly one move_to tool call per decision. x in [0.15,0.45], y in [-0.25,0.25], z in [0.005,0.35], gripper in [0,1], steps is an integer from 1 to 100. The harness linearly interpolates position from the current pose to your target over steps control ticks at nominal 10 Hz; the gripper command is applied throughout. A stationary target can be used to wait.
You have at most 60 decisions and 2500 control ticks. Use finish=true only to end the episode voluntarily (other numeric fields are then ignored). Your brief reason is logged and included in the subsequent episode history.
Do not seek external information; respond from the provided image and episode history alone. The robot stops automatically if the evaluator detects success."""

SCHEMA = {"type": "object", "properties": {
    "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
    "gripper": {"type": "number"}, "steps": {"type": "integer"},
    "finish": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["x", "y", "z", "gripper", "steps", "finish", "reason"],
    "additionalProperties": False}


def stitch_png_data_url(obs) -> str:
    top = Image.fromarray(obs.images["top"])
    side = Image.fromarray(obs.images["side"])
    h = top.height
    w = top.width + side.width
    canvas = Image.new("RGB", (w, h))
    canvas.paste(top, (0, 0))
    canvas.paste(side, (top.width, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="c9a22906-f9de-47cd-b94c-4fe43b4dd85d/deepseek-v4-flash")
    p.add_argument("--base-url", default="https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1")
    p.add_argument("--api-key-env", default="HUOSHAN_API_KEY")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-decisions", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--log-dir", default="logs_huoshan_dsv4flash_nocoord")
    args = p.parse_args()

    api_key = __import__("os").environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing {args.api_key_env} in environment")

    out = SIM / args.log_dir / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "policy_prompt.txt").write_text(PROMPT)
    protocol = {
        "seed": args.seed, "model_requested": args.model, "effort": args.effort,
        "coordinate_readout": False, "stitched_views": True,
        "input": "One stitched PNG per decision: left=top camera, right=side camera; plus budgets and self-issued action history (text only). No measured pose/joint state.",
        "max_decisions": args.max_decisions, "max_control_ticks": args.max_steps,
        "action_space": "absolute eef target x,y,z + gripper 0..1, interpolated over steps ticks at 10 Hz",
        "success_signal": "Boolean terminated from unchanged simulator (privileged); never shown to policy.",
        "protocol": "openai_chat_completions_images_only_no_coordinate_readout",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))

    client = httpx.Client(base_url=args.base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=args.timeout)
    emb = None
    history = []
    usage = []
    success = False
    status = "completed"
    reason = "decision_budget"
    calls = 0
    started = time.time()
    try:
        emb = YamBlockBowlEmbodiment()
        obs = emb.reset(Scene(id="nocoord-trial", instruction=PROMPT.splitlines()[0], init_seed=args.seed), seed=args.seed)
        live_images(obs)
        live_status("正在观察并思考；机械臂暂停等待", 0, 0)
        for n in range(args.max_decisions):
            if emb.num_steps >= args.max_steps:
                reason = "control_tick_budget"
                break
            data_url = stitch_png_data_url(obs)
            state = {"remaining_control_ticks": args.max_steps - emb.num_steps, "remaining_decisions": args.max_decisions - n}
            payload = {"current_observation": state, "episode_history": history}
            prompt = PROMPT + "\n" + json.dumps(payload)
            (out / f"{n:03d}_input.txt").write_text(prompt)
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Current stitched camera image (left=top, right=side):"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": json.dumps(payload)},
                ]},
            ]
            body = {"model": args.model, "messages": messages,
                    "tools": [{"type": "function", "function": {"name": "move_to", "description": "Move the arm", "parameters": SCHEMA}}],
                    "reasoning_effort": args.effort, "max_tokens": 16384}
            calls += 1
            live_status("deepseek-v4-flash 正在观察并思考；机械臂暂停等待", n + 1, emb.num_steps)
            last_error = "unknown error"
            resp = None
            for attempt in range(3):
                try:
                    resp = client.post("/chat/completions", json=body)
                except httpx.TransportError as exc:
                    last_error = str(exc)
                    time.sleep(1.0 * 2 ** attempt)
                    continue
                if resp.status_code == 200:
                    break
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (429,) and resp.status_code < 500:
                    raise RuntimeError(f"LLM request rejected — {last_error}")
                time.sleep(1.0 * 2 ** attempt)
            if resp is None or resp.status_code != 200:
                raise RuntimeError(f"LLM request failed after retries — {last_error}")
            data = resp.json()
            usage.append(data.get("usage") or {})
            message = data["choices"][0]["message"]
            action = None
            for call in message.get("tool_calls") or []:
                if call["function"]["name"] == "move_to":
                    action = json.loads(call["function"]["arguments"])
                    break
            if action is None and message.get("content"):
                text = str(message["content"])
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    try:
                        action = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        action = None
            if action is None:
                raise RuntimeError("No move_to tool call or JSON action in model output")
            (out / f"{n:03d}_action.json").write_text(json.dumps(action, indent=2))
            if action.get("finish"):
                reason = "policy_finished"
                break
            target = np.array([action[k] for k in ["x", "y", "z", "gripper"]], dtype=float)
            steps = action["steps"]
            if not np.isfinite(target).all() or not (np.all(target >= [.15, -.25, .005, 0]) and np.all(target <= [.45, .25, .35, 1])):
                raise RuntimeError("Out-of-bounds action; not corrected or retried")
            if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100:
                raise RuntimeError("Invalid action duration")
            initial = obs.state["eef_state"][:3].copy()
            executed = 0
            for tick in range(1, min(steps, args.max_steps - emb.num_steps) + 1):
                tick_started = time.monotonic()
                point = initial + (target[:3] - initial) * (tick / steps)
                result = emb.step(Action(data=np.array([*point, target[3]])))
                executed += 1
                obs = result.observation
                success = bool(result.terminated)
                live_images(obs)
                live_status("正在执行动作", n + 1, emb.num_steps)
                time.sleep(max(0, 0.1 - (time.monotonic() - tick_started)))
                if success:
                    break
            history.append({"before": state, "action": action, "executed_ticks": executed})
            (out / "episode_history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps({"decision": n + 1, "steps": emb.num_steps, "terminated": success}), flush=True)
            if success:
                reason = "simulator_success"
                break
        for camera in ["top", "side"]:
            Image.fromarray(obs.images[camera]).save(out / f"final_{camera}.png")
    except Exception as exc:
        status = "error"
        reason = type(exc).__name__ + ": " + str(exc)
        (out / "error.txt").write_text(traceback.format_exc())
    finally:
        final = {"status": status, "success_at_end": int(success) if status == "completed" else None,
                 "termination_reason": reason, "seed": args.seed, "model_requested": args.model,
                 "policy_calls": calls, "executed_commands": len(history),
                 "control_ticks": emb.num_steps if emb else 0,
                 "wall_time_s": time.time() - started, "usage_per_call": usage,
                 "protocol": protocol["protocol"]}
        (out / "result.json").write_text(json.dumps(final, indent=2))
        if emb:
            emb.close()
        client.close()
        live_status("测试成功：红色方块已放入碗中" if success and status == "completed" else
                    ("测试结束：未成功" if status == "completed" else "测试出错，请查看日志"),
                    len(history), emb.num_steps if emb else 0)
        print(json.dumps({"final": final, "directory": str(out)}), flush=True)


if __name__ == "__main__":
    main()
