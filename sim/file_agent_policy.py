"""FileAgentPolicy — an external agent (here: qwen3.8-max via the ZCode session)
acts as the LLM policy through a file handshake, with the *official*
inspect-robots-agent motion layer underneath.

Methodology note (must appear in any report):
  The rollout, scoring, toolset validation, move_to interpolation, action
  chunks, budgets, and logging are all the official inspect_robots /
  inspect_robots_agent code paths — identical to the K3 runs. The ONLY
  deviation is the transport: instead of an HTTP LLM call, act() writes the
  observation (camera PNGs + the same state text the LLM would receive) to a
  directory and waits for the external agent to write back one tool call as
  JSON. The external agent sees exactly what LLMAgentPolicy would send
  (images=always) and is held to the same 60-call budget.

Handshake protocol (per decision):
  policy writes:  <exchange>/decision_NNN/{obs.md, top.png, side.png, wrist.png,
                   request.json}
  agent writes:   <exchange>/decision_NNN/response.json   (atomic rename)
    {"name": "move_to",
     "arguments": {"note": "...", "targets": {"x":..,"y":..,"z":..,"gripper":..}}}
    or {"name": "done"|"give_up", "arguments": {"summary": "...", "hindsight": "..."}}
  policy validates via toolset.execute(); on error it writes err.txt in the
  same directory, deletes response.json, and waits again (each consumed
  response counts against the budget, like an LLM completion would).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import ToolResult, build_toolset
from inspect_robots_agent.policy import _state_lines, _step_label


def _constant_grip_chunk(chunk: ActionChunk, grip: float) -> ActionChunk:
    """Replace the motion layer's gripper interpolation with the commanded
    setpoint held constant across the chunk.

    The official toolset interpolates every dimension (gripper included) from
    the *measured* state. While a block is in the jaws the measured gripper
    reads ~0.66 (fingers physically blocked), so the first interpolated step of
    any following move_to exceeds yam_sim's 0.6 weld-release threshold and the
    grasp is dropped before the transport even starts. The gripper command is
    a setpoint, not a trajectory: holding it constant at the commanded value
    preserves the official position interpolation while keeping the weld
    (grip<=0.4 intent) or releasing it (grip>0.6 intent) exactly as commanded.
    """
    actions = []
    for a in chunk.actions:
        data = np.asarray(a.data, dtype=np.float64).copy()
        data[3] = grip
        actions.append(Action(data=data, meta=dict(a.meta)))
    return ActionChunk(actions=actions, control_hz=chunk.control_hz)

_GUIDANCE = """You are controlling a simulated robot embodiment through tool calls.
Each observation gives you the current proprioceptive state and camera images.
Work toward the user's goal in small, deliberate motions; re-check the
observation after every motion. Every move_to call must include a `note`:
in one or two sentences, say what you observe and why you chose this motion.
Call done when the goal is achieved; if it cannot be achieved call give_up.

Available tools (official inspect-robots-agent toolset):
- move_to: {"note": str, "targets": {"x": m, "y": m, "z": m, "gripper": 0..1}}
  Absolute grasp-point target in the arm base frame (table surface z=0),
  interpolated at a fixed safe speed; gripper 1=open, 0=closed.
- done: {"summary": str, "hindsight": str}
- give_up: {"reason": str, "hindsight": str}
"""


class FileAgentPolicy(PolicyBase):
    def __init__(
        self,
        exchange_dir: str | os.PathLike,
        max_llm_calls: int = 60,
        response_timeout_s: float = 7200.0,
        poll_interval_s: float = 0.5,
        transcript_echo: bool = True,
    ) -> None:
        self.config = PolicyConfig()
        self.info = PolicyInfo(name="agent-file", action_space=Box(shape=(1,)))
        self._exchange_root = Path(exchange_dir)
        self._max_llm_calls = max_llm_calls
        self._timeout = response_timeout_s
        self._poll = poll_interval_s
        self._echo_on = transcript_echo
        self._toolset = None
        self._state_labels = None
        self._embodiment_docs: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._calls_used = 0
        self._decision = 0
        self._goal = ""

    # -- lifecycle ---------------------------------------------------------

    def bind(self, embodiment_info) -> None:
        self._toolset = build_toolset(
            embodiment_info.action_space,
            embodiment_info.observation_space,
            embodiment_info.control_hz,
            0.1,
            images="always",
            pre_check=None,
        )
        self._state_labels = self._toolset.state_labels()
        self._embodiment_docs = getattr(embodiment_info, "docs", None)
        self.info = PolicyInfo(
            name="agent-file",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def reset(self, scene: Scene) -> None:
        self._goal = scene.instruction
        self._messages = [
            {"role": "system", "content": _GUIDANCE},
            {"role": "user", "content": f"Goal: {scene.instruction}"},
        ]
        self._calls_used = 0
        self._decision = 0
        self._trial_dir = self._exchange_root / f"trial-{scene.id}"
        self._trial_dir.mkdir(parents=True, exist_ok=True)
        self._echo(f"[agent] goal: {scene.instruction}")

    def on_trial_end(self, record, log_dir: str, run_id: str) -> None:
        messages = self.transcript()
        if not messages:
            return
        transcript_dir = Path(log_dir) / "transcripts" / run_id
        transcript_dir.mkdir(parents=True, exist_ok=True)
        trial_id = f"{record.scene_id}-e{record.epoch}"
        path = transcript_dir / f"{trial_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        record.metadata["transcript"] = f"transcripts/{run_id}/{trial_id}.jsonl"
        record.metadata["policy_transport"] = "file-handshake (external qwen3.8-max agent)"
        if self._calls_used:
            record.metadata["llm_usage"] = {"llm_calls": self._calls_used}

    def transcript(self) -> list[dict[str, Any]] | None:
        if not self._messages:
            return None
        return json.loads(json.dumps(self._messages, ensure_ascii=False, default=str))

    # -- the loop ----------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        assert self._toolset is not None, "act() before bind()"
        self._decision += 1
        dec_dir = self._trial_dir / f"decision_{self._decision:03d}"
        dec_dir.mkdir(parents=True, exist_ok=True)

        # Render exactly what the LLM would receive.
        lines = ["Current observation.", f"Instruction: {self._goal}"]
        step_label = _step_label(observation)
        if step_label:
            lines.append(step_label)
        lines.extend(_state_lines(observation, self._state_labels))
        lines.append(
            f"Budget: {self._calls_used}/{self._max_llm_calls} calls used."
        )
        for name, image in observation.images.items():
            Image.fromarray(np.asarray(image)).save(dec_dir / f"{name}.png")
            lines.append(f"camera {name!r}: saved as {name}.png")
        if self._embodiment_docs:
            lines.append("\nEmbodiment notes:\n" + self._embodiment_docs.strip())
        (dec_dir / "obs.md").write_text(
            _GUIDANCE + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )
        self._messages.append(
            {"role": "user", "content": "\n".join(lines), "images": sorted(observation.images)}
        )
        self._echo(
            f"[agent] >> decision {self._decision}: {len(observation.images)} camera(s), "
            + " | ".join(_state_lines(observation, self._state_labels))
        )

        err_path = dec_dir / "err.txt"
        resp_path = dec_dir / "response.json"
        err_path.unlink(missing_ok=True)
        resp_path.unlink(missing_ok=True)
        (dec_dir / "request.json").write_text(
            json.dumps(
                {"decision": self._decision, "calls_used": self._calls_used,
                 "calls_max": self._max_llm_calls}
            )
            + "\n",
            encoding="utf-8",
        )

        deadline = time.time() + self._timeout
        while True:
            if self._calls_used >= self._max_llm_calls:
                return self._forced_give_up(observation, "call budget exhausted")
            if time.time() > deadline:
                return self._forced_give_up(observation, "response timeout")
            if resp_path.exists():
                try:
                    payload = json.loads(resp_path.read_text(encoding="utf-8"))
                    name = str(payload["name"])
                    arguments = payload.get("arguments", {})
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                except Exception as exc:  # malformed response
                    self._calls_used += 1
                    self._feedback(dec_dir, resp_path, err_path, f"malformed response.json: {exc}")
                    deadline = time.time() + self._timeout
                    continue
                self._calls_used += 1
                # Hard information-set constraint (user 2026-09-14): the policy
                # decides from the top + side views only. The third 'wrist'
                # camera is a base-fixed mount (yam_sim never re-poses it), so
                # it carries no gripper-relative information anyway.
                cams = arguments.get("cameras")
                if isinstance(cams, list) and "wrist" in cams:
                    self._feedback(
                        dec_dir, resp_path, err_path,
                        "wrist view is off-limits for this policy: decide from "
                        "'top' and 'side' only",
                    )
                    deadline = time.time() + self._timeout
                    continue
                call = ToolCall(
                    id=f"d{self._decision}", name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
                self._echo(f"[agent] << tool_call {name}({json.dumps(arguments, ensure_ascii=False)})")
                self._messages.append(
                    {"role": "assistant", "tool_calls": [{"name": name, "arguments": arguments}]}
                )
                result = self._toolset.execute(call, observation)
                if result.error is not None:
                    self._feedback(dec_dir, resp_path, err_path, result.error)
                    deadline = time.time() + self._timeout
                    continue
                assert result.chunk is not None
                if name == "move_to" and "gripper" in arguments.get("targets", {}):
                    result = ToolResult(
                        chunk=_constant_grip_chunk(
                            result.chunk, float(arguments["targets"]["gripper"])
                        ),
                        note=result.note,
                        target=result.target,
                    )
                self._echo(f"[agent] -- {result.note}")
                self._messages.append({"role": "tool", "content": result.note})
                (dec_dir / "result.txt").write_text(result.note + "\n", encoding="utf-8")
                return result.chunk
            time.sleep(self._poll)

    # -- helpers -----------------------------------------------------------

    def _feedback(self, dec_dir: Path, resp_path: Path, err_path: Path, error: str) -> None:
        """Mirror the official loop's error turn: report and wait for a retry."""
        self._echo(f"[agent] -- tool error: {error}")
        self._messages.append({"role": "tool", "content": f"error: {error}"})
        resp_path.unlink(missing_ok=True)
        err_path.write_text(
            error + "\n\nFix the tool call and write response.json again.\n", encoding="utf-8"
        )

    def _forced_give_up(self, observation: Observation, why: str) -> ActionChunk:
        self._echo(f"[agent] -- {why}; forcing give_up")
        call = ToolCall(id="budget", name="give_up", arguments=json.dumps({"reason": why}))
        result = self._toolset.execute(call, observation)
        if result.error is not None or result.chunk is None:
            raise RuntimeError(f"forced give_up failed: {result.error}")
        return result.chunk

    def _echo(self, text: str) -> None:
        if self._echo_on:
            print(text, flush=True)
