"""Run the block-into-bowl sim eval with an LLM agent policy.

Reproduces RoboCurve's "GPT-6 Astra on robotic manipulation" methodology in
MuJoCo sim: the model sees camera renders + labeled Cartesian state, emits
absolute end-effector targets via tool calls; success = block resting in bowl.

Usage:
  python run_eval.py --scenes 1 --seed-start 0            # single smoke trial
  python run_eval.py --scenes 20 --seed-start 0           # full 20-trial run
"""

import argparse
import os
import time

from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots_agent import LLMAgentPolicy

from yam_sim import YamBlockBowlEmbodiment

INSTRUCTION = "Pick up the red block from the table and place it inside the bowl."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="k3-agent")
    p.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL"))
    p.add_argument("--api-key-env", default="KIMI_API_KEY")
    p.add_argument("--effort", default="low")
    p.add_argument("--scenes", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--max-llm-calls", type=int, default=60)
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    task = Task(
        name="block-into-bowl",
        scenes=[
            Scene(id=f"layout-{i}", instruction=INSTRUCTION, init_seed=i)
            for i in range(args.seed_start, args.seed_start + args.scenes)
        ],
        scorer=success_at_end(),
        max_steps=args.max_steps,
    )
    policy = LLMAgentPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        effort=args.effort,
        max_llm_calls=args.max_llm_calls,
        transcript_echo=True,
    )
    emb = YamBlockBowlEmbodiment()

    t0 = time.time()
    logs = eval(
        task,
        policy,
        emb,
        log_dir=args.log_dir,
        store_frames=False,
        fail_on_error=False,
    )
    dt = time.time() - t0
    for log in logs:
        print("=" * 60)
        print("status:", log.status)
        res = log.results
        print("results:", res.metrics if res else None)
    print(f"wall time: {dt / 60:.1f} min for {len(logs)} trial(s)")
    emb.close()


if __name__ == "__main__":
    main()
