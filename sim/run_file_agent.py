"""Run the block-into-bowl sim eval with the external file-handshake agent policy.

Same eval harness as run_eval.py (Task/scenes/success_at_end scorer/eval loop);
the policy transport is a file handshake instead of an HTTP LLM call, so an
external agent (qwen3.8-max in the ZCode session) can act as the brain.

Usage:
  python run_file_agent.py --scenes 1 --seed-start 0            # single trial
  python run_file_agent.py --scenes 20 --seed-start 0           # full run
"""

import argparse
import time

from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

from file_agent_policy import FileAgentPolicy
from yam_sim import YamBlockBowlEmbodiment

INSTRUCTION = "Pick up the red block from the table and place it inside the bowl."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--max-llm-calls", type=int, default=60)
    p.add_argument("--log-dir", default="logs_qwen38max")
    p.add_argument("--exchange-dir", default="agent_exchange")
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
    policy = FileAgentPolicy(
        exchange_dir=args.exchange_dir,
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
