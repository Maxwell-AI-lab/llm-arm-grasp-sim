"""Control experiment: scripted oracle policy across all 20 eval seeds.

If the sim is sound, a policy that knows the ground-truth block position
must succeed on every seed. Any failure here means the simulator (not the
LLM) is the bottleneck and the eval would be invalid.
"""

import numpy as np

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

BOWL = np.array([0.30, -0.14])


def run_seed(seed: int) -> bool:
    emb = YamBlockBowlEmbodiment()
    emb.reset(Scene(id=f"layout-{seed}", instruction="pick", init_seed=seed), seed=seed)

    def move_to(t, g, steps):
        s = emb._eef_state()
        r = None
        for i in range(1, steps + 1):
            a = i / steps
            r = emb.step(Action(data=np.array([
                s[0] + (t[0] - s[0]) * a,
                s[1] + (t[1] - s[1]) * a,
                s[2] + (t[2] - s[2]) * a,
                g,
            ])))
        return r

    b = emb._task_state()
    bx, by = b["block_x"], b["block_y"]
    move_to([bx, by, 0.12], 1.0, 20)
    move_to([bx, by, 0.015], 1.0, 25)
    move_to([bx, by, 0.015], 0.0, 15)
    move_to([bx, by, 0.18], 0.0, 30)
    move_to([BOWL[0], BOWL[1], 0.15], 0.0, 30)
    r = move_to([BOWL[0], BOWL[1], 0.15], 1.0, 15)
    for _ in range(40):
        r = emb.step(Action(data=np.array([BOWL[0], BOWL[1], 0.20, 1.0])))
        if r.terminated:
            break
    ok = bool(r.info["success"])
    emb.close()
    return ok


def main():
    results = {}
    for seed in range(20):
        results[seed] = run_seed(seed)
        print(f"seed {seed:2d}: {'SUCCESS' if results[seed] else 'FAIL'}")
    n = sum(results.values())
    print(f"\nOracle policy: {n}/20 = {n / 20:.0%}")


if __name__ == "__main__":
    main()
