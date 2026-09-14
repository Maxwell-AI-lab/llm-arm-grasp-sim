"""Free smoke test: drive the sim with hardcoded waypoints (no LLM).

Approach the seeded block, grasp it, carry it over the bowl, release, and
check the privileged success flag. Also saves a few frames for eyeballing.
"""

import numpy as np
from PIL import Image

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

BOWL = np.array([0.30, -0.14])


def move_to(emb, target_xyz, gripper, steps):
    """Linearly interpolate toward an absolute target over `steps` control steps."""
    start = emb._eef_state()
    result = None
    for i in range(1, steps + 1):
        a = i / steps
        tgt = np.array([
            start[0] + (target_xyz[0] - start[0]) * a,
            start[1] + (target_xyz[1] - start[1]) * a,
            start[2] + (target_xyz[2] - start[2]) * a,
            gripper,
        ])
        result = emb.step(Action(data=tgt))
        if result.terminated:
            break
    return result


def main():
    emb = YamBlockBowlEmbodiment()
    scene = Scene(id="layout-0", instruction="Pick up the red block and place it inside the bowl.", init_seed=0)
    obs = emb.reset(scene, seed=0)
    Image.fromarray(obs.images["top"]).save("smoke_top_0.png")

    block = np.array([obs_info for obs_info in [emb._task_state()]][0].values())  # not used; read directly
    b = emb._task_state()
    bx, by = b["block_x"], b["block_y"]
    print(f"block at ({bx:.3f}, {by:.3f})")

    r = move_to(emb, [bx, by, 0.15], 1.0, 20)          # hover above block
    r = move_to(emb, [bx, by, 0.035], 1.0, 20)         # descend around block
    Image.fromarray(r.observation.images["side"]).save("smoke_side_pregrasp.png")
    r = move_to(emb, [bx, by, 0.035], 0.0, 10)         # close gripper
    r = move_to(emb, [bx, by, 0.18], 0.0, 25)          # lift
    b2 = emb._task_state()
    print(f"after lift: block z={b2['block_z']:.3f} (should be ~0.15 if grasped)")
    r = move_to(emb, [BOWL[0], BOWL[1], 0.18], 0.0, 25)  # carry over bowl
    r = move_to(emb, [BOWL[0], BOWL[1], 0.10], 0.0, 10)
    r = move_to(emb, [BOWL[0], BOWL[1], 0.10], 1.0, 10)  # release
    Image.fromarray(r.observation.images["side"]).save("smoke_side_release.png")
    for _ in range(30):                                 # let it settle / detect
        r = emb.step(Action(data=np.array([BOWL[0], BOWL[1], 0.15, 1.0])))
        if r.terminated:
            break
    Image.fromarray(r.observation.images["top"]).save("smoke_top_end.png")
    print("terminated:", r.terminated, "| reason:", r.termination_reason, "| success:", r.info["success"])
    emb.close()


if __name__ == "__main__":
    main()
