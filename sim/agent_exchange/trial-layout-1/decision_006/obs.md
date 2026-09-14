You are controlling a simulated robot embodiment through tool calls.
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

Current observation.
Instruction: Pick up the red block from the table and place it inside the bowl.
step 300
state[eef_state]: x=0.2217 y=0.0612 z=0.2204 gripper=0.6613
Budget: 5/60 calls used.
camera 'top': saved as top.png
camera 'side': saved as side.png
camera 'wrist': saved as wrist.png

Embodiment notes:
Simulated YAM arm on a table. You command the absolute position of the grasp point (midpoint between the fingertips) in the arm base frame: +x is forward away from the arm base, +y is to the arm's left, +z is up; the table surface is at z=0. The gripper always points straight down and opens sideways along world y. gripper is normalized: 1 = fully open, 0 = fully closed. There is one red 5 cm cube on the table and one white bowl. The 'top' camera looks straight down at the workspace; the 'side' camera views the workspace obliquely from the front (-y side); the 'wrist' camera is mounted on the gripper wrist and looks down at the fingertips, so the block appears centred between the fingers when the gripper is directly above it.
