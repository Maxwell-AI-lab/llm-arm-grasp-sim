"""MuJoCo block-into-bowl simulation of the I2RT YAM arm as an Inspect Robots embodiment.

Reproduces the RoboCurve "GPT-6 Astra on robotic manipulation" setup in sim:
an LLM agent sees rendered cameras + a labeled Cartesian state and commands
absolute end-effector targets (x, y, z, gripper) that are IK-solved and
position-controlled in physics. Success (block resting inside the bowl) is
detected privileged, like a simulator capability of Inspect Robots.
"""

from __future__ import annotations

import json
import os
import pathlib

import mujoco
import numpy as np
from PIL import Image

from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

_XML = pathlib.Path(__file__).parent / "assets" / "i2rt_yam" / "block_bowl_scene.xml"

_CONTROL_HZ = 10.0
_IMG = 448
_ARM_JOINTS = 6
_GRIP_OPEN_CTRL = 0.0375  # left_finger ctrl fully open (ctrlrange 0..0.041)
_READY_GRASP = np.array([0.25, 0.0, 0.22])  # post-reset grasp-point target
# Desired link_6 orientation: grasp direction (col 2) straight down,
# gripper opening axis (col 0) along world +y.
_R_TARGET = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

# Action bounds (metres, normalized gripper) in the arm base frame (= world).
_ACT_LOW = np.array([0.15, -0.25, 0.005, 0.0])
_ACT_HIGH = np.array([0.45, 0.25, 0.35, 1.0])

_BLOCK_SPAWN_X = (0.20, 0.38)
_BLOCK_SPAWN_Y = (0.03, 0.20)

_BOWL_CENTER = np.array([0.30, -0.14])
_BOWL_INNER_R = 0.052
_BOWL_RIM_Z = 0.045


class YamBlockBowlEmbodiment:
    """Block-into-bowl world: one YAM arm, a red cube, a bowl, two cameras."""

    def __init__(self) -> None:
        from inspect_robots.embodiment import (
            AUTO_RESET,
            PRIVILEGED_SUCCESS,
            RENDERABLE,
            RESETTABLE,
            SEEDABLE,
            EmbodimentInfo,
        )

        self.model = mujoco.MjModel.from_xml_path(str(_XML))
        self.data = mujoco.MjData(self.model)
        # Sim-fidelity knobs: stiffen the stock PD position servos (kp 10-40 is
        # far softer than the real YAM drives and sags centimetres under
        # gravity), firm up the gripper pinch (kp=100 lets a 50 g cube slide
        # out), and raise the block's contact friction toward rubber-pad
        # behaviour.
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.model.actuator_gainprm[aid][0] *= 5.0   # kp (feedforward)
            self.model.actuator_biasprm[aid][1] *= 5.0   # -kp (position feedback)
            self.model.actuator_biasprm[aid][2] *= 2.0   # -kv (damping)
        gid_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.model.actuator_gainprm[gid_act][0] = 300.0   # kp feedforward
        self.model.actuator_biasprm[gid_act][1] = -300.0  # -kp position feedback (must match gain)
        gid_blk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
        self.model.geom_friction[gid_blk] = [1.2, 0.05, 0.01]
        # Stiffen gripper/block contacts (solref 20ms -> 4ms). The soft default
        # lets a grasped cube creep out of the pads over a few seconds.
        for body in ("lf_down", "rf_down", "lf_rot", "rf_rot", "block"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    self.model.geom_solref[g] = [0.004, 1.0]
        self.renderer = mujoco.Renderer(self.model, _IMG, _IMG)
        # Optional live web view: when YAM_LIVE_DIR is set, every control step
        # also renders two small frames into that directory (same thread that
        # steps the physics, so the GL context is never shared across threads).
        self._live_dir = os.environ.get("YAM_LIVE_DIR")
        if self._live_dir:
            pathlib.Path(self._live_dir).mkdir(parents=True, exist_ok=True)
            self._live_renderer = mujoco.Renderer(self.model, 256, 256)
        else:
            self._live_renderer = None
        self._n_sub = int(round(1.0 / _CONTROL_HZ / self.model.opt.timestep))
        self._grasp_sid = self.model.site("grasp_site").id
        self._body_id = self.model.body("link_6").id
        self._block_jid = self.model.joint("block_free").id
        self._block_qadr = self.model.jnt_qposadr[self._block_jid]
        self._block_dadr = self.model.jnt_dofadr[self._block_jid]
        self._block_bid = self.model.body("block").id
        self._joint_lo = self.model.jnt_range[:_ARM_JOINTS, 0].copy()
        self._joint_hi = self.model.jnt_range[:_ARM_JOINTS, 1].copy()
        self._success_streak = 0
        self.num_steps = 0
        self._weld_id = self.model.eq("grasp_weld").id
        self._grasped = False

        # Ready pose: IK from the home keyframe once at startup.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self._q_ready = self._ik(self.data.qpos[:_ARM_JOINTS].copy(), _READY_GRASP, iters=200)
        self._ready_gripper = 1.0

        # The fingers are L-shaped: the pinch happens at the curled tip pads,
        # not at the grasp_site. Locate the forward-most box pad on each finger
        # at the ready pose and use its midpoint as the pinch reference, so a
        # commanded (x, y, z) means "pinch the object at this point".
        self._lf_id = self.model.body("lf_down").id
        self._rf_id = self.model.body("rf_down").id
        self.data.qpos[:_ARM_JOINTS] = self._q_ready
        self.data.qpos[6] = _GRIP_OPEN_CTRL
        self.data.qpos[7] = -_GRIP_OPEN_CTRL
        mujoco.mj_forward(self.model, self.data)

        def _tip_pad_xpos(body_name: str) -> np.ndarray:
            bid = self.model.body(body_name).id
            cands = [
                g for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == bid and self.model.geom_type[g] == 6
            ]
            best = max(cands, key=lambda g: self.data.geom(g).xpos[0])
            return self.data.geom(best).xpos

        pinch = 0.5 * (_tip_pad_xpos("lf_down") + _tip_pad_xpos("rf_down"))
        self._pinch_offset = self.data.site(self._grasp_sid).xpos - pinch

        self.info = EmbodimentInfo(
            name="yam_block_bowl_sim",
            action_space=Box(
                shape=(4,),
                low=_ACT_LOW.copy(),
                high=_ACT_HIGH.copy(),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    frame="world",
                    rotation_repr="none",
                    dim_labels=("x", "y", "z", "gripper"),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec(name="top", height=_IMG, width=_IMG, channels=3),
                    CameraSpec(name="side", height=_IMG, width=_IMG, channels=3),
                    CameraSpec(name="wrist", height=_IMG, width=_IMG, channels=3),
                ),
                state_keys=frozenset({"eef_state"}),
                state=StateSpec(
                    fields=(
                        StateField(key="eef_state", shape=(4,), unit="m,norm"),
                    )
                ),
            ),
            control_hz=_CONTROL_HZ,
            is_simulated=True,
            capabilities=frozenset(
                {SEEDABLE, RESETTABLE, AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE}
            ),
            docs=(
                "Simulated YAM arm on a table. You command the absolute position of the "
                "grasp point (midpoint between the fingertips) in the arm base frame: "
                "+x is forward away from the arm base, +y is to the arm's left, +z is up; "
                "the table surface is at z=0. The gripper always points straight down "
                "and opens sideways along world y. gripper is normalized: 1 = fully open, "
                "0 = fully closed. There is one red 5 cm cube on the table and one white "
                "bowl. The 'top' camera looks straight down at the workspace; the 'side' "
                "camera views the workspace obliquely from the front (-y side); the "
                "'wrist' camera is mounted on the gripper wrist and looks down at the "
                "fingertips, so the block appears centred between the fingers when the "
                "gripper is directly above it."
            ),
        )

    # ------------------------------------------------------------------ API

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Repose the arm, open the gripper, and spawn the block per seed."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[:_ARM_JOINTS] = self._q_ready
        self.data.qpos[6] = _GRIP_OPEN_CTRL
        self.data.qpos[7] = -_GRIP_OPEN_CTRL
        rng = np.random.RandomState(seed if seed is not None else 0)
        bx = rng.uniform(*_BLOCK_SPAWN_X)
        by = rng.uniform(*_BLOCK_SPAWN_Y)
        self.data.qpos[self._block_qadr : self._block_qadr + 7] = [bx, by, 0.026, 1, 0, 0, 0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:_ARM_JOINTS] = self._q_ready
        self.data.ctrl[6] = _GRIP_OPEN_CTRL
        self.data.eq_active[self._weld_id] = 0
        self._grasped = False
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.5 / self.model.opt.timestep)):  # settle 0.5 s
            mujoco.mj_step(self.model, self.data)
        self.num_steps = 0
        self._success_streak = 0
        self._publish_live()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Track one absolute (x, y, z, gripper) target for one control period."""
        self.num_steps += 1
        target = np.clip(np.asarray(action.data, dtype=np.float64), _ACT_LOW, _ACT_HIGH)
        grasp_goal = target[:3] + self._pinch_offset
        self.data.ctrl[6] = target[3] * _GRIP_OPEN_CTRL
        # Split the control period into chunks; before each chunk, re-solve IK
        # from the measured state with a small tracking-error feedforward.
        # (Gain >= 1.0 overshoots and diverges with stiff servos; 0.5 lands
        # within ~3 mm and stays stable.)
        n_chunks = 6
        for _ in range(n_chunks):
            q_now = self.data.qpos[:_ARM_JOINTS].copy()
            grasp_now = self.data.site(self._grasp_sid).xpos
            aug = grasp_goal + 0.5 * (grasp_goal - grasp_now)
            q_tgt = self._ik(q_now, aug, iters=10)
            dq = np.clip(q_tgt - q_now, -0.2, 0.2)
            self.data.ctrl[:_ARM_JOINTS] = np.clip(q_now + dq, self._joint_lo, self._joint_hi)
            for _ in range(self._n_sub // n_chunks):
                mujoco.mj_step(self.model, self.data)
        # Convergence pass: PD sag in deep crouches leaves a residual; run extra
        # short correction bursts until the grasp point actually arrives (<=3 mm).
        for _ in range(20):
            grasp_now = self.data.site(self._grasp_sid).xpos
            err = grasp_goal - grasp_now
            if np.linalg.norm(err) < 0.003:
                break
            q_now = self.data.qpos[:_ARM_JOINTS].copy()
            q_tgt = self._ik(q_now, grasp_goal + 0.5 * err, iters=10)
            dq = np.clip(q_tgt - q_now, -0.05, 0.05)
            self.data.ctrl[:_ARM_JOINTS] = np.clip(q_now + dq, self._joint_lo, self._joint_hi)
            for _ in range(6):
                mujoco.mj_step(self.model, self.data)
        self._update_grasp(target[3])
        success = self._check_success()
        self._publish_live()
        return StepResult(
            observation=self._observe(None),
            reward=0.0,
            terminated=success,
            termination_reason="success" if success else None,
            truncated=False,
            info={"success": success, **self._task_state()},
        )

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()

    # ------------------------------------------------------------- internals

    def _update_grasp(self, gripper_target: float) -> None:
        """Sticky-grasp assist: weld the block to the wrist when the closing
        gripper is around it; release the weld when the gripper opens.

        MuJoCo rigid point contacts cannot reproduce the real rubber pads'
        friction and centering effect, so grasp retention is abstracted:
        while the jaws close with the cube inside, the cube is nudged toward
        the jaw centre (as real pads push it), and once centred it is welded
        to the wrist. Precise positioning, closing, transport, and release
        timing remain the policy's problem.
        """
        if not self._grasped and gripper_target < 0.4:
            pinch = self.data.site(self._grasp_sid).xpos - self._pinch_offset
            bp = self.data.body(self._block_bid).xpos
            offset = bp[:2] - pinch[:2]
            dist = np.linalg.norm(offset)
            near_z = -0.01 < bp[2] - pinch[2] < 0.04
            if dist < 0.033 and near_z:
                if dist > 0.012:
                    # Pads closing on an off-centre cube push it toward the
                    # jaw centre; emulate with a capped centering velocity.
                    v = -2.0 * offset
                    speed = np.linalg.norm(v)
                    if speed > 0.05:
                        v *= 0.05 / speed
                    self.data.qvel[self._block_dadr : self._block_dadr + 2] = v
                    return
                p6 = self.data.body(self._body_id).xpos
                q6 = self.data.body(self._body_id).xquat.copy()
                bq = self.data.qpos[self._block_qadr + 3 : self._block_qadr + 7].copy()
                centered = np.array([pinch[0], pinch[1], bp[2]])
                q6_inv = np.zeros(4)
                mujoco.mju_negQuat(q6_inv, q6)
                rel_pos = np.zeros(3)
                mujoco.mju_rotVecQuat(rel_pos, centered - p6, q6_inv)
                rel_quat = np.zeros(4)
                mujoco.mju_mulQuat(rel_quat, q6_inv, bq)
                eq_data = self.model.eq_data[self._weld_id]
                eq_data[0:3] = 0.0           # anchor
                eq_data[3:6] = rel_pos       # relpose position
                eq_data[6:10] = rel_quat     # relpose orientation
                eq_data[10] = 1.0            # torquescale
                # Snap the cube to the centred pose so physics and weld agree.
                self.data.qpos[self._block_qadr : self._block_qadr + 3] = centered
                self.data.eq_active[self._weld_id] = 1
                mujoco.mj_forward(self.model, self.data)
                self._grasped = True
        elif self._grasped and gripper_target > 0.6:
            self.data.eq_active[self._weld_id] = 0
            mujoco.mj_forward(self.model, self.data)
            self._grasped = False

    def _task_state(self) -> dict[str, float]:
        bp = self.data.body(self._block_bid).xpos
        d = float(np.linalg.norm(bp[:2] - _BOWL_CENTER))
        return {"block_x": float(bp[0]), "block_y": float(bp[1]), "block_z": float(bp[2]),
                "block_dist_to_bowl": d}

    def _check_success(self) -> bool:
        bp = self.data.body(self._block_bid).xpos
        bv = self.data.qvel[self._block_dadr : self._block_dadr + 3]
        inside = (
            np.linalg.norm(bp[:2] - _BOWL_CENTER) < _BOWL_INNER_R
            and bp[2] < _BOWL_RIM_Z
            and np.linalg.norm(bv) < 0.05
        )
        self._success_streak = self._success_streak + 1 if inside else 0
        return self._success_streak >= 5

    def _eef_state(self) -> np.ndarray:
        p = self.data.site(self._grasp_sid).xpos - self._pinch_offset
        g = float(self.data.qpos[6] / _GRIP_OPEN_CTRL)
        return np.array([p[0], p[1], p[2], np.clip(g, 0.0, 1.0)])

    def _observe(self, instruction: str | None) -> Observation:
        images = {}
        for cam in ("top", "side", "wrist"):
            self.renderer.update_scene(self.data, camera=cam)
            images[cam] = self.renderer.render().copy()
        return Observation(
            images=images,
            state={"eef_state": self._eef_state()},
            instruction=instruction,
        )

    def _publish_live(self) -> None:
        """Overwrite the live-view frames + state json (no-op unless enabled)."""
        if self._live_renderer is None:
            return
        for cam in ("top", "side"):
            self._live_renderer.update_scene(self.data, camera=cam)
            Image.fromarray(self._live_renderer.render()).save(
                pathlib.Path(self._live_dir) / f"{cam}.png"
            )
        st = self._eef_state()
        # Privileged fields (block pose, grasp flag) go to meta_priv.json only:
        # the policy's information set is cameras + eef_state, so the page and
        # the policy see the clean meta.json. Privileged data is for post-trial
        # auditing by the operator, never for in-trial decisions.
        meta = {
            "step": self.num_steps,
            "eef": [round(float(v), 4) for v in st],
        }
        tmp = pathlib.Path(self._live_dir) / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(pathlib.Path(self._live_dir) / "meta.json")
        priv = {
            **meta,
            "grasped": self._grasped,
            **{k: round(v, 4) for k, v in self._task_state().items()},
        }
        tmp = pathlib.Path(self._live_dir) / "meta_priv.json.tmp"
        tmp.write_text(json.dumps(priv))
        tmp.replace(pathlib.Path(self._live_dir) / "meta_priv.json")

    def _ik(self, q0: np.ndarray, target: np.ndarray, iters: int) -> np.ndarray:
        """Damped least-squares IK: grasp-site position + fixed link_6 orientation."""
        q = q0.copy()
        saved_qpos = self.data.qpos.copy()  # IK plays with the live state; restore it
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lam = 1e-3
        for _ in range(iters):
            self.data.qpos[:_ARM_JOINTS] = q
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site(self._grasp_sid).xpos
            R = self.data.body(self._body_id).xmat.reshape(3, 3)
            e_pos = target - pos
            e_rot = 0.5 * (
                np.cross(R[:, 0], _R_TARGET[:, 0])
                + np.cross(R[:, 1], _R_TARGET[:, 1])
                + np.cross(R[:, 2], _R_TARGET[:, 2])
            )
            e = np.concatenate([e_pos, 0.3 * e_rot])
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self._grasp_sid)
            mujoco.mj_jacBody(self.model, self.data, None, jacr, self._body_id)
            J = np.vstack([jacp[:, :_ARM_JOINTS], 0.3 * jacr[:, :_ARM_JOINTS]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            q = np.clip(q + dq, self._joint_lo, self._joint_hi)
            if np.linalg.norm(dq) < 1e-6:
                break
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return q
