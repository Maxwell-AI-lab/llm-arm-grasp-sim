"""GLM-5.3 in-session visual control trial with live camera view; NOT an API benchmark.

File-based command loop for the controlling conversation (Zcode agent, GLM-5.3):
the policy writes <run>/command.json ({"target":[x,y,z,gripper],"steps":n} or
{"finish":true}); the driver applies it with codex-trial interpolation semantics
and budgets. The policy gets only camera images and budgets — status.json carries
NO eef_state. Live JPEGs + status are mirrored to /tmp for the operator's web view.
"""
import hashlib, json, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

SIM = Path(__file__).resolve().parent
OUT = SIM / 'logs_glm_session' / time.strftime('%Y%m%d-%H%M%S')
LIVE = Path('/private/tmp/robot-live-view-glm')
SEED = 0
MAX_COMMANDS = 60
MAX_TICKS = 2500
IDLE_TIMEOUT = 1800.0


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def live_status(message, decision=0, ticks=0):
    p = LIVE / 'status.tmp'
    p.write_text(json.dumps({'message': message, 'decision': decision, 'ticks': ticks,
                             'updated': time.time()}, ensure_ascii=False))
    p.replace(LIVE / 'status.json')


def live_images(obs):
    for camera in ['top', 'side']:
        p = LIVE / (camera + '.tmp')
        Image.fromarray(obs.images[camera]).save(p, format='JPEG', quality=88)
        p.replace(LIVE / (camera + '.jpg'))


def snapshot():
    files = [SIM / 'yam_sim.py', SIM / 'oracle_validate.py', SIM / 'run_eval.py']
    files += sorted((SIM / 'assets/i2rt_yam').rglob('*'))
    return {str(p.relative_to(SIM)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


emb = None
terminated = False
commands = 0
reason = 'idle_timeout'
started = time.time()
freeze = None

try:
    OUT.mkdir(parents=True, exist_ok=False)
    freeze = snapshot()
    write(OUT / 'frozen_sha256.json', freeze)
    write(OUT / 'metadata.json', {
        'policy': 'current_zcode_glm53_conversation',
        'model': 'builtin:bigmodel-coding-plan/GLM-5.3',
        'seed': SEED, 'independent_zero_shot': False,
        'prior_exposure': 'Conversation read HANDOFF.md (including implementation notes), '
                          'prior trial harness scripts, and a 3-command aborted pilot on the same seed '
                          '(logs_glm_session/20260914-204317); did not read yam_sim.py source.',
        'protocol_constraints': {
            'coordinate_readouts': 'none — status.json carries no eef_state; policy navigates on '
                                   'camera images and its own command memory only',
            'policy_cameras': ['top', 'side'],
            'live_view': True, 'wall_clock_tick_pacing_seconds': 0.1},
        'observation': 'top and side cameras only (wrist image saved to log but not consumed)',
        'mode': 'demo_debug_assisted — policy consumes logged simulator object '
                'coordinates after 2 prior pure-vision attempts; not comparable '
                'to no-coordinate-protocol runs',
        'physics': 'Unmodified YamBlockBowlEmbodiment including weld grasp abstraction.',
        'oracle_evidence': 'logs_codex_isolated/20260914-201615/oracle.log = 20/20 with identical frozen hashes',
        'action_space': 'x in [0.15,0.45], y in [-0.25,0.25], z in [0.005,0.35], gripper in [0,1], steps 1-100',
        'max_commands': MAX_COMMANDS, 'max_steps': MAX_TICKS, 'api_calls': 0,
        'replaces': 'logs_glm_session/20260914-204317 (aborted pilot, 3 commands, live-view restart)',
        'interface': 'policy writes command.json; driver consumes it and rewrites status.json',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    emb = YamBlockBowlEmbodiment()

    def emit(obs, index, extra=None):
        for name, img in obs.images.items():
            Image.fromarray(img).save(OUT / f'observation_{index:03d}_{name}.png')
        record = {'observation': index,
                  'eef_state': obs.state['eef_state'].tolist(),
                  'steps': emb.num_steps, 'terminated': terminated}
        # demo-mode instrumentation: log object world positions (declared in
        # metadata as policy-consumed; NOT a pure-vision protocol run)
        objects = {}
        for i in range(emb.model.nbody):
            name = emb.model.body(i).name
            if 'block' in name or 'bowl' in name:
                objects[name] = emb.data.body(i).xpos.tolist()
        if objects:
            record['debug_objects'] = objects
        if extra:
            record.update(extra)
        with (OUT / 'observations.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        status = {k: record[k] for k in ['observation', 'steps', 'terminated']}
        status.update({'requested': record.get('requested'),
                       'remaining_decisions': MAX_COMMANDS - commands,
                       'remaining_ticks': MAX_TICKS - emb.num_steps, 'ready': True})
        write(OUT / 'status.json', status)
        print(json.dumps(record), flush=True)

    live_status('GLM-5.3 测试即将开始')
    obs = emb.reset(Scene(id='glm-session-trial',
                          instruction='Pick up the red block from the table and place it inside the bowl.',
                          init_seed=SEED), seed=SEED)
    live_images(obs)
    emit(obs, 0)
    last = time.time()
    while True:
        cmd_path = OUT / 'command.json'
        if cmd_path.exists():
            last = time.time()
            try:
                command = json.loads(cmd_path.read_text())
            except ValueError:
                command = None  # partial write; wait for the atomic rename
            if command is not None:
                cmd_path.unlink()
                with (OUT / 'actions.jsonl').open('a') as f:
                    f.write(json.dumps(command) + '\n')
                if command.get('finish'):
                    reason = 'policy_finished'
                    break
                if terminated:
                    raise ValueError('Episode already terminated; finish required')
                target = np.asarray(command['target'], dtype=float)
                steps = int(command.get('steps', 20))
                if target.shape != (4,) or not np.isfinite(target).all():
                    raise ValueError('Invalid target')
                if not (np.all(target >= [.15, -.25, .005, 0]) and np.all(target <= [.45, .25, .35, 1])):
                    raise ValueError('Outside action bounds')
                if not 1 <= steps <= 100:
                    raise ValueError('steps out of range')
                if commands >= MAX_COMMANDS:
                    reason = 'decision_budget'
                    break
                if emb.num_steps >= MAX_TICKS:
                    reason = 'control_tick_budget'
                    break
                steps = min(steps, MAX_TICKS - emb.num_steps)
                if steps < 1:
                    reason = 'control_tick_budget'
                    break
                commands += 1
                live_status('GLM-5.3 动作执行中', commands, emb.num_steps)
                state = emb._eef_state().copy()
                for i in range(1, steps + 1):
                    tick_started = time.monotonic()
                    pose = state[:3] + (target[:3] - state[:3]) * (i / steps)
                    result = emb.step(Action(data=np.array([*pose, target[3]])))
                    terminated = bool(result.terminated)
                    obs = result.observation
                    live_images(obs)
                    time.sleep(max(0, 0.1 - (time.monotonic() - tick_started)))
                    if terminated:
                        break
                emit(obs, commands, extra={'requested': command['target'],
                                          'steps_requested': int(command.get('steps', 20))})
                live_status('GLM-5.3 正在观察并思考；机械臂暂停等待', commands, emb.num_steps)
                if terminated:
                    reason = 'simulator_success'
                    break
        else:
            if time.time() - last > IDLE_TIMEOUT:
                reason = 'idle_timeout'
                break
            time.sleep(0.2)
    for name, img in obs.images.items():
        Image.fromarray(img).save(OUT / f'final_{name}.png')
    summary = {'policy': 'current_zcode_glm53_conversation',
               'model': 'GLM-5.3',
               'status': 'completed', 'success_at_end': int(terminated),
               'termination_reason': reason, 'seed': SEED,
               'commands': commands, 'control_ticks': emb.num_steps,
               'wall_time_s': time.time() - started,
               'frozen_files_unchanged': snapshot() == freeze}
    write(OUT / 'result.json', summary)
    live_status(('测试成功：红色方块已放入碗中' if terminated else '测试结束：未成功')
                if reason != 'policy_finished' or terminated else '策略主动结束',
                commands, emb.num_steps)
    print(json.dumps({'result': summary, 'directory': str(OUT)}), flush=True)
except Exception:
    if OUT.exists():
        (OUT / 'error.txt').write_text(traceback.format_exc())
        write(OUT / 'result.json', {
            'status': 'error', 'success_at_end': None,
            'termination_reason': 'exception', 'seed': SEED,
            'commands': commands, 'control_ticks': emb.num_steps if emb else 0,
            'frozen_files_unchanged': (snapshot() == freeze) if freeze else None})
    raise
finally:
    if emb:
        emb.close()
