"""One pre-registered, source-isolated visual policy trial via native Codex auth.

This is a new Codex CLI protocol, not the inspect-robots-agent API protocol.
Controller input is constructed by allowlist; privileged state is never serialized.
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

PUBLIC=Path('/private/tmp/robot-policy-random-p526y1j2')
SIM=Path(__file__).resolve().parent
OUT=SIM/'logs_codex_isolated'/time.strftime('%Y%m%d-%H%M%S')
CODEX='/Applications/ChatGPT.app/Contents/Resources/codex'
DISABLED=['shell_tool','unified_exec','apps','plugins','hooks','browser_use',
 'browser_use_external','computer_use','in_app_browser','image_generation',
 'multi_agent','memories','shell_snapshot','goals','tool_suggest',
 'workspace_dependencies','code_mode','code_mode_only','realtime_conversation']
DISABLED+=['skill_search']
ENV={k:v for k,v in os.environ.items() if k in
 {'PATH','HOME','TMPDIR','LANG','LC_ALL','USER','LOGNAME','SSL_CERT_FILE','SSL_CERT_DIR'}}
SEED=2081712759
SCHEMA={'type':'object','properties':{
 'x':{'type':'number'},'y':{'type':'number'},'z':{'type':'number'},
 'gripper':{'type':'number'},'steps':{'type':'integer'},
 'finish':{'type':'boolean'},'reason':{'type':'string'}},
 'required':['x','y','z','gripper','steps','finish','reason'],'additionalProperties':False}
PROMPT='''You are a robot control policy. Pick up the red block from the table and place it inside the bowl.
You receive two current camera images (first top, second side),
and the complete text history of this episode. Choose the next action using these observations.
Coordinates are in metres in the arm base frame: +x is forward from the arm base,
+y is to its left, +z is upward, and table height is z=0. The robot has a single gripper
pointing downward with fixed orientation. The commanded reference is the midpoint between
the fingertips. gripper=1 opens it and gripper=0 closes it. No measured pose or gripper state is provided.
The red cube has side length 0.05 m. The top camera looks down; the side camera looks
obliquely from the front (-y side). There is no supplied object-coordinate readout.
Return only the JSON action. x in [0.15,0.45], y in [-0.25,0.25], z in [0.005,0.35],
gripper in [0,1], steps is an integer from 1 to 100. The harness linearly interpolates
position from the current measured pose to your target over steps control ticks at nominal
10 Hz; the gripper command is applied throughout. A stationary target can be used to wait.
You have at most 60 decisions and 2500 control ticks. Use finish=true only to end the
episode voluntarily (other numeric fields are then ignored). Your brief reason is logged
and included in the subsequent episode history. No file, shell, web, image-read or other
tools may be used. Do not seek external information; respond from the provided images and
episode history alone. The robot stops automatically if the evaluator detects success.
'''

LIVE=Path('/private/tmp/astra-live-ugle_noj')
def live_status(message, decision=0, ticks=0):
    p=LIVE/'status.tmp'
    p.write_text(json.dumps({'message':message,'decision':decision,'ticks':ticks,'updated':time.time()},ensure_ascii=False))
    p.replace(LIVE/'status.json')
def live_images(obs):
    for camera in ['top','side']:
        p=LIVE/(camera+'.tmp')
        Image.fromarray(obs.images[camera]).save(p,format='JPEG',quality=88)
        p.replace(LIVE/(camera+'.jpg'))

def write(path,value): path.write_text(json.dumps(value,indent=2))
def hashes():
    files=[SIM/'yam_sim.py',SIM/'oracle_validate.py',SIM/'run_eval.py']
    files+=sorted((SIM/'assets/i2rt_yam').rglob('*'))
    return {str(p.relative_to(SIM)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
def args():
    a=['/usr/bin/sandbox-exec','-f',str(PUBLIC/'isolation.sb'),CODEX,
       'exec','--ignore-user-config','--ignore-rules','--ephemeral',
       '--skip-git-repo-check','--sandbox','read-only','-C',str(PUBLIC),
       '--model','gpt-6-astra','-c','model_reasoning_effort="high"',
       '-c','web_search="disabled"','-c','project_doc_max_bytes=0',
       '-c','tools.view_image=false','-c',f'sqlite_home="{PUBLIC}/runtime"',
       '-c',f'log_dir="{PUBLIC}/runtime/log"','-c','approval_policy="never"',
       '--json','--output-schema',str(PUBLIC/'action_schema.json')]
    for n in DISABLED: a+=['--disable',n]
    return a

def main():
    live_status('正在运行参考策略检查…')
    OUT.mkdir(parents=True,exist_ok=False)
    freeze=hashes(); write(OUT/'frozen_sha256.json',freeze)
    protocol={'seed':SEED,'trials_planned':1,'restarts_allowed':0,'model_requested':'gpt-6-astra',
      'effort':'high','max_decisions':60,'max_control_ticks':2500,'max_inference_seconds':240,
      'policy':'codex_exec_fresh_context_with_episode_history',
      'model_context':'Generic Codex instructions plus fixed task prompt, current two camera images, remaining budgets, and current-episode self-issued action history only; no measured coordinate readouts.',
      'original_inspect_robots_agent_protocol':False,
      'live_view':True,'wall_clock_tick_pacing_seconds':0.1,'operator_action_override':False,'task_state_to_policy':False,
      'success_signal':'Boolean terminated from unchanged simulator; no privileged coordinates in policy input.',
      'limitations':['Simulator includes weld-assisted grasping.','Not a real-hardware experiment.',
         'Different harness from original API benchmark.','No assertion about unknown training-data exposure.'],
      'disabled_features':DISABLED,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write(OUT/'protocol.json',protocol); (OUT/'policy_prompt.txt').write_text(PROMPT)
    for name in ['isolation.sb','read_checks.json','smoke_events.jsonl','smoke_stderr.txt']:
        shutil.copy2(PUBLIC/name,OUT/name)
    # A fresh empty runtime database for actual policy calls, independent of connection smoke.
    (PUBLIC/'runtime').rename(PUBLIC/'preflight_runtime')
    for name in ['read_checks.json','smoke_events.jsonl','smoke_stderr.txt']:
        (PUBLIC/name).unlink()
    write(PUBLIC/'action_schema.json',SCHEMA)
    oracle=subprocess.run([sys.executable,'-u','oracle_validate.py'],cwd=SIM,
                          capture_output=True,text=True,timeout=240)
    (OUT/'oracle.log').write_text(oracle.stdout+oracle.stderr)
    if oracle.returncode or 'Oracle policy: 20/20 = 100%' not in oracle.stdout:
        write(OUT/'result.json',{'status':'oracle_failed','success_at_end':None,'trial_started':False})
        raise RuntimeError('Oracle check failed; trial not started')
    print('ORACLE_PASS; isolated trial starting; logs='+str(OUT),flush=True)
    emb=None; history=[]; usage=[]; success=False; status='completed'; reason='decision_budget'; calls=0
    started=time.time()
    try:
        emb=YamBlockBowlEmbodiment()
        obs=emb.reset(Scene(id='independent-trial',instruction=PROMPT.splitlines()[0],init_seed=SEED),seed=SEED)
        live_images(obs)
        for n in range(60):
            if emb.num_steps>=2500: reason='control_tick_budget'; break
            images=[]
            for camera in ['top','side']:
                path=PUBLIC/f'observation_{n:03d}_{camera}.png'
                Image.fromarray(obs.images[camera]).save(path)
                shutil.copy2(path,OUT/path.name); images+=['--image',str(path)]
            state={'remaining_control_ticks':2500-emb.num_steps,'remaining_decisions':60-n}
            payload={'current_observation':state,'episode_history':history}
            prompt=PROMPT+'\n'+json.dumps(payload)
            (OUT/f'{n:03d}_input.txt').write_text(prompt)
            calls+=1
            live_status('Astra 正在观察并思考；机械臂暂停等待',n+1,emb.num_steps)
            r=subprocess.run(args()+images+['-'],input=prompt,cwd=PUBLIC,env=ENV,
                             capture_output=True,text=True,timeout=240)
            (OUT/f'{n:03d}_events.jsonl').write_text(r.stdout)
            (OUT/f'{n:03d}_stderr.txt').write_text(r.stderr)
            if r.returncode: raise RuntimeError('Codex inference failed; inspect saved stderr')
            events=[json.loads(x) for x in r.stdout.splitlines() if x.strip()]
            for event in events:
                if event.get('type')=='turn.failed': raise RuntimeError('Backend failed turn')
                if event.get('type')=='error':
                    if event.get('message','').startswith('Reconnecting...'): continue
                    raise RuntimeError('Backend error event')
                item=event.get('item',{})
                if item.get('type')=='error' and item.get('message','').startswith('Falling back from WebSockets to HTTPS transport.'): continue
                if item and item.get('type') not in ['agent_message','reasoning']:
                    raise RuntimeError('Disallowed tool/event attempted; trial invalidated')
                if event.get('type')=='turn.completed': usage.append(event.get('usage',{}))
            messages=[e['item']['text'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
            if not messages: raise RuntimeError('No policy action')
            action=json.loads(messages[-1]); write(OUT/f'{n:03d}_action.json',action)
            if action['finish']: reason='policy_finished'; break
            target=np.array([action[k] for k in ['x','y','z','gripper']],dtype=float)
            steps=action['steps']
            if not np.isfinite(target).all() or not (np.all(target>=[.15,-.25,.005,0]) and np.all(target<=[.45,.25,.35,1])):
                raise RuntimeError('Out-of-bounds action; not corrected or retried')
            if not isinstance(steps,int) or isinstance(steps,bool) or not 1<=steps<=100:
                raise RuntimeError('Invalid action duration')
            live_status('正在执行 Astra 的动作',n+1,emb.num_steps)
            initial=obs.state['eef_state'][:3].copy()
            executed=0
            for tick in range(1,min(steps,2500-emb.num_steps)+1):
                tick_started=time.monotonic()
                point=initial+(target[:3]-initial)*(tick/steps)
                result=emb.step(Action(data=np.array([*point,target[3]])))
                executed+=1; obs=result.observation; success=bool(result.terminated)
                live_images(obs)
                live_status('正在执行 Astra 的动作',n+1,emb.num_steps)
                time.sleep(max(0,0.1-(time.monotonic()-tick_started)))
                if success: break
            history.append({'before':state,'action':action,'executed_ticks':executed})
            write(OUT/'episode_history.json',history)
            print(json.dumps({'decision':n+1,'steps':emb.num_steps,'terminated':success}),flush=True)
            if success: reason='simulator_success'; break
        for camera in ['top','side']: Image.fromarray(obs.images[camera]).save(OUT/f'final_{camera}.png')
    except Exception as exc:
        status='error'; reason=type(exc).__name__+': '+str(exc)
        (OUT/'error.txt').write_text(traceback.format_exc())
    finally:
        same=hashes()==freeze
        final={'status':status,'success_at_end':int(success) if status=='completed' else None,
               'termination_reason':reason,'seed':SEED,'model_requested':'gpt-6-astra',
               'policy_calls':calls,'executed_commands':len(history),'control_ticks':emb.num_steps if emb else 0,
               'wall_time_s':time.time()-started,'usage_per_call':usage,
               'frozen_files_unchanged':same,'source_read_denial_preflight_passed':True,
               'isolation_valid':same and status=='completed','protocol':protocol['policy']}
        write(OUT/'result.json',final)
        live_status('测试成功：红色方块已放入碗中' if success and status=='completed' else ('测试结束：未成功' if status=='completed' else '测试出错，请查看日志'),len(history),emb.num_steps if emb else 0)
        if emb: emb.close()
        print(json.dumps({'final':final,'directory':str(OUT)}),flush=True)

if __name__=='__main__': main()
