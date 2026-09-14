"""Exploratory current-session visual control; NOT an independent API benchmark.

Only camera images and measured eef_state are returned to the controller.
The oracle implementation has already been seen by the current conversation.
No API keys, external model calls, or scene/physics modifications are used.
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from yam_sim import YamBlockBowlEmbodiment

out=Path(__file__).resolve().parent/'logs_codex_session'/time.strftime('%Y%m%d-%H%M%S')
out.mkdir(parents=True,exist_ok=False)
emb=YamBlockBowlEmbodiment()
start=time.time(); actions=[]; terminated=False; index=0
metadata={'policy':'current_codex_conversation','seed':0,'independent_zero_shot':False,
 'prior_exposure':'Conversation has read simulator source, oracle and prior K3 summary.',
 'observation':'top and side cameras; measured eef_state only',
 'physics':'Unmodified YamBlockBowlEmbodiment including weld grasp abstraction.',
 'max_commands':60,'max_steps':2500,'api_calls':0}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
def emit(obs):
    paths={}
    for name,img in obs.images.items():
        path=out/f'{index:03d}_{name}.png'; Image.fromarray(img).save(path); paths[name]=str(path)
    record={'observation':index,'eef_state':obs.state['eef_state'].tolist(),'images':paths,
            'steps':emb.num_steps,'terminated':terminated}
    with (out/'observations.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
try:
    emit(emb.reset(Scene(id='layout-0',instruction='Pick up the red block from the table and place it inside the bowl.',init_seed=0),seed=0))
    for line in sys.stdin:
        command=json.loads(line)
        if command.get('finish'): break
        if terminated: raise ValueError('Episode already terminated; finish required')
        target=np.asarray(command['target'],dtype=float)
        steps=int(command.get('steps',20))
        if target.shape!=(4,) or not np.isfinite(target).all(): raise ValueError('Invalid target')
        if not (np.all(target>=[.15,-.25,.005,0]) and np.all(target<=[.45,.25,.35,1])): raise ValueError('Outside action bounds')
        if not 1<=steps<=100 or len(actions)>=60 or emb.num_steps+steps>2500: raise ValueError('Budget exceeded')
        state=emb._eef_state().copy()
        actions.append(command)
        with (out/'actions.jsonl').open('a') as f: f.write(json.dumps(command)+'\n')
        for i in range(1,steps+1):
            pose=state[:3]+(target[:3]-state[:3])*(i/steps)
            result=emb.step(Action(data=np.array([*pose,target[3]])))
            terminated=bool(result.terminated)
            if terminated: break
        index+=1; emit(result.observation)
    summary={**metadata,'status':'success','success_at_end':int(terminated),
             'commands':len(actions),'steps':emb.num_steps,'wall_time_s':time.time()-start}
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'result':summary,'directory':str(out)}),flush=True)
except Exception:
    (out/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    emb.close()
