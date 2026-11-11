"""Fixed1000 fast adapter training. Cache only frozen neural history, never physics."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from fast_motor import HERE,FastMotor,load_base,save,load,sha256,verify_hashes
from fast_data import prepare,cached,neutral,metrics
from check_fast import equal

SOURCE_NAMES=('PREREG.json','fast_motor.py','fast_data.py','check_fast.py','FAST_CHECKS.json','train_fast.py','verify_fast_saved.py','evaluate_fast.py','verify_fast_actual.py','EVAL_PREREG.json')


def main(args):
    torch.set_num_threads(1);torch.manual_seed(20261110);np.random.seed(20261110)
    checks=json.loads((HERE/'FAST_CHECKS.json').read_text())
    if not checks['all_passed']:raise ValueError('fast offline checks required')
    verify_hashes(checks['source_hashes']);verify_hashes(checks['data_hashes'])
    sources={str(HERE/n):sha256(HERE/n) for n in SOURCE_NAMES}
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();model=FastMotor(load_base());episodes,data=prepare(model)
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.001)
    config=dict(kind='fast_residual_fixed1000',steps=1000,physical_eval_only_step=1000,seed=20261110,lr=.001,sources=sources,data_hashes=data,
                loss='equal4 actual teacher postprime bounded action MSE; frozen sequentialbasehidden cached; no addedrecurrence',
                architectures=model.identity(),prereg=json.loads((HERE/'PREREG.json').read_text()))
    (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
    curve=[dict(step=0,postprime_MSE=metrics(model,episodes),neutral=neutral(model),wall_s=time.monotonic()-started)]
    with (out/'TRAINING.jsonl').open('x') as stream:
        def record(row):
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
        record(curve[0])
        for step in range(1,1001):
            opt.zero_grad(set_to_none=True);losses={}
            for e in episodes:
                prediction=cached(model,e)
                loss=(prediction[960:].tanh()-e['y'][960:]).square().mean()/len(episodes)
                loss.backward();losses[e['name']]=float(loss.detach())*len(episodes)
            grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
            if any(p.requires_grad or p.grad is not None for p in model.base.parameters()):raise ValueError('base freeze violated')
            opt.step();row=dict(step=step,postprime_MSE=losses,gradient_norm=float(grad),wall_s=time.monotonic()-started)
            if step%100==0:
                row['metrics_after_update']=metrics(model,episodes);row['neutral']=neutral(model)
            curve.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if step%50==0:print(json.dumps(row),flush=True)
            if step%100==0:
                verify_hashes(sources);verify_hashes(data)
                rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
                ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng)
                restored,p=load(ck,digest)
                if not equal(model.state_dict(),restored.state_dict()) or not equal(opt.state_dict(),p['optimizer']) or not equal(rng,p['rng']) or not equal(curve,p['curve']):raise ValueError('saved exactness failed')
                # Constructor-based reload checks must not perturb the live training RNG.
                torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
                subprocess.run([sys.executable,'-B',str(HERE/'verify_fast_saved.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True,stdout=subprocess.DEVNULL)
                print(json.dumps(dict(checkpoint=str(ck),sha256=digest,step=step,saved_state_independent_pass=True)),flush=True)
    result=dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=1000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,final=curve[-1],sources=sources,data_hashes=data)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
