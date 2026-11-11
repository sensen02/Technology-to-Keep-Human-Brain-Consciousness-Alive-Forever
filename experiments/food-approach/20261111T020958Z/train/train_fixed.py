"""Fixed5000 nonlinear readout updates; frozen causal parent cached once."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,ROOT,PARENT,PARENT_SHA,load_parent,NonlinearMotor,save,load_checked,sha256,verify_hashes,exact
from data import prepare,cached,invariance

def main(args):
 torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 pre=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not pre['all_passed']:raise ValueError('preflight required')
 verify_hashes(pre['source_hashes']);verify_hashes(pre['data_hashes'])
 parent,_=load_parent();m=NonlinearMotor(parent);e,data=prepare(m)
 sources={str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','check_train.py','PREFLIGHT.json','verify_saved.py','train_fixed.py')};sources[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json')
 out=Path(args.out).resolve()
 if not out.is_relative_to(HERE):raise ValueError('train scoped output required')
 out.mkdir(parents=True,exist_ok=False);opt=torch.optim.Adam(m.adapter_parameters(),lr=.001)
 config=dict(kind='fixed_nonlinear_positive_readout5000',physical_eval_only_step=5000,seed=20261111,lr=.001,steps=5000,immediate_parent_checkpoint=str(PARENT),immediate_parent_sha256=PARENT_SHA,frozen_motor_parent_checkpoint=m.parent.identity()['parent_checkpoint'],frozen_motor_parent_sha256=m.parent.identity()['parent_sha256'],source_hashes=sources,data_hashes=data,prereg=json.loads((HERE/'PREREG.json').read_text()),new_parameters=22378,frozen_parent_parameters=197608)
 (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 started=time.monotonic()
 with torch.no_grad():initial=float((cached(m,e)[960:].tanh()-e['y'][960:]).square().mean())
 curve=[dict(step=0,loss_after_update=initial,wall_s=0.)]
 with (out/'TRAINING.jsonl').open('x') as stream:
  stream.write(json.dumps(curve[0])+'\n');stream.flush();print(json.dumps(curve[0]),flush=True)
  for step in range(1,5001):
   opt.zero_grad(set_to_none=True);mean=cached(m,e);loss=(mean[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   if any(q.requires_grad or q.grad is not None for q in m.parent.parameters()):raise ValueError('frozen parent violated')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started)
   if step%250==0:
    with torch.no_grad():row['loss_after_update']=float((cached(m,e)[960:].tanh()-e['y'][960:]).square().mean())
    print(json.dumps(row),flush=True)
   if step in (2500,5000):row['invariance']=invariance(m)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (2500,5000):
    verify_hashes(sources);verify_hashes(data);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,m,opt,config,curve,rng)
    restored,p=load_checked(ck,digest,require_final=False)
    if not exact(m.state_dict(),restored.state_dict()) or not exact(opt.state_dict(),p['optimizer']) or not exact(config,p['config']) or not exact(curve,p['curve']) or not exact(rng,p['rng']):raise ValueError('live saved fullstate differs')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,live_fullstate_exact=True,fresh_saved_checks=True)),flush=True)
 result=dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=5000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,initial_loss=initial,final=curve[-1],physical_steps=0,source_hashes=sources,data_hashes=data)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
