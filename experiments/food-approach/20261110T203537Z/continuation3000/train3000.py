"""Exact fixed1000->3000 lower-lr continuation; old source/data immutable."""
from pathlib import Path
import argparse,json,sys,time,subprocess
from copy import deepcopy
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,save,sha256,verify_hashes
from fast_data import prepare,cached,metrics,neutral
from check_fast import equal
PARENT=ROOT/'TRAIN_FAST_1000/checkpoint_001000.pt'
PARENT_SHA='3dd80394239504ec8b88f4cfe07c79569cf08329a9320c749c3c7b25035fda28'
SOURCE_NAMES=['PREREG.json','train3000.py','verify3000.py','check_resume.py','PREFLIGHT.json']


def main(args):
 torch.set_num_threads(1)
 m,p=load(PARENT,PARENT_SHA)
 if p['curve'][-1]['step']!=1000 or len(p['curve'])!=1001:raise ValueError('requires exact parent1000')
 verify_hashes(p['config']['sources']);verify_hashes(p['config']['data_hashes'])
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 model_exact=equal(m.state_dict(),p['model']);optimizer_exact=equal(opt.state_dict(),p['optimizer'])
 if not(model_exact and optimizer_exact) or len(opt.state)!=6 or not all(float(s['step'])==1000 for s in opt.state.values()):raise ValueError('exact model/Adam1000 resume required')
 if len(opt.param_groups)!=1 or opt.param_groups[0]['lr']!=.001:raise ValueError('parent lr/group drift')
 episodes,data=prepare(m)
 if data!=p['config']['data_hashes']:raise ValueError('dataset identity differs from parent')
 sources={str(HERE/n):sha256(HERE/n) for n in SOURCE_NAMES}
 sources.update(p['config']['sources'])
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 curve=deepcopy(p['curve'])
 config=deepcopy(p['config']);config.update(kind='fast_residual_exact1000_to3000_lowerlr',steps=3000,physical_eval_only_step=3000,lr=.0001,sources=sources,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,continuation_prereg=json.loads((HERE/'PREREG.json').read_text()),parent_steps=1000,additional_steps=2000)
 opt.param_groups[0]['lr']=.0001
 after=deepcopy(opt.state_dict());old=deepcopy(p['optimizer']);old['param_groups'][0]['lr']=.0001
 if not equal(after,old):raise ValueError('more than lr changed at resume')
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 rng_exact=equal(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 if not rng_exact:raise ValueError('RNG resume failed')
 resumed=metrics(m,episodes)
 if resumed!=curve[-1]['metrics_after_update']:raise ValueError('resumed predictions differ parent final')
 handover=dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,model_exact=model_exact,optimizer_exact_before_lr=optimizer_exact,only_lr_changed=equal(after,old),rng_exact=rng_exact,curve_prefix_exact=equal(curve,p['curve']),old_lr=.001,new_lr=.0001,all_counters=[float(s['step']) for s in opt.state.values()],metrics_exact=bool(resumed==curve[-1]['metrics_after_update']),sources=sources,data_hashes=data)
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),curve=curve,config=config),out/'RESUME_STATE.pt')
 handover['resume_state_sha256']=sha256(out/'RESUME_STATE.pt')
 (out/'RESUME_CHECKS.json').write_text(json.dumps(handover,indent=2)+'\n')
 (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 started=time.monotonic()
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as stream:
  for step in range(1001,3001):
   if time.monotonic()-started>4*3600:raise TimeoutError('fixed4h budget exceeded')
   opt.zero_grad(set_to_none=True);losses={}
   for e in episodes:
    pred=cached(m,e);loss=(pred[960:].tanh()-e['y'][960:]).square().mean()/len(episodes)
    loss.backward();losses[e['name']]=float(loss.detach())*len(episodes)
   grad=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   if any(q.requires_grad or q.grad is not None for q in m.base.parameters()):raise ValueError('base freeze changed')
   opt.step();row=dict(step=step,postprime_MSE=losses,gradient_norm=float(grad),wall_s=time.monotonic()-started,continuation_stage=True)
   if step%500==0:row.update(metrics_after_update=metrics(m,episodes),neutral=neutral(m))
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%100==0:print(json.dumps(row),flush=True)
   if step%500==0:
    verify_hashes(sources);verify_hashes(data)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,m,opt,config,curve,rng)
    mm,pp=load(ck,digest)
    if not equal(mm.state_dict(),m.state_dict()) or not equal(pp['optimizer'],opt.state_dict()) or not equal(pp['curve'],curve) or not equal(pp['rng'],rng):raise ValueError('saved state exactness failed')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify3000.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,independent_saved_pass=True)),flush=True)
 result=dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=3000,additional_steps=2000,checkpoint=str(ck),sha256=digest,final=curve[-1],wall_s=time.monotonic()-started,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,resume_checks=str(out/'RESUME_CHECKS.json'),sources=sources,data_hashes=data)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args())
