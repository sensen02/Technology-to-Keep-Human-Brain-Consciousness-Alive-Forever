"""Exact3000->5000 corrective-label dataset intervention; unsafe flags immutable."""
from pathlib import Path
import argparse,json,sys,time,subprocess
from copy import deepcopy
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,save,sha256,verify_hashes
from fast_data import cached,neutral,metrics
from check_fast import equal
from data5000 import prepare,PARENT_SHA
PARENT=ROOT/'continuation3000/TRAIN/checkpoint_003000.pt'
SOURCE_NAMES=['PREREG.json','data5000.py','check5000.py','PREFLIGHT.json','train5000.py','verify_saved5000.py']

def main(args):
 torch.set_num_threads(1)
 checks=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not checks['all_passed']:raise ValueError('corrective label/cachedobjective preflight required')
 verify_hashes(checks['source_hashes']);verify_hashes(checks['data_hashes'])
 m,p=load(PARENT,PARENT_SHA)
 if p['curve'][-1]['step']!=3000 or len(p['curve'])!=3001:raise ValueError('exact parent3000 required')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 if not(equal(m.state_dict(),p['model']) and equal(opt.state_dict(),p['optimizer']) and len(opt.state)==6 and all(float(s['step'])==3000 for s in opt.state.values())):raise ValueError('exact model/optimizer3000 resume failed')
 if len(opt.param_groups)!=1 or opt.param_groups[0]['lr']!=.0001:raise ValueError('parentlr mustnotchange')
 episodes,data,admission=prepare(m)
 if data!=checks['data_hashes']:raise ValueError('prepared data differs from checked corrective evidence')
 if not all(data[q]==v for q,v in p['config']['data_hashes'].items()):raise ValueError('old data modified or excluded')
 sources={**p['config']['sources'],**{str(HERE/n):sha256(HERE/n) for n in SOURCE_NAMES}}
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 curve=deepcopy(p['curve']);config=deepcopy(p['config'])
 config.update(kind='fast_residual_exact3000_to5000_failedstate_corrective_supervision',steps=5000,physical_eval_only_step=5000,lr=.0001,sources=sources,data_hashes=data,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_steps=3000,additional_steps=2000,corrective_prereg=json.loads((HERE/'PREREG.json').read_text()),corrective_examples=admission,loss='equal6 trajectory postprime boundedactionMSE averages; old4x1920 + corrective2x960; correctiveweight1/3; actualfailedmixedobs validteacherlabels only')
 initial=metrics(m,episodes)
 if {k:initial[k] for k in p['curve'][-1]['metrics_after_update']}!=p['curve'][-1]['metrics_after_update']:raise ValueError('old4 resumed predictions changed')
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
 if not equal(rng,p['rng']):raise ValueError('RNG resume mismatch')
 resume=dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,config=config)
 torch.save(resume,out/'RESUME_STATE.pt')
 report=dict(model_exact=equal(m.state_dict(),p['model']),optimizer_exact=equal(opt.state_dict(),p['optimizer']),rng_exact=equal(rng,p['rng']),curve_prefix_exact=equal(curve,p['curve']),lr_unchanged=.0001,counters=[float(s['step']) for s in opt.state.values()],parent_sha256=PARENT_SHA,initial_new_sixway_metrics=initial,loss_weight_each=1/6,corrective_total_weight=1/3,original_safety_flags_still_false=True,resume_state_sha256=sha256(out/'RESUME_STATE.pt'))
 (out/'RESUME_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');(out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 print(json.dumps(dict(resumed3000=True,initial_new_sixway_metrics=initial,corrective_examples=admission)),flush=True)
 started=time.monotonic()
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as stream:
  for step in range(3001,5001):
   if time.monotonic()-started>4*3600:raise TimeoutError('fixed4h wallcap')
   opt.zero_grad(set_to_none=True);losses={}
   for e in episodes:
    pred=cached(m,e);loss=(pred[960:].tanh()-e['y'][960:]).square().mean()/6
    loss.backward();losses[e['name']]=float(loss.detach())*6
   grad=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinitegradient')
   if any(q.requires_grad or q.grad is not None for q in m.base.parameters()):raise ValueError('basefreezeviolated')
   opt.step();row=dict(step=step,postprime_MSE=losses,gradient_norm=float(grad),wall_s=time.monotonic()-started,corrective_dataset_stage=True)
   if step%1000==0:row.update(metrics_after_update=metrics(m,episodes),neutral=neutral(m))
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%200==0:print(json.dumps(row),flush=True)
   if step%1000==0:
    verify_hashes(sources);verify_hashes(data)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,m,opt,config,curve,rng)
    mm,pp=load(ck,digest)
    if not(equal(mm.state_dict(),m.state_dict()) and equal(pp['optimizer'],opt.state_dict()) and equal(pp['rng'],rng) and equal(pp['curve'],curve)):raise ValueError('savedstateexactnessfailed')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved5000.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,savedindependentPASS=True)),flush=True)
 result=dict(status='CORRECTIVE_TRAINED_NOT_PHYSICALLY_EVALUATED',steps=5000,additional_steps=2000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,final=curve[-1],corrective_safety_flags_remain_false=True,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,sources=sources,data_hashes=data)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args())
