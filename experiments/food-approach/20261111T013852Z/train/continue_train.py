"""Preflight then fixed2000->10000, cached2730positive readout only."""
import argparse,json,sys,time,subprocess
from pathlib import Path
from copy import deepcopy
import numpy as np,torch
from model_io import HERE,OLD,PARENT,PARENT_SHA,FROZEN_PARENT,FROZEN_SHA,load_parent,save,load_checked,sha256,verify_hashes,exact
from data import prepare,cached,invariance

def main(a):
 torch.set_num_threads(1);model,parent=load_parent();e,hashes=prepare(model)
 if hashes!=parent['config']['data_hashes']:raise ValueError('original teacher data identity differs')
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.003);opt.load_state_dict(parent['optimizer'])
 checks=dict(model_exact=exact(model.state_dict(),parent['state_dict']),optimizer_exact=exact(opt.state_dict(),parent['optimizer']),twoAdamstates=len(opt.state)==2,allcounters2000=all(float(s['step'])==2000 for s in opt.state.values()))
 with torch.no_grad():prod,states=model.sequence(e['x'][:,None],e['u'][:,None]);prediction=cached(model,e)
 checks['actual2880cache_production_tol1e12']=float((prod[:,0]-prediction).abs().max())<=1e-12
 checks.update(invariance(model))
 if not all(checks.values()):raise ValueError('preflight failed '+json.dumps(checks))
 sources=dict(parent['config']['source_hashes']);sources.update({str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','continue_train.py','verify_saved.py')})
 correction=OLD/'PARENT_METADATA_CORRECTION.json';sources[str(correction)]=sha256(correction)
 hashes=dict(hashes);hashes[str(PARENT)]=PARENT_SHA;hashes[str(FROZEN_PARENT)]=FROZEN_SHA
 curve=deepcopy(parent['curve'])
 config=dict(kind='exact_positive_readout10000',immediate_parent_checkpoint=str(PARENT),immediate_parent_sha256=PARENT_SHA,frozen_motor_parent_checkpoint=str(FROZEN_PARENT),frozen_motor_parent_sha256=FROZEN_SHA,parent_metadata_correction=str(correction),parent_metadata_correction_sha256=sha256(correction),source_hashes=sources,data_hashes=hashes,physical_eval_only_step=10000,start_step=2000,end_step=10000,additional_updates=8000,lr=.001,prereg=json.loads((HERE/'PREREG.json').read_text()),ancestral_metadata_warning='Oldconfig.parent_checkpoint wrongancestor; actualidentity/frozenweights remainauthoritative. Immediateheadparent andfrozenmotorparent recordedseparately here.')
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
 before=deepcopy(opt.state_dict());opt.param_groups[0]['lr']=.001;expected=deepcopy(before);expected['param_groups'][0]['lr']=.001
 checks['only_lr_changed']=exact(expected,opt.state_dict());checks['parent_curve_exact']=exact(curve,parent['curve'])
 torch.set_rng_state(parent['rng']['torch']);np.random.set_state(parent['rng']['numpy']);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());checks['rng_exact']=exact(rng,parent['rng'])
 if not all(checks.values()):raise ValueError('resume failed')
 torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),curve=curve,rng=rng,config=config),out/'RESUME_STATE.pt')
 initial=float((prediction[960:].tanh()-e['y'][960:]).square().mean())
 report=dict(all_passed=True,checks=checks,initial_actual2000_postmetric=initial,resume_state_sha256=sha256(out/'RESUME_STATE.pt'),source_hashes=sources,data_hashes=hashes)
 (out/'RESUME_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');(out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 started=time.monotonic();print(json.dumps(dict(start_step=2000,initial_postmetric=initial,preflight_pass=True)),flush=True)
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as log:
  for step in range(2001,10001):
   if time.monotonic()-started>14400:raise TimeoutError('4h fixed wallcap')
   opt.zero_grad(set_to_none=True);mean=cached(model,e);loss=(mean[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
   grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   if any(p.grad is not None or p.requires_grad for p in model.parent.parameters()):raise ValueError('frozen parent changed')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started,continuation_stage=True);curve.append(row)
   log.write(json.dumps(row)+'\n');log.flush()
   if step%1000==0:print(json.dumps(row),flush=True)
   if step in (4000,7000,10000):
    verify_hashes(sources);verify_hashes(hashes);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    with torch.no_grad():row['postmetric']=float((cached(model,e)[960:].tanh()-e['y'][960:]).square().mean())
    ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng);fresh,p=load_checked(ck,digest,require_final=False)
    if not all((exact(model.state_dict(),fresh.state_dict()),exact(opt.state_dict(),p['optimizer']),exact(rng,p['rng']),exact(config,p['config']),exact(curve,p['curve']))):raise ValueError('live saved mismatch')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,independent_saved_pass=True)),flush=True)
 result=dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=10000,additional_updates=8000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,initial_postmetric=initial,final=curve[-1],physical_steps=0)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
