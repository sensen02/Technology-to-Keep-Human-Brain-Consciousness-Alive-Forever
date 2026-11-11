"""Conditional exact2000->fixed8000 with RH-before-action loss weighting only."""
import argparse,json,sys,time,subprocess
from pathlib import Path
from copy import deepcopy
import numpy as np,torch
from model_io import HERE,ROOT,OLD,PARENT,PARENT_SHA,FROZEN_PARENT,FROZEN_SHA,load_parent,save,load_checked,sha256,verify_hashes,exact
from data import prepare,cached,invariance,objective


def main(args):
 torch.set_num_threads(1)
 feasibility=json.loads((HERE/'FEASIBILITY.json').read_text());pre=json.loads((HERE/'PREFLIGHT.json').read_text())
 if not feasibility['supported'] or not pre['all_passed']:raise ValueError('predeclared feasibility/preflight gate rejects optimization')
 verify_hashes(pre['source_hashes']);verify_hashes(pre['data_hashes'])
 model,parent=load_parent();e,hashes,alignment=prepare(model)
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.003);opt.load_state_dict(parent['optimizer'])
 checks=dict(actual_model_exact=exact(model.state_dict(),parent['state_dict']),actual_optimizer_exact=exact(opt.state_dict(),parent['optimizer']),actual_two_states=len(opt.state)==2,actual_all_counters2000=all(float(s['step'])==2000 for s in opt.state.values()),parent_curve1_to2000=[r['step'] for r in parent['curve']]==list(range(1,2001)))
 if not all(checks.values()):raise ValueError('exact saved parent required')
 out=Path(args.out).resolve()
 if not out.is_relative_to(HERE):raise ValueError('scoped output required')
 out.mkdir(parents=True,exist_ok=False)
 sidecar=out/'LOSS_WEIGHTS.npz';np.savez_compressed(sidecar,weights=e['weights'].numpy(),RH_loaded=e['loaded_mask'].numpy(),before_indices=e['before_indices'])
 (out/'ALIGNMENT.json').write_text(json.dumps(alignment,indent=2)+'\n')
 sources=dict(parent['config']['source_hashes']);sources.update({str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','feasibility.py','FEASIBILITY.json','check_train.py','PREFLIGHT.json','train_weighted.py','verify_saved.py')})
 sources[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json');correction=OLD/'PARENT_METADATA_CORRECTION.json';sources[str(correction)]=sha256(correction)
 hashes.update({str(PARENT):PARENT_SHA,str(FROZEN_PARENT):FROZEN_SHA,str(sidecar):sha256(sidecar),str(out/'ALIGNMENT.json'):sha256(out/'ALIGNMENT.json')})
 curve=deepcopy(parent['curve']);oldopt=deepcopy(opt.state_dict());opt.param_groups[0]['lr']=.001;expected=deepcopy(oldopt);expected['param_groups'][0]['lr']=.001
 checks['sole_optimizer_lr_change']=exact(expected,opt.state_dict());checks['parent_fullcurve_exact']=exact(curve,parent['curve'])
 torch.set_rng_state(parent['rng']['torch']);np.random.set_state(parent['rng']['numpy']);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());checks['actual_rng_exact']=exact(rng,parent['rng'])
 if not all(checks.values()):raise ValueError('resume state differs')
 config=dict(kind='fixed_RHstance_weighted_readout8000',immediate_parent_checkpoint=str(PARENT),immediate_parent_sha256=PARENT_SHA,frozen_motor_parent_checkpoint=str(FROZEN_PARENT),frozen_motor_parent_sha256=FROZEN_SHA,physical_eval_only_step=8000,start_step=2000,end_step=8000,additional_updates=6000,lr=.001,loss='sum(before_action_RHload>=.1W_RH7_weight4_else1 * boundedactionerror²)/sum(actualpostweights)',weight_sidecar=str(sidecar),weight_sidecar_sha256=sha256(sidecar),alignment=alignment,feasibility_supported=True,feasibility_sha256=sha256(HERE/'FEASIBILITY.json'),source_hashes=sources,data_hashes=hashes,prereg=json.loads((HERE/'PREREG.json').read_text()),parent_metadata_correction=str(correction),parent_metadata_correction_sha256=sha256(correction),limitations='Lossandlrchange, noequalbudgetuniformcontrol, weightedteacherfitnotphysicalsuccess; inheritedcurve1..2000nostep0invented; oldmetadatawrongancestorpathretainedexternallycorrected.')
 torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),curve=curve,rng=rng,config=config),out/'RESUME_STATE.pt')
 with torch.no_grad():initial_prediction=cached(model,e);initial=dict(uniform=float(objective(initial_prediction,e,False)),weighted=float(objective(initial_prediction,e)))
 (out/'RESUME_CHECKS.json').write_text(json.dumps(dict(all_passed=all(checks.values()),checks=checks,old_lr=.003,new_lr=.001,initial_new_objective_metrics=initial,resume_state_sha256=sha256(out/'RESUME_STATE.pt'),source_hashes=sources),indent=2)+'\n')
 (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
 started=time.monotonic();print(json.dumps(dict(start_step=2000,initial=initial,feasibility_supported=True)),flush=True)
 with (out/'TRAINING_ADDITIONAL.jsonl').open('x') as stream:
  for step in range(2001,8001):
   if time.monotonic()-started>14400:raise TimeoutError('fixed4h cap')
   opt.zero_grad(set_to_none=True);pred=cached(model,e);loss=objective(pred,e);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
   if any(p.requires_grad or p.grad is not None for p in model.parent.parameters()):raise ValueError('frozen parent changed')
   opt.step();row=dict(step=step,weighted_loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-started,weighted_continuation_stage=True)
   if step%1000==0:
    with torch.no_grad():after=cached(model,e);row.update(weighted_postmetric=float(objective(after,e)),uniform_postmetric=float(objective(after,e,False)))
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step%1000==0:print(json.dumps(row),flush=True)
   if step in (5000,8000):
    verify_hashes(sources);verify_hashes(hashes);rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    row['invariance']=invariance(model)
    if not all(row['invariance'].values()):raise ValueError('neutral/right invariance failed')
    ck=out/f'checkpoint_{step:06d}.pt';digest=save(ck,model,opt,config,curve,rng);fresh,payload=load_checked(ck,digest,require_final=False)
    if not all((exact(model.state_dict(),fresh.state_dict()),exact(opt.state_dict(),payload['optimizer']),exact(rng,payload['rng']),exact(curve,payload['curve']),exact(config,payload['config']))):raise ValueError('live-before-save/full-state reload differs')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--sha256',digest,'--out',str(out/f'VERIFY_{step:06d}.json')],check=True)
    print(json.dumps(dict(step=step,checkpoint=str(ck),sha256=digest,independent_saved_pass=True)),flush=True)
 result=dict(status='WEIGHTED_TRAINED_NOT_PHYSICALLY_EVALUATED',steps=8000,additional_updates=6000,checkpoint=str(ck),sha256=digest,wall_s=time.monotonic()-started,initial_metrics=initial,final=curve[-1],feasibility_supported=True,physics_integration_steps=0,source_hashes=sources,data_hashes=hashes)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args())
