"""CPU1 conditionalfeedback value/gradient/storage-readiness gates, no Core."""
import argparse,copy,sys,time
from sources import ROOT,read,save,verify
from utils import module,tensor_digest,max_error,production_feedback,output_errors,torch_integrate
PARENT_ROOT='/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration'
PARENT_SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-preflight',action='store_true');a=p.parse_args()
 if not a.execute_parent_preflight:p.error('Parenttargetfeedbackpreflight required')
 if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
 start=time.monotonic()
 from pathlib import Path
 from learning import cpu1,rng_snapshot,rng_equal,restore_rng,adapter_anchor,forward,slice_data,objective,resource_gate,time_weights
 cpu1()
 import torch,numpy as np,random
 import common,model
 from scan import make_scan
 from data import prepare,production_sequence,dev_sources
 b=read(ROOT/'INPUT_BINDINGS.json');verify(b['source_hashes']);verify(b['data_hashes']);torch.manual_seed(20261114);np.random.seed(20261114);random.seed(20261114)
 setup=time.monotonic();m,pp=common.initialize();data=prepare(m);sc=make_scan(m);python_scan=make_scan(m,script=False);setup_seconds=time.monotonic()-setup;checks={};valueproof=[]
 parentio=module('_targetpreflight_parentstrict',Path(PARENT_ROOT)/'common.py');parent,_=parentio.load_checkpoint(Path(PARENT_ROOT)/'TRAIN/checkpoint_000600.pt',PARENT_SHA,final=True)
 checks['parent_deployment_allweights_exact']=max_error(parent.state_dict(),m.state_dict())==0
 initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot();olddigest=tensor_digest(m.old.state_dict())
 checks['twelve_sources_nine_targets_six_v1']=data['physical'].shape==(1920,12,112) and data['oldafter'].shape==(1920,12,256) and int(data['target_selector'].sum())==9 and int(data['v1'].sum())==6
 w=time_weights(1920,data['y'],data['valid']);checks['maskedseqmass1']=float((w.sum(0)-1).abs().max())<=1e-12 and bool((w[data['valid']==0]==0).all())
 # Torch reimplementation must reproduce the13actual NumPyvalidatedsource targets.
 accumulator=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/accumulator_replay_diagnostic');am=module('_targetpreflight_actual_accumulator_metadata',accumulator/'metadata.py');ameta=am.qualify();actual_errors=[]
 with torch.no_grad():
  for e in ameta['entries']:
   folder=Path(e['folder']);ident=am.identity_contract(am.read(folder/'IDENTITY.json'))
   with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z:t={k:z[k].copy() for k in ('target','qpos','time')}
   with np.load(e['sample_path'],allow_pickle=False) as z:
    key='Y42' if e['teacher'] and not e['collection'] else 'actually_applied_action';action=z[key][e['start']:].copy();X=z['X238'][e['start']:].copy();ms=np.rint(z['time'][e['start']:]*1000).astype(int)
   bounds=torch.tensor(ident['actuator']['target_bounds_rad'],dtype=torch.float64);rate=torch.tensor(ident['actuator']['max_rate_rad_s'],dtype=torch.float64)
   predicted=torch_integrate(torch.tensor(action),torch.tensor(t['target'][5799]),bounds,rate)[:len(t['target'])-5800];err=float((predicted-torch.tensor(t['target'][5800:])).abs().max());actual_errors.append(dict(id=e['id'],max_target_error=err))
   qa=ameta['mapping']['qpos_indices'];half=torch.tensor(ameta['mapping']['halfspan42'],dtype=torch.float64);q0=torch.tensor(ameta['mapping']['q0_42'],dtype=torch.float64);target=torch.tensor(t['target'][ms-1]);q=torch.tensor(t['qpos'][ms-1][:,qa]);error=((target-q)/half).clamp(-10,10).float().double();offset=((target-q0)/half).clamp(-10,10).float().double()
   checks[e['id']+'_torch_actualtarget_and_float32encoding']=err<=1e-12 and torch.equal(error,torch.tensor(X[:,42:84])) and torch.equal(offset,torch.tensor(X[:,126:168]))
 # All12+8deployment copies, with actual variablelength DEV and nofakeY.
 parity_start=time.monotonic()
 with torch.no_grad():
  for i,raw in enumerate(data['raw']+dev_sources(b)):
   nm,nh=production_sequence(m,raw);pm,ph=production_sequence(parent,raw);tag=raw['entry']['id']+str(i)
   checks[tag+'_unchanged_deployment_migration']=torch.equal(nm,pm) and torch.equal(nh,ph)
   valueproof.append(dict(id=raw['entry']['id'],rows=len(raw['x']),migration_mean_exact=torch.equal(nm,pm),migration_state_exact=torch.equal(nh,ph),training=i<12))
  for i,raw in enumerate(data['raw']):
   one=slice_data(data,i,i+1);forced=forward(sc,one,forced=True);actualmean,actualh=production_sequence(m,raw)
   checks[raw['entry']['id']+'_teacherforced_actualX']=float((forced[0][:,0]-actualmean).abs().max())<=1e-12 and float((forced[2][:,0]-actualh[:,256:]).abs().max())<=1e-12
   predicted=forward(sc,one);full,old=production_feedback(m,raw,data,i);errs=output_errors(tuple(v[:,0] for v in predicted),full)
   checks[raw['entry']['id']+'_selfpredtarget_fulloldproduction']=max(errs)<=1e-12 and torch.equal(old,data['oldafter'][:,i]);valueproof[i]['conditional_full_production_errors']=errs
  scripted=forward(sc,data);ordinary=forward(python_scan,data);checks['full1920_Python_JIT_all6outputs']=max(output_errors(scripted,ordinary))<=1e-12
  del scripted,ordinary,nm,nh,pm,ph,forced,predicted,full,old
 parity_seconds=time.monotonic()-parity_start
 # Preserve the previously qualified numericalactorinput contract, not autograd0.
 with torch.no_grad():
  x=data['x'][0,0:1].clone();u=data['u'][0,0:1];state=m.new_state_from_old(tuple(s[0:1].clone() for s in data['initial_old']));base,features,oldstate=m.cached_features(x,u,state[:3]);mean0,state0,res0=m.step_details(x,u,state)
  for lo,hi in ((42,84),(195,237)):
   xx=x.clone();xx[:,lo:hi]+=.03125;bb,ff,hh=m.cached_features(xx,u,state[:3]);mm,ss,rr=m.step_details(xx,u,state)
   checks['numeric_excluded_'+str(lo)+'_'+str(hi)]=torch.equal(base,bb) and torch.equal(features,ff) and torch.equal(mean0,mm) and torch.equal(res0,rr) and all(torch.equal(a,b) for a,b in zip(oldstate,hh))
  xx=x.clone();xx[:,126:168]+=.03125;bb,ff,hh=m.cached_features(xx,u,state[:3]);checks['numeric126_oldprior_present_old256_independent']=not torch.equal(base,bb) and all(torch.equal(a,b) for a,b in zip(oldstate,hh)) and torch.equal(ff[:,-42:],bb)
 # Explicit differentiable target-canary: no no_grad around predictedT/prior.
 small=slice_data(data,0,12,T=32);canary=small['initial_target'].clone().requires_grad_(True);small=dict(small,initial_target=canary);out=forward(sc,small);probe=out[0].square().mean()+out[4].square().mean();probe.backward();checks['target_feedback_gradient_canary']=canary.grad is not None and bool(torch.isfinite(canary.grad).all()) and float(canary.grad.abs().max())>0.;del out,probe;m.zero_grad(set_to_none=True)
 # Quantizedcast finite-difference is an approximate diagnostic, not strictrealfloatgrad.
 anchor=adapter_anchor(m);small=slice_data(data,0,12,T=32);out=forward(sc,small);loss,_=objective(out,small,m,anchor);loss.backward();fd=[];del out,loss
 for parameter,index in ((m.head.bias,(0,)),(m.encoder.weight,(0,0))):
  analytic=float(parameter.grad[index]);value=float(parameter[index]);epsilon=1e-4;vals=[]
  with torch.no_grad():
   for sign in (1.,-1.):
    parameter[index]=value+sign*epsilon;out=forward(sc,small);loss,_=objective(out,small,m,anchor);vals.append(float(loss));del out,loss
   parameter[index]=value
  numerical=(vals[0]-vals[1])/(2*epsilon);absolute=abs(numerical-analytic);relative=absolute/max(abs(numerical),abs(analytic),1e-12);fd.append(dict(analytic=analytic,numerical=numerical,absolute_error=absolute,relative_error=relative,passed=absolute<=1e-3 or relative<=.05))
 checks['FD2coords_quantized_approximation']=all(r['passed'] for r in fd);m.zero_grad(set_to_none=True)
 # FULL1920 B12 versus3completeB4, globaldenominators12/9/6.
 opt1=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);out=forward(sc,data);loss,_=objective(out,data,m,anchor);fullvalue=float(loss.detach());loss.backward();grads=[p.grad.detach().clone() for p in m.adapter_parameters()];del out,loss;opt1.step();fullstate={k:v.detach().clone() for k,v in m.state_dict().items()};fulladam=copy.deepcopy(opt1.state_dict());del opt1
 m.load_state_dict(initial,strict=True);opt=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt.zero_grad(set_to_none=True);microvalue=0.;timer=time.monotonic()
 for lo in (0,4,8):
  d=slice_data(data,lo,lo+4);out=forward(sc,d);loss,_=objective(out,d,m,anchor);loss.backward();microvalue+=float(loss.detach());del out,loss,d
 ge=max(float((p.grad-g).abs().max()) for p,g in zip(m.adapter_parameters(),grads));opt.step();update=time.monotonic()-timer;ae=max_error(fulladam,opt.state_dict());pe=max_error(fullstate,m.state_dict())
 checks['full1920_loss_full12_micro4x3']=abs(fullvalue-microvalue)<=1e-10;checks['full1920_all8gradient']=ge<=1e-10;checks['full1920_Adam_params']=ae<=1e-10 and pe<=1e-10
 checks['frozenoldweights_exact']=tensor_digest(m.old.state_dict())==olddigest and all(not p.requires_grad and p.grad is None for p in m.old.parameters())
 m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);restore_rng(rng);checks['initialweights_RNG_restored']=max_error(initial,m.state_dict())==0 and rng_equal(rng,rng_snapshot())
 checks['no_Core_physics_teacherbackend']=not any(k in sys.modules for k in ('mujoco','physics','flygym','static_teacher','student_wave'))
 allowance=3*(setup_seconds+parity_seconds)*1.3;estimated=600*update*1.3+allowance;timing=estimated<=5400;rss=resource_gate(start);verify(b['source_hashes']);verify(b['data_hashes'])
 save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),all_correctness_checks_passed=bool(all(checks.values())),checks=checks,timing_status='READY_FIXED600_5400S' if timing else 'BLOCKED_BEFORE_TRAINING_TIMING',actual13_target_errors=actual_errors,migration20_and_conditional12=valueproof,finite_difference32approximation=fd,full1920_loss_error=abs(fullvalue-microvalue),gradient_error=ge,Adam_error=ae,parameter_error=pe,actual_micro4x3_update_s=update,three_freshverify_allowance_s=allowance,predicted600_total_s_with_margin=estimated,peak_RSS_MiB=rss,source_hashes=b['source_hashes'],data_hashes=b['data_hashes'],training_updates=0,optimizer_probe_updates=2,Core_constructions=0,physical_steps=0,conditional_notphysicalcounterfactual=True))
 if not all(checks.values()):raise ValueError('Conditionaltargetfeedbackpreflightfailed')
 if not timing:raise ValueError('Timingblockedbeforetraining,noautoshrinkextension')
 print('TARGET_FEEDBACK_PREFLIGHT_READY_FIXED600_5400S',flush=True)


if __name__=='__main__':main()
