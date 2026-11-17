"""CPU1 newstatepairing qualification; fullB12 reference3072MiB, training1800."""
import argparse,copy,time,resource,sys
from sources import ROOT,read,save,verify

def resource_gate(start):
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
 if rss>3072:raise RuntimeError('Statealignedreferencepreflight RSS3072MiBexceeded')
 if time.monotonic()-start>7200:raise RuntimeError('Preflightwall7200exceeded,noextension')
 return rss

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-state-preflight',action='store_true');a=p.parse_args()
 if not a.execute_parent_state_preflight:p.error('Explicitstatealignedpreflightrequired')
 if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
 start=time.monotonic()
 from learning import cpu1,rng_snapshot,rng_equal,restore_rng,tf_forward,forward,slice_data,tf_objective,conditional_objective
 cpu1()
 import torch
 import common
 from data import prepare,production_sequence,dev_sources
 from scan import make_scan,make_actual_scan
 from utils import production_feedback,output_errors,max_error,tensor_digest
 b=read(ROOT/'INPUT_BINDINGS.json');verify(b['source_hashes']);verify(b['data_hashes']);setup=time.monotonic();m,parent=common.initialize();oldm,oldpayload=common.load_resume_parent();parent_adam=copy.deepcopy(parent['optimizer_state_dict']);data=prepare(m);condscan=make_scan(m);tfscan=make_actual_scan(m);condpython=make_scan(m,script=False);tfpython=make_actual_scan(m,script=False);setup_seconds=time.monotonic()-setup
 initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot();oldhash=tensor_digest(m.old.state_dict());anchor=parent['config']['initial_anchor'];checks={'strictactual600':parent['step']==600,'allparentweights_exact':max_error(initial,oldm.state_dict())==0,'actual21773labels9target6prior':int(data['valid'].sum())==21773 and int(data['target_selector'].sum())==9 and int(data['v1'].sum())==6};rows=[];timer=time.monotonic()
 with torch.no_grad():
  for i,raw in enumerate(data['raw']+dev_sources(b)):
   actual,h=production_sequence(m,raw);previous,ph=production_sequence(oldm,raw);checks[raw['entry']['id']+str(i)+'_deployedmigrationexact']=torch.equal(actual,previous) and torch.equal(h,ph)
   if i<12:
    one=slice_data(data,i,i+1);tf=tf_forward(tfscan,one);cond=forward(condscan,one);full,old=production_feedback(m,raw,data,i);errs=output_errors(tuple(v[:,0] for v in cond),full)
    checks[raw['entry']['id']+'_TF_publicproduction']=float((tf[0][:,0]-actual).abs().max())<=1e-12 and float((tf[2][:,0]-h[:,256:]).abs().max())<=1e-12 and torch.equal(h[:,:256],data['oldafter'][:,i])
    checks[raw['entry']['id']+'_conditionalfullproduction']=max(errs)<=1e-12 and torch.equal(old,data['oldafter'][:,i]);rows.append(dict(id=raw['entry']['id'],conditional_errors=errs))
   resource_gate(start)
  tf=tf_forward(tfscan,data);cond=forward(condscan,data);ptf=tf_forward(tfpython,data);pcond=forward(condpython,data)
  checks['PythonJITfull1920bothbranches']=max(output_errors(tf,ptf)+output_errors(cond,pcond))<=1e-12
  for i,raw in enumerate(data['raw']):
   actual,h=production_sequence(m,raw);full,old=production_feedback(m,raw,data,i);checks[raw['entry']['id']+'_fullB12bothbranches']=float((tf[0][:,i]-actual).abs().max())<=1e-12 and float((tf[2][:,i]-h[:,256:]).abs().max())<=1e-12 and max(output_errors(tuple(v[:,i] for v in cond),full))<=1e-12 and torch.equal(old,data['oldafter'][:,i])
  del tf,cond,ptf,pcond
  for lo in (0,4,8):
   part=slice_data(data,lo,lo+4);tf=tf_forward(tfscan,part);cond=forward(condscan,part)
   for j in range(4):
    i=lo+j;raw=data['raw'][i];actual,h=production_sequence(m,raw);full,old=production_feedback(m,raw,data,i);checks[raw['entry']['id']+'_micro4bothbranches']=float((tf[0][:,j]-actual).abs().max())<=1e-12 and float((tf[2][:,j]-h[:,256:]).abs().max())<=1e-12 and max(output_errors(tuple(v[:,j] for v in cond),full))<=1e-12 and torch.equal(old,data['oldafter'][:,i])
   del tf,cond,part
  del actual,h,previous,ph,full,old
 parity_seconds=time.monotonic()-timer
 # Frozen integrator unchanged: reproduce13source targets andfloat32encoding.
 from pathlib import Path
 import numpy as np
 from utils import module,torch_integrate
 am=module('_statealigned_accumulator_metadata',Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/accumulator_replay_diagnostic/metadata.py'));ameta=am.qualify()
 with torch.no_grad():
  for entry in ameta['entries']:
   folder=Path(entry['folder']);ident=am.identity_contract(am.read(folder/'IDENTITY.json'))
   with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z:targets=z['target'].copy();qpos=z['qpos'].copy()
   with np.load(entry['sample_path'],allow_pickle=False) as z:
    k='Y42' if entry['teacher'] and not entry['collection'] else 'actually_applied_action';actions=z[k][entry['start']:].copy();X=z['X238'][entry['start']:].copy();ms=np.rint(z['time'][entry['start']:]*1000).astype(int)
   bounds=torch.tensor(ident['actuator']['target_bounds_rad'],dtype=torch.float64);rate=torch.tensor(ident['actuator']['max_rate_rad_s'],dtype=torch.float64);pred=torch_integrate(torch.tensor(actions),torch.tensor(targets[5799]),bounds,rate)[:len(targets)-5800];target=torch.tensor(targets[ms-1]);q=torch.tensor(qpos[ms-1][:,ameta['mapping']['qpos_indices']]);half=torch.tensor(ameta['mapping']['halfspan42'],dtype=torch.float64);q0=torch.tensor(ameta['mapping']['q0_42'],dtype=torch.float64)
   checks[entry['id']+'_13actualtarget_encoding']=float((pred-torch.tensor(targets[5800:])).abs().max())<=1e-12 and torch.equal(((target-q)/half).clamp(-10,10).float().double(),torch.tensor(X[:,42:84])) and torch.equal(((target-q0)/half).clamp(-10,10).float().double(),torch.tensor(X[:,126:168]))
 small=slice_data(data,0,12,T=32)
 with torch.no_grad():
  tf=tf_forward(tfscan,small);cond=forward(condscan,small);tl,_=tf_objective(tf,small);cl,_=conditional_objective(cond,small,m,anchor);changed=dict(small,y=small['y']+.1);tl2,_=tf_objective(tf,changed);cl2,_=conditional_objective(cond,changed,m,anchor)
  perturbed=(cond[0]+.2,cond[1],cond[2],cond[3],cond[4],cond[5]+.2);cl3,_=conditional_objective(perturbed,small,m,anchor);tl3,_=tf_objective(tf,small)
  checks['action_labels_supervise_TF_only']=float(tl)!=float(tl2) and torch.equal(cl,cl2) and torch.equal(cl,cl3) and torch.equal(tl,tl3)
  changedtf=(tf[0]+.2,tf[1],tf[2],(tf[0]+.2).tanh());tl4,_=tf_objective(changedtf,small);checks['TFperturb_changes_actionloss']=float(tl4)!=float(tl)
  checks['assisted3_no_targetloss']=int(small['target_selector'].sum())==9 and all(float(small['target_selector'][i])==0 for i,e in enumerate(data['raw']) if e['entry'].get('collection') and e['entry']['teacher'])
  del tf,cond,tl,cl,tl2,cl2,cl3,tl3,tl4
 initialT=small['initial_target'].clone().requires_grad_(True);canarydata=dict(small,initial_target=initialT);out=forward(condscan,canarydata);probe=out[0].square().mean()+out[4].square().mean();probe.backward();checks['conditional_target_gradient_nonzero']=initialT.grad is not None and bool(torch.isfinite(initialT.grad).all()) and float(initialT.grad.abs().max())>0.;del out,probe;m.zero_grad(set_to_none=True)
 # Finitedifference only32tickdiagnostic, complete1920reference below remainsrequired.
 tf=tf_forward(tfscan,small);loss,_=tf_objective(tf,small);loss.backward();del tf,loss
 cond=forward(condscan,small);loss,_=conditional_objective(cond,small,m,anchor);loss.backward();del cond,loss;fd=[]
 for parameter,index in ((m.head.bias,(0,)),(m.encoder.weight,(0,0))):
  analytical=float(parameter.grad[index]);value=float(parameter[index]);values=[];eps=1e-4
  with torch.no_grad():
   for sign in (1.,-1.):
    parameter[index]=value+sign*eps;tf=tf_forward(tfscan,small);loss,_=tf_objective(tf,small);v=float(loss);del tf,loss;cond=forward(condscan,small);loss,_=conditional_objective(cond,small,m,anchor);values.append(v+float(loss));del cond,loss
   parameter[index]=value
  numerical=(values[0]-values[1])/(2*eps);ae=abs(numerical-analytical);re=ae/max(abs(numerical),abs(analytical),1e-12);fd.append(dict(analytic=analytical,numerical=numerical,absolute_error=ae,relative_error=re,passed=ae<=1e-3 or re<=.05))
 checks['FD2quantized_samegate']=all(x['passed'] for x in fd);m.zero_grad(set_to_none=True)
 # Branchsequential fullB12reference, releasing eachgraph before nextbranch.
 opt1=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt1.load_state_dict(copy.deepcopy(parent_adam));tf=tf_forward(tfscan,data);loss,_=tf_objective(tf,data);reference=float(loss.detach());loss.backward();del tf,loss
 cond=forward(condscan,data);loss,_=conditional_objective(cond,data,m,anchor);reference+=float(loss.detach());loss.backward();del cond,loss;grad=[p.grad.detach().clone() for p in m.adapter_parameters()];opt1.step();fullstate={k:v.detach().clone() for k,v in m.state_dict().items()};fulladam=copy.deepcopy(opt1.state_dict());del opt1
 m.load_state_dict(initial,strict=True);opt=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt.load_state_dict(copy.deepcopy(parent_adam));opt.zero_grad(set_to_none=True);timer=time.monotonic();microvalue=0.
 for lo in (0,4,8):
  part=slice_data(data,lo,lo+4);tf=tf_forward(tfscan,part);loss,_=tf_objective(tf,part);microvalue+=float(loss.detach());loss.backward();del tf,loss
  cond=forward(condscan,part);loss,_=conditional_objective(cond,part,m,anchor);microvalue+=float(loss.detach());loss.backward();del cond,loss,part
 ge=max(float((p.grad-g).abs().max()) for p,g in zip(m.adapter_parameters(),grad));opt.step();update=time.monotonic()-timer;ae=max_error(fulladam,opt.state_dict());pe=max_error(fullstate,m.state_dict())
 checks['full1920_branchseq_reference_micro_loss']=abs(reference-microvalue)<=1e-10;checks['all8gradients']=ge<=1e-10;checks['Adam_and_parameters']=ae<=1e-10 and pe<=1e-10
 checks['oldweights_frozen']=tensor_digest(m.old.state_dict())==oldhash and all(p.grad is None and not p.requires_grad for p in m.old.parameters())
 m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);opt.load_state_dict(copy.deepcopy(parent_adam));restore_rng(rng);checks['parent600weights_Adam_RNG_unpolluted']=max_error(initial,m.state_dict())==0 and max_error(opt.state_dict(),parent_adam)==0 and max_error(parent['optimizer_state_dict'],parent_adam)==0 and rng_equal(rng,rng_snapshot())
 checks['no_Core_physics_teacherbackend']=not any(n in sys.modules for n in ('physics','mujoco','flygym','static_teacher','student_wave'))
 allowance=3*(setup_seconds+parity_seconds)*1.3;evaluation_allowance=(setup_seconds+parity_seconds)*1.3;estimate=300*update*1.3+allowance+evaluation_allowance;timing=estimate<=7200;rss=resource_gate(start);verify(b['source_hashes']);verify(b['data_hashes'])
 save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),checks=checks,timing_status='READY_STATE_ALIGNED600_TO900' if timing else 'BLOCKED_BEFORE_TRAINING_TIMING',migration20_TF12_conditional12=rows,FD=fd,full1920_loss_error=abs(reference-microvalue),gradient_error=ge,Adam_error=ae,parameter_error=pe,actual_two_branch_micro4x3_update_s=update,predicted300total_withverification_s=estimate,three_fresh_save_allowance_s=allowance,evaluation_allowance_s=evaluation_allowance,peak_RSS_MiB=rss,resource_scope='fullB12reference_preflight_only',preflight_RSS_limit_MiB=3072,training_RSS_limit_MiB=1800,wall_limit_s=7200,integrator13_numeric_replayed=True,source_hashes=b['source_hashes'],data_hashes=b['data_hashes'],training_updates=0,optimizer_probe_updates=2,new_training_updates=0,physical_steps=0))
 if not all(checks.values()):raise ValueError('Statealignedpreflightfailed')
 if not timing:raise ValueError('Statealignedtimingblockedbeforetraining')
 print('STATE_ALIGNED_READY600_TO900',flush=True)
if __name__=='__main__':main()
