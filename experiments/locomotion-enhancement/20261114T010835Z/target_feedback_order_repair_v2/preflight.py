"""All12fullhistory exactcomponentorder qualification at strictsaved300, no training."""
import argparse,copy,time,sys
from sources import ROOT,read,save,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-order-preflight',action='store_true');a=p.parse_args()
 if not a.execute_parent_order_preflight:p.error('Parentorderrepairpreflightrequired')
 if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
 start=time.monotonic()
 from learning import cpu1,rng_snapshot,rng_equal,restore_rng,forward,slice_data,objective,resource_gate
 cpu1()
 import torch
 import common
 from data import prepare,production_sequence,dev_sources
 from scan import make_scan
 from utils import production_feedback,output_errors,max_error,tensor_digest
 b=read(ROOT/'INPUT_BINDINGS.json');verify(b['source_hashes']);verify(b['data_hashes']);setup=time.monotonic();m,parent=common.initialize();parent_adam=copy.deepcopy(parent['optimizer_state_dict']);data=prepare(m);sc=make_scan(m);py=make_scan(m,script=False);setup_seconds=time.monotonic()-setup
 initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot();oldhash=tensor_digest(m.old.state_dict());checks={'resume_strict300':parent['step']==300,'9target6prior':int(data['target_selector'].sum())==9 and int(data['v1'].sum())==6};rows=[];timer=time.monotonic()
 with torch.no_grad():
  for i,raw in enumerate(data['raw']+dev_sources(b)):
   prod,h=production_sequence(m,raw);checks[raw['entry']['id']+str(i)+'_finiteactualX']=bool(torch.isfinite(prod).all() and torch.isfinite(h).all())
   if i<12:
    one=slice_data(data,i,i+1);out=forward(sc,one);full,old=production_feedback(m,raw,data,i);err=output_errors(tuple(v[:,0] for v in out),full);forced=forward(sc,one,forced=True)
    checks[raw['entry']['id']+'_conditionalexactorder']=max(err)<=1e-12 and torch.equal(old,data['oldafter'][:,i]);checks[raw['entry']['id']+'_forcedactualX']=float((forced[0][:,0]-prod).abs().max())<=1e-12 and float((forced[2][:,0]-h[:,256:]).abs().max())<=1e-12
    rows.append(dict(id=raw['entry']['id'],errors=err,all6outputs_exact=all(v==0. for v in err)))
   resource_gate(start)
  out=forward(sc,data);po=forward(py,data);checks['PythonJITfull1920']=max(output_errors(out,po))<=1e-12
  for i,raw in enumerate(data['raw']):
   full,old=production_feedback(m,raw,data,i);checks[raw['entry']['id']+'_B12fullproduction']=max(output_errors(tuple(v[:,i] for v in out),full))<=1e-12 and torch.equal(old,data['oldafter'][:,i])
  del out,po
  for lo in (0,4,8):
   four=forward(sc,slice_data(data,lo,lo+4))
   for j in range(4):
    i=lo+j;full,old=production_feedback(m,data['raw'][i],data,i);checks[data['raw'][i]['entry']['id']+'_micro4fullproduction']=max(output_errors(tuple(v[:,j] for v in four),full))<=1e-12 and torch.equal(old,data['oldafter'][:,i])
   del four
  del prod,h,full,old,forced
 parity_seconds=time.monotonic()-timer;anchor=parent['config']['initial_anchor']
 # Retain all originalactual13target/float32source qualification as immutableproof;
 # data.prepare independently rebuilds12source normalizers andactualtargetindices.
 from pathlib import Path
 import numpy as np
 from utils import module,torch_integrate
 am=module('_orderrepair_actual_accumulator_metadata',Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/accumulator_replay_diagnostic/metadata.py'));ameta=am.qualify()
 with torch.no_grad():
  for entry in ameta['entries']:
   folder=Path(entry['folder']);ident=am.identity_contract(am.read(folder/'IDENTITY.json'))
   with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z:targets=z['target'].copy();qpos=z['qpos'].copy()
   with np.load(entry['sample_path'],allow_pickle=False) as z:
    k='Y42' if entry['teacher'] and not entry['collection'] else 'actually_applied_action';actions=z[k][entry['start']:].copy();X=z['X238'][entry['start']:].copy();ms=np.rint(z['time'][entry['start']:]*1000).astype(int)
   bounds=torch.tensor(ident['actuator']['target_bounds_rad'],dtype=torch.float64);rate=torch.tensor(ident['actuator']['max_rate_rad_s'],dtype=torch.float64);pred=torch_integrate(torch.tensor(actions),torch.tensor(targets[5799]),bounds,rate)[:len(targets)-5800]
   target=torch.tensor(targets[ms-1]);q=torch.tensor(qpos[ms-1][:,ameta['mapping']['qpos_indices']]);half=torch.tensor(ameta['mapping']['halfspan42'],dtype=torch.float64);q0=torch.tensor(ameta['mapping']['q0_42'],dtype=torch.float64)
   checks[entry['id']+'_13torchactualtarget_encoding']=float((pred-torch.tensor(targets[5800:])).abs().max())<=1e-12 and torch.equal(((target-q)/half).clamp(-10,10).float().double(),torch.tensor(X[:,42:84])) and torch.equal(((target-q0)/half).clamp(-10,10).float().double(),torch.tensor(X[:,126:168]))
 small=slice_data(data,0,12,T=32);target=small['initial_target'].clone().requires_grad_(True);small=dict(small,initial_target=target);out=forward(sc,small);canary=out[0].square().mean()+out[4].square().mean();canary.backward();checks['targetgradient_canary']=target.grad is not None and bool(torch.isfinite(target.grad).all()) and float(target.grad.abs().max())>0.;del out,canary;m.zero_grad(set_to_none=True)
 small=slice_data(data,0,12,T=32);out=forward(sc,small);loss,_=objective(out,small,m,anchor);loss.backward();fd=[];del out,loss
 for parameter,index in ((m.head.bias,(0,)),(m.encoder.weight,(0,0))):
  analytical=float(parameter.grad[index]);value=float(parameter[index]);values=[];epsilon=1e-4
  with torch.no_grad():
   for sign in (1.,-1.):
    parameter[index]=value+sign*epsilon;out=forward(sc,small);loss,_=objective(out,small,m,anchor);values.append(float(loss));del out,loss
   parameter[index]=value
  numerical=(values[0]-values[1])/(2*epsilon);ae=abs(numerical-analytical);re=ae/max(abs(numerical),abs(analytical),1e-12);fd.append(dict(analytic=analytical,numerical=numerical,absolute_error=ae,relative_error=re,passed=ae<=1e-3 or re<=.05))
 checks['FDsame_registered_gate']=all(r['passed'] for r in fd);m.zero_grad(set_to_none=True)
 opt1=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt1.load_state_dict(copy.deepcopy(parent_adam));out=forward(sc,data);loss,_=objective(out,data,m,anchor);fullvalue=float(loss.detach());loss.backward();grad=[p.grad.detach().clone() for p in m.adapter_parameters()];del out,loss;opt1.step();fullstate={k:v.detach().clone() for k,v in m.state_dict().items()};adamstate=copy.deepcopy(opt1.state_dict());del opt1
 m.load_state_dict(initial,strict=True);opt=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt.load_state_dict(copy.deepcopy(parent_adam));opt.zero_grad(set_to_none=True);timer=time.monotonic();micro=0.
 for lo in (0,4,8):
  d=slice_data(data,lo,lo+4);out=forward(sc,d);loss,_=objective(out,d,m,anchor);loss.backward();micro+=float(loss.detach());del out,d,loss
 ge=max(float((p.grad-g).abs().max()) for p,g in zip(m.adapter_parameters(),grad));opt.step();update=time.monotonic()-timer;ae=max_error(adamstate,opt.state_dict());pe=max_error(fullstate,m.state_dict())
 checks['full1920loss12vs3micro']=abs(fullvalue-micro)<=1e-10;checks['all8gradients']=ge<=1e-10;checks['Adam_and_parameters']=ae<=1e-10 and pe<=1e-10
 checks['oldweights_unchanged']=tensor_digest(m.old.state_dict())==oldhash and all(p.grad is None and not p.requires_grad for p in m.old.parameters())
 m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);opt.load_state_dict(copy.deepcopy(parent_adam));restore_rng(rng);checks['300weights_Adam_RNG_restored']=max_error(initial,m.state_dict())==0 and max_error(opt.state_dict(),parent_adam)==0 and max_error(parent['optimizer_state_dict'],parent_adam)==0 and rng_equal(rng,rng_snapshot())
 checks['no_Core_physical_teacherbackend']=not any(n in sys.modules for n in ('physics','mujoco','flygym','static_teacher','student_wave'))
 allowance=3*(setup_seconds+parity_seconds)*1.3;estimate=300*update*1.3+allowance;timing=estimate<=5400;rss=resource_gate(start);verify(b['source_hashes']);verify(b['data_hashes'])
 save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),checks=checks,timing_status='READY_RESUME300_TO600' if timing else 'BLOCKED_BEFORE_RESUME_TIMING',all12conditional=rows,FD=fd,loss_error=abs(fullvalue-micro),gradient_error=ge,Adam_error=ae,parameter_error=pe,micro4x3_update_s=update,predicted_remaining300_s_withverification=estimate,peak_RSS_MiB=rss,source_hashes=b['source_hashes'],data_hashes=b['data_hashes'],training_updates=0,optimizer_probe_updates=2,new_training_updates=0,physical_steps=0))
 if not all(checks.values()):raise ValueError('Orderrepair300qualificationfailed')
 if not timing:raise ValueError('Orderrepairtimingblockedbeforetraining')
 print('ORDER_REPAIR_READY_RESUME300_TO600',flush=True)


if __name__=='__main__':main()
