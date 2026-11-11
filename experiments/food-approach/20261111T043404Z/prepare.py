"""Zero-integration checks and immutable generation-one proposals before physics."""
from pathlib import Path
import json,copy,sys
import numpy as np
import torch
from model_io import ROOT,PARENT,PARENT_SHA,legacy,sha256,load_checked
from cem import propose,update,objective

def main():
 torch.set_num_threads(1);parent,p=legacy.load_checked(PARENT,PARENT_SHA)
 mapping_path=ROOT.parent/'20261111T024558Z/JOINT_MAPPING.json';seal=json.loads((mapping_path.parent/'ARTIFACT_SHA256.json').read_text())
 if sha256(mapping_path)!=seal['files'][str(mapping_path)]['sha256']:raise ValueError('sealed joint map changed')
 mapping=json.loads(mapping_path.read_text());names=['joint_RHCoxa_pitch','joint_RHTrochanter_pitch','joint_RHTibia_pitch']
 checks={'three_actual_joint_names':all(mapping['joint_names'][i]==n for i,n in zip((29,32,34),names)),'same_parent_checkpoint':sha256(PARENT)==PARENT_SHA}
 rng=np.random.default_rng(20261111);before=copy.deepcopy(rng.bit_generator.state);d,b,a=propose([0.,.05,0.],[.05]*3,rng)
 rng2=np.random.default_rng();rng2.bit_generator.state=before;d2,b2,a2=propose([0.,.05,0.],[.05]*3,rng2)
 checks['PCG64_roundtrip_exact']=np.array_equal(d,d2) and b==b2 and a==a2
 checks['antithetic_order_exact']=np.allclose(d[0]+d[1],[0.,.1,0.],rtol=0,atol=1e-16) and np.allclose(d[2]+d[3],[0.,.1,0.],rtol=0,atol=1e-16)
 checks['proposal_bounds']=bool((np.abs(d)<=.25).all())
 nm,ns,elite=update([0.,.05,0.],[.05]*3,d,[.6,.5,.5,.7]);expected=d[[1,2]]
 checks['elite_tie_break_and_exact_update']=elite==[1,2] and np.array_equal(nm,.25*np.array([0.,.05,0.])+.75*expected.mean(0)) and np.array_equal(ns,np.clip(.3*np.full(3,.05)+.7*expected.std(0,ddof=0),.01,.10))
 fake=dict(autonomous=dict(gates=dict(support=True,nonfoot=True,com=True,forward=True),slip_ratio_max=.52),criteria=dict(all_six=True,strict_nonfoot_zero=True,turn_sign_magnitude=True))
 checks['objective_whole_slip']=objective(fake)==.52;fake['criteria']['all_six']=False
 checks['objective_constraint_penalty']=objective(fake)==1010.52
 files=[ROOT/n for n in ('PREREG.json','model_io.py','cem.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py')]
 files += [ROOT.parent/'20261111T024558Z'/n for n in ('evaluate_motor.py','verify_actual.py')]+[mapping_path]
 hashes={str(f):sha256(f) for f in files}
 template=json.loads((ROOT.parent/'20261111T010218Z/evaluation/EVAL_PREREG.json').read_text());(ROOT/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 proposal_path=ROOT/'GEN01_PROPOSAL.json';proposal=dict(generation=1,mean=[0.,.05,0.],sigma=[.05]*3,rng_before=b,rng_after=a,deltas=d.tolist(),source_hashes=hashes)
 with proposal_path.open('x') as f:json.dump(proposal,f,indent=2);f.write('\n')
 streams={}
 for cond in ('neutral','right'):
  path=ROOT.parent/'20261111T010218Z/evaluation'/('EVAL_'+cond)/'neural_decisions.npz'
  with np.load(path,allow_pickle=False) as z:streams[cond]={k:z[k].copy() for k in ('X238','command','mean','hidden_after')}
 candidates=[]
 for idx,delta in enumerate(d):
  cid=f'g01_c{idx:02d}';folder=ROOT/cid;folder.mkdir(exist_ok=False);ck=folder/'candidate.pt'
  payload=copy.deepcopy(p);payload.update(kind='CEM3bias',generation=1,candidate_index=idx,delta=delta.tolist(),proposal_path=str(proposal_path),proposal_sha256=sha256(proposal_path),parent_sha256=PARENT_SHA,calibration_source_hashes={**hashes,str(proposal_path):sha256(proposal_path)})
  for row,value in zip((29,32,34),delta):payload['state_dict']['readout.bias'][row]+=float(value)
  torch.save(payload,ck);digest=sha256(ck);m,_=load_checked(ck,digest)
  for cond,z in streams.items():
   state=m.initial_state(1);means=[];states=[]
   with torch.no_grad():
    for x,u in zip(z['X238'],z['command']):
     mean,state=m.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state);means.append(mean[0].numpy());states.append(torch.cat(state,-1)[0].numpy())
   checks[cid+'_'+cond+'_all2880means_exact']=np.array_equal(means,z['mean']);checks[cid+'_'+cond+'_all192states_exact']=np.array_equal(states,z['hidden_after'])
  candidates.append(dict(id=cid,generation=1,index=idx,delta=delta.tolist(),checkpoint=str(ck),sha256=digest))
  (folder/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n');(folder/'READINESS.json').write_text(json.dumps(dict(all_passed=True,source_hashes=hashes),indent=2)+'\n')
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=hashes,joint_names=names,initial_candidates=candidates,physical_steps=0,optimizer_weight_updates=0)
 (ROOT/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'GEN01_FROZEN.json').write_text(json.dumps(dict(generation=1,proposal_sha256=sha256(proposal_path),candidates=candidates,all_checkpoints_saved_before_any_generation_physics=True),indent=2)+'\n')
 import evaluate_motor,verify_actual
 for f in evaluate_motor.source_paths():
  if not f.is_file():raise ValueError('preCore missing '+str(f))
 if not report['all_passed']:raise ValueError('preparation failure '+json.dumps(checks))
 print(json.dumps(dict(all_passed=True,checks=len(checks),candidates=candidates,physical_steps=0)),flush=True)
if __name__=='__main__':main()
