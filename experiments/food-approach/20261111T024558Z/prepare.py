"""Pre-physics checkpoint creation, actual joint mapping and full neural invariance."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
import model_io as io
ROOT=io.ROOT

def main():
 torch.set_num_threads(1);parent,p=io.legacy.load_checked(io.PARENT,io.PARENT_SHA)
 sys.path.insert(0,str(io.legacy.BASE));import student_wave as sw
 core=sw.Core(adhesion=False);before=float(core.d.time)
 names=[core.plant.joint(int(j)).name for j in core.actuator.joint_ids]
 if len(names)!=42 or not all(n.startswith('joint_RH') for n in names[28:35]):raise ValueError('RH row mapping incorrect '+str(names))
 mapping=dict(joint_names=names,RH_rows=list(range(28,35)),RH_names=names[28:35],physics_steps=0,time_unchanged=float(core.d.time)==before)
 (ROOT/'JOINT_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
 files=[ROOT/n for n in ('PREREG.json','model_io.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py','JOINT_MAPPING.json')]
 hashes={str(f):io.sha256(f) for f in files}
 # Explicit original evaluation prereg keeps unchanged step2000 loader metadata.
 template=json.loads((io.OLD/'evaluation/EVAL_PREREG.json').read_text())
 (ROOT/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 candidates={};checks={'joint_mapping_RH7':True,'mapping_zero_integration':mapping['time_unchanged']}
 original_streams={}
 for condition in ('neutral','left','right'):
  path=io.OLD/'evaluation'/('EVAL_'+condition)/'neural_decisions.npz'
  with np.load(path,allow_pickle=False) as z:original_streams[condition]={k:z[k].copy() for k in ('X238','command','mean','policy_action','hidden_after')}
 for c in json.loads((ROOT/'PREREG.json').read_text())['candidates']:
  cid=c['id'];folder=ROOT/cid;folder.mkdir(exist_ok=False)
  payload=copy.deepcopy(p);payload.update(candidate_id=cid,alpha=c['alpha'],parent_sha256=io.PARENT_SHA,calibration_source_hashes=hashes)
  for k in ('readout.weight','readout.bias'):payload['state_dict'][k][28:35]*=c['alpha']
  ck=folder/'candidate.pt';torch.save(payload,ck);digest=io.sha256(ck);model,_=io.load_checked(ck,digest)
  for condition in ('neutral','right'):
   z=original_streams[condition];state=model.initial_state(1);means=[];states=[]
   with torch.no_grad():
    for x,u in zip(z['X238'],z['command']):
     mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state);means.append(mean[0].numpy());states.append(torch.cat(state,-1)[0].numpy())
   checks[cid+'_'+condition+'_all2880_means_exact']=np.array_equal(means,z['mean'])
   checks[cid+'_'+condition+'_all192states_exact']=np.array_equal(states,z['hidden_after'])
  candidates[cid]=dict(alpha=c['alpha'],checkpoint=str(ck),sha256=digest)
  (folder/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 # Required files/strict JSON ready BEFORE all physics.
 prep=dict(all_passed=all(checks.values()),checks=checks,source_hashes=hashes,candidates=candidates,joint_mapping=mapping,scope='no integration; only Core construction for runtime joint identity')
 (ROOT/'PREPARATION.json').write_text(json.dumps(prep,indent=2)+'\n')
 for c in candidates:
  (ROOT/c/'READINESS.json').write_text(json.dumps(dict(all_passed=prep['all_passed'],source_hashes=hashes),indent=2)+'\n')
 import evaluate_motor
 for path in evaluate_motor.source_paths():
  if not path.is_file():raise ValueError('missing source '+str(path))
 if not prep['all_passed']:raise ValueError('preflight checks failed')
 print(json.dumps(dict(all_passed=True,checks=len(checks),candidates=candidates,RH_names=mapping['RH_names'])),flush=True)
if __name__=='__main__':main()
