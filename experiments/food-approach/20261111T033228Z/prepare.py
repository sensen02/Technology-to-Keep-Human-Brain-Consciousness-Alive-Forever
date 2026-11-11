"""Freeze exact bias candidates and negative-command invariance before physics."""
from pathlib import Path
import json,copy
import numpy as np
import torch
import model_io as io
ROOT=io.ROOT;PREVIOUS=ROOT.parent/'20261111T024558Z'
def main():
 torch.set_num_threads(1);parent,p=io.legacy.load_checked(io.PARENT,io.PARENT_SHA)
 mapping=json.loads((PREVIOUS/'JOINT_MAPPING.json').read_text())
 sealed=json.loads((PREVIOUS/'ARTIFACT_SHA256.json').read_text())
 if io.sha256(PREVIOUS/'JOINT_MAPPING.json')!=sealed['files'][str(PREVIOUS/'JOINT_MAPPING.json')]['sha256']:raise ValueError('sealed mapping changed')
 plan=json.loads((ROOT/'PREREG.json').read_text())
 for c in plan['candidates']:
  if mapping['joint_names'][c['index']]!=c['joint']:raise ValueError('wrong neuron/actuator mapping')
 files=[ROOT/n for n in ('PREREG.json','model_io.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py')]
 files += [PREVIOUS/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','JOINT_MAPPING.json')]
 hashes={str(f):io.sha256(f) for f in files}
 template=json.loads((io.OLD/'evaluation/EVAL_PREREG.json').read_text())
 (ROOT/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 candidates={};checks={'sealed_joint_mapping':True,'parent_exact_sha':io.sha256(io.PARENT)==io.PARENT_SHA}
 streams={}
 for cond in ('neutral','right'):
  with np.load(io.OLD/'evaluation'/('EVAL_'+cond)/'neural_decisions.npz',allow_pickle=False) as z:streams[cond]={k:z[k].copy() for k in ('X238','command','mean','hidden_after')}
 for c in plan['candidates']:
  folder=ROOT/c['id'];folder.mkdir(exist_ok=False)
  payload=copy.deepcopy(p);payload.update(candidate_id=c['id'],bias_index=c['index'],bias_delta=c['bias_delta'],parent_sha256=io.PARENT_SHA,calibration_source_hashes=hashes)
  payload['state_dict']['readout.bias'][c['index']]+=c['bias_delta']
  ck=folder/'candidate.pt';torch.save(payload,ck);digest=io.sha256(ck);model,_=io.load_checked(ck,digest)
  for cond,z in streams.items():
   state=model.initial_state(1);means=[];states=[]
   with torch.no_grad():
    for x,u in zip(z['X238'],z['command']):
     mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state);means.append(mean[0].numpy());states.append(torch.cat(state,-1)[0].numpy())
   checks[c['id']+'_'+cond+'_2880means_exact']=np.array_equal(means,z['mean']);checks[c['id']+'_'+cond+'_192hidden_exact']=np.array_equal(states,z['hidden_after'])
  candidates[c['id']]=dict(bias_index=c['index'],joint=c['joint'],bias_delta=c['bias_delta'],checkpoint=str(ck),sha256=digest)
  (folder/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=hashes,candidates=candidates,physical_steps=0,optimizer_updates=0)
 (ROOT/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
 for cid in candidates:(ROOT/cid/'READINESS.json').write_text(json.dumps(dict(all_passed=report['all_passed'],source_hashes=hashes),indent=2)+'\n')
 import evaluate_motor,verify_actual
 for f in evaluate_motor.source_paths():
  if not f.is_file():raise ValueError('preCore missing path '+str(f))
 if not report['all_passed']:raise ValueError('preparation failed')
 print(json.dumps(dict(all_passed=True,checks=len(checks),candidates=candidates)),flush=True)
if __name__=='__main__':main()
