"""Parent-released zero-Core model preflight and all4 immutable proposal/checkpoints."""
import argparse,json,copy,os
from pathlib import Path
import numpy as np,torch
from model_io import *
def dump_new(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('CODE_ONLY until parent resource release')
 torch.set_num_threads(1);torch.manual_seed(20261111)
 import evaluate_motor,verify_actual
 for idx in range(4):evaluate_motor.build(ROOT/f'c{idx:02d}')
 verify_actual.build()
 parent,p=legacy.load_checked(PARENT,PARENT_SHA);mapping=verify_mapping();prop=proposal();checks={}
 rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=prop['rng_before'];z=rng.standard_normal((2,3));d=np.clip(np.stack((.005*z[0],-.005*z[0],.005*z[1],-.005*z[1])),-.02,.02)
 checks['PCG64_roundtrip_exact']=np.array_equal(d,prop['deltas']) and rng.bit_generator.state==prop['rng_after'];checks['antithetic_exact']=np.array_equal(d[0],-d[1]) and np.array_equal(d[2],-d[3]);checks['fixed_bounds']=bool(np.max(np.abs(d))<=.02);checks['mapping_actual3LH']=mapping['global_rows']==[8,11,13] and mapping['local_rows']==[1,4,6];checks['parent_SHA']=sha256(PARENT)==PARENT_SHA
 fixtures=[dict(index=i,integrity_all_pass=True,engineering_pass=i in (1,2),slip=[.6,.49,.49,.7][i]) for i in range(4)];checks['selection_tie_index']=selection(fixtures)['index']==1;checks['selection_none']=selection([dict(r,engineering_pass=False) for r in fixtures]) is None
 paths=[ROOT/n for n in ('PREREG.json','EVAL_PREREG.json','model_io.py','prepare.py','evaluate_motor.py','verify_actual.py','pipeline.py')]+[PARENT_DIR/'model_io.py',ROOT.parent/'model.py',ROOT.parent/'evaluate_motor.py',ROOT.parent/'verify_actual.py',legacy.XML,legacy.MAPPING_SOURCE]
 hashes={str(p):sha256(p) for p in paths};prop['source_hashes']=hashes;proposal_path=ROOT/'PROPOSAL.json';dump_new(proposal_path,prop)
 streams={}
 for condition in ('neutral','right'):
  path=PARENT_DIR/('EVAL_'+condition)/'neural_decisions.npz'
  with np.load(path,allow_pickle=False) as data:streams[condition]={k:data[k].copy() for k in ('X238','command','mean','hidden_after')}
 other=[i for i in range(42) if i not in GLOBAL_ROWS];candidates=[]
 # Zero-delta fixture does not construct a Core and is not an actual candidate.
 parent.load_state_dict(expected_state(p['state_dict'],[0.,0.,0.]),strict=True);checks['zero_delta_full_state_exact']=exact(parent.state_dict(),p['state_dict'])
 for condition,data in streams.items():
  with torch.no_grad():zero_means,zero_h=parent.sequence(torch.from_numpy(data['X238'])[:,None],torch.from_numpy(data['command'])[:,None])
  checks['zero_delta_'+condition+'_allmeans_exact']=np.array_equal(zero_means[:,0].numpy(),data['mean']);checks['zero_delta_'+condition+'_all192_exact']=np.array_equal(torch.cat(zero_h,-1)[:,0].numpy(),data['hidden_after'])
 for idx,delta in enumerate(d):
  folder=ROOT/f'c{idx:02d}';folder.mkdir(exist_ok=False);ck=folder/'candidate.pt'
  payload=dict(format_version=FORMAT,identity=parent.identity(),state_dict=expected_state(p['state_dict'],delta),optimizer=None,training_updates=0,parent_training_steps=4000,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,candidate_index=idx,delta=delta.tolist(),mapping=mapping,proposal_path=str(proposal_path),proposal_sha256=sha256(proposal_path),calibration_source_hashes=hashes)
  torch.save(payload,ck);digest=sha256(ck);model,loaded=load_checked(ck,digest,require_frozen=False);checks[f'c{idx:02d}_exact_delta_only']=exact(model.state_dict(),expected_state(p['state_dict'],delta))
  for condition,data in streams.items():
   x=torch.from_numpy(data['X238'])[:,None];u=torch.from_numpy(data['command'])[:,None]
   with torch.no_grad():means,h=model.sequence(x,u)
   state=torch.cat(h,-1)[:,0].numpy();values=means[:,0].numpy();checks[f'c{idx:02d}_{condition}_all192_hidden_exact']=np.array_equal(state,data['hidden_after'])
   checks[f'c{idx:02d}_{condition}_non3_means_exact']=np.array_equal(values[:,other],data['mean'][:,other])
   if condition=='neutral':checks[f'c{idx:02d}_neutral_all2880_means_exact']=np.array_equal(values,data['mean'])
  template=json.loads((ROOT/'EVAL_PREREG.json').read_text());dump_new(folder/'EVAL_PREREG.json',template);dump_new(folder/'READINESS.json',dict(all_passed=True,source_hashes=hashes))
  candidates.append(dict(id=folder.name,index=idx,delta=delta.tolist(),checkpoint=str(ck),sha256=digest))
 if not all(checks.values()):dump_new(ROOT/'PREPARATION_FAILURE.json',dict(checks=checks));raise ValueError('candidate model preflight failed')
 dump_new(ROOT/'FROZEN.json',dict(kind=FORMAT,generations=1,proposal_sha256=sha256(proposal_path),candidates=candidates,all4_hashes_saved_before_any_actual=True,physical_steps=0))
 for c in candidates:load_checked(c['checkpoint'],c['sha256'])
 report=dict(all_passed=False,checks=checks,source_hashes=hashes,input_stream_hashes={str(PARENT_DIR/('EVAL_'+c)/'neural_decisions.npz'):sha256(PARENT_DIR/('EVAL_'+c)/'neural_decisions.npz') for c in streams},proposal_sha256=sha256(proposal_path),frozen_sha256=sha256(ROOT/'FROZEN.json'),candidates=candidates,Core_constructions=0,physical_steps=0,optimizer_updates=0);dump_new(ROOT/'PREPARATION.json',report)
 import evaluate_motor,verify_actual
 for c in candidates:evaluate_motor.build(ROOT/c['id']);evaluate_motor.source_paths(ROOT/c['id'])
 verify_actual.build();report['all_passed']=True;temporary=ROOT/'PREPARATION.json.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(ROOT/'PREPARATION.json');print(json.dumps(dict(all_passed=True,checks=len(checks),candidates=candidates,Core_constructions=0,physical_steps=0)),flush=True)
if __name__=='__main__':main()
