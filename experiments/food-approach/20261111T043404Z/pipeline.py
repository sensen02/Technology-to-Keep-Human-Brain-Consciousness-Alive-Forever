"""Finite physical-objective CEM, durable immutable generation proposals and results."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,sys,subprocess,copy,time
import numpy as np
import torch
from model_io import ROOT,PARENT,PARENT_SHA,legacy,sha256,load_checked
from cem import propose,update,objective

def dump(path,value):
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def main():
 torch.set_num_threads(1);prep=json.loads((ROOT/'PREPARATION.json').read_text())
 if not prep['all_passed']:raise ValueError('preflight failed')
 legacy.verify_hashes(prep['source_hashes']);plan=json.loads((ROOT/'PREREG.json').read_text())
 model,parent=legacy.load_checked(PARENT,PARENT_SHA);template=json.loads((ROOT/'EVAL_PREREG.json').read_text())
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');rng=np.random.default_rng(20261111);mean=np.array([0.,.05,0.]);sigma=np.full(3,.05);history=[];selected=None;started=time.monotonic()
 def command(label,argv):
  log=ROOT/(label+'.log')
  with log.open('x') as f:rc=subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError(label+' exit '+str(rc))
 def one(candidate,condition):
  folder=ROOT/candidate['id'];ck=folder/'candidate.pt';digest=candidate['sha256'];out=folder/('EVAL_'+condition);audit=folder/('VERIFY_'+condition+'.json')
  if (out/'RESULT.json').exists() and audit.exists():
   a=json.loads(audit.read_text());r=json.loads((out/'RESULT.json').read_text())
   if not a['integrity_all_pass'] or a['checkpoint_sha256']!=digest or r['checkpoint_sha256']!=digest:raise ValueError('existing candidate result binding failure')
   for key,filename in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz')]:
    if sha256(out/filename)!=r[key]['sha256']:raise ValueError('existing trajectory changed')
  else:
   if out.exists():raise RuntimeError('incomplete actual trajectory retained, cannot silently rerun '+str(out))
   command(candidate['id']+'_'+condition,[ROOT/'evaluate_motor.py','--checkpoint',ck,'--sha256',digest,'--condition',condition,'--out',out,'--training-complete'])
   command(candidate['id']+'_audit_'+condition,[ROOT/'verify_actual.py','--run',out,'--out',audit])
   r=json.loads((out/'RESULT.json').read_text());a=json.loads(audit.read_text())
  if not a['integrity_all_pass'] or not all(a['checks'].values()):raise ValueError('integrity failure not CEM measurement')
  value=dict(candidate=candidate['id'],generation=candidate['generation'],index=candidate['index'],delta=candidate['delta'],checkpoint=str(ck),sha256=digest,condition=condition,engineering_pass=bool(r['engineering_pass']),integrity_all_pass=True,objective=objective(r),slip=r['autonomous']['slip_ratio_max'],forward=r['autonomous']['forward_signed_native_per_s'],yaw=r['yaw_change_rad'],gates=r['autonomous']['gates'],criteria=r['criteria'],cycles=r['autonomous_cycles']['complete_per_leg'],audit_checks=len(a['checks']),result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),audit=str(audit),audit_sha256=sha256(audit))
  print(json.dumps(value),flush=True);return value
 for gen in range(1,5):
  proposal_path=ROOT/f'GEN{gen:02d}_PROPOSAL.json'
  if proposal_path.exists():
   proposal=json.loads(proposal_path.read_text())
   if proposal['mean']!=mean.tolist() or proposal['sigma']!=sigma.tolist():raise ValueError('resume optimizer state mismatch')
   if proposal['rng_before']!=rng.bit_generator.state:raise ValueError('resume RNG differs')
   rng.bit_generator.state=proposal['rng_after'];deltas=np.asarray(proposal['deltas'])
  else:
   deltas,before,after=propose(mean,sigma,rng);proposal=dict(generation=gen,mean=mean.tolist(),sigma=sigma.tolist(),rng_before=before,rng_after=after,deltas=deltas.tolist(),source_hashes=prep['source_hashes'])
   dump(proposal_path,proposal)
  candidates=[]
  for idx,delta in enumerate(deltas):
   cid=f'g{gen:02d}_c{idx:02d}';folder=ROOT/cid;ck=folder/'candidate.pt';folder.mkdir(exist_ok=True)
   if not ck.exists():
    p=copy.deepcopy(parent);p.update(kind='CEM3bias',generation=gen,candidate_index=idx,delta=delta.tolist(),proposal_path=str(proposal_path),proposal_sha256=sha256(proposal_path),parent_sha256=PARENT_SHA,calibration_source_hashes={**prep['source_hashes'],str(proposal_path):sha256(proposal_path)})
    for row,d in zip((29,32,34),delta):p['state_dict']['readout.bias'][row]+=float(d)
    torch.save(p,ck);(folder/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n');(folder/'READINESS.json').write_text(json.dumps(dict(all_passed=True,source_hashes=prep['source_hashes']),indent=2)+'\n')
   digest=sha256(ck);load_checked(ck,digest)
   candidates.append(dict(id=cid,generation=gen,index=idx,delta=delta.tolist(),checkpoint=str(ck),sha256=digest))
  frozen=ROOT/f'GEN{gen:02d}_FROZEN.json'
  if frozen.exists():
   if json.loads(frozen.read_text())['candidates']!=candidates:raise ValueError('frozen candidates changed')
  else:dump(frozen,dict(generation=gen,proposal_sha256=sha256(proposal_path),candidates=candidates,all_checkpoints_saved_before_any_generation_physics=True))
  results={};errors={}
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(one,c,'left'):c['index'] for c in candidates}
   for future in as_completed(futures):
    idx=futures[future]
    try:results[idx]=future.result()
    except BaseException as exc:errors[idx]=repr(exc)
    dump(ROOT/'LIVE_PROGRESS.json',dict(generation=gen,results=results,errors=errors,history=history,scope='mutable pointer excluded from all scientific source maps'))
  if errors:
   dump(ROOT/f'GEN{gen:02d}_EXECUTION_ERRORS.json',dict(errors=errors,completed_results=results));raise RuntimeError('all settled jobs joined; invalid rollout not optimization measurement')
  ordered=[results[i] for i in range(4)];scores=[r['objective'] for r in ordered]
  passing=[r for r in ordered if r['engineering_pass']]
  entry=dict(generation=gen,proposal=str(proposal_path),frozen=str(frozen),mean_before=mean.tolist(),sigma_before=sigma.tolist(),results=ordered)
  if passing:
   selected=min(passing,key=lambda r:(r['slip'],r['index']));entry['selected']=selected;history.append(entry);dump(ROOT/f'GEN{gen:02d}_RESULT.json',entry)
   proof=dict(candidate=selected['candidate'],generation=gen,index=selected['index'],checkpoint=selected['checkpoint'],checkpoint_sha256=selected['sha256'],proposal_path=str(proposal_path),proposal_sha256=sha256(proposal_path),frozen_candidates_path=str(frozen),frozen_candidates_sha256=sha256(frozen),left_result=selected['result'],left_result_sha256=selected['result_sha256'],left_audit=selected['audit'],left_audit_sha256=selected['audit_sha256'],engineering_pass=selected['engineering_pass'],integrity_all_pass=selected['integrity_all_pass'],selection_rule='Complete all4 currentgeneration, minimum(slip,index) among actualfullpass, firstpassinggeneration',all_generation_scores=ordered)
   selected_path=ROOT/'SELECTED.json'
   if selected_path.exists():
    if json.loads(selected_path.read_text())!=proof:raise ValueError('immutable selected proof changed')
   else:
    with selected_path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
   break
  mean_new,sigma_new,elite=update(mean,sigma,deltas,scores);entry.update(elite_indices=elite,elite_scores=[scores[i] for i in elite],mean_after=mean_new.tolist(),sigma_after=sigma_new.tolist(),rng_after=rng.bit_generator.state);history.append(entry)
  dump(ROOT/f'GEN{gen:02d}_RESULT.json',entry);mean,sigma=mean_new,sigma_new
 confirmations=[]
 if selected:
  c=next(c for c in candidates if c['id']==selected['candidate'])
  with ThreadPoolExecutor(max_workers=2) as pool:
   fs=[pool.submit(one,c,condition) for condition in ('neutral','right')];failures=[]
   for f in fs:
    try:confirmations.append(f.result())
    except BaseException as exc:failures.append(repr(exc))
  if failures:dump(ROOT/'CONFIRMATION_EXECUTION_ERRORS.json',dict(errors=failures,completed=confirmations));raise RuntimeError('confirmation entry/integrity failure')
 final=dict(status='COMPLETE',generations=len(history),left_actual_rollouts=sum(len(h['results']) for h in history),history=history,selected=None if selected is None else selected['candidate'],selected_checkpoint=None if selected is None else selected['checkpoint'],selected_checkpoint_sha256=None if selected is None else selected['sha256'],selected_left=selected,selected_proof=None if selected is None else str(ROOT/'SELECTED.json'),confirmations=confirmations,selected_all_three_pass=bool(selected and len(confirmations)==2 and all(r['engineering_pass'] for r in confirmations)),elapsed_s=time.monotonic()-started,scope='finite actual-objective ANN bias optimization, fixed motor engineering only; no food/native/continuous-range qualification')
 dump(ROOT/'FINAL_SUMMARY.json',final);legacy.verify_hashes(prep['source_hashes']);print(json.dumps(dict(status='COMPLETE',generations=len(history),left_actual_rollouts=final['left_actual_rollouts'],selected_all_three_pass=final['selected_all_three_pass'])),flush=True)
if __name__=='__main__':main()
