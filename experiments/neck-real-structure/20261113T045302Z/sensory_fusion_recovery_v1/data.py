"""Correct only causal-target firstrow shape using already extracted actual sensory data."""
from common import *
_original=module('_fusion_recovery_frozen_data_helpers',ORIGINAL/'data.py');weights_for=_original.weights_for;tensors=_original.tensors

def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def causal_target(desired,np):return np.concatenate((np.zeros_like(desired[:1]),desired[:-1]),axis=0)
def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('Inspect before re-running recovery')
 desc=read(ORIGINAL/'DATASET.json');proof=read(ORIGINAL/'SENSORY_REPLAY.json');jobs=read(ORIGINAL/'JOBS.json');log=ORIGINAL/'PREFLIGHT_EXECUTION.log'
 if not proof['all_bitexact'] or proof['actual_visual_cases']!=8 or proof['actual_odor_cases']!=12 or len(jobs)!=1 or jobs[0]['stage']!='PREFLIGHT' or jobs[0]['exit_code']!=1 or 'Fusionpreflight actual27full1920' not in log.read_text():raise ValueError('Exact original failedpreflight and completed sensoryextraction required')
 if (ORIGINAL/'TRAIN').exists() or (ORIGINAL/'FROZEN.json').exists():raise ValueError('Original failure must be pretraining only')
 verify(desc['source_hashes']);references=dict(desc['source_hashes']);before={str(ORIGINAL/n):sha(ORIGINAL/n) for n in ('DATASET.json','SENSORY_REPLAY.json','data.py','JOBS.json','PREFLIGHT_EXECUTION.log','PIPELINE.log')};references.update(before);checks={};outputs={};shapes={}
 for split,expected in (('train',27),('development',8)):
  old=desc['outputs'][split];p=Path(old['path'])
  if sha(p)!=old['sha256']:raise ValueError('Original extracted dataset bytes changed')
  references[str(p)]=old['sha256']
  with np.load(p,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
  original_shape=list(value['teacher'].shape);desired=value['current_desired'];target=causal_target(desired,np);weights=weights_for(target,np)
  checks[split+'_original_bug3839_preserved']=original_shape==[3839,expected,1];checks[split+'_correct_target1920']=target.shape==(1920,expected,1);checks[split+'_firstrow0']=bool(np.array_equal(target[:1],np.zeros_like(desired[:1])));checks[split+'_alllater_previousdesired']=bool(np.array_equal(target[1:],desired[:-1]));checks[split+'_shape_body_u']=value['body111'].shape==(1920,expected,111) and value['modalities'].shape==(1920,expected,2);checks[split+'_weights_sum1']=bool(np.max(abs(weights.sum(0)-1))<1e-12)
  value['teacher']=target;value['weights']=weights
  # Reopen original source only to prove all science inputs were carried unchanged.
  with np.load(p,allow_pickle=False) as z:
   for key in ('body111','modalities','current_desired','mean','std'):checks[split+'_actual_source_exact_'+key]=bool(np.array_equal(value[key],z[key]))
  checks[split+'_finite']=all(bool(np.isfinite(v).all()) for v in value.values());out=HERE/(split+'.npz');np.savez_compressed(out,**value);outputs[split]=dict(path=str(out),sha256=sha(out));shapes[split]=dict(original_bad_target=original_shape,new_target=list(target.shape),source_npz=str(p),source_sha256=old['sha256'])
 if not all(checks.values()):raise ValueError('Recovery shape/unchanged input checks '+str(checks))
 verify(references);save(HERE/'ORIGINAL_FAILURE.json',dict(status='ORIGINAL122_PREFLIGHT_CAUSAL_TARGET_SHAPE_ERROR_PRESERVED',original_preflight_exit1=True,original_no_training=True,original_no_body=True,original_files_sha256=before,recovery_only_one_zero_row_then_previousdesired=True,new_physics_steps=0));replay=dict(proof,reused_original_completed_extraction=True,original_replay_path=str(ORIGINAL/'SENSORY_REPLAY.json'),original_replay_sha256=sha(ORIGINAL/'SENSORY_REPLAY.json'),sensory_models_reloaded_for_extraction=False);save(HERE/'SENSORY_REPLAY.json',replay);references[str(HERE/'SENSORY_REPLAY.json')]=sha(HERE/'SENSORY_REPLAY.json');references[str(HERE/'ORIGINAL_FAILURE.json')]=sha(HERE/'ORIGINAL_FAILURE.json')
 save(HERE/'RECOVERY_DATA_PROOF.json',dict(all_passed=True,checks=checks,shapes=shapes,original_source_hashes=references,new_training_updates=0,new_physics_steps=0,no_repeated_sensory_extraction=True));references[str(HERE/'RECOVERY_DATA_PROOF.json')]=sha(HERE/'RECOVERY_DATA_PROOF.json');new=dict(desc,kind='CAUSAL_TARGET_SHAPE_RECOVERY_OF_ACTUAL_SENSOR27',outputs=outputs,source_hashes=references,recovery_original_dataset=str(ORIGINAL/'DATASET.json'),recovery_proof_sha256=sha(HERE/'RECOVERY_DATA_PROOF.json'),recovery_only_causal_target_and_weights=True);save(HERE/'DATASET.json',new);print('CAUSAL_TARGET1920_RECOVERED_ACTUAL_SOURCE27_UNCHANGED_NO_REPEATED_SENSORY_EXTRACTION')
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
