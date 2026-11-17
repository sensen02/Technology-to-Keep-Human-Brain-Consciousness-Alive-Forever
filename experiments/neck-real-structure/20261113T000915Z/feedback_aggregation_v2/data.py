"""Exact archived4 + onpolicy3 actual111 inputs; explicit instructionteacher offline only."""
from common import *
def action_weights(cue,np):
 active=np.abs(cue[:,:,0])>.5;weights=np.zeros_like(cue,dtype=np.float64)
 for j in range(cue.shape[1]):
  on=active[:,j];off=~on
  if on.any() and off.any():weights[on,j,0]=.5/on.sum();weights[off,j,0]=.5/off.sum()
  else:weights[:,j,0]=1./1920.
 return weights

def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('No blind aggregate dataset overwrite')
 bp,b,digest=binding();hashes={};meta=[]
 for p in (OLD/'DATASET.json',OLD/'train.npz',OLD/'development.npz',BODY/'BODY_SUMMARY.json'):hashes[str(p)]=sha(p)
 prior_dataset=read(OLD/'DATASET.json');verify(prior_dataset['source_hashes']);hashes.update(prior_dataset['source_hashes'])
 for row in prior_dataset['outputs'].values():
  if sha(row['path'])!=row['sha256']:raise ValueError('Prior input dataset changed')
 with np.load(OLD/'train.npz',allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 if base['body111'].shape!=(1920,4,111) or base['cue'].shape!=(1920,4,1):raise ValueError('Exact original4 fullhistory inputs required')
 body=[base['body111']];cue=[base['cue']];original=[base['teacher']]
 for j,name in enumerate(TRAIN[:4]):meta.append(dict(split='train',condition=name,origin='original_archived_train',original_index=j,actual_neural_rows=2880,autonomous_start_index=960,allowed111_inputs_bitexact=True))
 for condition in TRAIN[4:]:
  folder=BODY/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');path=folder/'neural_decisions.npz'
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or sha(folder/'RESULT.json')!=a['result_sha256'] or r['brain_checkpoint_sha256']!=PARENT_SHA or r['neck_source_binding_sha256']!=digest:raise ValueError('Actual completed onpolicy source-neuralbody audit required')
  if sha(path)!=r['neural_decisions']['sha256']:raise ValueError('ActualonpolicyneuralartifactSHA')
  verify(r['source_hashes']);hashes.update(r['source_hashes']);hashes.update({str(p):sha(p) for p in (folder/'RESULT.json',folder/'VERIFY.json',folder/'SOURCE_QUEUE.json',path)})
  with np.load(path,allow_pickle=False) as z:
   if z['X238'].shape!=(3080,238) or z['raw_turn'].shape!=(3080,):raise ValueError('Onpolicy3080 actualrows required, do not use old960 suffix')
   raw=z['X238'][1160:3080].copy();c=z['cue'][1160:3080].copy()[:,None];t=z['time'][1160:3080].copy();y=z['raw_turn'][1160:3080].copy()[:,None]
  if np.max(abs(t-(5.8+np.arange(1920)*.005)))>1e-8 or np.any(~np.isin(c,[-1.,0.,1.])) or not np.isfinite(raw).all():raise ValueError('Actual onpolicy timing/cue/finite111 contract')
  x=raw[:,MASK];body.append(x[:,None,:]);cue.append(c[:,None,:]);original.append(y[:,None,:]);meta.append(dict(split='train',condition=condition,origin='actual_onpolicy_parent_model_body_trial',actual_neural_rows=3080,autonomous_start_index=1160,actual_raw_X238_field=True,source=str(folder),fixed_archived_trajectory_not_online_environment=True))
 trainbody=np.concatenate(body,1);traincue=np.concatenate(cue,1);orig=np.concatenate(original,1)
 if trainbody.shape!=(1920,7,111) or not np.array_equal(trainbody[:,:4],base['body111']) or not np.array_equal(traincue[:,:4],base['cue']):raise ValueError('Base4exact + onpolicy3 aggregate dimensions')
 with np.load(OLD/'development.npz',allow_pickle=False) as z:dev={k:z[k].copy() for k in z.files}
 datasets={'train':dict(body111=trainbody,cue=traincue,teacher=.8*traincue,original_teacher=orig,weights=action_weights(traincue,np),mean=base['mean'],std=base['std']), 'development':dict(body111=dev['body111'],cue=dev['cue'],teacher=.8*dev['cue'],original_teacher=dev['teacher'],weights=action_weights(dev['cue'],np),mean=base['mean'],std=base['std'])}
 for j,name in enumerate(DEV):meta.append(dict(split='development',condition=name,origin='old_archived_reversal_not_fresh_holdout',original_index=j,instruction_target_replaces_retinal_teacher=True))
 for split,value in datasets.items():
  if not np.isfinite(value['body111']).all() or not np.isfinite(value['teacher']).all() or np.max(abs(value['weights'].sum(0)-1))>1e-12:raise ValueError('Exact finite unmodified feedback and actionmass required')
  np.savez_compressed(HERE/(split+'.npz'),**value)
 norm=(trainbody-base['mean'])/base['std'];summary=dict(normalizer_mean_std_parent_exact=True,no_clipping=True,old4_normalized_absolute_max=float(abs(norm[:,:4]).max()),new3_normalized_absolute_max=float(abs(norm[:,4:]).max()),new3_raw_absolute_max=float(abs(trainbody[:,4:]).max()),coldAN_DN_brain_initial_state=True)
 verify(hashes);save(HERE/'DATASET.json',dict(kind='ACTUAL_ONPOLICY_FEEDBACK_AGGREGATION_7_SEQUENCES',source_binding=str(bp),source_binding_sha256=digest,source_hashes=hashes,outputs={k:dict(path=str(HERE/(k+'.npz')),sha256=sha(HERE/(k+'.npz'))) for k in datasets},rows=meta,allowed_raw_indices=MASK,distribution=summary,teacher='.8*explicitbraininstructioncue_training_only',original_weak_retinal_and_actual_turn_retained_not_input=True,new_training_updates=0,new_physics_steps=0))
 print('ACTUAL_AGGREGATE7_TRAIN_AND2_DEVELOPMENT_SAVED_NO_PHYSICS')
def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(data,indices=None):
 import torch
 return tuple(torch.from_numpy(data[k] if indices is None else data[k][:,indices]) for k in ('body111','cue','teacher','weights'))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-aggregation')
