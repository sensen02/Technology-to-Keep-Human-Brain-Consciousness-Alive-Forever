"""Parent-only actual receptor replay plus declared offline withdrawal augmentations."""
from common import *
def weights_for(y,np):
 out=np.zeros_like(y)
 for j in range(y.shape[1]):
  a=np.abs(y[:,j,0]);groups=(a<=.02,a>.1,(a>.02)&(a<=.1));mass=(.5,.4,.1);den=sum(v for g,v in zip(groups,mass) if g.any())
  for g,v in zip(groups,mass):
   if g.any():out[g,j,0]=v/den/g.sum()
 return out
def ramp_target(original,cut,np):
 t=np.arange(1920,dtype=np.float64)*.005;s=np.clip((t-cut)/.25,0.,1.);return original*(1.-s[:,None])
def actual_case(name,parent,torch,np):
 folder=(OLD/'physical_B' if name.startswith('B_') else BODY)/'trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');p=folder/'neural_decisions.npz';hashes={str(folder/n):sha(folder/n) for n in ('RESULT.json','VERIFY.json')}
 if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json') or sha(p)!=r['neural_decisions']['sha256']:raise ValueError('Actualcaseaudit '+name)
 if r.get('behavior_checkpoint_sha256',r.get('odor_checkpoint_sha256'))!=PARENT_SHA:raise ValueError('Same66factualteacher required')
 hashes[str(p)]=r['neural_decisions']['sha256'];hashes.update(r['source_hashes']);verify(hashes);old=name.startswith('B_');start=960 if old else 1160;rows=2880 if old else 3080
 keys=('time','receptors','raw_turn','behavior_hidden_before','behavior_hidden_after') if old else ('time','actor_receptors','odor_raw_turn','odor_hidden_before','odor_hidden_after')
 with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in keys}
 receptor=d['receptors' if old else 'actor_receptors'];turn=d['raw_turn' if old else 'odor_raw_turn'];before=d['behavior_hidden_before' if old else 'odor_hidden_before'];after=d['behavior_hidden_after' if old else 'odor_hidden_after'];expected_time=(1. if old else 0.)+np.arange(rows)*.005
 if receptor.shape!=(rows,2) or before.shape!=(rows,32) or turn.shape!=(rows,) or np.max(abs(d['time']-expected_time))>1e-8 or not np.isfinite(receptor).all() or (receptor<0).any() or (receptor>1).any():raise ValueError('Actualreceptor/clock/range')
 if np.max(abs(before[:start+1]))!=0 or not np.array_equal(before[1:],after[:-1]):raise ValueError('No warmh32/reset continuity')
 h=parent.initial_state(1);raws=[];states=[]
 with torch.no_grad():
  for k in range(1920):
   j=k+start
   if not np.array_equal(h[0].numpy(),before[j]):raise ValueError('Actualhbefore '+name)
   latent,h=parent.mean_step(torch.from_numpy(receptor[j:j+1]),h);value=latent[:,0].tanh();raws.append(float(value[0]));states.append(h[0].numpy().copy())
 pred=np.asarray(raws)[:,None];state=np.asarray(states)
 if not np.array_equal(pred[:,0],turn[start:]) or not np.array_equal(state,after[start:]):raise ValueError('Original66factualraw_h32notbitexact '+name)
 return receptor[start:].copy(),pred,dict(condition=name,actual_rows=rows,ANNsuffix=[start,rows],actor_field='receptors' if old else 'actor_receptors',all_parent66f_raw_h32_bitexact=True,hidden_continuous=True),hashes

def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('InspectpriorDataset')
 diagnosis=read(DIAGNOSTIC)
 if not diagnosis['reproduction_checks_pass'] or not all(diagnosis['checks'].values()) or diagnosis['extra_training_updates']!=0 or diagnosis['new_physics_steps']!=0:raise ValueError('Completed actual failurediagnostic required')
 parent,p=initialize();parent.requires_grad_(False);before=hashes_of(parent);hashes={str(DIAGNOSTIC):sha(DIAGNOSTIC)};hashes.update(p['config']['source_hashes']);hashes.update(p['config']['data_hashes']);actual={};proofs=[]
 for name in B_NAMES+CURRENT:
  z,u,proof,h=actual_case(name,parent,torch,np);actual[name]=(z,u);proofs.append(proof);hashes.update(h)
 from model import production
 samples=[];meta=[];cutchecks={}
 for name in B_NAMES+CURRENT:
  z,original=actual[name];target=original.copy();cut=None
  if name in ('B_noodor','O_noodor'):target=np.zeros_like(original)
  elif name=='O_left_odorcut':cut=1.6;target=ramp_target(original,cut,np);cutchecks['actual_actor_receptorcut_zero']=bool(np.max(abs(z[320:]))==0)
  samples.append(dict(receptors=z,teacher=target,original_teacher=original));meta.append(dict(condition=name,source_kind='ACTUAL_66F_SAMEWEIGHT_BODY_RECEPTOR_REPLAY',new_physical_trial=False,fresh_holdout=False,withdrawal_target_cut_s=cut,zero_neutral_target=name in ('B_noodor','O_noodor'),runtime_cut_flag_not_input=True))
 for name,cut in zip(AUG,CUTS):
  original_z,original_u=actual['O_left'];z=original_z.copy();index=int(round(cut/.005));z[index:]=0
  with torch.no_grad():pred,latent,h=production(parent,torch.from_numpy(z[:,None]));u=pred[:,0].numpy().copy()
  cutchecks[name+'_actual_source_prefix_receptors']=bool(np.array_equal(z[:index],original_z[:index]));cutchecks[name+'_raw_prefix_exact']=bool(np.array_equal(u[:index],original_u[:index]));cutchecks[name+'_future_receptors_zero']=bool(np.max(abs(z[index:]))==0);target=ramp_target(u,cut,np);samples.append(dict(receptors=z,teacher=target,original_teacher=u));meta.append(dict(condition=name,source_kind='DECLARED_OFFLINE_WITHDRAWAL_AUGMENTATION_FROM_ACTUAL_O_LEFT',source_condition='O_left',cut_s=cut,new_physical_trial=False,fresh_holdout=False,withdrawal_target_cut_s=cut,runtime_cut_flag_not_input=True))
 value={k:np.stack([s[k] for s in samples],1) for k in samples[0]};value['weights']=weights_for(value['teacher'],np)
 if value['receptors'].shape!=(1920,16,2) or value['teacher'].shape!=(1920,16,1) or np.max(abs(value['weights'].sum(0)-1))>1e-12 or not all(np.isfinite(v).all() for v in value.values()) or not all(cutchecks.values()):raise ValueError('Exact16data/weights/augmentation')
 for j,row in enumerate(meta):
  if row.get('withdrawal_target_cut_s') is not None:
   cut=row['withdrawal_target_cut_s'];t=np.arange(1920)*.005;cutchecks[row['condition']+'_target_matches_original_before_cut']=bool(np.array_equal(value['teacher'][t<=cut,j],value['original_teacher'][t<=cut,j]));cutchecks[row['condition']+'_target0_afterquartersec']=bool(np.max(abs(value['teacher'][t>=cut+.25+1e-9,j]))==0)
 if not all(cutchecks.values()):raise ValueError('Declared losslabelramp violated')
 if hashes_of(parent)!=before:raise ValueError('Original66fteacher changed')
 verify(hashes);path=HERE/'train.npz';np.savez_compressed(path,**value);save(HERE/'PARENT_REPLAY.json',dict(all_bitexact=True,rows=proofs,actual_cases=13,augmentation_checks=cutchecks,parent_parameters_unchanged=True,parent_parameter_hashes=before,alpha=float(parent.ALPHA),identity=parent.identity(),no_hidden_reset=True,new_physics_steps=0));hashes[str(HERE/'PARENT_REPLAY.json')]=sha(HERE/'PARENT_REPLAY.json');save(HERE/'DATASET.json',dict(kind='ACTUAL13_PLUS3_OFFLINE_RECEPTOR_WITHDRAWAL_SEQUENCES',source_hashes=hashes,outputs=dict(train=dict(path=str(path),sha256=sha(path))),rows=meta,parent_sha256=PARENT_SHA,full_history=1920,effective_batch=16,teacher='Original66fcontinuousraw;declaredwithdrawalofflinequartersecondramp;B/Onoodor0',loss_labels_not_network_inputs=True,new_physics_steps=0));print('ACTUAL13_PARENT66F_BITEXACT_PLUS3_DECLARED_WITHDRAWAL_AUGMENTATIONS')
def arrays(split='train'):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(d):
 import torch
 return tuple(torch.from_numpy(d[k]) for k in ('receptors','teacher','weights'))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
