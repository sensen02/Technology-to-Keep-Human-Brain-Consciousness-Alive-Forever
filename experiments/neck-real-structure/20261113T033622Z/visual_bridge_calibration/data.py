"""Parent-only actual RGB neural replay. Labels/geometry never become visual input."""
from common import *
def weights_for(target,np):
 out=np.zeros_like(target,dtype=np.float64)
 for j in range(target.shape[1]):
  a=np.abs(target[:,j,0]);groups=(a<=.02,a>.1,(a>.02)&(a<=.1));masses=(.5,.4,.1);total=sum(m for g,m in zip(groups,masses) if g.any())
  for group,mass in zip(groups,masses):
   if group.any():out[group,j,0]=mass/total/group.sum()
 return out

def replay_visual(condition,vision,torch,np):
 folder=VIS_BODY/'trials'/condition;rp=folder/'RESULT.json';vp=folder/'VERIFY.json';r=read(rp);v=read(vp)
 if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(rp) or r['visual_checkpoint_sha256']!=VISION_SHA:raise ValueError('Actualoldvisionbodyaudit required '+condition)
 verify(r['source_hashes']);hashes=dict(r['source_hashes']);paths=[rp,vp,folder/'neural_decisions.npz',folder/'capture_rgb.npz',folder/'CAPTURES.json']
 for name,key in [('neural_decisions.npz','neural_decisions'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata')]:
  if sha(folder/name)!=r[key]['sha256']:raise ValueError('Actual visualartifactSHA '+name)
 hashes.update({str(p):sha(p) for p in paths});cap=read(folder/'CAPTURES.json')
 with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:
  keys=('X238','held_features','raw_turn','latent','time','image_capture_id','image_capture_time_s','image_release_time_s','vision_hidden_before','vision_hidden_after','raw_released_image_capture_id','actor_frame_frozen');a={k:z[k].copy() for k in keys}
 with np.load(folder/'capture_rgb.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('Only actualrgb key expected')
  rgb=z['rgb'].copy()
 if rgb.shape!=(770,2,32,32,3) or rgb.dtype!=np.uint8 or len(cap)!=770 or a['X238'].shape!=(2880,238) or a['held_features'].shape!=(2880,16) or a['vision_hidden_after'].shape!=(2880,32):raise ValueError('ActualRGB/neural header shapes')
 expected_raw=49+np.arange(2880)//4;expected_actor=np.minimum(expected_raw,529) if condition=='V_flip_fixedframe' else expected_raw;expected_freeze=np.arange(2880)>=1920 if condition=='V_flip_fixedframe' else np.zeros(2880,dtype=bool)
 if not np.array_equal(a['raw_released_image_capture_id'],expected_raw) or not np.array_equal(a['image_capture_id'],expected_actor) or not np.array_equal(a['actor_frame_frozen'],expected_freeze):raise ValueError('Releasedrealframe/declaredflipcontrol IDs')
 if np.max(abs(a['time']-(1.+np.arange(2880)*.005)))>1e-8:raise ValueError('Actualbodydecisionglobalclock')
 before={k:v.detach().clone() for k,v in vision.state_dict().items()};h=vision.initial_state(1);cache=None;last=-1;rawturn=[];lat=[];states=[];maxerr={k:0. for k in ('held_features','raw_turn','latent','vision_hidden_before','vision_hidden_after')}
 with torch.no_grad():
  for k in range(2880):
   fid=int(expected_actor[k]);row=cap[fid];time_s=float(a['time'][k])
   if row['capture_id']!=fid or abs(row['capture_time_s']-fid*.02)>1e-10 or abs(row['release_time_s']-(fid*.02+.02))>1e-10 or row['release_time_s']>time_s+1e-8 or abs(a['image_capture_time_s'][k]-row['capture_time_s'])>1e-10 or abs(a['image_release_time_s'][k]-row['release_time_s'])>1e-10 or hashlib.sha256(rgb[fid].tobytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('ActualRGB capture/release/hash/future')
   if fid!=last:cache=vision.encode(torch.from_numpy(rgb[fid:fid+1,None]))[0];last=fid
   actual={'held_features':cache[0].numpy().copy(),'vision_hidden_before':h[0].numpy().copy()}
   if k<960:t=cache.new_zeros(1);l=t.clone()
   else:t,l,h=vision.step(cache,h)
   actual.update(raw_turn=t[0].numpy().copy(),latent=l[0].numpy().copy(),vision_hidden_after=h[0].numpy().copy())
   for key,value in actual.items():
    maxerr[key]=max(maxerr[key],float(np.max(abs(value-a[key][k]))))
    if not np.array_equal(value,a[key][k]):raise ValueError('Frozen99698actualRGB replay not bitexact '+condition+' '+key+' row'+str(k))
   rawturn.append(float(t[0]));lat.append(float(l[0]));states.append(h[0].numpy().copy())
 if any(not torch.equal(vision.state_dict()[k],v) for k,v in before.items()):raise ValueError('Frozenvisiontensorchanged')
 u=np.asarray(rawturn,dtype=np.float64)[960:,None];x=a['X238'][960:,MASK];target=np.concatenate((np.zeros((1,1)),u[:-1]),0)
 if not np.isfinite(x).all() or not np.isfinite(u).all() or abs(u).max()>1:raise ValueError('Finite actualvisual andbody111')
 proof=dict(condition=condition,kind='ACTUAL_PAIRED_RENDERED_RGB_VISUAL_NEURAL_REPLAY',all_old_cache_turn_latent_h32_bitexact=True,maximum_errors=maxerr,raw_capture_min=49,raw_capture_max=768,actor_capture_max=int(expected_actor.max()),captures_not_yet_released_excluded=[769],visual_warm_hidden_held_zero=True,explicit_flip_actor_freeze=condition=='V_flip_fixedframe',actual_closed_occlusion_rgb=condition=='V_left_occluded',no_condition_or_stimulus_as_neural_input=True,actual_global_decision_rows=2880,training_suffix=[960,2880],new_physics_steps=0)
 return dict(body111=x,visual_input=u,teacher=target,current_visual=u.copy()),proof,hashes

def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('No dataset overwrite')
 source_review=SCOPE/'visual_source_audit/NEURAL_VISUAL_SOURCE_REVIEW.json';review=read(source_review)
 if not review['all_checks_passed'] or not all(review['checks'].values()):raise ValueError('Sourcearchiveaudit required')
 visual,p=load_visual();bp,b,digest=binding();groups={'train':[],'development':[]};proofs=[];hashes={str(source_review):sha(source_review)};hashes.update(p['source_hashes']);hashes.update(p['data_hashes']);meta=[]
 for name in NORMAL+DEV:
  sample,proof,refs=replay_visual(name,visual,torch,np);split='train' if name in NORMAL else 'development';groups[split].append(sample);proofs.append(proof);hashes.update(refs);meta.append(dict(split=split,condition=name,input_source_kind=proof['kind'],actual_paired_RGB=True,visual_signal_generated_from_actual_RGB=True,fresh_holdout=False))
 for name in AUX:
  folder=BODY/'trials'/name;r=read(folder/'RESULT.json');v=read(folder/'VERIFY.json');path=folder/'neural_decisions.npz'
  if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(folder/'RESULT.json') or sha(path)!=r['neural_decisions']['sha256']:raise ValueError('Actualbody auxiliary archiveaudit')
  verify(r['source_hashes']);hashes.update(r['source_hashes']);hashes.update({str(p):sha(p) for p in (folder/'RESULT.json',folder/'VERIFY.json',path)})
  with np.load(path,allow_pickle=False) as z:
   if z['X238'].shape!=(3080,238) or np.max(abs(z['time'][1160:]-(5.8+np.arange(1920)*.005)))>1e-8:raise ValueError('Auxactualbody3080/1160suffix')
   x=z['X238'][1160:,MASK].copy()
  zero=np.zeros((1920,1),dtype=np.float64);groups['train'].append(dict(body111=x,visual_input=zero,teacher=zero.copy(),current_visual=zero.copy()));meta.append(dict(split='train',condition=name,input_source_kind='EXPLICIT_ZERO_VISUAL_NEURAL_INPUT_INTERVENTION_ON_ACTUAL_BODY',actual_paired_RGB=False,no_visual_image_or_hidden_state_created=True,fresh_holdout=False))
 m,parent=initialize();outputs={}
 for split,samples in groups.items():
  value={k:np.stack([s[k] for s in samples],1) for k in samples[0]};value.update(weights=weights_for(value['teacher'],np),mean=parent['state_dict']['observation_mean'].numpy().copy(),std=parent['state_dict']['observation_std'].numpy().copy())
  if value['body111'].shape!=(1920,9 if split=='train' else 2,111) or np.max(abs(value['weights'].sum(0)-1))>1e-12 or not all(np.isfinite(v).all() for v in value.values()):raise ValueError('Fullactualdataset finite/dim/weights')
  out=HERE/(split+'.npz');np.savez_compressed(out,**value);outputs[split]=dict(path=str(out),sha256=sha(out))
 verify(hashes);save(HERE/'VISUAL_REPLAY.json',dict(all_bitexact=True,rows=proofs,visual_sha256=VISION_SHA,frozen_visual_parameters_unchanged=True,no_neural_input_from_labels=True,new_physics_steps=0));hashes[str(HERE/'VISUAL_REPLAY.json')]=sha(HERE/'VISUAL_REPLAY.json')
 save(HERE/'DATASET.json',dict(status='ACTUAL_RGB_SCALAR_INPUTS_AND_EXPLICIT_ZERO_NEURAL_AUXILIARIES',source_hashes=hashes,outputs=outputs,rows=meta,allowed_raw_indices=MASK,visual_sha256=VISION_SHA,parent_sha256=PARENT_SHA,source_binding_sha256=digest,teacher='one-decision delayed actualvisualsignal, initialDNhold0',no_fake_aux_rgb=True,normalizer_parent_exact=True,new_physics_steps=0))
 print('ACTUALRGB_REPLAY_BITEXACT_6TRAIN_2CONTROL_AND3_EXPLICIT_NEURALZERO_AUX_SAVED')
def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(d):
 import torch
 return tuple(torch.from_numpy(d[k]) for k in ('body111','visual_input','teacher','weights'))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-visual-calibration')
