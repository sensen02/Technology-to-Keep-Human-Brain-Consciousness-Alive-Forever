"""Parent actual24 archivedstate inputs. Teacherbound onlyoffline; no fake sensoryframes."""
from common import *
def weights_for(y,np):
 w=np.zeros_like(y,dtype=np.float64)
 for j in range(y.shape[1]):
  a=np.abs(y[:,j,0]);groups=(a<=.02,a>.1,(a>.02)&(a<=.1));masses=(.5,.4,.1);total=sum(m for g,m in zip(groups,masses) if g.any())
  for group,mass in zip(groups,masses):
   if group.any():w[group,j,0]=mass/total/group.sum()
 return w
def desired_teacher(u,kind,np):
 if kind=='visual':desired=u[:,0:1]
 elif kind=='odor':desired=u[:,1:2]
 elif kind=='joint':
  v=u[:,0:1];o=u[:,1:2];desired=(v*np.abs(v)+o*np.abs(o))/(np.abs(v)+np.abs(o)+1e-8);desired=np.clip(desired,-.78,.78)
 elif kind=='neuralzero':desired=np.zeros((1920,1),dtype=np.float64)
 else:raise ValueError('Offlinekind')
 return desired.copy(),np.concatenate((np.zeros((1,1),dtype=np.float64),desired[:-1]),0)
def actual_archive(folder,condition,vision,odor,latest,torch,np):
 r=read(folder/'RESULT.json');v=read(folder/'VERIFY.json');verify(r['source_hashes'])
 if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(folder/'RESULT.json') or r['visual_checkpoint_sha256']!=VISION_SHA:raise ValueError('Completeactualarchiveauditrequired')
 for filename,key in [('neural_decisions.npz','neural_decisions'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata')]:
  if sha(folder/filename)!=r[key]['sha256']:raise ValueError('ActualartifactSHA '+filename)
 with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:
  names=['X238','time','image_capture_id','held_features','vision_hidden_before','vision_hidden_after','visual_raw_turn','visual_latent']+(['actor_receptors'] if latest else []);a={k:z[k].copy() for k in names}
 with np.load(folder/'capture_rgb.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('ActualRGBonly')
  rgb=z['rgb'].copy()
 caps=read(folder/'CAPTURES.json')
 if a['X238'].shape!=(3080,238) or a['held_features'].shape!=(3080,16) or rgb.shape!=(770,2,32,32,3) or rgb.dtype!=np.uint8 or len(caps)!=770 or np.max(abs(a['time']-np.arange(3080)*.005))>1e-8:raise ValueError('Actual3080/770shapeclock')
 if latest and a['actor_receptors'].shape!=(3080,2):raise ValueError('Actualactorreceptorsnotavailable')
 vh=vision.initial_state(1);oh=odor.initial_state(1);held=vision.head.weight.new_zeros((1,16));last=-1;vs=[];osig=[];zero_receptors=np.zeros(2);inputhash=hashlib.sha256();maxerr={k:0. for k in ('held_features','vision_hidden_before','vision_hidden_after','visual_raw_turn','visual_latent')}
 with torch.no_grad():
  for k in range(3080):
   before=vh[0].numpy().copy();fid=int(a['image_capture_id'][k]);ms=k*5;rawfid=-1 if ms<20 else (ms-20)//20;expected=min(rawfid,529) if condition=='V_flip_fixedframe' and ms>=10600 else rawfid
   if fid!=expected:raise ValueError('Actualreleasedactorimageclock '+condition)
   if fid>=0:
    cap=caps[fid]
    if cap['capture_id']!=fid or abs(cap['capture_time_s']-fid*.02)>1e-10 or abs(cap['release_time_s']-(fid*.02+.02))>1e-10 or cap['release_time_s']>ms*.001+1e-8 or hashlib.sha256(rgb[fid].tobytes()).hexdigest()!=cap['rgb_sha256']:raise ValueError('Capture/release/RGBhash')
    if fid!=last:held=vision.encode(torch.from_numpy(rgb[fid][None,None]))[0];last=fid
   receptor=a['actor_receptors'][k] if latest else zero_receptors;inputhash.update(np.asarray(receptor,dtype=np.float64).tobytes())
   if k<1160:vr=held.new_zeros(1);vl=vr.clone();orr=vr.clone()
   else:vr,vl,vh=vision.step(held,vh);odcmd,oh=odor.step(torch.tensor(receptor[None],dtype=torch.float64),oh);orr=odcmd[:,1]
   measured=dict(held_features=held[0].numpy().copy(),vision_hidden_before=before,vision_hidden_after=vh[0].numpy().copy(),visual_raw_turn=np.asarray(float(vr[0])),visual_latent=np.asarray(float(vl[0])))
   for key,value in measured.items():
    maxerr[key]=max(maxerr[key],float(np.max(abs(value-a[key][k]))))
    if not np.array_equal(value,a[key][k]):raise ValueError('Frozen99698notexact '+condition+' '+key+' row'+str(k))
   vs.append(float(vr[0]));osig.append(float(orr[0]))
 x=a['X238'][1160:,MASK].copy();u=np.stack((vs,osig),axis=-1)[1160:];kind='joint' if condition.startswith('F_') else 'odor' if condition.startswith('O_') else 'visual';desired,y=desired_teacher(u,kind,np)
 hashes=dict(r['source_hashes']);hashes.update({str(p):sha(p) for p in (folder/'RESULT.json',folder/'VERIFY.json',folder/'neural_decisions.npz',folder/'capture_rgb.npz',folder/'CAPTURES.json')})
 proof=dict(condition=condition,archive=str(folder),actualRGBreplay99698bitexact=True,visual_errors=maxerr,odor_input_kind='actual_actor_receptors_same_as_recorded66f' if latest else 'declared_physically_absent_zero_receptors_natural_newNNbias',new_odor_output_not_claimed_equal_old66f=True,actual_receptor_stream_sha256=inputhash.hexdigest(),old_archive_integrity_passed=True,source_field_semantics='Labels/environmentnotactorinputs;kindonlyselectsofflineloss',rows=3080,ANNsuffix=[1160,3080],teacher_kind=kind,foodarrival_claim=False)
 return dict(body111=x,modalities=u,teacher=y,current_desired=desired),proof,hashes

def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('Inspect prior aggregate artifacts')
 ob=odor_binding();vision,odor,vp,op=load_sensory();fixedvision={k:v.detach().clone() for k,v in vision.state_dict().items()};fixedodor={k:v.detach().clone() for k,v in odor.state_dict().items()};summary=read(BODY/'SENSORY_BODY_SUMMARY.json');review=read(BODY/'PARENT_READY_REVIEW.json');jobs=read(BODY/'JOBS.json')
 if summary['kind']!='SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1' or not summary['all14_integrity_verified'] or not review['integrity_all_pass'] or review['summary_sha256']!=sha(BODY/'SENSORY_BODY_SUMMARY.json') or len(jobs)!=30 or not all(j['exit_code']==0 for j in jobs):raise ValueError('Actualall14bodyqualification')
 groups={'train':[],'development':[]};metadata=[];proofs=[];hashes={str(p):sha(p) for p in (BODY/'SENSORY_BODY_SUMMARY.json',BODY/'PARENT_READY_REVIEW.json',BODY/'JOBS.json',HERE/'PARENT_ODOR_BINDING.json')};hashes.update(ob['source_hashes']);hashes.update(vp['source_hashes']);hashes.update(vp['data_hashes']);actuals={}
 for name in LATEST:
  sample,proof,refs=actual_archive(BODY/'trials'/name,name,vision,odor,True,torch,np);groups['train'].append(sample);actuals[name]=sample;proofs.append(proof);hashes.update(refs);metadata.append(dict(split='train',name='LATEST_'+name,actual_paired_RGB_and_receptor=True,source_kind='ACTUAL_LATEST14_BODY_INPUTS',newodor_output_recomputed=True,fresh_holdout=False))
 for name in NORMAL+DEV:
  sample,proof,refs=actual_archive(VISBODY/'trials'/name,name,vision,odor,False,torch,np);split='train' if name in NORMAL else 'development';groups[split].append(sample);proofs.append(proof);hashes.update(refs);metadata.append(dict(split=split,name='PRIOR_'+name if split=='train' else name,actual_paired_RGB_and_receptor=False,source_kind='ACTUAL_PRIOR_VISUAL_BODY_WITH_DECLARED_PHYSICAL_ZERO_ODOR_RECEPTORS',natural_noodor_NNsignal_not_neuralzero=True,fresh_holdout=False))
 for name in AUX:
  x=actuals[name]['body111'].copy();u=np.zeros((1920,2));desired,y=desired_teacher(u,'neuralzero',np);groups['train'].append(dict(body111=x,modalities=u,teacher=y,current_desired=desired));metadata.append(dict(split='train',name='ZERO_'+name,source_kind='EXPLICIT_BOTH_NEURAL_INPUTS_ZERO_ON_ACTUAL_BODY',no_RGB_or_receptors_created=True,actual_paired_RGB_and_receptor=False,fresh_holdout=False))
 if any(not torch.equal(vision.state_dict()[k],v) for k,v in fixedvision.items()) or any(not torch.equal(odor.state_dict()[k],v) for k,v in fixedodor.items()):raise ValueError('Frozenmodalnetworkchanged')
 m,parent=initialize();outputs={}
 for split,samples in groups.items():
  value={k:np.stack([s[k] for s in samples],1) for k in samples[0]};value.update(weights=weights_for(value['teacher'],np),mean=parent['state_dict']['observation_mean'].numpy().copy(),std=parent['state_dict']['observation_std'].numpy().copy())
  if value['body111'].shape!=(1920,24 if split=='train' else 2,111) or value['modalities'].shape!=(1920,24 if split=='train' else 2,2) or value['teacher'].shape!=(1920,24 if split=='train' else 2,1) or np.max(abs(value['weights'].sum(0)-1))>1e-12 or not all(np.isfinite(v).all() for v in value.values()):raise ValueError('Correct1920aggregate24shapes/weights/finite')
  path=HERE/(split+'.npz');np.savez_compressed(path,**value);outputs[split]=dict(path=str(path),sha256=sha(path))
 verify(hashes);save(HERE/'SENSORY_REPLAY.json',dict(all_actual_visual_bitexact=True,all_old_actor_receptor_inputs_preserved=True,newodor_output_expected_different_from_old66f=True,rows=proofs,odor_sha256=ob['sha256'],frozen_modal_parameters_unchanged=True,new_physics_steps=0));hashes[str(HERE/'SENSORY_REPLAY.json')]=sha(HERE/'SENSORY_REPLAY.json');save(HERE/'DATASET.json',dict(status='ACTUAL14_PLUS6_PLUS4_ZERO_AUX_AGGREGATION',source_hashes=hashes,outputs=outputs,rows=metadata,source_counts=dict(actual_latest14=14,priorvisual6=6,explicitneuralzero4=4,total24=24),visual_sha256=VISION_SHA,odor_sha256=ob['sha256'],parent_sha256=PARENT_SHA,normalizer_parent_exact=True,offline_jointteacher_bound_only=True,new_physics_steps=0));print('ACTUAL14+6+4_SOURCE_BOUND_DATASET24_PREPARED')
def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(d):
 import torch
 return tuple(torch.from_numpy(d[k]) for k in ('body111','modalities','teacher','weights'))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
