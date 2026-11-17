"""Parent-only real sensory replay/source extraction; offlinepairings explicit, no invented images."""
from common import *
def weights_for(target,np):
 out=np.zeros_like(target)
 for j in range(target.shape[1]):
  a=np.abs(target[:,j,0]);groups=(a<=.02,a>.1,(a>.02)&(a<=.1));mass=(.5,.4,.1);den=sum(m for g,m in zip(groups,mass) if g.any())
  for g,m in zip(groups,mass):
   if g.any():out[g,j,0]=m/den/g.sum()
 return out
def source_case(folder):
 r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
 if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actualsourcecaseaudit '+str(folder))
 hashes={str(folder/n):sha(folder/n) for n in ('RESULT.json','VERIFY.json')};hashes.update(r['source_hashes']);verify(hashes)
 for key in ('neural_decisions','capture_rgb','captures_metadata','geometry'):
  if key in r:
   value=r[key]
   if sha(value['path'])!=value['sha256']:raise ValueError('ActualartifactSHA')
   hashes[value['path']]=value['sha256']
 return r,a,hashes

def visual_replay(name,vision,torch,np):
 folder=VISBODY/'trials'/name;r,a,refs=source_case(folder)
 if r['brain_checkpoint_sha256']!=PARENT_SHA or r['visual_checkpoint_sha256']!=VISION_SHA or not a['saved_actual_RGB_source_queue_brain_motor_replay']:raise ValueError('Latestvisual2123 actualRGBsource required')
 with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:record={k:z[k].copy() for k in ('X238','time','physical_step_ms','image_valid','image_capture_id','image_capture_time_s','image_release_time_s','held_features','vision_hidden_before','vision_hidden_after','visual_raw_turn','visual_latent')}
 with np.load(folder/'capture_rgb.npz',allow_pickle=False) as z:rgb=z['rgb'].copy()
 caps=read(folder/'CAPTURES.json')
 if rgb.shape!=(770,2,32,32,3) or rgb.dtype!=np.uint8 or record['X238'].shape!=(3080,238) or len(caps)!=770:raise ValueError('Latest actual3080/770shape')
 if np.max(abs(record['time']-np.arange(3080)*.005))>1e-8 or not np.array_equal(record['physical_step_ms'],np.arange(3080)*5):raise ValueError('Realvisualclock')
 for i,cap in enumerate(caps):
  if cap['capture_id']!=i or abs(cap['capture_time_s']-i*.02)>1e-10 or abs(cap['release_time_s']-i*.02-.02)>1e-10 or hashlib.sha256(rgb[i].tobytes()).hexdigest()!=cap['rgb_sha256']:raise ValueError('ActualRGBbytes/captureclock')
 h=vision.initial_state(1);cache=vision.head.weight.new_zeros((1,16));last=-1;turns=[];errors={k:0. for k in ('held_features','vision_hidden_before','vision_hidden_after','visual_raw_turn','visual_latent')}
 with torch.no_grad():
  for k in range(3080):
   rawfid=(k*5-20)//20;valid=rawfid>=0;fid=min(rawfid,529) if name=='V_flip_fixedframe' and k>=2120 else rawfid
   if bool(record['image_valid'][k])!=valid or int(record['image_capture_id'][k])!=fid:raise ValueError('ReleasedactualactorID')
   if valid:
    cap=caps[fid]
    if cap['release_time_s']>record['time'][k]+1e-8 or abs(record['image_capture_time_s'][k]-cap['capture_time_s'])>1e-10 or abs(record['image_release_time_s'][k]-cap['release_time_s'])>1e-10:raise ValueError('Futureimage')
    if fid!=last:cache=vision.encode(torch.from_numpy(rgb[fid:fid+1,None]))[0];last=fid
   before=h.clone()
   if k<1160:raw=cache.new_zeros(1);latent=raw.clone()
   else:raw,latent,h=vision.step(cache,h)
   values=dict(held_features=cache[0].numpy(),vision_hidden_before=before[0].numpy(),vision_hidden_after=h[0].numpy(),visual_raw_turn=raw[0].numpy(),visual_latent=latent[0].numpy())
   for key,value in values.items():
    error=float(np.max(abs(value-record[key][k])));errors[key]=max(errors[key],error)
    if not np.array_equal(value,record[key][k]):raise ValueError('Actualvisualbitexact '+name+' '+key+' row'+str(k))
   turns.append(float(raw[0]))
 return record['X238'][1160:,MASK].copy(),np.asarray(turns)[1160:,None],dict(condition=name,all_actual_RGB_cache_h32_raw_latent_exact=True,rows=3080,ANNsuffix=[1160,3080],maximum_errors=errors,first_invalid_no_fake_RGB=True,actorfreeze=name=='V_flip_fixedframe'),refs

def odor_replay(name,odor,torch,np):
 folder=(B if name.startswith('B_') else C)/'trials'/name;r,a,refs=source_case(folder)
 if r['behavior_checkpoint_sha256']!=ODOR_SHA:raise ValueError('Sameactual66f source required')
 with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('X238','time','receptors','concentration','filter_before','filter_after','pending_before','pending_after','raw_command','raw_turn','behavior_hidden_before','behavior_hidden_after')}
 if d['X238'].shape!=(2880,238) or d['receptors'].shape!=(2880,2) or np.max(abs(d['time']-(1.+np.arange(2880)*.005)))>1e-8:raise ValueError('Actualodor2880clock')
 h=odor.initial_state(1);turns=[];alpha=-np.expm1(-.005/.05);filterstate=np.zeros(2);pending=np.zeros(2);errors={k:0. for k in ('raw_command','raw_turn','behavior_hidden_before','behavior_hidden_after')}
 with torch.no_grad():
  for k in range(2880):
   before=h.clone()
   if k<960:raw=h.new_zeros((1,2))
   else:
    if not np.array_equal(d['receptors'][k],pending) or not np.array_equal(d['filter_before'][k],filterstate) or not np.array_equal(d['pending_before'][k],pending):raise ValueError('Realpreviouspendingcausality')
    Cnow=d['concentration'][k];filterstate=filterstate+alpha*(Cnow/(1.+Cnow)-filterstate);pending=filterstate.copy()
    if not np.array_equal(d['filter_after'][k],filterstate) or not np.array_equal(d['pending_after'][k],pending):raise ValueError('Actualreceptorchain')
    raw,h=odor.step(torch.from_numpy(d['receptors'][k:k+1]),h)
   values=dict(raw_command=raw[0].numpy(),raw_turn=raw[0,1].numpy(),behavior_hidden_before=before[0].numpy(),behavior_hidden_after=h[0].numpy())
   for key,value in values.items():
    error=float(np.max(abs(value-d[key][k])));errors[key]=max(errors[key],error)
    if not np.array_equal(value,d[key][k]):raise ValueError('Actualodorbitexact '+name+' '+key+' row'+str(k))
   if float(raw[0,0])!=0:raise ValueError('Odor structural speed mustzero')
   turns.append(float(raw[0,1]))
 return d['X238'][960:,MASK].copy(),np.asarray(turns)[960:,None],dict(condition=name,same66f=True,actor_input='actualreceptors no intervention B/C',ANNsuffix=[960,2880],all_rawturn_h32_filter_pending_exact=True,maximum_errors=errors),refs

def zero_receptor_neural(odor,torch,np):
 h=odor.initial_state(1);out=[]
 with torch.no_grad():
  for _ in range(1920):raw,h=odor.step(torch.zeros((1,2),dtype=torch.float64),h);out.append(float(raw[0,1]))
 return np.asarray(out,dtype=np.float64)[:,None]
def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('Inspectbeforeanyrepeat')
 summary=read(VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json');review=read(VISBODY/'PARENT_READY_REVIEW.json');jobs=read(VISBODY/'JOBS.json')
 if not(summary['normal6_all_functional_pass'] and summary['control2_all_pass'] and summary['all10_integrity_verified'] and len(jobs)==22 and all(v['exit_code']==0 for v in jobs)):raise ValueError('Visualindependentgate notpassed')
 if review['summary_sha256']!=sha(VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json'):raise ValueError('VisualsummaryproofSHA')
 vision,odor,vp,op=load_sensory();vision_before=tensor_hashes(vision);odor_before=tensor_hashes(odor);refs={str(VISBODY/n):sha(VISBODY/n) for n in ('NEURAL_VISUAL_BODY_SUMMARY.json','PARENT_READY_REVIEW.json','JOBS.json')}
 for key in ('source_hashes','data_hashes'):refs.update(vp[key]);refs.update(op['config'][key])
 verify(refs);vs={};osamples={};proofs=[]
 for name in NORMAL+DEV[:2]:x,u,proof,h=visual_replay(name,vision,torch,np);vs[name]=(x,u);proofs.append(proof);refs.update(h)
 for name in BNAMES+CNAMES:x,u,proof,h=odor_replay(name,odor,torch,np);osamples[name]=(x,u);proofs.append(proof);refs.update(h)
 natural=zero_receptor_neural(odor,torch,np);zero=np.zeros((1920,1),dtype=np.float64);train=[];dev=[];metadata=[]
 def add(split,name,x,v,o,desired,kind,**extra):
  if x.shape!=(1920,111) or v.shape!=(1920,1) or o.shape!=v.shape or desired.shape!=v.shape:raise ValueError('Sampleshape')
  target=np.concatenate((zero,desired[:-1]),0);sample=dict(body111=x.copy(),modalities=np.concatenate((v,o),1),teacher=target,current_desired=desired.copy());(train if split=='train' else dev).append(sample);metadata.append(dict(split=split,condition=name,input_source_kind=kind,fresh_holdout=False,physical_simultaneous_multisensory=False,**extra))
 for name in NORMAL:x,v=vs[name];add('train','VIS_'+name,x,v,natural,v,'ACTUAL_RGB_WITH_GENERATED_NATURAL_ZERO_RECEPTOR_ODOR_NEURAL_INPUT',no_fake_RGB=True,odor_input_not_exactzero=True)
 for name in BNAMES+CNAMES:x,o=osamples[name];add('train',name,x,zero,o,o,'ACTUAL_ODOR_RECEPTORS_WITH_EXPLICIT_VISUAL_NEURALZERO',visual_neuralzero_not_fakeRGB=True)
 for vname,oname in zip(NORMAL,BNAMES):
  x,v=vs[vname];o=osamples[oname][1];desired=(v*np.abs(v)+o*np.abs(o))/(np.abs(v)+np.abs(o)+1e-8);add('train','PAIR_'+vname,x,v,o,desired,'OFFLINE_NON_SIMULTANEOUS_MODAL_NEURAL_STREAM_PAIRING',visual_source=vname,odor_source=oname,teacher_engineering_only_not_runtime=True)
 for name in AUX:
  folder=AUXBODY/'trials'/name;r,a,h=source_case(folder);refs.update(h)
  with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:
   if z['X238'].shape!=(3080,238):raise ValueError('Realauxbodyshape')
   x=z['X238'][1160:,MASK].copy()
  add('train',name,x,zero,zero,zero,'EXPLICIT_BOTH_MODAL_NEURALZERO_ON_ACTUAL_BODY',not_actual_sensory_trajectory=True)
 for name in DEV[:2]:x,v=vs[name];add('development',name,x,v,natural,v,'KNOWN_VISUAL_CONTROL_WITH_GENERATED_NATURAL_NOODOR')
 for name in CNAMES:x,o=osamples[name];add('development','DEV_'+name,x,zero,o,o,'DUPLICATED_KNOWN_ODOR_C_DEVELOPMENT_NOT_HOLDOUT')
 m,parent,old=initialize();outputs={}
 for split,samples in (('train',train),('development',dev)):
  value={k:np.stack([q[k] for q in samples],1) for k in samples[0]};value.update(weights=weights_for(value['teacher'],np),mean=m.observation_mean.numpy().copy(),std=m.observation_std.numpy().copy())
  if len(samples)!=(27 if split=='train' else 8) or np.max(abs(value['weights'].sum(0)-1))>1e-12 or not all(np.isfinite(q).all() for q in value.values()):raise ValueError('Full27source data/finite/mass')
  p=HERE/(split+'.npz');np.savez_compressed(p,**value);outputs[split]=dict(path=str(p),sha256=sha(p))
 if tensor_hashes(vision)!=vision_before or tensor_hashes(odor)!=odor_before:raise ValueError('Frozenmodalparametersdrift')
 verify(refs);save(HERE/'SENSORY_REPLAY.json',dict(all_bitexact=True,rows=proofs,actual_visual_cases=8,actual_odor_cases=12,natural_noodor_generated_not_actual_world=True,natural_noodor_RMS=float(np.sqrt(np.mean(natural**2))),frozen_visual_hashes=vision_before,frozen_odor_hashes=odor_before,noDtraining=True,new_physics_steps=0));refs[str(HERE/'SENSORY_REPLAY.json')]=sha(HERE/'SENSORY_REPLAY.json')
 save(HERE/'DATASET.json',dict(kind='ACTUAL_SENSOR_SINGLEMODAL_AND_DECLARED_OFFLINEPAIR27',source_hashes=refs,outputs=outputs,rows=metadata,train_count=27,development_count=8,source_counts=dict(visual6=6,odorB6=6,odorC6=6,offlinepair6=6,neuralzeroaux3=3),allowed_raw_indices=MASK,source_binding_sha256=NECK_SHA,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,teacher='singlemodal source NN output / pairedengineeringstrength onlyloss, delayedone step first0',same_parent_normalizer=True,no_neural_input_from_labels_XYZ_time=True,old_D_delayedcut_failure_retained=True,new_physics_steps=0));print('ACTUAL8RGB_12ODOR_BITEXACT_WITH27_DECLARED_SOURCE_ROWS_NO_NEW_PHYSICS')
def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(d):
 import torch
 return tuple(torch.from_numpy(d[k]) for k in ('body111','modalities','teacher','weights'))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
