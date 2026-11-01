"""Saved-array-only warmup audit. No model imports, forward, or integration."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-warmup-offline/20261101T190431Z')
R=Path('/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z')
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):
 (O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def episodes(mask,idx,t):
 # Every adjacent saved bit is retained; no merging, interpolation or debounce.
 cuts=np.r_[0,np.flatnonzero(np.diff(mask.astype(int)))+1,len(mask)]
 return [dict(state=bool(mask[a]),first_index=int(idx[a]),last_index=int(idx[b-1]),first_time_s=float(t[a]),last_time_s=float(t[b-1]),observed_span_s=float(t[b-1]-t[a]),sample_count=int(b-a),left_censored=bool(a==0),right_censored=bool(b==len(mask))) for a,b in zip(cuts[:-1],cuts[1:])]
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'METRICS.json').exists(), 'sealed/result directory cannot be rerun'
 files=[S/n for n in ['BODY.npz','GAIT.npz','MODEL.xml','PREREGISTRATION.json','GAINS.json','GAIT_RETARGET.json','run_physical_walk_prewarm.py','HANDOVER_AFTER.npz']]+[R/'STANDARD_GAIT_FK.npz',R/'REFERENCE_METADATA.json',Path(__file__)]
 hashes={str(p):sha(p) for p in files}
 if (O/'PREREGISTRATION.json').exists():
  dump('PRE_RUN_CORRECTION.json',dict(reason='Initial exit1 before metrics: first worldbody child is ground, not free Thorax. Select named Thorax; definitions unchanged.',initial_script_sha256=json.loads((O/'PREREGISTRATION.json').read_text())['source_sha256'][str(Path(__file__))],corrected_source_sha256=hashes,first_exit_code=1,model_load=0,forward=0,body_step=0,neural_step=0))
 else: dump('PREREGISTRATION.json',dict(source_sha256=hashes,scope='Pure saved-array analysis; no model load/forward/body/neural integration',stages={'settle':[0,.15],'ramp':[.15,.25],'steady':[.25,1.3]},interval_convention='left closed/right open using rounded saved 0.5ms grid; 1.3s after-withdrawal BODY row excluded',contact='saved geometric foot_contact Boolean; load separately Fz>0; neither implies sufficient support',episodes='unmerged sampled bit runs, observed span=(last-first) time; stage edges censored, no reconstructed microevents',thresholds='descriptive only; no >=4-foot or 60ms gate',coordinates='root translation/quaternion only if free root body is Thorax; Tarsus5 origins, not contact material points',joint_error='reconstruct declared periodic GAIT and ramp at saved time; also last integration start time dt earlier; not a torque reproduction',limitations='0.5ms samples cannot certify 12.5us event stream, exact slip, full mesh clearance, biological gait or autonomous control'))
 with np.load(S/'BODY.npz') as z:b={k:z[k].copy() for k in z.files}
 with np.load(S/'GAIT.npz') as z:g={k:z[k].copy() for k in z.files}
 prereg=json.loads((S/'PREREGISTRATION.json').read_text()); weight=json.loads((S/'GAINS.json').read_text())['weight']
 t=b['time'];ticks=np.rint(t/.0005).astype(int);assert np.max(abs(t-ticks*.0005))<1e-9 and np.all(np.diff(ticks)==1)
 idx=np.flatnonzero(ticks<2600);assert len(idx)==2600 and ticks[2600]==2600
 tree=ET.parse(S/'MODEL.xml');root=tree.getroot().find("worldbody/body[@name='Thorax']");assert root.get('name')=='Thorax' and (root.find('freejoint') is not None or any(j.get('type')=='free' for j in root.findall('joint')))
 # The free-joint quaternion is world orientation of the Thorax root body.
 q=b['qpos'][idx];quat=q[:,[4,5,6,3]];rot=Rotation.from_quat(quat).as_matrix()
 assert np.max(abs(q[:,:3]-b['thorax'][idx]))<1e-10
 foot=np.einsum('nli,nij->nlj',b['foot_origins'][idx]-q[:,:3,None].transpose(0,2,1),rot)
 qa=g['qpos_addresses'].astype(int);assert len(qa)==42 and len(set(qa))==42
 key=np.fromstring(tree.getroot().find('keyframe/key').get('qpos'),sep=' ');assert np.allclose(key,b['qpos'][0],rtol=0,atol=1e-12)
 gait=CubicSpline(g['phase'],g['qpos_hinges'],axis=0,bc_type='periodic')
 def target(tt):
  ramp=np.clip((tt-.15)/.10,0,1);phase=np.maximum(0,tt-.15)*2*np.pi*5
  return (1-ramp[:,None])*key[qa]+ramp[:,None]*gait(phase%(2*np.pi))
 desired=target(t[idx]);previous=target(np.maximum(0,t[idx]-prereg['dt']));err=q[:,qa]-desired
 summaries={};all_episodes={}
 for label,lo,hi in [('settle',0,300),('ramp',300,500),('steady',500,2600)]:
  sel=idx[(ticks[idx]>=lo)&(ticks[idx]<hi)];local=np.searchsorted(idx,sel);f=b['foot_force'][sel,:,2];a=b['assist'][sel]
  legs={};stageepisodes={}
  for j,leg in enumerate(LEGS):
   ee=episodes(b['foot_contact'][sel,j],sel,t[sel]);stageepisodes[leg]=ee
   complete=[e for e in ee if not e['left_censored'] and not e['right_censored']]
   legs[leg]=dict(body_xyz_min=foot[local,j].min(0),body_xyz_max=foot[local,j].max(0),body_xyz_span=np.ptp(foot[local,j],axis=0),world_z_min=float(b['foot_origins'][sel,j,2].min()),world_z_max=float(b['foot_origins'][sel,j,2].max()),contact_sample_fraction=float(b['foot_contact'][sel,j].mean()),positive_Fz_sample_fraction=float((f[:,j]>0).mean()),Fz_min=float(f[:,j].min()),Fz_max=float(f[:,j].max()),Fz_mean=float(f[:,j].mean()),complete_contact_spans_s=[e['observed_span_s'] for e in complete if e['state']],complete_noncontact_spans_s=[e['observed_span_s'] for e in complete if not e['state']],transitions=int(np.count_nonzero(np.diff(b['foot_contact'][sel,j]))))
  summaries[label]=dict(n_samples=len(sel),first_s=float(t[sel[0]]),last_s=float(t[sel[-1]]),root_displacement=q[local[-1],:3]-q[local[0],:3],com_start=b['com'][sel[0]],com_end=b['com'][sel[-1]],com_z_min=float(b['com'][sel,2].min()),nonfoot_samples=int(np.count_nonzero(b['nonfoot_count'][sel])),foot_total_Fz_over_weight_min=float((f.sum(1)/weight).min()),foot_total_Fz_over_weight_mean=float((f.sum(1)/weight).mean()),positive_load_feet_histogram=np.bincount((f>0).sum(1),minlength=7),root_assist_max=float(abs(a[:,:6]).max()),leg_assist_norm_mean=float(np.linalg.norm(a[:,6:],axis=1).mean()),leg_assist_norm_max=float(np.linalg.norm(a[:,6:],axis=1).max()),external_max=float(abs(b['external'][sel]).max()),joint_target_rms_rad=float(np.sqrt(np.mean(err[local]**2))),joint_target_max_rad=float(abs(err[local]).max()),joint_last_step_target_rms_rad=float(np.sqrt(np.mean((q[local][:,qa]-previous[local])**2))),LF_body_y_min=float(foot[local,0,1].min()),RF_body_y_max=float(foot[local,3,1].max()),legs=legs)
  all_episodes[label]=stageepisodes
 np.savez_compressed(O/'ANALYSIS_ARRAYS.npz',body_indices=idx,time_s=t[idx],phase_rad=np.maximum(0,t[idx]-.15)*2*np.pi*5%(2*np.pi),foot_thorax=foot,joint_target=desired,joint_error=err,joint_last_step_target=previous)
 dump('METRICS.json',dict(model_sha256=hashes[str(S/'MODEL.xml')],weight_model=weight,body_sample_dt_s=.0005,physical_dt_s=prereg['dt'],excluded_boundary_index=2600,stages=summaries,scope='No behavior pass threshold. Cartesian endpoints are Tarsus5 body origins; signed force is saved world Fz.'))
 dump('EPISODES.json',all_episodes)
 assert hashes=={str(p):sha(p) for p in files}
 dump('EXECUTION.json',dict(model_load=0,forward=0,body_step=0,neural_step=0,source_hashes_unchanged=True,exit_code=0))
 print(json.dumps({k:{n:v for n,v in s.items() if n!='legs'} for k,s in summaries.items()},ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
if __name__=='__main__':main()
