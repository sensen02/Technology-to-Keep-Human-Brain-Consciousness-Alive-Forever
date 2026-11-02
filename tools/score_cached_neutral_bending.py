"""Original neutral body-cache bending score: FK only, no body/neural integration."""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,math,hashlib
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
T=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz')
F=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z/FROZEN_REFERENCE.json')
O=Path('/home/sensen/Desktop/engine修复验收/s4-neutral-bending-cache-score/20261102T042848Z')
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def geometric(points):
 a=points[:,:,:-2]-points[:,:,1:-1];b=points[:,:,2:]-points[:,:,1:-1]
 return np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1))
def main():
 import mujoco as mj
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 O.mkdir(parents=True,exist_ok=True);assert F.exists(),'Neutralgate must have passed and frozen reference first'
 assert not (O/'RESULT.json').exists(),'Never overwrite accepted score'
 sources=[Path(__file__),B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'CHUNKS.json',B/'CONTACT_SCHEMA.json',T,F];hashes={str(p):sha(p) for p in sources}
 dump(O/('CACHED_ARRAY_ATTEMPT.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Compute motion and support on original neutral cache only; no second neutral body trial',source_sha256=hashes,windows_s=[.1,.3],marker='Same four body origins as newphysicaltraining, source13/37times nearest.5msendpoint <=.25ms',force='Eachcontact positive nonfoot floor Fz sum, actual native12.5us endpoints, no margin/phase heuristic; aggregate not substituted',planned=dict(model_compiles=1,kinematics=37,forward=0,body_step=0,neural_step=0,training=0)))
 spec=_load_mjcf(B/'RUN/MODEL.xml');model=spec.compile();data=mj.MjData(model);assert (model.nq,model.nv,model.nu,model.nmesh)==(49,48,90,71)
 body=np.load(B/'RUN/BODY.npz');target=np.load(T);reference=json.loads(F.read_text());idx=np.array([np.argmin(abs(body['time']-(1.3+t))) for t in target['time_relative_s']]);timeerror=body['time'][idx]-(1.3+target['time_relative_s']);assert abs(timeerror).max()<=.00025+1e-10
 ids=np.array([[model.body(leg+'Coxa').id,model.body(leg+('Trochanter' if leg[1]=='F' else 'Femur')).id,model.body(leg+'Tibia').id,model.body(leg+'Tarsus1').id] for leg in LEGS]);points=[]
 for index in idx:
  data.qpos[:]=body['qpos'][index];mj.mj_kinematics(model,data);points.append(data.xpos[ids].copy())
 points=np.array(points);prediction=geometric(points);observed=target['angles_rad'];assert prediction.shape==observed.shape
 floor=set(np.flatnonzero(model.geom_type==mj.mjtGeom.mjGEOM_PLANE).tolist());native=[]
 used_chunks=[]
 for entry in json.loads((B/'CHUNKS.json').read_text()):
  path=Path(entry['file']);assert sha(path)==entry['sha256']
  with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
  mask=(z['kind']==0)&(z['step']>104000)&(z['step']<=128000)
  if not mask.any():continue
  used_chunks.append(dict(file=str(path),sha256=entry['sha256']))
  for i in np.flatnonzero(mask):
   contacts=z['contact_rows'][z['contact_offsets'][i]:z['contact_offsets'][i+1]];nfpositive=0.
   for row in contacts:
    g1=int(row[0]);g2=int(row[1])
    if int(row[4])<0 and (g1 in floor or g2 in floor):nfpositive+=max(float(row[17])*(1 if g1 in floor else -1),0.)
   assert z['assist_root_max'][i]==0 and z['assist_leg_norm'][i]==0 and z['external_max'][i]==0 and z['warning_count'][i]==0
   native.append([float(z['time'][i]),*z['foot_force'][i,:,2],nfpositive,float(z['nonfoot_force'][i,2]),float(z['nonfoot_count'][i]),float(z['com'][i,2]),*z['root'][i],float(np.mean(z['ctrl'][i]**2))])
 native=np.array(native);assert native.shape==(24000,19);assert np.array_equal(np.round((native[:,0]-1.3)/.0000125).astype(int),np.arange(1,24001))
 q0=np.array(reference['q0']);q=native[:,14:18];dot=abs(q@q0)/np.linalg.norm(q,axis=1);angle=2*np.arccos(np.clip(dot,0,1));W=reference['weight'];answers={}
 for nframes,steps in [(13,8000),(37,24000)]:
  n=native[:steps];motiondiff=prediction[:nframes]-observed[:nframes];terms=dict(motion=float(np.mean(motiondiff[target['angle_valid'][:nframes]]**2)),height=float(np.mean(((n[:,10]-reference['h0'])/reference['h0'])**2)),foot=float(np.mean((n[:,1:7].sum(1)/W-1)**2)),nonfoot=float(np.mean(n[:,7]/W)),rotation=float(.01*np.mean(angle[:steps]**2)),ctrl=float(.001*np.mean(n[:,18])),reg=0.)
  hits=np.flatnonzero(n[:,7]>0)
  answers[str(nframes)]=dict(source_samples=nframes,duration_s=steps*.0000125,total_loss=sum(terms.values()),terms=terms,first_positiveNFz_s=None if not len(hits) else float(n[hits[0],0]-1.3),positiveNFz_endpoints=int(len(hits)),mean_foot_load=float(n[:,1:7].sum(1).mean()/W),per_leg_bending_MSE=np.mean(motiondiff**2,axis=(0,2)).tolist())
 np.savez_compressed(O/'CACHED_TARGET_AND_SUPPORT.npz',points=points,prediction_angles_rad=prediction,target_angles_rad=observed,mask=target['angle_valid'],source_frame=target['source_frame'],source_nominal_time_s=target['time_relative_s'],body_indices=idx,physical_absolute_time_s=body['time'][idx],time_error_s=timeerror,native_loss_rows=native)
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',dict(passed=True,source_sha256=hashes,used_native_chunks=used_chunks,windows=answers,reference=reference,calls=dict(model_compile=1,kinematics=37,forward=0,body_step=0,network=0,neural_step=0,training=0),limits='Cache FKderivedposes only; native forces from original archived trajectory, no neutral body rerun; all37validation includes trainingprefix.'))
 print(json.dumps(answers,indent=2))
if __name__=='__main__':main()
