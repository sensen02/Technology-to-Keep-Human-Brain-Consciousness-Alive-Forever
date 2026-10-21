"""Data-driven keypoint pose fit: BBNPYX tracked keypoints -> our model's leg joints.

Kinematics only (mj_kinematics, no dynamics, no contact, no stepping). The dataset
declares no length unit, so scale is an explicit swept parameter and the whole
curve is reported; the fit target is keypoint residual only.
"""
from pathlib import Path
import json,time,hashlib,zipfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
SCALES=[0.1,0.25,0.4,0.5,0.75,1.0,1.25,1.5,2.0]
NOMINAL_SCALE=1.0
FRAME_STRIDE=20
KEY=['ThCx','CxTr','TrFe','FeTi','TiTar','Tar']
# Declared correspondence. Front legs (LF/RF) have a separate Trochanter body; mid/hind legs
# (LM/LH/RM/RH) do NOT - their trochanter joints sit on <side>Femur, so CxTr maps to Femur and
# the dataset TrFe keypoint has no counterpart there (declared missing, not silently substituted).
BODY_FRONT={'ThCx':'Coxa','CxTr':'Trochanter','TrFe':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
BODY_MIDHIND={'ThCx':'Coxa','CxTr':'Femur','FeTi':'Tibia','TiTar':'Tarsus1','Tar':'Tarsus5'}
BODY=BODY_FRONT
LEGS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import mujoco
 from scipy.optimize import least_squares
 from engine import canonical_physics as cp
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-keypoint-pose-fit-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='DATA_DRIVEN_KEYPOINT_FIT_NOT_MEASURED_STANCE_NOT_PRODUCT',scales=SCALES,nominal_scale=NOMINAL_SCALE,frame_stride=FRAME_STRIDE,
        caveats=['single trial fly01_t001, tethered on a spherical treadmill: n=1','dataset declares no length unit, hence the scale sweep','the animal is not this model specimen; this is a data-driven candidate posture, not a measured stance','fit target is keypoint residual only; support margin/duration/contact count were NOT used to select the pose'])
 paths=[Path(__file__),DATA/'readme.csv',DATA/'modelAngles_fly01_t001.csv',DATA/'BodyCoords3D_fly01_t001.csv',DATA/'provenance.json',POSE,cp.CANONICAL_XML]
 hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
 fk=0
 try:
  lines=(DATA/'BodyCoords3D_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
  V=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2)
  assert V.shape==(334,len(cols)) and np.isfinite(V).all()
  idx={c:i for i,c in enumerate(cols)}
  def col(name,frame,axes='xyz'):
   return np.array([V[frame,idx[f'{name}_{ax}']] for ax in axes])
  m=cp.load_model();d=mujoco.MjData(m)
  with np.load(POSE) as z:canon=z['projected_seated_qpos'].copy()
  d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
  def body_id(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
  def rel_world(name,ref='Thorax'):
   return d.xpos[body_id(name)].copy(),d.xmat[body_id(ref)].reshape(3,3).copy(),d.xpos[body_id(ref)].copy()
  oT,R_T,_=rel_world('Thorax')
  land_model={('Th-'+('lWing' if n=='LWing' else 'rWing')):d.xmat[body_id('Thorax')].reshape(3,3).T@(d.xpos[body_id(n)]-d.xpos[body_id('Thorax')]) for n in ('LWing','RWing')}
  land_model['Th-apex']=np.zeros(3)   # DECLARED PROXY: thorax-frame origin stands in for the dataset apex
  frames=list(range(0,V.shape[0],FRAME_STRIDE))
  def leg_joints(side):
   js=[]
   for grp,n in (('Coxa',3),('Trochanter',3),('Tibia',1)):
    for a in (('pitch','roll','yaw') if n==3 else ('pitch',)):
     name=f'joint_{side}{grp}_{a}';j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);assert j>=0,name
     js.append(j)
   return js
  body_ids_cache={}
  def bodies_for(side):
   if side in body_ids_cache:return body_ids_cache[side]
   spec=BODY_FRONT if side in ('LF','RF') else BODY_MIDHIND
   ids=[]
   for key,bn in spec.items():
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+bn)
    if bid<0:raise ValueError(f'body {side+bn} does not exist for keypoint {key}')
    ids.append((key,bid))
   body_ids_cache[side]=ids
   return ids
  def leg_points(side,joint_ids,angles):
   d.qpos[:]=canon
   for j,a in zip(joint_ids,angles):d.qpos[m.jnt_qposadr[j]]=a
   nonlocal fk;fk+=1;mujoco.mj_kinematics(m,d)
   RT=d.xmat[body_id('Thorax')].reshape(3,3);oT2=d.xpos[body_id('Thorax')]
   return np.array([RT.T@(d.xpos[bid]-oT2) for _,bid in bodies_for(side)])
  def transform(scale,frame):
   """Declared: R from the apex/lWing/rWing triangle, translation maps apex to the thorax origin."""
   a=col('Th-lWing',frame)
   b=col('Th-rWing',frame)
   X=np.stack([a,np.zeros(3),b])           # dataset triangle (apex at origin)
   Y=np.stack([land_model['Th-lWing'],np.zeros(3),land_model['Th-rWing']])
   Xc=X-X.mean(0);Yc=Y-Y.mean(0)
   U,S,Vt=np.linalg.svd(Xc.T@Yc);dd=np.sign(np.linalg.det(Vt.T@U.T));D=np.diag([1,1,dd])
   R=Vt.T@D@U.T;t=-scale*R@np.zeros(3)
   resid=float(np.sqrt(np.mean(np.sum((scale*X@R.T+t-Y)**2,axis=1))))
   return R,t,resid
  results={}
  for scale in SCALES:
   results[scale]={}
   for ds,side in LEGS.items():
    ids=leg_joints(side)
    lo=[];hi=[]
    for j in ids:
     q0,q1=m.jnt_range[j]
     if not m.jnt_limited[j]:q0,q1=-2.5,2.5
     lo.append(q0);hi.append(q1)
    lo=np.array(lo);hi=np.array(hi)
    per_frame=[]
    for frame in frames:
     R,t,lres=transform(scale,frame)
     keys=[k for k,_ in bodies_for(side)]
     tgt=np.array([scale*R@col(f'{ds}-{k}',frame)+t for k in keys])
     x0=canon[[m.jnt_qposadr[j] for j in ids]]
     sol=least_squares(lambda x: (leg_points(side,ids,x)-tgt).ravel(),x0,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
     res=sol.fun.reshape(-1,3);per_frame.append(dict(frame=frame,rms=float(np.sqrt(np.mean(res**2))),max=float(np.abs(res).max()),cost=float(sol.cost),nfev=int(sol.nfev),status=int(sol.status),angles=sol.x.tolist(),landmark_residual=lres))
    A=np.array([p['angles'] for p in per_frame]);med=np.median(A,axis=0)
    results[scale][ds]=dict(model_side=side,per_frame_frame=[p['frame'] for p in per_frame],per_frame_rms=[p['rms'] for p in per_frame],per_frame_max=[p['max'] for p in per_frame],per_frame_nfev=[p['nfev'] for p in per_frame],per_frame_status=[p['status'] for p in per_frame],median_angles=med.tolist(),median_angles_deg=[float(np.degrees(x)) for x in med],angle_frame_spread_max_deg=float(np.degrees(np.max(A.max(0)-A.min(0)))),rms_median_frame=float(np.median([p['rms'] for p in per_frame])),rms_worst_frame=float(np.max([p['rms'] for p in per_frame])),landmark_residual_median=float(np.median([p['landmark_residual'] for p in per_frame])),joint_range_violations=int(np.sum((med<lo-1e-9)|(med>hi+1e-9))))
   print('scale',scale,'done',flush=True)
  # Declared convention diagnostics at the nominal scale: can frame handedness or
  # left/right pairing explain the residual? Every variant is reported; none is adopted silently.
  def variant_targets(mirror,swap,frame,scale):
   R,t,_=transform(scale,frame);out={}
   for ds,side in LEGS.items():
    key=ds.replace('L','X').replace('R','L').replace('X','R') if swap else ds
    vs=[col(f'{key}-{k}',frame) for k,_ in bodies_for(side)]
    vs=[np.array([v[0],-v[1],v[2]]) if mirror else v for v in vs]
    out[ds]=np.array([scale*R@v+t for v in vs])
   return out
  diagnostics={}
  for label,mirror,swap in (('as_is',False,False),('mirror_y',True,False),('swap_LR',False,True),('mirror_y_swap_LR',True,True)):
   per={}
   for ds,side in LEGS.items():
    ids=leg_joints(side);lo=[];hi=[]
    for j in ids:
     q0,q1=m.jnt_range[j]
     if not m.jnt_limited[j]:q0,q1=-2.5,2.5
     lo.append(q0);hi.append(q1)
    rms=[]
    for frame in frames:
     tgt=variant_targets(mirror,swap,frame,NOMINAL_SCALE)[ds]
     x0=canon[[m.jnt_qposadr[j] for j in ids]]
     sol=least_squares(lambda x:(leg_points(side,ids,x)-tgt).ravel(),x0,bounds=(np.array(lo),np.array(hi)),max_nfev=400)
     rms.append(float(np.sqrt(np.mean(sol.fun.reshape(-1,3)**2))))
    per[ds]=dict(median_rms=round(float(np.median(rms)),4),worst_rms=round(float(np.max(rms)),4))
   diagnostics[label]=per
  r['declared_keypoint_map']={s:[k for k,_ in bodies_for(s)] for s in set(LEGS.values())}
  r['convention_diagnostics_at_scale_1']=dict(variants=diagnostics,note='each variant refits all six legs at the nominal scale; all variants reported, none adopted silently')
  r['fit']={str(k):v for k,v in results.items()}
  d.qpos[:]=canon;mujoco.mj_kinematics(m,d)
  span={ds:float(np.linalg.norm(d.xpos[body_id(side+'Tarsus5')]-d.xpos[body_id(side+'Coxa')])) for ds,side in LEGS.items()}
  dspan={ds:float(np.linalg.norm(col(f'{ds}-Tar',0)-col(f'{ds}-ThCx',0))) for ds in LEGS}
  r['chain_span_diagnostic']=dict(declared='model Coxa->Tarsus5 distance at the canonical pose vs dataset ThCx->Tar distance at frame 0; ratio is a scale estimate, not a fitted unit claim',model=span,dataset=dspan,ratio_dataset_over_model={k:round(dspan[k]/span[k],4) for k in span})
  r['fk_calls']=fk
  # candidate pose at the declared nominal scale, canonical root/non-leg joints preserved
  nom=results[NOMINAL_SCALE]
  cand=canon.copy();viol=[]
  for ds,side in LEGS.items():
   ids=leg_joints(side);med=np.array(nom[ds]['median_angles'])
   for j,a in zip(ids,med):
    q0,q1=m.jnt_range[j]
    if not m.jnt_limited[j]:q0,q1=-2.5,2.5
    if a<q0-1e-9 or a>q1+1e-9:viol.append(dict(joint=m.joint(j).name,value=float(a),range=[float(q0),float(q1)]))
    cand[m.jnt_qposadr[j]]=min(max(a,q0),q1)
  r['candidate_pose']=dict(qpos=cand.tolist(),joint_range_violations=viol,projection='values outside a joint range are clipped to that range (declared engineering projection); none were clipped' if not viol else 'values outside a joint range were clipped to that range (declared engineering projection)')
  np.savez_compressed(out/'candidate_pose.npz',projected_seated_qpos=cand)
  # keypoint clearance fact for the candidate: contact and foot point heights
  d.qpos[:]=cand;mujoco.mj_kinematics(m,d)
  feet=[body_id(f'{s}Tarsus5') for s in LEGS.values()]
  r['candidate_fact_report']=dict(tarsus5_lowest_and_highest_model_z=[float(min(d.xpos[f][2] for f in feet)),float(max(d.xpos[f][2] for f in feet))],note='fact only; no support search was performed and no contact count was used to select the pose')
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(11,4))
  for ds,side in LEGS.items():axes[0].plot(SCALES,[nom[ds]['rms_median_frame'] if False else results[s][ds]['rms_median_frame'] for s in SCALES],'o-',label=f'{ds}->{side}')
  axes[0].set(xlabel='declared dataset scale',ylabel='median-frame keypoint RMS (model length units)',title='fit residual vs declared scale (whole curve)');axes[0].legend(fontsize=7)
  frame=frames[len(frames)//2];R,t,lres=transform(NOMINAL_SCALE,frame);tgt=np.array([NOMINAL_SCALE*R@col(f'L1-{k}',frame)+t for k in KEY]);pts=leg_points('LF',leg_joints('LF'),np.array(nom['L1']['median_angles']))
  for i,k in enumerate(KEY):
   axes[1].plot([tgt[i,1]],[tgt[i,2]],'o',color='tab:blue');axes[1].plot([pts[i,1]],[pts[i,2]],'x',color='tab:red')
  axes[1].plot(tgt[:,1],tgt[:,2],'-',color='tab:blue',label='dataset keypoints (transformed)');axes[1].plot(pts[:,1],pts[:,2],'--',color='tab:red',label='model LF keypoints (fitted)')
  axes[1].set(xlabel='thorax-frame y',ylabel='thorax-frame z',title=f'LF keypoints, scale={NOMINAL_SCALE}, frame={frame}');axes[1].legend(fontsize=7)
  fig.suptitle('Data-driven keypoint fit (kinematics only) — NOT standing, NOT a measured stance',fontsize=9);fig.tight_layout();fig.savefig(out/'keypoint-fit.png');plt.close(fig)
  r['status']='PASS_KEYPOINT_FIT_REPORTED_ALL_SCALES'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
  (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(out);print(r['status']);print('fk calls',fk);print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
