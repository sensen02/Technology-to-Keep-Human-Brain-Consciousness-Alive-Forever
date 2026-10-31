"""P3 saved-walking phase selection by native no-assistance mechanics, not neural outcomes."""
from pathlib import Path
import json,hashlib,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from scipy.spatial import ConvexHull
from flygym.compose.fly.musculoskeletal import _load_mjcf
from run_physical_walk_prewarm import contacts
S=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-handover-phase-selection/20261028T064000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 def conv(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)
 p.write_text(json.dumps(x,indent=2,default=conv)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();sources=[S/'BODY.npz',S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',Path(__file__)];hashes={str(x):sha(x) for x in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,candidate_times=np.round(np.arange(.80,1.0001,.005),3),scope='Offline instantaneous mechanics only. No body integration, no neural rollout or runtime controller. Rank is chosen before any new neural trial.',eligibility='>=3 complete >=10ms air bouts for each leg since0.25s; >=3 current ground-contact feet; no nonfoot; warning0',score='predicted_10ms_downward_root_displacement/0.1 + predicted_10ms_angular_excursion/0.1 + abs(native_foot_load_ratio-1)*0.05 + assist_norm*0.0001; lower better. Projected excursions use current qvel and native no-assist qacc, not a forward simulation. Secondary sort time descending.',physiology='NOT_ESTABLISHED'))
 with np.load(S/'BODY.npz') as f:b={k:f[k] for k in f.files}
 before=np.load(S/'HANDOVER_BEFORE.npz');m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0);weight=m.body_mass.sum()*abs(m.opt.gravity[2]);mass=m.body_mass.sum();rows=[];dt=float(np.median(np.diff(b['time'])));root=m.body('Thorax').id
 for target in np.round(np.arange(.80,1.0001,.005),3):
  i=int(np.argmin(abs(b['time']-target)));d.qpos[:]=b['qpos'][i];d.qvel[:]=b['qvel'][i];d.act[:]=b['activation'][i];d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1
  f,nf,fc,nc=contacts(m,d);ratio=float(f[:,2].sum()/weight);totalacc=(f.sum(0)+nf)/mass+np.array(m.opt.gravity);durable=[]
  for j in range(6):
   air=(~b['foot_contact'][:,j])&(b['time']>=.25);pad=np.r_[False,air,False].astype(int);st=np.flatnonzero(np.diff(pad)==1);en=np.flatnonzero(np.diff(pad)==-1);valid=(en<len(b['time']))&(en<=i);durable.append(int(np.sum(valid&((en-st)*dt>=.010-1e-9))))
  assist=b['assist'][i].copy()
  if target==1.:assist=before['qfrc_applied'].copy()
  h=.010;down=max(0.,-float(d.qvel[2]*h+.5*d.qacc[2]*h*h));angular=float(np.linalg.norm(d.qvel[3:6]*h+.5*d.qacc[3:6]*h*h));eligible=bool(min(durable)>=3 and fc.sum()>=3 and nc==0 and not d.warning.number.any())
  # Signed support-polygon margin in world XY, positive inside; diagnostic only (not rank gate).
  pts=b['foot_origins'][i,fc,:2];margin=None
  if len(pts)>=3:
   try:
    hull=ConvexHull(pts);e=hull.equations;margin=float(np.min(-(e[:,:2]@d.subtree_com[root,:2]+e[:,2])/np.linalg.norm(e[:,:2],axis=1)))
   except Exception:pass
  score=down/.1+angular/.1+abs(ratio-1)*.05+float(np.linalg.norm(assist[6:]))*.0001
  rows.append(dict(target_time_s=float(target),saved_time_s=float(b['time'][i]),saved_index=i,eligible=eligible,complete_air_bouts_ge10ms=durable,feet_contacting=int(fc.sum()),nonfoot_count=nc,native_foot_load_ratio=ratio,native_total_com_acceleration_world=totalacc,native_root_qacc=d.qacc[:6].copy(),root_qvel=d.qvel[:6].copy(),root_angspeed=float(np.linalg.norm(d.qvel[3:6])),native_predicted_10ms_downward_root_displacement=down,native_predicted_10ms_angular_excursion=angular,assist_norm=float(np.linalg.norm(assist[6:])),support_polygon_margin=margin,score=score))
 ranked=sorted([r for r in rows if r['eligible']],key=lambda r:(r['score'],-r['target_time_s']));assert ranked,'No mechanically admissible phase with3completedcycles';selected=ranked[0]
 dump(O/'CANDIDATES.json',rows);dump(O/'RANKING.json',ranked);dump(O/'SELECTION.json',dict(selected=selected,rank_count=len(ranked),scope='Pre-neural mechanical phase selection only; not a prediction or certificate of future sustained support. Required actual same-execution walking + neural test next. Post-neural controller unchanged.',counts=counts));dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-start,source_unchanged=all(sha(Path(k))==v for k,v in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(selected=selected,rank_count=len(ranked),counts=counts),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,indent=2),flush=True)
if __name__=='__main__':main()
