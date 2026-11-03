"""Stage-A failure diagnostic: open up the instant the abdomen lands.

mode run      : one instrumented real trial of the chosen parameters (dense 12.5us
                records, contact identities, muscle activations).
mode capacity : at the saved posture just before the first non-foot contact, rebuild
                the conditional static system and ask whether ANY activation within
                bounds could hold that posture (mapped66 / independent74 / independent90).
Nothing here changes the model, the reference or any gate.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,inspect,traceback,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s4-stage-a-failure-diagnosis/20261102T232000Z')
S=Path('/home/sensen/Desktop/engine修复验收/s4-stage-a-pool-modes/20261102T203810Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
A=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
LEGS=('LF','LM','LH','RF','RM','RH');DT=.0000125;FLOOR=1e-4
def js(p):return json.loads(Path(p).read_text())
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def groups():
 m=js(B/'RUN/MAPPING.json');roots=ar(B/'IDENTITIES.npz')['root_ids']
 g=np.full(len(roots),-1,int);pools={}
 for row in m['rows']:
  for i in row['local_indices']:g[int(i)]=LEGS.index(row['leg'])
  key=tuple(row['local_indices'])
  if key:
   if key not in pools:pools[key]=len(pools)
 G=np.zeros((90,66))
 for u in range(90):
  key=tuple(m['rows'][u]['local_indices'])
  if key:G[u,pools[key]]=1.
 return roots,g,G,pools

def run():
 best=js(S/'BEST_SO_FAR_R2.json');z=np.asarray(best['z'],float)
 import train_stage_a_pool_modes as st
 folder=O/'TRACE';folder.parent.mkdir(parents=True,exist_ok=True)
 folder,rows,summary=st.trial(folder,z,.2,keep=('fk',))
 ref=st.reference();metrics=st.gate_metrics(rows,.2,ref)
 t=rows[:,0]-1.3;vz=np.gradient(rows[:,10],DT);support=rows[:,1:7].sum(1)/ref['weight']
 np.savez_compressed(O/'FAILURE_TRACE.npz',relative_time_s=t,foot_fz_per_leg=rows[:,1:7],nonfoot_positive=rows[:,7],nonfoot_signed=rows[:,8],nonfoot_contact_count=rows[:,9],com_z=rows[:,10],com_vz=vz,root_qpos=rows[:,11:18],ctrl_mean_square=rows[:,18],foot_support_over_weight=support)
 body=ar(folder/'RUN/BODY.npz')
 np.savez_compressed(O/'BODY_HALF_MS.npz',**{k:body[k] for k in body})
 ev=js(folder/'FIRST_NONFOOT_EVENT.json') if (folder/'FIRST_NONFOOT_EVENT.json').exists() else None
 win=np.flatnonzero(t>0)
 snap=[]
 for k in win:
  if t[k]>0.03:break
  if k%8==0:snap.append(dict(t_ms=round(float(t[k])*1000,3),per_leg=[round(float(x),3) for x in rows[k,1:7]],foot_support_W=round(float(support[k]),4),nf_positive=round(float(rows[k,7]),2),com_z=round(float(rows[k,10]),4),com_vz=round(float(vz[k]),2)))
 out=dict(scope='one instrumented real trial of the stage-A best parameters; diagnostics only',z=best['z'],J_at_best=best['J'],metrics=metrics,first_nonfoot_event=ev,
  boundary=dict(com_z=float(rows[0,10]),com_vz=float(vz[0]),foot_support_W=float(support[0]),per_leg=[float(x) for x in rows[0,1:7]],legs_in_contact=int(np.sum(rows[0,1:7]>0))),
  snapshot_first_30ms=snap,calls=summary['counters'],wall_s=summary['wall_s'])
 dump(O/'TRACE_SUMMARY.json',out)
 print(json.dumps(dict(boundary=out['boundary'],first_nonfoot=ev,metrics=metrics),ensure_ascii=False,indent=1))
 return out

def capacity():
 import mujoco
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 from scipy.optimize import linprog
 tr=ar(O/'FAILURE_TRACE.npz');body=ar(O/'BODY_HALF_MS.npz');roots,g,G,pools=groups()
 t=tr['relative_time_s'];nf=tr['nonfoot_positive']
 idx=int(np.flatnonzero(nf>0)[0]);t_first=float(t[idx])
 # posture used for the static question: the last saved 0.5 ms sample strictly before the event
 half=body['time']-1.3
 cand=np.flatnonzero(half<t_first)
 k=int(cand[-1]) if len(cand) else 0
 qpos=body['qpos'][k].copy();act=body['activation'][k].copy();ctrl=body['command'][k].copy()
 m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();d=mujoco.MjData(m)
 d.qpos[:]=qpos
 for j in range(42):
  qa=int(m.jnt_qposadr[np.flatnonzero(m.jnt_type==3)[j]])
 d.qvel[:]=0.;d.act[:]=act;d.ctrl[:]=ctrl;d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.;d.qacc_warmstart[:]=0.
 mujoco.mj_forward(m,d)
 moment=np.zeros((90,m.nv))
 if d.actuator_moment.shape==(90,m.nv):moment[:]=d.actuator_moment
 else:
  for u in range(90):
   off=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);moment[u,d.moment_colind[off:off+n]]=d.actuator_moment[off:off+n]
 names=[m.geom(i).name or '' for i in range(m.ngeom)];planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
 feet=[];nonfoot=[];limits=[];force=np.zeros(6)
 for j in range(d.ncon):
  co=d.contact[j];a1,a2=int(co.geom1),int(co.geom2)
  if not (planes[a1] or planes[a2]):continue
  gid=a2 if planes[a1] else a1;nm=names[gid]
  fine=next((i for i,l in enumerate(LEGS) if nm.startswith(l) and 'Tars' in nm),-1)
  if fine<0:nonfoot.append(nm);continue
  mujoco.mj_contactForce(m,d,j,force)
  fz=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if planes[a1] else -1)
  feet.append(dict(leg=LEGS[fine],geom=nm,fz=fz,mu=float(d.contact[j].friction[0]),frame=co.frame.reshape(3,3).copy(),pos=co.pos.copy(),dim=int(d.contact[j].dim)))
 # static gain via MuJoCo: gain at zero velocity is actuator_gainprm-derived; use mju_muscleGain
 gain=np.zeros(90);bias=np.zeros(90)
 for u in range(90):
  prm=np.ascontiguousarray(np.asarray(m.actuator_gainprm[u],dtype=float)[:9])
  L=float(d.actuator_length[u]);lr=m.actuator_lengthrange[u];acc0=float(m.actuator_acc0[u])
  gain[u]=float(mujoco.mju_muscleGain(L,0.,np.asarray(lr,dtype=float),acc0,prm))
  bias[u]=float(mujoco.mju_muscleBias(L,np.asarray(lr,dtype=float),acc0,prm))
 A=moment.T*gain
 base=(d.qfrc_passive-d.qfrc_bias)+moment.T@bias
 # foot contact Jacobians in world coordinates
 Jf=[]
 for f in feet:
  bid=m.geom_bodyid[[i for i in range(m.ngeom) if names[i]==f['geom']][0]]
  jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,f['pos'],int(bid))
  Jf.append(jac)
 rows_out=[];witnesses={}
 for arm,P in [('mapped66',G),('independent74',np.eye(90)[:,np.array(sorted(set(range(90))-{u for u in range(90) if not js(B/'RUN/MAPPING.json')['rows'][u]['local_indices']}))]),('independent90',np.eye(90))]:
  n=P.shape[1];E=[A@P];cols=[n]
  for Jc in Jf:E.append(Jc.T);cols.append(3)
  b=-base-A@np.full(90,FLOOR)
  E=np.column_stack(E);ncols=E.shape[1]
  bounds=[(0.,1.-FLOOR)]*n
  for f in feet:bounds+=[(-np.inf,np.inf),(-np.inf,np.inf),(0.,np.inf)]
  U=[];row=0
  for i,f in enumerate(feet):
   z0=n+3*i;mu=f['mu']
   for sx,sy in [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)]:
    r=np.zeros(ncols);r[z0]=sx;r[z0+1]=sy;r[z0+2]=-mu;U.append(r)
  U=np.array(U) if U else None;v=np.zeros(len(U)) if U is not None else None
  rs=np.maximum(1,np.maximum(abs(E).max(1),abs(b)))
  res=linprog(np.zeros(ncols),A_eq=E/rs[:,None],b_eq=b/rs,A_ub=(U if U is not None else None),b_ub=v,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  ok=False
  if res.success:
   x=res.x;ok=bool(np.max(abs(E@x-b),initial=0.)<=1e-8 and np.max((U@x if U is not None else np.zeros(1)),initial=0.)<=1e-8)
   if ok:witnesses[arm]=dict(activation90=(FLOOR+P@x[:n]).tolist(),foot_fz=[float(x[n+3*i+2]) for i in range(len(feet))])
  # quantify the distance to feasibility: smallest t with |E x - b| <= t * row_scale
  nv=ncols+1
  Es=E/rs[:,None];bs=b/rs
  Ef=np.vstack([np.hstack([Es,-np.ones((len(Es),1))]),np.hstack([-Es,-np.ones((len(Es),1))])])
  bf=np.concatenate([bs,-bs])
  if U is not None:
   Uu=np.hstack([U,np.zeros((len(U),1))]);vf=v
   Aub=np.vstack([Ef,Uu]);bub=np.concatenate([bf,vf])
  else:
   Aub=Ef;bub=bf
  bnds=[(0.,1.-FLOOR)]*n+[(-np.inf,np.inf)]*(ncols-n)+[(0.,np.inf)]
  c=np.zeros(nv);c[-1]=1.
  gapres=linprog(c,A_eq=np.zeros((0,nv)),b_eq=np.zeros(0),A_ub=Aub,b_ub=bub,bounds=bnds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  gap=float(gapres.x[-1]) if gapres.success else None
  # cross-check: the scaled gap must agree with the feasibility LP
  consistent=(gap is not None and ((gap<=1e-9)==ok))
  rows_out.append(dict(arm=arm,status=int(res.status),feasible_static_witness=ok,n_vars=int(ncols),feet=len(feet),min_relative_equation_slack=gap,gap_solver_status=int(gapres.status),
   slack_units='scaled relative equation slack; 0 means feasible, larger means farther from any static balance',gap_agrees_with_feasibility_lp=bool(consistent)))
 out=dict(scope='conditional static capability at the saved posture immediately before the abdomen contact; qvel forced to zero on purpose, so this is a posture question, not a trajectory prediction',
  question_time_relative_s=t_first,posture_body_time_s=float(body['time'][k]),posture_relative_s=float(half[k]),
  foot_contacts=[dict(leg=f['leg'],geom=f['geom'],actual_fz_at_pose=f['fz'],mu=f['mu'],dim=f['dim']) for f in feet],
  nonfoot_contact_geoms=nonfoot,actual_activation_mean=float(act.mean()),actual_activation_max=float(act.max()),
  arms=rows_out,witnesses=witnesses,
  calls=dict(model_load=1,forward=1,linprog=len(rows_out),body_step=0,neural_step=0),
  limits='Only this one posture/contact set. A feasible witness here means the network was not delivering what the muscles could; infeasibility means this posture cannot be held statically even with full recruitment, which does not by itself forbid a dynamic gait.')
 dump(O/'CAPACITY_AT_FAILURE.json',out);print(json.dumps({k:v for k,v in out.items() if k!='witnesses'},ensure_ascii=False,indent=1))
 return out

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','capacity']);a=ap.parse_args()
 globals()[a.mode]()
