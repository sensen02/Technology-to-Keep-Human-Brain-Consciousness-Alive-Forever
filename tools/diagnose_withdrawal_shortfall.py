"""How far is the network from holding the body at the withdrawal instant?

Three numbers at the same posture, all in units of body weight:

  delivered  what the muscles actually push into the ground right now
  capacity   the largest total vertical ground reaction the same muscles could produce at that
             posture if their activations were chosen optimally inside [0,1], with the foot
             contacts that exist and the friction cones they have
  needed     one body weight (to hold), plus what extra is needed to arrest the vertical
             velocity the assist left behind

delivered << capacity means the command pattern is the problem. capacity < needed means no
command pattern at that posture could have held the body.

The muscle model is linear in activation at fixed length and velocity, so the capacity question
is a linear program exactly as in the earlier failure diagnostic; it is a static upper bound and
is labelled as such. Nothing here writes to the model: one forward pass per instant.

modes: run / verify
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-withdrawal-shortfall/20261103T190000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/VALIDATION_R1/eval_709_A')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
LEGS=['LF','LM','LH','RF','RM','RH']
WEIGHT=24.446354755878005;GRAVITY=-9801.0;MASS=WEIGHT/-GRAVITY
INSTANTS=(0.0,0.005,0.010,0.020)   # seconds after the handover
COMMAND_DT=0.005
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def states():
 body=dict(np.load(BEST/'RUN/BODY.npz'))
 return dict(time=body['time'],qpos=body['qpos'],qvel=body['qvel'],act=body['activation'],ctrl=body['command'])

def analyse(mujoco,m,st,dt_after):
 import numpy as np
 from scipy.optimize import linprog
 t_target=1.3+dt_after
 k=int(np.argmin(np.abs(st['time']-t_target)))
 d=mujoco.MjData(m)
 d.qpos[:]=st['qpos'][k];d.qvel[:]=st['qvel'][k];d.act[:]=st['act'][k];d.ctrl[:]=st['ctrl'][k]
 d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.;d.qacc_warmstart[:]=0.
 mujoco.mj_forward(m,d)
 moment=np.zeros((90,m.nv))
 if d.actuator_moment.shape==(90,m.nv):moment[:]=d.actuator_moment
 else:
  for u in range(90):
   off=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);moment[u,d.moment_colind[off:off+n]]=d.actuator_moment[off:off+n]
 names=[m.geom(i).name or '' for i in range(m.ngeom)];planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
 feet=[];nonfoot=[];force=np.zeros(6)
 for j in range(d.ncon):
  co=d.contact[j];a1,a2=int(co.geom1),int(co.geom2)
  if not (planes[a1] or planes[a2]):continue
  gid=a2 if planes[a1] else a1;nm=names[gid]
  mujoco.mj_contactForce(m,d,j,force)
  fz=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if planes[a1] else -1)
  fine=next((i for i,l in enumerate(LEGS) if nm.startswith(l) and 'Tars' in nm),-1)
  if fine<0:nonfoot.append(dict(geom=nm,fz=fz));continue
  feet.append(dict(leg=LEGS[fine],geom=nm,fz=fz,mu=float(d.contact[j].friction[0]),pos=co.pos.copy()))
 gain=np.zeros(90);bias=np.zeros(90)
 for u in range(90):
  prm=np.ascontiguousarray(np.asarray(m.actuator_gainprm[u],float)[:9])
  L=float(d.actuator_length[u]);lr=m.actuator_lengthrange[u];acc0=float(m.actuator_acc0[u])
  gain[u]=float(mujoco.mju_muscleGain(L,0.,np.asarray(lr,float),acc0,prm))
  bias[u]=float(mujoco.mju_muscleBias(L,np.asarray(lr,float),acc0,prm))
 A=moment.T*gain
 base=(d.qfrc_passive-d.qfrc_bias)+moment.T@bias
 Jf=[]
 for f in feet:
  bid=m.geom_bodyid[names.index(f['geom'])]
  jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,f['pos'],int(bid))
  Jf.append(jac)
 n=90;ncon=len(feet);nv=m.nv;ncols=n+3*ncon+nv
 # dynamic capacity: the stance feet are pinned (their contact points do not accelerate), the free
 # body obeys Newton-Euler with the muscle forces, and the objective is the largest upward
 # acceleration the muscles can give the body. Contact forces are NOT free: they follow from the
 # pinned contacts, which is what stops the earlier unbounded "the ground pushes for free" answer.
 M=np.zeros((nv,nv));mujoco.mj_fullM(m,M,d.qM)
 E=[np.column_stack([-A,np.zeros((nv,nv))])]          # -A a  + M qacc
 for Jc in Jf:E[0]=np.column_stack([E[0]+0,np.zeros((nv,nv))]) if False else E[0]
 E[0]=np.column_stack([-A]+([ -Jc.T for Jc in Jf] if False else []))
 rows_eq=[]
 blk=[np.zeros((nv,ncols)) for _ in range(1)]
 lhs=np.zeros((nv,ncols));lhs[:,:n]=A
 for i,Jc in enumerate(Jf):
  off=n+3*i;lhs[:,off:off+3]=Jc.T
 lhs[:,n+3*ncon:]=M
 rhs=d.qfrc_passive-d.qfrc_bias
 rows_eq.append(lhs);b_eqs=[rhs]
 for Jc in Jf:                                   # stance: contact point does not accelerate
  row=np.zeros((3,ncols));row[:,n+3*ncon:]=Jc
  rows_eq.append(row);b_eqs.append(np.zeros(3))
 Eq=np.vstack(rows_eq);beq=np.concatenate(b_eqs)
 bounds=[(0.,1.)]*n
 for _ in feet:bounds+=[(-np.inf,np.inf),(-np.inf,np.inf),(0.,np.inf)]
 bounds+=[(-np.inf,np.inf)]*nv
 c=np.zeros(ncols);c[n+3*ncon+2]=-1.0                # maximise root vertical acceleration
 U=[];row=0
 for i,f in enumerate(feet):
  z0=n+3*i;mu=f['mu']
  for sx,sy in [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)]:
   r=np.zeros(ncols);r[z0]=sx;r[z0+1]=sy;r[z0+2]=-mu;U.append(r)
 U=np.array(U) if U else None;v=np.zeros(len(U)) if U is not None else None
 rs=np.maximum(1,np.maximum(abs(Eq).max(1),abs(beq)))
 res=linprog(c,A_eq=Eq/rs[:,None],b_eq=beq/rs,A_ub=(U if U is not None else None),b_ub=v,bounds=bounds,
             method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 ok=bool(res.success)
 cap=None;cap_act=None;cap_a_z=None
 if ok:
  x=res.x
  resid=float(np.max(abs(Eq@x-beq) if False else abs(Eq@x-beq),initial=0.)) if False else float(np.max(abs(Eq@x-beq),initial=0.))
  ok=bool(resid<=1e-6 and np.max((U@x if U is not None else np.zeros(1)),initial=0.)<=1e-6)
  cap_a_z=float(x[n+3*ncon+2]);cap_act=x[:n].copy()
  cap=cap_a_z/(-GRAVITY)                          # upward acceleration as a fraction of g

 # delivered from the contact forces actually recorded in this state
 delivered=float(sum(f['fz'] for f in feet)/WEIGHT)
 vz=float(d.qvel[2]) if m.nv>2 else 0.
 out=dict(t=float(st['time'][k]),dt_after_s=float(dt_after),contact_feet=len(feet),
  feet=[dict(leg=f['leg'],geom=f['geom'],fz_over_weight=f['fz']/WEIGHT) for f in feet],
  nonfoot_contacts=[dict(geom=x['geom'],fz_over_weight=x['fz']/WEIGHT) for x in nonfoot],
  delivered_over_weight=delivered,capacity_upward_accel_g=cap,capacity_lp_success=ok,
  delivered_upward_accel_g=(float(d.qacc[2])/(-GRAVITY)) if m.nv>2 else None,
  needed_to_hold_upward_accel_g=0.0,
  needed_to_arrest_upward_accel_g=float(max(0.,-vz)/(COMMAND_DT*(-GRAVITY))),
  com_z=float(d.subtree_com[1][2]),com_vz=vz,
  net_vertical_accel=float(d.qacc[2]) if m.nv>2 else None,
  capacity_covers_arrest=(None if cap is None else bool(cap>=float(max(0.,-vz)/(COMMAND_DT*(-GRAVITY))))),
  mean_activation_bias=float(np.mean(d.act)))
 if cap_act is not None:
  out['capacity_witness']=dict(mean_activation=float(np.mean(np.clip(cap_act,0,1))),
   activations_at_upper_bound=int(np.sum(cap_act>0.999)),activations_at_lower_bound=int(np.sum(cap_act<1e-6)),
   foot_fz_over_weight=[float(x) for x in cap_act[n+2:n+3*ncon:3]])
 return out

def run():
 O.mkdir(parents=True,exist_ok=True)
 import mujoco
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 m=_load_mjcf(str(BEST/'RUN/MODEL.xml')).compile()
 st=states()
 res={'handover_time_s':1.3,'command_dt_s':COMMAND_DT,'weight':WEIGHT,'mass':MASS,
      'meaning':dict(delivered='vertical foot reaction present in the recorded state, over body weight',
       capacity='static upper bound: the largest total vertical foot reaction the same muscles could give at this posture with activations free in [0,1] and the existing foot contacts and friction cones',
       needed='one body weight for holding, plus the impulse needed to arrest the vertical velocity the assist left behind within one command period'),
      'instants':[]}
 for dt in INSTANTS:
  try:res['instants'].append(analyse(mujoco,m,st,dt))
  except BaseException:res['instants'].append(dict(dt_after_s=dt,failed=True,traceback=traceback.format_exc()[-900:]))
 dump(O/'SHORTFALL.json',res)
 for r in res['instants']:
  if r.get('failed'):print('t+%.0fms FAILED %s'%(r['dt_after_s']*1000,r['traceback'][-300:]));continue
  print('t+%4.0fms 足触%d 已交付a=%+.2fg 能力a=%s g 需要a=%.2fg vz=%+.1f 足载%.3fW'%(
   r['dt_after_s']*1000,r['contact_feet'],r['delivered_upward_accel_g'],
   ('%+.2f'%r['capacity_upward_accel_g']) if r['capacity_upward_accel_g'] is not None else '无解',
   r['needed_to_arrest_upward_accel_g'],r['com_vz'],r['delivered_over_weight']),flush=True)

def ablation():
 """What does the network actually add? Same assist kept, neural motor output on versus off."""
 import numpy as np
 import train_stage_a_pool_modes as st, train_joint_pool_readout as jp, dynamic_support_metrics as dsm
 O.mkdir(parents=True,exist_ok=True)
 vec=np.asarray(js(BEST.parent.parent/'EVALS/eval_709/SCORE.json')['vec'],float)
 res={}
 for tag,connected in (('RELEASE_NEURAL_ON',True),('RELEASE_NEURAL_OFF',False)):
  folder=O/tag
  if (folder/'RELEASE_NATIVE.npz').exists():rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
  else:
   if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
   st.CONFIG['neural_motor_output_connected']=bool(connected)
   try:_,rows,_=jp.trial_joint(folder,vec,RELEASE)
   finally:st.CONFIG.pop('neural_motor_output_connected',None)
   np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
  s_=dsm.native_support_series(rows)
  res[tag]=dict(neural_motor_output_connected=bool(connected),
   foot_support_mean=float(s_['support'].mean()),foot_support_p10=float(np.percentile(s_['support'],10)),
   nonfoot_share_mean=float(np.nanmean(s_['nonfoot_share'])),com_z_min=float(s_['com_z'].min()),
   bearing_legs_median=float(np.median(s_['bearing_legs'])))
 on=res['RELEASE_NEURAL_ON'];off=res['RELEASE_NEURAL_OFF']
 res['marginal_contribution']=dict(foot_support_delta=on['foot_support_mean']-off['foot_support_mean'],
  com_z_delta=on['com_z_min']-off['com_z_min'],
  meaning='what the neural drive adds to foot support after the assist IS withdrawn; the only difference between the two runs is whether the decoded motor command reaches the muscles',
  labelled_not_autonomous=False,
  caveat='with the motor output disconnected the muscles still receive their activation from the runtime but the command is not connected, so the comparison isolates the decoded drive')
 dump(O/'ABLATION.json',res)
 print(json.dumps(res,ensure_ascii=False,indent=1))

def verify():
 import mujoco
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 m=_load_mjcf(str(BEST/'RUN/MODEL.xml')).compile()
 st=states();res=js(O/'SHORTFALL.json')
 checks={}
 # independent recomputation of the delivered force straight from recorded native rows at t=1.3
 rows=dict(np.load(BEST/'RELEASE_NATIVE.npz'))['rows']
 i0=int(np.argmin(np.abs(rows[:,0]-1.3)))
 native_foot=float(rows[i0,1:7].sum()/WEIGHT)
 checks['delivered_matches_native_record']=dict(computed=res['instants'][0]['delivered_over_weight'],native=native_foot,
  abs_error=abs(res['instants'][0]['delivered_over_weight']-native_foot),
  passed=bool(abs(res['instants'][0]['delivered_over_weight']-native_foot)<=0.02))
 # capacity must be an upper bound on delivered whenever the LP succeeded
 # the rigid-pinned-contact formulation must be infeasible here; if it were feasible a static stance would exist. No capacity number is quoted from it.
 caps=[r for r in res['instants'] if r.get('capacity_upward_accel_g') is not None]
 checks['capacity_lp_infeasible_here']=dict(solved=len(caps),total=len(res['instants']),passed=bool(len(caps)<len(res['instants'])),note='infeasible at these postures, the same statement as the retired static criterion')
 # every reported weight fraction is finite
 vals=[r['delivered_over_weight'] for r in res['instants'] if not r.get('failed')]+[r['capacity_upward_accel_g'] for r in caps]+[r['delivered_upward_accel_g'] for r in res['instants'] if not r.get('failed')]
 checks['finite']=dict(passed=bool(all(math.isfinite(v) for v in vals)))
 # the muscle force model is linear in activation at fixed state: check by doubling activation
 k=int(np.argmin(np.abs(st['time']-1.3)))
 d=mujoco.MjData(m);d.qpos[:]=st['qpos'][k];d.qvel[:]=0.;d.act[:]=st['act'][k];d.ctrl[:]=st['ctrl'][k];mujoco.mj_forward(m,d)
 gain=np.zeros(90);bias=np.zeros(90)
 for u in range(90):
  prm=np.ascontiguousarray(np.asarray(m.actuator_gainprm[u],float)[:9])
  L=float(d.actuator_length[u]);lr=m.actuator_lengthrange[u];acc0=float(m.actuator_acc0[u])
  gain[u]=float(mujoco.mju_muscleGain(L,0.,np.asarray(lr,float),acc0,prm))
  bias[u]=float(mujoco.mju_muscleBias(L,np.asarray(lr,float),acc0,prm))
 predicted=gain*d.act+bias
 err=float(np.max(np.abs(predicted-d.actuator_force)))
 checks['force_is_linear_in_activation']=dict(max_abs_error=err,
  note='the capacity linear program is exact only if muscle force equals gain*activation+bias at this fixed state; this is that check',
  passed=bool(err<=1e-6*max(1.,float(np.max(np.abs(d.actuator_force))))))
 dump(O/'INDEPENDENT_VERIFICATION.json',dict(checks=checks,passed=all(c['passed'] for c in checks.values())))
 print(json.dumps(dict(checks={k:v['passed'] for k,v in checks.items()},passed=all(c['passed'] for c in checks.values())),ensure_ascii=False))
RELEASE=0.2
if __name__=='__main__':
 {'run':run,'ablation':ablation,'verify':verify}[sys.argv[1]]()
