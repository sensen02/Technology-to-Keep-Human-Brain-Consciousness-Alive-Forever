"""Fixed saved-pose ideal-contact static feasibility. NOT neural standing."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
# Declared optional override: an explicit pose file (same npz key) so a second pose can be
# run through the identical gate without touching the canonical default.
if len(sys.argv)>1: POSE=Path(sys.argv[1])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
 import mujoco
 from scipy.optimize import linprog
 import scipy
 from engine import canonical_physics as cp
 from run_s2_fixed_stimulus import dense_moment
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-support-gate-'+time.strftime('%Y%m%dT%H%M%S'))
 out.mkdir(exist_ok=False)
 report=dict(status='RUNNING',manual_research_not_product=True,standing=False,steps=0,forward_calls=0,results=[])
 paths=[Path(__file__),ROOT/'S2_SUPPORT_GATE_PROTOCOL.zh-CN.md',ROOT/'run_s2_fixed_stimulus.py',Path(cp.__file__),POSE,cp.CANONICAL_XML]
 hashes={str(p):sha(p) for p in paths}
 report.update(source_input_hashes=hashes,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__))
 def forward():
  report['forward_calls']+=1
  if report['forward_calls']>110: raise ValueError('forward budget')
  mujoco.mj_forward(m,d)
  if np.any(d.warning.number): raise ValueError('MuJoCo warning')
  if not all(np.isfinite(x).all() for x in (d.qacc,d.actuator_force,d.qfrc_passive,d.qfrc_bias)): raise ValueError('nonfinite forward')
 try:
  m=cp.load_model();d=mujoco.MjData(m)
  with np.load(POSE,allow_pickle=False) as p: pose=p['projected_seated_qpos'].copy()
  if pose.shape!=(m.nq,): raise ValueError('pose shape')
  if m.nu!=90 or m.nv!=48 or m.na!=90: raise ValueError('unsupported dimensions')
  if not np.array_equal(m.actuator_actadr,np.arange(m.nu)) or not np.all(m.actuator_actnum==1): raise ValueError('activation layout')
  if not np.all(m.actuator_gaintype==int(mujoco.mjtGain.mjGAIN_MUSCLE)) or not np.all(m.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_MUSCLE)): raise ValueError('not native muscles')
  if np.any(m.actuator_forcelimited) or np.any(m.jnt_actfrclimited) or np.any(m.actuator_actearly) or np.any(m.body_gravcomp) or np.any(m.dof_frictionloss) or np.any(m.tendon_frictionloss): raise ValueError('unsupported force coupling/clipping')
  if not np.all(m.actuator_ctrllimited): raise ValueError('unbounded controls')
  if m.nplugin: raise ValueError('plugins unsupported')
  d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;forward()
  for j in range(m.njnt):
   if m.jnt_limited[j] and int(m.jnt_type[j]) in (2,3):
    q=pose[m.jnt_qposadr[j]]
    if q<m.jnt_range[j,0]-1e-10 or q>m.jnt_range[j,1]+1e-10: raise ValueError('saved pose violates scalar range')
  for c in d.contact:
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (c.geom1,c.geom2)]
   if c.efc_address>=0 and not any('Tarsus' in n for n in names): raise ValueError('nonfoot active contact')
  base=d.qfrc_actuator.copy();passive=d.qfrc_passive.copy();bias=d.qfrc_bias.copy();moment=dense_moment(m,d)
  if not np.allclose(moment.T@d.actuator_force,base,atol=1e-9,rtol=1e-10): raise ValueError('moment factorization')
  target=bias-passive-base;R=np.zeros((m.nv,m.nu))
  for i in range(m.nu):
   d.act[:]=0;d.act[i]=1;forward();R[:,i]=d.qfrc_actuator-base
  errors=[]
  for a in (np.full(m.nu,.37),np.linspace(0,1,m.nu),np.arange(m.nu)%2):
   d.act[:]=a;forward();pred=base+R@a;errors.append(float(np.max(np.abs(pred-d.qfrc_actuator))))
   if not np.allclose(pred,d.qfrc_actuator,atol=1e-9,rtol=1e-10): raise ValueError('nonaffine muscle mapping')
  d.act[:]=0;forward()
  if np.asarray(d.efc_J).size!=d.nefc*m.nv: raise ValueError('requires dense efc J')
  E=np.asarray(d.efc_J).reshape(d.nefc,m.nv).copy();lims=[]
  allowed={int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT),int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)}
  for i,t in enumerate(d.efc_type):
   if int(t) in allowed: lims.append(E[i])
   elif int(t) not in {int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)}: raise ValueError('unsupported constraint type')
  limit=np.array(lims).reshape(-1,m.nv).T;contacts=[];blocks={'diamond':[],'outer':[],'union':[]}
  for i,c in enumerate(d.contact):
   if c.efc_address<0: continue
   if c.dim!=3: raise ValueError('requires condim3')
   J=np.zeros((3,m.nv))
   for g,sign in ((c.geom2,1),(c.geom1,-1)):
    b=int(m.geom_bodyid[g]);jp=np.zeros_like(J);jr=np.zeros_like(J)
    mujoco.mj_jac(m,d,jp,jr,c.pos,b);J+=sign*jp
   frame=np.asarray(c.frame).reshape(3,3);mu=np.asarray(c.friction)[:2]
   # 'union' (diamond + outer = 8 rays) is the decisive test: the true pyramidal cone contains
   # both subsets, so a certificate valid for the union proves infeasibility for the full cone,
   # while feasibility under any subset already proves feasibility for the full cone.
   for kind,xy in [('diamond',[(1,0),(-1,0),(0,1),(0,-1)]),('outer',[(1,1),(1,-1),(-1,1),(-1,-1)]),
                   ('union',[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)])]:
    B=np.array([frame[0]+x*mu[0]*frame[1]+y*mu[1]*frame[2] for x,y in xy]).T
    blocks[kind].append(J.T@B)
   contacts.append(dict(index=i,geom1=int(c.geom1),geom2=int(c.geom2),dist=float(c.dist),pos=np.array(c.pos).tolist(),frame=frame.tolist(),mu=mu.tolist(),J=J.tolist()))
  meta=dict(nv=m.nv,nu=m.nu,ncon=d.ncon,active_contacts=contacts,limit_count=limit.shape[1],affine_max_errors=errors,gravity=m.opt.gravity.tolist(),body_mass=m.body_mass.tolist(),ctrlrange=m.actuator_ctrlrange.tolist(),actrange=m.actuator_actrange.tolist(),actlimited=m.actuator_actlimited.tolist(),actuator_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)],gainprm=m.actuator_gainprm.tolist(),biasprm=m.actuator_biasprm.tolist(),dynprm=m.actuator_dynprm.tolist(),units='model force / model torque; mass and biological absolute force uncalibrated')
  dump(out/'metadata.json',meta)
  raw=dict(qpos=pose,R=R,base=base,passive=passive,bias=bias,target=target,moment=moment,limit=limit)
  for kind in blocks: raw['contact_'+kind]=np.hstack(blocks[kind]) if blocks[kind] else np.zeros((m.nv,0))
  np.savez_compressed(out/'raw.npz',**raw)
  for floor in ('native','zero'):
   lo=m.actuator_ctrlrange[:,0].copy() if floor=='native' else np.zeros(m.nu);hi=m.actuator_ctrlrange[:,1].copy()
   limited=m.actuator_actlimited.astype(bool);lo[limited]=np.maximum(lo[limited],m.actuator_actrange[limited,0]);hi[limited]=np.minimum(hi[limited],m.actuator_actrange[limited,1])
   if np.any(lo>hi): raise ValueError('empty activation interval')
   for kind in blocks:
    A=np.hstack([R,raw['contact_'+kind],limit]);scale=np.maximum(1,np.maximum(np.abs(target),np.max(np.abs(A),axis=1)));As=A/scale[:,None];b=target/scale
    bounds=list(zip(lo,hi))+[(0,None)]*(A.shape[1]-m.nu)
    res=linprog(np.zeros(A.shape[1]),A_eq=As,b_eq=b,bounds=bounds,method='highs',options=dict(time_limit=10,maxiter=10000,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    entry=dict(floor=floor,contact_model=kind,solver_status=int(res.status),message=res.message,classification='INDETERMINATE')
    saved=dict(A=A,target=target,scale=scale,lo=lo,hi=hi)
    if res.status==2: entry['classification']='NUMERICALLY_INFEASIBLE_FOR_STATED_RELAXATION'
    if res.status==0:
     x=res.x;residual=A@x-target;scaled=float(np.max(np.abs(residual)/scale));violation=float(max(0,np.max(lo-x[:m.nu]),np.max(x[:m.nu]-hi),-np.min(x[m.nu:],initial=0)))
     if scaled>1e-7 or violation>1e-7: raise ValueError('LP witness reconstruction failed')
     d=mujoco.MjData(m);d.qpos[:]=pose;d.act[:]=x[:m.nu];d.ctrl[:]=x[:m.nu]
     forward();muscle_error=float(np.max(np.abs(d.qfrc_actuator-base-R@x[:m.nu])))
     if not np.allclose(d.qfrc_actuator,base+R@x[:m.nu],rtol=1e-10,atol=1e-9): raise ValueError('witness muscle replay failed')
     entry.update(classification='RELAXED_WITNESS',scaled_max_residual=scaled,bound_violation=violation,max_abs_model_force_residual=float(np.max(np.abs(residual[:3]))),max_abs_model_torque_residual=float(np.max(np.abs(residual[3:]))),witness_muscle_error=muscle_error,actual_forward_qacc_max=float(np.max(np.abs(d.qacc))),actual_forward_constraint_not_lp=True)
     saved.update(x=x,residual=residual,actual_qacc=d.qacc.copy(),actual_act_dot=d.act_dot.copy(),actual_qfrc_constraint=d.qfrc_constraint.copy(),actual_qfrc_actuator=d.qfrc_actuator.copy())
    np.savez_compressed(out/(floor+'_'+kind+'.npz'),**saved);report['results'].append(entry);dump(out/'report.json',report)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,4));labels=[r['floor']+' / '+r['contact_model'] for r in report['results']];vals=[r.get('scaled_max_residual',0) for r in report['results']];ax.bar(labels,vals);ax.set_ylabel('scaled equilibrium residual (witness only)');ax.set_title('Fixed-pose ideal-contact LP; NOT standing / NOT biological calibration')
  for i,r in enumerate(report['results']): ax.text(i,vals[i],r['classification'],rotation=15,fontsize=7,ha='center')
  fig.tight_layout();fig.savefig(out/'support.png');plt.close(fig)
  report['status']='COMPLETE'
 except Exception as exc: report.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
 finally:
  report['source_input_hashes_after']={p:sha(p) for p in hashes};report['hashes_unchanged']=report['source_input_hashes_after']==hashes
  if not report['hashes_unchanged']: report['status']='FAIL'
  dump(out/'report.json',report)
 print(out,flush=True)
 return 0 if report['status']=='COMPLETE' else 1
if __name__=='__main__': raise SystemExit(main())
