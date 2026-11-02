"""S2/S4 saved-state conditional static capacity. No simulation or neural action.
collect: one unchanged compile, two forwards per pose. solve: exactly nine LPs.
verify: fresh arrays only, independent scalar reconstruction, no solve/model.
Does not seal; parent review must precede a single readonly seal.
"""
from pathlib import Path
import sys, os, json, hashlib, math, shutil, traceback
import numpy as np
R=Path(__file__).resolve().parents[1]
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s2-mapped-support-capacity/20261102T170741Z'
B=D/'s5-banc888-physical-handover/20261101T220018Z'
E=D/'s2-early-muscle-support/20261102T053639Z'
C=D/'s2-reference-muscle-operating-range/20261102T155412Z'
S=D/'s5-slow-reference-handover/20261102T165227Z/SLOW_2P5HZ_PHASE_MATCHED'
OLD=D/'s5-fixed-muscle-authority/20261028T081000Z'
SHA='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
FLOOR=1e-4
ARMS=['mapped66','independent74','independent90']
def js(p): return json.loads(Path(p).read_text())
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):
 (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def err(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0))
def matrix_moment(d):
 out=np.zeros((90,48))
 if d.actuator_moment.shape==out.shape:return d.actuator_moment.copy()
 for u in range(90):
  j=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);out[u,d.moment_colind[j:j+n]]=d.actuator_moment[j:j+n]
 return out
def collect():
 O.mkdir(parents=True,exist_ok=False)
 calls=dict(model_compile=0,forward=0,contact_jacobian=0,muscleGain=0,muscleBias=0,linprog=0,body_step=0,neural_step=0,network=0,training=0)
 paths=[Path(__file__),R/'AGENTS.md',R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'RUN/MAPPING.json',B/'IDENTITIES.npz',E/'IDENTITY_AND_MAPPING.json',E/'TRAIN_000_BASELINE_AUDIT.npz',C/'AUDIT.npz',C/'COMPILED_ACTUAL_ARRAYS.npz',C/'MODEL_METADATA.json',S/'RUN/BODY.npz',S/'PARAMETERS.npz',OLD/'STATIC_DETAIL.json',R/'diagnose_full90_vs_trusted66.py',R/'diagnose_fixed_muscle_authority.py',R/'tools/audit_muscle_phase_capacity.py',D/'s5-muscle-phase-capacity/20261101T193630Z/STATIC_ARRAYS.npz',D/'s2-static-lp-audit-20261020T100000Z/RESULT.json']
 hashes={str(p):sha(p) for p in paths};assert hashes[str(B/'RUN/MODEL.xml')]==SHA
 dump('SOURCE_HASHES_BEFORE.json',hashes)
 dump('PREREGISTRATION.json',dict(head='dd1a550c6e8fe3fdffbf4fe0a6467986d03c4b0d',stage='S2/S4',poses=[dict(label='initial0',source=str(B/'RUN/BODY.npz'),index=0,time=0),dict(label='original_release1p3',source=str(B/'RUN/BODY.npz'),index=2600,time=1.3),dict(label='slow_release1p25',source=str(S/'RUN/BODY.npz'),index=2500,time=1.25)],model_sha256=SHA,budget='one compile, at most two forwards/pose, total6; exactly9 scipy linprog calls; no additional solve, residual minimization, search or integration',equation='moment.T*gain_static @ activation + sum(J_world_contact.T @ F_world) + qfrc_passive_static + moment.T@muscle_bias_static - qfrc_bias_static = 0, all48 DOF',activation='[1e-4,1]; mapped66 hypothetical equal ACTIVATION (not equal command),16fixedfloor; independent74 relaxes sharing only; independent90 also relaxes mapping without adding neural connections',contact='Only original current geometric foot-floor candidates: plane and original leg prefix+Tars name classification, each exact contact.pos, Fz>=0, |Fx|+|Fy|<=actual contact.friction[0]*Fz. No nonfoot bearing or fabricated origins.',model_scope='SIMPLIFIED STATIC CONDITION MODEL EXCLUDING JOINT-LIMIT AND OTHER CONSTRAINT REACTIONS. Thus infeasible is not a necessary-relaxation certificate of full mechanics. Joint-limit rows and range proximity recorded. Contact forces are relaxed independently of original soft solver, gap, compliance and complementarity. Even feasible is not soft-contact stability or neural reachability.',static='saved complete q unchanged; offline clone qvel=0; qacc=0 assumed in equilibrium equation, not actual forward qacc or runtime/history edit. Springs/internal passive at qvel0 retained. Original actual q/v/act/ctrl separately checked. No root servo/force, no Fmax/model changes.',solver=dict(method='highs',objective='zero feasibility',row_scaling='each equation divided by max(1,maxabs coefficients,abs RHS); invertible numerical row conditioning only, not physical normalization',primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,witness_equilibrium_maxabs_native=1e-8,witness_friction_max_violation=1e-8,witness_activation_atol=1e-10,infeasible='no witness; status/message only; no residual solve or retries'),units='native force and torque separately listed; absolute biological units NOT_ESTABLISHED',no_runtime_write=True,STDP='deferred',source_hashes=hashes))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 try:
  b=ar(B/'RUN/BODY.npz');s=ar(S/'RUN/BODY.npz');q=np.array([b['qpos'][0],b['qpos'][2600],s['qpos'][2500]])
  old=js(OLD/'STATIC_DETAIL.json');phase=ar(D/'s5-muscle-phase-capacity/20261101T193630Z/STATIC_ARRAYS.npz')
  duplicates=dict(scope='bounded named packages/tools only, not archive exhaustive',old081_q_maxabs=[err(x,old['qpos']) for x in q],old081_constraints='native qvel0 but joint-limit reactions included; force-scale grid; different q for all3',old083='full90 uses old081 same different q plus limits; no reuse',phase193630=dict(target='dynamic q/v torque substitution to muscle+assist, not static contact equilibrium',q_saved_in_static_arrays=False,poses=phase['times']),old20261020_source=js(D/'s2-static-lp-audit-20261020T100000Z/RESULT.json')['source'],cache_reuse='early baseline and fullcycle reused for16actualdirection and identity; none stores identical full static contact LP for chosen3poses')
  dump('HISTORY_DUPLICATE_CHECK.json',duplicates)
  mapping=js(B/'RUN/MAPPING.json')['rows'];early=js(E/'IDENTITY_AND_MAPPING.json');assert mapping==early['mapping']['rows']
  groups={};pg=np.full(90,-1,int)
  for r in mapping:
   key=tuple(r['local_indices'])
   if key:
    if key not in groups:groups[key]=len(groups)
    pg[r['actuator_index']]=groups[key]
  assert len(groups)==66 and sum(pg<0)==16 and sum(r['status']=='partial' for r in mapping)==7 and sum(r['status']=='complete' for r in mapping)==67
  G=np.zeros((90,66))
  for u,g in enumerate(pg):
   if g>=0:G[u,g]=1
  perms=[G,np.eye(90)[:,pg>=0],np.eye(90)]
  ident=ar(B/'IDENTITIES.npz');dump('ROOT_IDENTITY_SCHEMA.json',{k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in ident.items()})
  # Source pool roots and local indices are tied to actual authoritative tier array.
  rootkey=next(k for k,v in ident.items() if v.shape==(12000,) and 'root' in k.lower())
  roots=ident[rootkey]
  for r in mapping:
   assert list(roots[r['local_indices']].astype(object))==r['tier_root_ids']
  dump('AUTHORITATIVE_MAPPING.json',dict(mapping=mapping,pool_group=pg,root_source=str(B/'IDENTITIES.npz'),root_key=rootkey,counts=dict(complete=67,partial=7,unresolved=16,pools=66),shared_groups=[dict(group=g,indices=np.flatnonzero(pg==g)) for g in range(66) if sum(pg==g)>1],root_identity_exact=True,provenance=early['mapping']['provenance'],no_new_root_or_connection=True))
  import mujoco as mj
  from flygym.compose.fly.musculoskeletal import _load_mjcf
  m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();calls['model_compile']+=1
  compiled={k:getattr(m,k).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)};cache=ar(C/'COMPILED_ACTUAL_ARRAYS.npz');assert all(np.array_equal(v,cache[k]) for k,v in compiled.items())
  np.savez_compressed(O/'COMPILED_MODEL.npz',**compiled)
  assert (m.nq,m.nv,m.nu,m.na,m.nmesh,m.nmocap,m.neq)==(49,48,90,90,71,0,0)
  assert [m.actuator(u).name for u in range(90)]==[r['actuator'] for r in mapping]
  hinge=np.arange(6,48);jointnames=[m.joint(int(m.dof_jntid[j])).name for j in hinge]
  common=dict(pool_group=pg,unresolved_mask=pg<0,G=G,P74=perms[1],P90=perms[2],root_ids=roots,joint_names=np.array(jointnames),hinge=hinge)
  np.savez_compressed(O/'GROUP_MASKS.npz',**common)
  details=[]
  for pi,(label,z,i) in enumerate([('initial0',b,0),('original_release1p3',b,2600),('slow_release1p25',s,2500)]):
   d=mj.MjData(m)
   for dest,src in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('qfrc_applied','assist'),('xfrc_applied','external')]:getattr(d,dest)[:]=z[src][i]
   d.time=z['time'][i];mj.mj_forward(m,d);calls['forward']+=1
   actualmom=matrix_moment(d);actualgain=np.array([mj.mju_muscleGain(d.actuator_length[u],d.actuator_velocity[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain']+=90
   error=dict(saved_force=err(d.actuator_force,z['muscle_force'][i]),moment_force=err(actualmom.T@d.actuator_force,d.qfrc_actuator),saved_com=err(d.subtree_com[m.body('Thorax').id],z['com'][i]));assert max(error.values())<1e-8
   a=dict(qpos=d.qpos.copy(),qvel_actual=d.qvel.copy(),activation_actual=d.act.copy(),command_actual=d.ctrl.copy(),actual_moment=actualmom,actual_gain=actualgain,actual_force=d.actuator_force.copy(),actual_qfrc_bias=d.qfrc_bias.copy(),actual_qfrc_passive=d.qfrc_passive.copy(),original_assist=d.qfrc_applied.copy(),original_external=d.xfrc_applied.copy())
   d.qvel[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0;mj.mj_forward(m,d);calls['forward']+=1
   moment=matrix_moment(d);gain=np.array([mj.mju_muscleGain(d.actuator_length[u],0,m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);bias=np.array([mj.mju_muscleBias(d.actuator_length[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_biasprm[u,:9]) for u in range(90)]);calls['muscleGain']+=90;calls['muscleBias']+=90
   contacts=[];excluded=[];J=[];positions=[];bodypositions=[];rootangular=[]
   for k in range(d.ncon):
    co=d.contact[k];g1,g2=int(co.geom1),int(co.geom2);p1=m.geom_type[g1]==mj.mjtGeom.mjGEOM_PLANE;p2=m.geom_type[g2]==mj.mjtGeom.mjGEOM_PLANE
    if not (p1 or p2):continue
    g=g2 if p1 else g1;name=m.geom(g).name;foot=any(name.startswith(l) for l in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in name
    info=dict(original_contact_index=k,geom=name,geom_id=g,body_id=int(m.geom_bodyid[g]),plane_geom_id=g1 if p1 else g2,position=co.pos.copy(),frame=co.frame.copy(),dist=float(co.dist),includemargin=float(co.includemargin),condim=int(co.dim),friction=co.friction.copy(),mu=float(co.friction[0]),solref=co.solref.copy(),solimp=co.solimp.copy(),efc_address=int(co.efc_address))
    if not foot:excluded.append(info);continue
    jp=np.zeros((3,48));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,co.pos,int(m.geom_bodyid[g]));calls['contact_jacobian']+=1
    J.append(jp);positions.append(co.pos.copy());bodypositions.append(d.xpos[m.body('Thorax').id].copy());rootangular.append(d.cdof[3:6,:3].copy());contacts.append(info)
   J=np.asarray(J).reshape(-1,3,48);A=moment.T*gain[None,:];base=d.qfrc_passive+moment.T@bias-d.qfrc_bias
   assert err(moment,actualmom)<1e-10 and err(A[:6],0)<1e-10
   active_limits=np.flatnonzero(np.asarray(d.efc_type)==int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT))
   fraction=(d.qpos[m.jnt_qposadr[1:]]-m.jnt_range[1:,0])/(m.jnt_range[1:,1]-m.jnt_range[1:,0])
   a.update(moment=moment,gain=gain,bias=bias,length=d.actuator_length.copy(),static_velocity=d.actuator_velocity.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),A90=A,base=base,J=J,positions=np.array(positions).reshape(-1,3),root_position=np.array(bodypositions).reshape(-1,3),root_angular=np.array(rootangular).reshape(-1,3,3),mu=np.array([x['mu'] for x in contacts]),native_forward_qacc_static=d.qacc.copy(),assumed_equilibrium_qacc=np.zeros(48),joint_range_fraction=fraction,active_joint_limit_efc_rows=active_limits)
   np.savez_compressed(O/f'POSE_{pi}.npz',**a)
   details.append(dict(label=label,time=float(z['time'][i]),q_sha256=hashlib.sha256(q[pi].tobytes()).hexdigest(),contacts=contacts,excluded_nonfoot_floor=excluded,errors=error,active_joint_limit_rows=active_limits,near_range_1percent=np.flatnonzero((fraction<=.01)|(fraction>=.99)),force_residual_units='nativeforce DOF0:3; nativetorque DOF3:48'))
   for arm,P in zip(ARMS,perms):
    eq=np.column_stack([A@P,J.transpose(2,0,1).reshape(48,-1)])
    rhs=-base-A@np.full(90,FLOOR);n=eq.shape[1];ub=np.zeros((4*len(J),n));bounds=np.array([[0,1-FLOOR]]*P.shape[1]+[[-np.inf,np.inf],[-np.inf,np.inf],[0,np.inf]]*len(J))
    for c,mu in enumerate(a['mu']):
     for f,(sx,sy) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):ub[4*c+f,P.shape[1]+3*c:P.shape[1]+3*c+3]=[sx,sy,-mu]
    scale=np.maximum(1,np.maximum(abs(eq).max(1),abs(rhs)))
    np.savez_compressed(O/f'LP_{pi}_{arm}.npz',A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=np.zeros(len(ub)),bounds=bounds,row_scale=scale,P=P)
  dump('POSE_CONTACTS.json',details);dump('COLLECT_EXECUTION.json',dict(calls=calls,exit_code=0,compiled_arrays_exact=len(compiled),weight=float(m.body_mass.sum()*abs(m.opt.gravity[2])),model_parameters_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in compiled.items())))
  missing=[];e=ar(E/'TRAIN_000_BASELINE_AUDIT.npz');c=ar(C/'AUDIT.npz');release=int(np.argmin(abs(e['time']-1.3)));rh=jointnames.index('joint_RHTrochanter_pitch')
  for r in mapping:
   u=r['actuator_index']
   if pg[u]>=0:continue
   coeff=e['moment_90x48'][release,u,hinge]*e['gain'][release,u];cycle=c['coefficient_42x90'][:,:,u]
   directions=[dict(joint=jointnames[j],release_coefficient=float(coeff[j]),cycle_min=float(cycle[:,j].min()),cycle_max=float(cycle[:,j].max())) for j in range(42) if max(abs(coeff[j]),abs(cycle[:,j]).max())>1e-12]
   missing.append(dict(**r,missing_reason=r['source_annotation']['reason'],known_roots_not_in_tier=False,metadata_alias_unknown='no authorized new alias/root inferred',mechanical_muscle_exists=True,current_directions=directions,RH_trochanter_positive_help_at_original_actual_state=bool(coeff[rh]>1e-12),RH_coefficient=float(coeff[rh]),support_interpretation='These are known joint-coordinate directions, not demonstrated stance load directions; sign alone does not establish useful bearing. Old independent90 box already included all16.'))
  dump('UNRESOLVED16_AUTHORITY.json',missing)
  assert all(sha(p)==h for p,h in hashes.items());dump('SOURCE_HASHES_AFTER_COLLECT.json',hashes)
  print(json.dumps(dict(calls=calls,contacts=[len(v['contacts']) for v in details],active_limits=[len(v['active_joint_limit_rows']) for v in details])))
 except Exception:
  dump('COLLECT_FAILURE.json',dict(calls=calls,error=traceback.format_exc()));raise

def solve():
 from scipy.optimize import linprog
 assert not (O/'RESULT.json').exists()
 rows=[];calls=0
 for pi in range(3):
  for arm in ARMS:
   p=ar(O/f'LP_{pi}_{arm}.npz');a=p['A_eq'];b=p['b_eq'];scale=p['row_scale'];bounds=[(None if np.isneginf(lo) else lo,None if np.isposinf(hi) else hi) for lo,hi in p['bounds']]
   assert calls<9
   result=linprog(np.zeros(a.shape[1]),A_eq=a/scale[:,None],b_eq=b/scale,A_ub=p['A_ub'] if len(p['A_ub']) else None,b_ub=p['b_ub'] if len(p['b_ub']) else None,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});calls+=1
   dump('SOLVER_CALL_GATE.json',dict(calls=calls,maximum=9,solves=[dict(pose=x['pose'],arm=x['arm']) for x in rows]+[dict(pose=pi,arm=arm)],no_other_solver_calls=True))
   row=dict(pose=pi,arm=arm,status=int(result.status),success=bool(result.success),message=result.message,nit=int(result.nit),witness=False,equilibrium_residual_maxabs_native=None)
   if result.success:
    x=result.x;res=a@x-b;fr=p['A_ub']@x;act=FLOOR+p['P']@x[:p['P'].shape[1]]
    mx=err(res,0);fv=float(np.max(fr,initial=0));bounds_ok=bool(np.all(x>=p['bounds'][:,0]-1e-10) and np.all(x<=p['bounds'][:,1]+1e-10))
    passed=mx<=1e-8 and fv<=1e-8 and bounds_ok
    np.savez_compressed(O/f'SOLUTION_{pi}_{arm}.npz',x=x,activation90=act,residual=res,friction_residual=fr)
    row.update(witness=passed,equilibrium_residual_maxabs_native=mx,root_force_maxabs_native=err(res[:3],0),torque_maxabs_native=err(res[3:],0),friction_max_violation=fv,bounds_ok=bounds_ok,gate='PASS' if passed else 'NUMERICAL_GATE_FAIL')
   else:row['gate']='NO_WITNESS_SOLVER_INFEASIBLE' if result.status==2 else 'NO_WITNESS_SOLVER_UNKNOWN'
   rows.append(row);print(json.dumps(row),flush=True)
 assert calls==9
 dump('RESULT.json',dict(rows=rows,calls=dict(linprog=9,model_compile=0,forward=0,body_step=0,neural_step=0,network=0,training=0),scope='Only three fixed saved q, current foot candidate geometry, unchanged parameters and explicitly simplified force assumptions; no full-physics impossibility or autonomous success. No inferred neural connection.'))

def verify():
 # Independent scalar construction, no MuJoCo/scipy imported and no LP solved.
 mp=js(O/'AUTHORITATIVE_MAPPING.json');rows=mp['mapping'];g=ar(O/'GROUP_MASKS.npz');roots=ar(B/'IDENTITIES.npz')[mp['root_key']];groups={};PG=[]
 for r in rows:
  ids=tuple(r['local_indices']);assert list(roots[list(ids)].astype(object))==r['tier_root_ids']
  if ids and ids not in groups:groups[ids]=len(groups)
  PG.append(groups[ids] if ids else -1)
 assert PG==g['pool_group'].tolist() and sum(v<0 for v in PG)==16 and len(groups)==66
 assert np.array_equal(g['unresolved_mask'],np.array(PG)<0)
 assert sum(r['status']=='complete' for r in rows)==67 and sum(r['status']=='partial' for r in rows)==7
 errors=dict(A90=0.,base=0.,eq=0.,rhs=0.,friction=0.,root_jac_translation=0.,root_jac_rotation=0.,witness=0.)
 result=js(O/'RESULT.json');contact=js(O/'POSE_CONTACTS.json');verified=[]
 for pi in range(3):
  p=ar(O/f'POSE_{pi}.npz');nc=len(p['J']);AA=np.array([[float(p['moment'][u,j])*float(p['gain'][u]) for u in range(90)] for j in range(48)])
  base=np.array([float(p['qfrc_passive'][j])-float(p['qfrc_bias'][j])+math.fsum(float(p['moment'][u,j])*float(p['bias'][u]) for u in range(90)) for j in range(48)])
  errors['A90']=max(errors['A90'],err(AA,p['A90']));errors['base']=max(errors['base'],err(base,p['base']))
  for c in range(nc):
   errors['root_jac_translation']=max(errors['root_jac_translation'],err(p['J'][c,:,:3],np.eye(3)))
   for j in range(3):
    # Free-root native angular dofs are body-local axes recorded as world cdof.
    pred=np.cross(p['root_angular'][c,j],p['positions'][c]-p['root_position'][c]);errors['root_jac_rotation']=max(errors['root_jac_rotation'],err(pred,p['J'][c,:,3+j]))
   assert contact[pi]['contacts'][c]['mu']==p['mu'][c]
  for arm in ARMS:
   z=ar(O/f'LP_{pi}_{arm}.npz');P=z['P'];n=P.shape[1];eq=np.empty_like(z['A_eq'])
   # Rebuild permissions from authoritative pool data, rather than trusting LP P.
   expected=np.zeros_like(P)
   if arm=='mapped66':
    for u,v in enumerate(PG):
     if v>=0:expected[u,v]=1
   else:
    indices=[u for u,v in enumerate(PG) if v>=0] if arm=='independent74' else list(range(90))
    for j,u in enumerate(indices):expected[u,j]=1
   assert np.array_equal(P,expected)
   for j in range(48):
    for k in range(n):eq[j,k]=math.fsum(float(AA[j,u])*float(expected[u,k]) for u in range(90))
    for c in range(nc):
     for k in range(3):eq[j,n+c*3+k]=p['J'][c,k,j]
   rhs=np.array([-base[j]-FLOOR*math.fsum(float(AA[j,u]) for u in range(90)) for j in range(48)])
   errors['eq']=max(errors['eq'],err(eq,z['A_eq']));errors['rhs']=max(errors['rhs'],err(rhs,z['b_eq']))
   ub=np.zeros_like(z['A_ub'])
   for c in range(nc):
    for k,(sx,sy) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):ub[4*c+k,n+3*c:n+3*c+3]=[sx,sy,-p['mu'][c]]
   errors['friction']=max(errors['friction'],err(ub,z['A_ub']));assert not z['b_ub'].any()
   rr=next(r for r in result['rows'] if r['pose']==pi and r['arm']==arm)
   path=O/f'SOLUTION_{pi}_{arm}.npz'
   if rr['success']:
    w=ar(path);x=w['x'];a=np.array([FLOOR+math.fsum(float(expected[u,k])*float(x[k]) for k in range(n)) for u in range(90)])
    res=np.array([math.fsum(float(eq[j,k])*float(x[k]) for k in range(len(x)))-rhs[j] for j in range(48)])
    errors['witness']=max(errors['witness'],err(res,w['residual']),err(a,w['activation90']))
    assert err(res,0)<=1e-8 and np.min(a)>=FLOOR-1e-10 and np.max(a)<=1+1e-10
    for c in range(nc):
     fx,fy,fz=x[n+3*c:n+3*c+3];assert fz>=-1e-10 and abs(fx)+abs(fy)<=p['mu'][c]*fz+1e-8
   else:assert not path.exists() and not rr['witness']
   verified.append(dict(pose=pi,arm=arm,solver_repeated=False,status=rr['status'],witness_verified=rr['witness']))
 assert max(errors.values())<1e-8,errors
 assert js(O/'SOLVER_CALL_GATE.json')['calls']==9
 hashes=js(O/'SOURCE_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in hashes.items())
 assert len(js(O/'UNRESOLVED16_AUTHORITY.json'))==16
 dump('INDEPENDENT_ARRAY_VERIFICATION.json',dict(errors=errors,rows=verified,mapping_root_exact=True,source_unchanged=True,calls=dict(model_compile=0,forward=0,solver=0,body_step=0,neural_step=0),method='fresh purearrays; scalar fsum matrix/permissions/base/contactroot wrench and witness reconstruction, no main construction helper imported',exit_code=0))
 print(json.dumps(errors))
if __name__=='__main__':{'collect':collect,'solve':solve,'verify':verify}[sys.argv[1]]()
