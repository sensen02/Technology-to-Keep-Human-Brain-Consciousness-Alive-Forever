"""S2/S4 native static constraint completion; capture once, arrays-only solve/review.
No simulation, neural calls, controller, defaults edits, or automatic sealing.
"""
from pathlib import Path
import sys, json, hashlib, math, shutil, traceback
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
D=Path('/home/sensen/Desktop/engine修复验收')
S=D/'s2-mapped-support-capacity/20261102T170741Z'
O=D/'s2-native-support-constraints/20261102T172633Z'
XML=D/'s5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml'
ARMS=['mapped66','independent74','independent90']
MODES=['native','outerbox']
FLOOR=1e-4

def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):
 (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def err(a,b=0):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0))
def moment(d):
 if d.actuator_moment.shape==(90,48):return d.actuator_moment.copy()
 a=np.zeros((90,48))
 for u in range(90):
  off=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);a[u,d.moment_colind[off:off+n]]=d.actuator_moment[off:off+n]
 return a

def capture():
 O.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),XML,R/'AGENTS.md',R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md']+sorted(p for p in S.iterdir() if p.is_file())
 dump('SOURCE_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs})
 dump('PREREGISTRATION.json',dict(head='8e1ee0ca646e370413bd32b5637bdf7df5bae2d9',stage='S2/S4',source=str(S),scope='Three same saved q; qvel0; clear assist/external/warmstart; unchanged original3470 model/469 arrays. Independent reactions are a conditional static model, not calibrated soft solver/complementarity or neural reachability.',budgets=dict(model_compile=1,static_forward=3,gain=0,bias=0,contactJac=0,feasibilityLP=18,FarkasLP_max=3,body=0,neural=0,network=0,training=0,plasticity=0),reuse='POSE A90/gain/bias/passive/bias/root physical J/base, all old48eq muscle blocks and RHS, G66/P74/P90, floor1e-4; old9innerLP solver outputs cached only',models=dict(native='actual efc contact rows transposed; pyramidal nonnegative ray reactions; elliptic inner polyhedral rays if encountered',outerbox='old physical world J.T F, Fz>=0, abs(Fx)<=mu Fz and abs(Fy)<=mu Fz'),internal='retain all active equality free, frictionloss +/- actual efc_frictionloss, limits nonnegative actual native J, selfcontact native reaction cone; exclude all nonfoot-floor bearing explicitly',conditioning='invertible positive rowmax scaling only',gates=dict(native_eq=1e-8,friction=1e-8,bounds=1e-10,activation_floor=FLOOR,dual_stationarity=1e-8,dual_nonnegative=-1e-10,dual_margin=-1+1e-8),Farkas='For E x=b,H x<=h with all finite bounds stacked in H: E.T y+H.T z=0,z>=0,b.T y+h.T z<=-1; y split +/-; raw and conditioned checks plus vector magnitude/relative residual; floating numerical certificate, not symbolic proof. Only independent90 outerbox at most once/pose if status2.',no_seal_before_parent=True))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 calls=dict(model_compile=0,forward=0,muscleGain=0,muscleBias=0,contact_jacobian=0,body_step=0,neural_step=0,network=0,training=0,plasticity=0,linprog=0)
 try:
  import mujoco as mj
  from flygym.compose.fly.musculoskeletal import _load_mjcf
  m=_load_mjcf(str(XML)).compile();calls['model_compile']+=1
  compiled={k:getattr(m,k).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)}
  np.savez_compressed(O/'COMPILED_MODEL.npz',**compiled)
  old=ar(S/'COMPILED_MODEL.npz');assert set(compiled)==set(old) and len(old)==469
  assert all(np.array_equal(v,old[k]) for k,v in compiled.items())
  cone=int(m.opt.cone);sparse=bool(mj.mj_isSparse(m));summary=[]
  dump('MODEL_SCHEMA.json',dict(mujoco=mj.__version__,cone=cone,cone_name=str(m.opt.cone),sparse=sparse,jacobian_option=int(m.opt.jacobian),nq=m.nq,nv=m.nv,nu=m.nu,nmocap=m.nmocap,neq=m.neq,compiled_arrays_exact=469,xml_sha=sha(XML),constraint_types={k:int(v) for k,v in mj.mjtConstraint.__members__.items()}))
  assert (m.nq,m.nv,m.nu,m.nmocap,m.neq)==(49,48,90,0,0)
  for pi in range(3):
   p=ar(S/f'POSE_{pi}.npz');d=mj.MjData(m)
   d.qpos[:]=p['qpos'];d.qvel[:]=0;d.act[:]=p['activation_actual'];d.ctrl[:]=p['command_actual'];d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.qacc_warmstart[:]=0
   d.time=js(S/'POSE_CONTACTS.json')[pi]['time']
   mj.mj_forward(m,d);calls['forward']+=1
   raw=d.efc_J.copy();J=np.zeros((d.nefc,m.nv))
   if sparse:
    for row in range(d.nefc):
     off=int(d.efc_J_rowadr[row]);n=int(d.efc_J_rownnz[row]);J[row,d.efc_J_colind[off:off+n]]=raw[off:off+n]
   else:J[:]=raw.reshape(d.nefc,m.nv)
   contacts=[]
   for ci in range(d.ncon):
    c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2);pl1=m.geom_type[g1]==mj.mjtGeom.mjGEOM_PLANE;pl2=m.geom_type[g2]==mj.mjtGeom.mjGEOM_PLANE
    floor=bool(pl1 or pl2);g=g2 if pl1 else g1;name=m.geom(g).name
    foot=bool(floor and any(name.startswith(l) for l in ['LF','LM','LH','RF','RM','RH']) and 'Tars' in name)
    count=1 if c.dim==1 else (2*(int(c.dim)-1) if cone==0 else int(c.dim))
    rows=list(range(int(c.efc_address),int(c.efc_address)+count)) if c.efc_address>=0 else []
    contacts.append(dict(index=ci,geom1=g1,geom2=g2,name1=m.geom(g1).name,name2=m.geom(g2).name,floor=floor,foot_floor=foot,selfcontact=not floor,dim=int(c.dim),efc_address=int(c.efc_address),rows=rows,pos=c.pos.copy(),frame=c.frame.reshape(3,3).copy(),friction=c.friction.copy(),dist=float(c.dist),includemargin=float(c.includemargin),exclude=int(c.exclude)))
   state=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),qfrc_applied=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy(),qacc_warmstart_after=d.qacc_warmstart.copy(),forward_qacc=d.qacc.copy(),assumed_equilibrium_qacc=np.zeros(48),J=J,efc_J_raw=raw,efc_type=d.efc_type.copy(),efc_id=d.efc_id.copy(),efc_force=d.efc_force.copy(),efc_frictionloss=d.efc_frictionloss.copy(),efc_pos=d.efc_pos.copy(),qfrc_constraint=d.qfrc_constraint.copy(),moment=moment(d),length=d.actuator_length.copy(),velocity=d.actuator_velocity.copy(),qfrc_passive=d.qfrc_passive.copy(),qfrc_bias=d.qfrc_bias.copy(),actuator_force=d.actuator_force.copy())
   for name in ['efc_J_rowadr','efc_J_rownnz','efc_J_colind']:state[name]=getattr(d,name).copy()
   np.savez_compressed(O/f'NATIVE_{pi}.npz',**state)
   dump(f'CONTACTS_{pi}.json',contacts)
   diffs={k:err(state[k],p[oldkey]) for k,oldkey in [('qpos','qpos'),('moment','moment'),('length','length'),('velocity','static_velocity'),('qfrc_passive','qfrc_passive'),('qfrc_bias','qfrc_bias'),('forward_qacc','native_forward_qacc_static')]}
   diffs['reused_force']=err(d.actuator_force,p['gain']*d.act+p['bias'])
   diffs['constraint_projection']=err(J.T@d.efc_force,d.qfrc_constraint)
   assert max(diffs.values())<=1e-8,diffs
   assert not d.qvel.any() and not d.qfrc_applied.any() and not d.xfrc_applied.any()
   # Every active row gets a declared policy; nothing silently omitted.
   covered=set();footrows=[];internal=[];ibounds=[];excluded=[];limitinfo=[]
   for c in contacts:
    rows=c['rows'];covered.update(rows)
    if c['floor'] and not c['foot_floor']:excluded+=rows;continue
    if not rows:continue
    if cone==0 or c['dim']==1:
     cols=J[rows].T;bounds=[[0,np.inf]]*len(rows)
    else:
     # finite polyhedral inner approximation: positive normal and +/- each tangent ray.
     dim=c['dim'];rays=[]
     for t in range(1,dim):
      for sign in [1,-1]:
       r=np.zeros(dim);r[0]=1;r[t]=sign*c['friction'][t-1];rays.append(r)
     cols=J[rows].T@np.array(rays).T;bounds=[[0,np.inf]]*len(rays)
    if c['foot_floor']:footrows.append(cols)
    else:
     internal.extend(cols.T);ibounds.extend(bounds)
   for row,typ in enumerate(d.efc_type):
    if row in covered:continue
    typ=int(typ);covered.add(row)
    if typ==0:bound=[-np.inf,np.inf]
    elif typ in [1,2]:bound=[-float(d.efc_frictionloss[row]),float(d.efc_frictionloss[row])]
    elif typ in [3,4]:bound=[0,np.inf]
    else:raise AssertionError(('unclassified active constraint',row,typ))
    internal.append(J[row]);ibounds.append(bound)
    if typ==3:
     ji=int(d.efc_id[row]);qa=int(m.jnt_qposadr[ji]);da=int(m.jnt_dofadr[ji]);q=float(d.qpos[qa]);lo,hi=map(float,m.jnt_range[ji]);side='lower' if abs(q-lo)<=abs(q-hi) else 'upper'
     expected=np.zeros(48);expected[da]=1 if side=='lower' else -1
     assert err(J[row],expected)<=1e-12
     limitinfo.append(dict(row=row,joint_id=ji,joint=m.joint(ji).name,q=q,range=[lo,hi],side=side,dof=da,native_sign=float(J[row,da]),direction_error=err(J[row],expected)))
   assert covered==set(range(d.nefc))
   nativefoot=np.column_stack(footrows) if footrows else np.zeros((48,0));internal=np.array(internal).reshape(-1,48).T;ibounds=np.array(ibounds).reshape(-1,2)
   feet=[c for c in contacts if c['foot_floor']]
   oldcontacts=js(S/'POSE_CONTACTS.json')[pi]['contacts'];assert len(feet)==len(oldcontacts)==[9,5,7][pi]
   jacerr=0.
   for c,oc,jp in zip(feet,oldcontacts,p['J']):
    assert c['index']==oc['original_contact_index'] and c['dim']==3 and err(c['pos'],oc['position'])==0
    # Contact relative velocity J is geom2 minus geom1; plane is static.
    sign=1 if m.geom_type[c['geom1']]==mj.mjtGeom.mjGEOM_PLANE else -1
    local=sign*np.asarray(c['frame'])@jp
    if cone==0:
     expected=np.array([local[0]+c['friction'][t]*sg*local[t+1] for t in range(2) for sg in [1,-1]])
    else:expected=local
    jacerr=max(jacerr,err(J[c['rows']],expected))
   assert jacerr<=1e-8,jacerr
   counts={k:int(np.sum(d.efc_type==int(v))) for k,v in mj.mjtConstraint.__members__.items()}
   assert len(limitinfo)==[0,1,0][pi] and not excluded
   np.savez_compressed(O/f'REACTIONS_{pi}.npz',native_foot_columns=nativefoot,internal_columns=internal,internal_bounds=ibounds)
   summary.append(dict(pose=pi,nefc=d.nefc,counts=counts,contacts=len(contacts),foot_floor=len(feet),selfcontact=sum(c['selfcontact'] for c in contacts),excluded_nonfoot_floor_rows=excluded,limits=limitinfo,native_world_J_error=jacerr,reused_state_errors=diffs,forward_qacc_max=err(d.qacc),all_active_rows_accounted=True))
   dump('CAPTURE_PROGRESS.json',dict(calls=calls,poses=summary))
  dump('CAPTURE_RESULT.json',dict(passed=True,calls=calls,poses=summary,compiled_arrays_exact=469,actual_dynamic_state_not_repeated=True))
  assert all(sha(p)==h for p,h in js(O/'SOURCE_HASHES_BEFORE.json').items())
 except Exception:
  dump('CAPTURE_FAILURE.json',dict(calls=calls,error=traceback.format_exc()));raise

def build():
 assert not (O/'MATRIX_BUILD.json').exists()
 for pi in range(3):
  p=ar(S/f'POSE_{pi}.npz');r=ar(O/f'REACTIONS_{pi}.npz')
  for arm in ARMS:
   old=ar(S/f'LP_{pi}_{arm}.npz');n=old['P'].shape[1]
   for mode in MODES:
    contact=r['native_foot_columns'] if mode=='native' else old['A_eq'][:,n:]
    E=np.column_stack([old['A_eq'][:,:n],contact,r['internal_columns']]);bounds=np.concatenate([old['bounds'][:n],np.array([[0,np.inf]]*contact.shape[1]) if mode=='native' else old['bounds'][n:],r['internal_bounds']])
    U=np.zeros((0 if mode=='native' else 4*len(p['J']),E.shape[1]))
    if mode=='outerbox':
     for c,mu in enumerate(p['mu']):
      for k,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):U[4*c+k,n+3*c:n+3*c+3]=[sx,sy,-mu]
    b=old['b_eq'];scale=np.maximum(1,np.maximum(abs(E).max(1),abs(b)));uscale=np.maximum(1,abs(U).max(1)) if len(U) else np.ones(0)
    np.savez_compressed(O/f'LP_{pi}_{arm}_{mode}.npz',E=E,b=b,U=U,v=np.zeros(len(U)),bounds=bounds,row_scale=scale,ineq_scale=uscale,P=old['P'],muscle_count=n,contact_count=contact.shape[1])
 dump('MATRIX_BUILD.json',dict(cases=18,old_muscle_blocks_and_rhs_exact=True,old_inner_reference='cached old RESULT.json only; not solved',calls=dict(model_compile=0,forward=0,solver=0)))

def Hsystem(p):
 n=p['E'].shape[1];rows=list(p['U']);rhs=list(p['v']);labels=[f'friction:{i}' for i in range(len(rows))]
 for i,(lo,hi) in enumerate(p['bounds']):
  if np.isfinite(lo):
   r=np.zeros(n);r[i]=-1;rows.append(r);rhs.append(-lo);labels.append(f'lower:{i}')
  if np.isfinite(hi):
   r=np.zeros(n);r[i]=1;rows.append(r);rhs.append(hi);labels.append(f'upper:{i}')
 return np.array(rows).reshape(-1,n),np.array(rhs),labels

def solve():
 from scipy.optimize import linprog
 assert not (O/'RESULT.json').exists()
 calls=0;certcalls=0;rows=[];certs=[]
 for pi in range(3):
  for arm in ARMS:
   for mode in MODES:
    name=f'{pi}_{arm}_{mode}';p=ar(O/f'LP_{name}.npz');E,b,U,v=p['E'],p['b'],p['U'],p['v'];scale=p['row_scale'];us=p['ineq_scale']
    result=linprog(np.zeros(E.shape[1]),A_eq=E/scale[:,None],b_eq=b/scale,A_ub=U/us[:,None] if len(U) else None,b_ub=v/us if len(U) else None,bounds=[(None if np.isneginf(lo) else lo,None if np.isposinf(hi) else hi) for lo,hi in p['bounds']],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});calls+=1
    row=dict(pose=pi,arm=arm,contact_model=mode,status=int(result.status),success=bool(result.success),message=result.message,nit=int(result.nit),witness=False)
    if result.success:
     x=result.x;res=E@x-b;fr=U@x-v;nv=int(p['muscle_count']);act=FLOOR+p['P']@x[:nv]
     bo=bool(np.all(x>=p['bounds'][:,0]-1e-10) and np.all(x<=p['bounds'][:,1]+1e-10));fv=float(np.max(fr,initial=0));passed=err(res)<=1e-8 and fv<=1e-8 and bo and np.min(act)>=FLOOR-1e-10 and np.max(act)<=1+1e-10
     np.savez_compressed(O/f'WITNESS_{name}.npz',x=x,activation90=act,residual_raw=res,residual_scaled=res/scale,friction_residual=fr)
     row.update(witness=bool(passed),residual_native=err(res),residual_scaled=err(res/scale),root_force_residual=err(res[:3]),torque_residual=err(res[3:]),friction_violation=fv,bounds_ok=bo)
    rows.append(row);dump('SOLVER_CALL_GATE.json',dict(feasibility_calls=calls,certificate_calls=certcalls,maximum_feasibility=18,maximum_certificate=3));print(json.dumps(row),flush=True)
  # Only predeclared independent90 outerbox, once/pose, no adaptive retry.
  target=next(r for r in rows if r['pose']==pi and r['arm']=='independent90' and r['contact_model']=='outerbox')
  if target['status']==2:
   p=ar(O/f'LP_{pi}_independent90_outerbox.npz');E,b=p['E'],p['b'];H,h,labels=Hsystem(p);es=p['row_scale'];hs=np.maximum(1,np.maximum(abs(H).max(1),abs(h)));Ec=E/es[:,None];bc=b/es;Hc=H/hs[:,None];hc=h/hs
   # Ecy + Hcz stationarity; free equality dual via yplus-yminus.
   M=np.column_stack([Ec.T,-Ec.T,Hc.T]);margin=np.r_[bc,-bc,hc]
   result=linprog(np.zeros(M.shape[1]),A_eq=M,b_eq=np.zeros(E.shape[1]),A_ub=margin[None,:],b_ub=[-1.],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});certcalls+=1
   cr=dict(pose=pi,status=int(result.status),message=result.message,established=False,kind='floating numerical separation only; not symbolic exact proof')
   payload=dict(E=E,b=b,H=H,h=h,eq_scale=es,ineq_scale=hs,H_labels=np.array(labels))
   if result.success:
    w=result.x;yc=w[:48]-w[48:96];zc=w[96:];y=yc/es;z=zc/hs;st=E.T@y+H.T@z;sc=Ec.T@yc+Hc.T@zc;mg=float(b@y+h@z);mc=float(bc@yc+hc@zc);den=np.abs(E).T@np.abs(y)+np.abs(H).T@np.abs(z);rel=float(np.max(abs(st)/np.maximum(1,den),initial=0));neg=float(np.min(z,initial=0))
    passed=err(st)<=1e-8 and err(sc)<=1e-8 and neg>=-1e-10 and mg<=-1+1e-8 and mc<=-1+1e-8
    payload.update(y=y,z=z,y_scaled=yc,z_scaled=zc,split_solution=w,stationarity_raw=st,stationarity_scaled=sc,margin_raw=mg,margin_scaled=mc)
    cr.update(established=bool(passed),stationarity_native=err(st),stationarity_scaled=err(sc),stationarity_relative=rel,margin_native=mg,margin_scaled=mc,dual_min=neg,y_max=err(y),z_max=err(z),summand_magnitude_max=err(den),margin_summand_L1=float(abs(b*y).sum()+abs(h*z).sum()),interpretation='Unit margin requires numerical scaling; tolerances and finite precision do not establish a symbolic mathematical proof.')
   np.savez_compressed(O/f'FARKAS_{pi}.npz',**payload);certs.append(cr);dump('SOLVER_CALL_GATE.json',dict(feasibility_calls=calls,certificate_calls=certcalls,maximum_feasibility=18,maximum_certificate=3));print(json.dumps(cr),flush=True)
 assert calls==18 and certcalls<=3
 dump('RESULT.json',dict(rows=rows,certificates=certs,calls=dict(feasibilityLP=calls,FarkasLP=certcalls,model_compile=0,forward=0,body=0,neural=0),scope='Only these three q/contact/muscle-bound conditional matrices. Status2 is solver report, not automatic certificate. No global pose/mechanical impossibility, softsolver calibration, neural reachability or autonomous support claim.'))

if __name__=='__main__':{'capture':capture,'build':build,'solve':solve}[sys.argv[1]]()
