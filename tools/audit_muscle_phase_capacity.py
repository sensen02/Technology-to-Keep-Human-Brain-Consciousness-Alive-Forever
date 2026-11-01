"""Eight frozen states: total muscle+held-assist torque substitution; never integrates."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from flygym.compose.fly.musculoskeletal import _load_mjcf
ROOT=Path(__file__).resolve().parents[1]
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-phase-capacity/20261101T193630Z')
H=Path('/home/sensen/Desktop/engine修复验收/s5-handover-torque-substitution/20261101T191552Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-torque-diagnostic/20261028T060000Z')
TIMES=np.array([1.1,1.125,1.15,1.175,1.2,1.225,1.25,1.275]);LOW=.0001;UP=1.-LOW;LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):
 (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();calls=dict(model_load=0,forward=0,body_step=0,neural_step=0)
 sources=[S/n for n in ['BODY.npz','MODEL.xml','MAPPING.json','GAIT.npz','PREREGISTRATION.json']]+[H/'RESULT.json',H/'FIT_ARRAYS.npz',H/'MANIFEST_SHA256.json',OLD/'PREREGISTRATION.json',OLD/'CAPACITY.json',ROOT/'diagnose_walk_neural_torque.py',ROOT/'tools/audit_handover_torque_substitution.py',Path(__file__)]
 hashes={str(p):sha(p) for p in sources};model_sha=hashes[str(S/'MODEL.xml')];assert model_sha=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
 dump('PREREGISTRATION.json',dict(source_sha256=hashes,model_sha256=model_sha,times_s=TIMES,selection='Fixed last complete0.2s assisted period; no phase search or new handover. t1.3 bounded results reused, no new forward.',target='Actual muscle generalized torque at saved(q,v,act) + recorded held qfrc_applied. At unchanged q,v, predict passive muscle torque + A*activation. This is instantaneous torque substitution, not equilibrium or a control command.',static_forward_budget='1 compile, exactly3forward per state: actual/act0/act1;24total; no integration',linearity_gate='saved muscle_force, moment.T@force and affine actual-activation reconstruction all maxabs<=1e-8; stop and preserve failure otherwise',activation_box=dict(lower=LOW,upper=1.,status='declared sustained-drive static approximation, NOT an XML actlimit or proof of immediate dynamic reachability'),families={'tied66':'74 mapped muscle rows share66 equal-activation pool increments;16 unresolved held at floor','independent74':'mapped rows independent; shared-pool equality relaxed','relaxed90':'all90 independent, including16 unmapped, not neural available'},forms={'box':'0<=x<=.9999','positive_cone':'x>=0, no upper; algebraic extrapolation, not biologically/MuJoCo-admissible activation','linear_span':'x signed and unbounded; algebraic extrapolation only'},objective='unweighted half squared L2 torque residual on42 actual hinge DOFs; denominator norm(total target), same across all fits at a phase',solvers='scipy lsq_linear BVLS tol1e-12 maxiter2000 for box/cone; numpy lstsq rcond1e-12 for span',verification='independent bounds, gradient/KKT, box convex supporting-plane lower bound; cone dual feasibility/gap to numerical tolerance1e-8; orthogonal span projection with rank threshold1e-12. No exact arithmetic certificate claimed.',history='060 already used same model and correct total target on058 sparse trajectory states; not current121 last-cycle states, no74/cone/span.191552Z1.3 reused. Not first correct-target analysis.',units='native model torque; absolute biological calibration NOT_ESTABLISHED'))
 try:
  with np.load(S/'BODY.npz') as z:b={k:z[k].copy() for k in z.files}
  mapping=json.loads((S/'MAPPING.json').read_text())['rows'];groups={}
  for r in mapping:
   if r['local_indices']:groups.setdefault(tuple(r['local_indices']),[]).append(r['actuator_index'])
  mapped=np.array([r['actuator_index'] for r in mapping if r['local_indices']],int);G=np.zeros((90,len(groups)))
  for j,ids in enumerate(groups.values()):G[ids,j]=1
  assert len(groups)==66 and len(mapped)==74
  permissions={'tied66':G,'independent74':np.eye(90)[:,mapped],'relaxed90':np.eye(90)}
  model=_load_mjcf(str(S/'MODEL.xml')).compile();calls['model_load']+=1
  assert (model.nq,model.nv,model.nu,model.na)==(49,48,90,90)
  assert [model.actuator(i).name for i in range(90)]==[r['actuator'] for r in mapping]
  assert np.array_equal(model.actuator_actadr,np.arange(90))
  hinge=np.array([i for i in range(model.nv) if model.jnt_type[model.dof_jntid[i]]==mujoco.mjtJoint.mjJNT_HINGE]);assert np.array_equal(hinge,np.arange(6,48))
  jointnames=[model.joint(int(model.dof_jntid[i])).name for i in hinge]
  legcols={L:np.array([j for j,n in enumerate(jointnames) if n.startswith('joint_'+L)]) for L in LEGS};assert all(len(ix)==7 for ix in legcols.values())
  ids=np.array([np.argmin(abs(b['time']-t)) for t in TIMES]);assert np.max(abs(b['time'][ids]-TIMES))<1e-9
  arrays={k:[] for k in ['moment','f_actual','f0','f1','A','passive','actual_torque','assist','target','activation']};states=[]
  for k,i in enumerate(ids):
   d=mujoco.MjData(model);d.qpos[:]=b['qpos'][i];d.qvel[:]=b['qvel'][i];d.act[:]=b['activation'][i];d.ctrl[:]=b['command'][i];d.time=b['time'][i]
   mujoco.mj_forward(model,d);calls['forward']+=1;fa=d.actuator_force.copy();ta=d.qfrc_actuator.copy();moment=np.zeros((90,48))
   for j in range(90):
    a=int(d.moment_rowadr[j]);n=int(d.moment_rownnz[j]);moment[j,d.moment_colind[a:a+n]]=d.actuator_moment[a:a+n]
   d.act[:]=0;mujoco.mj_forward(model,d);calls['forward']+=1;f0=d.actuator_force.copy()
   d.act[:]=1;mujoco.mj_forward(model,d);calls['forward']+=1;f1=d.actuator_force.copy()
   A=moment.T*(f1-f0)[None,:];passive=moment.T@f0;assist=b['assist'][i].copy();target=ta+assist;activation=b['activation'][i].copy()
   errors=dict(saved_force=float(abs(fa-b['muscle_force'][i]).max()),moment_torque=float(abs(moment.T@fa-ta).max()),affine_force=float(abs(f0+(f1-f0)*activation-fa).max()),affine_torque=float(abs(passive+A@activation-ta).max()))
   assert max(errors.values())<=1e-8,errors
   assert abs(A[:6]).max()<1e-10 and abs(assist[:6]).max()==0
   for name,v in dict(moment=moment,f_actual=fa,f0=f0,f1=f1,A=A,passive=passive,actual_torque=ta,assist=assist,target=target,activation=activation).items():arrays[name].append(v)
   states.append(dict(requested_s=float(TIMES[k]),saved_s=float(b['time'][i]),body_index=int(i),phase_rad=float(max(0,b['time'][i]-.15)*10*np.pi%(2*np.pi)),foot_fz=b['foot_force'][i,:,2],loaded_feet=int((b['foot_force'][i,:,2]>0).sum()),com=b['com'][i],foot_contact=b['foot_contact'][i],nonfoot_count=int(b['nonfoot_count'][i]),actual_muscle_torque_norm=float(np.linalg.norm(ta[hinge])),held_assist_norm=float(np.linalg.norm(assist[hinge])),total_target_norm=float(np.linalg.norm(target[hinge])),activation_min=float(activation.min()),activation_max=float(activation.max()),activation_below_declared_floor=int((activation<LOW-1e-12).sum()),activation_above_one=int((activation>1+1e-12).sum()),linearity_errors=errors))
  arrays={k:np.array(v) for k,v in arrays.items()}
  np.savez_compressed(O/'STATIC_ARRAYS.npz',**arrays,body_indices=ids,times=TIMES,hinge_indices=hinge,group_matrix=G,mapped_indices=mapped,joint_names=np.array(jointnames))
  dump('MODEL_IDENTITY.json',dict(model_sha256=model_sha,joint_names=jointnames,actuator_names=[r['actuator'] for r in mapping],groups=[dict(local_indices=list(key),actuator_indices=val) for key,val in groups.items()],actual_actlimited=model.actuator_actlimited,actual_actrange=model.actuator_actrange,ctrlrange=model.actuator_ctrlrange,whole_saved_activation_min=float(b['activation'].min()),whole_saved_activation_max=float(b['activation'].max()),selected_activation_min=float(arrays['activation'].min()),selected_activation_max=float(arrays['activation'].max()),actual_joint_count=len(jointnames),coordinate_warning='Actual7DOF/leg: coxa3,trochanter3,tibia1; no independenttarsus hinge. Derived from compiled joint identities, not reference joint names.'))
  raw={};results=[]
  for k in range(8):
   A=arrays['A'][k];passive=arrays['passive'][k];target=arrays['target'][k];base=passive+A@np.full(90,LOW);y=target[hinge]-base[hinge];den=np.linalg.norm(target[hinge]);fits=[]
   for family,P in permissions.items():
    B=A[hinge]@P;u,s,vh=np.linalg.svd(B,full_matrices=False);rank=int((s>s[0]*1e-12).sum());span_lower=float(np.linalg.norm(y-u[:,:rank]@(u[:,:rank].T@y)))
    for form in ['box','positive_cone','linear_span']:
     if form=='linear_span':
      x,_,_,_=np.linalg.lstsq(B,y,rcond=1e-12);success=True;message='signed unbounded numerical least squares'
     else:
      fit=lsq_linear(B,y,bounds=(0.,UP if form=='box' else np.inf),method='bvls',tol=1e-12,max_iter=2000);x=fit.x;success=bool(fit.success);message=fit.message
     assert success,(k,family,form,message)
     r=B@x-y;grad=B.T@r;obj=.5*float(r@r);prediction=base+A@P@x;deficit=target[hinge]-prediction[hinge]
     if form=='box':
      linear_min=float(np.minimum(0,UP*grad).sum());lb=obj+linear_min-float(grad@x);gap=obj-lb;kkt=np.where(x<1e-8,np.minimum(grad,0),np.where(x>UP-1e-8,np.maximum(grad,0),grad));dual_violation=0.
     elif form=='positive_cone':
      lb=-float(y@r)-.5*float(r@r);gap=obj-lb;kkt=np.where(x<1e-8,np.minimum(grad,0),grad);dual_violation=float(max(0.,-grad.min()))
     else:lb=.5*span_lower**2;gap=obj-lb;kkt=grad;dual_violation=0.
     legdetail={}
     for L,jx in legcols.items():
      worst=int(jx[np.argmax(abs(deficit[jx]))]);legdetail[L]=dict(residual_l2=float(np.linalg.norm(deficit[jx])),max_deficit_joint=jointnames[worst],signed_target_minus_prediction=float(deficit[worst]),direction_needed='positive' if deficit[worst]>0 else 'negative',target=float(target[hinge[worst]]),prediction=float(prediction[hinge[worst]]))
     key=f'p{k}_{family}_{form}';raw[key+'_x']=x;raw[key+'_prediction']=prediction;raw[key+'_residual']=r;raw[key+'_gradient']=grad
     fits.append(dict(family=family,form=form,solver_success=success,message=message,residual_l2=float(np.linalg.norm(r)),relative_residual=float(np.linalg.norm(r)/den),objective_half_squared=obj,numerical_lower_bound_half_squared=float(lb),numerical_optimality_gap=float(gap),kkt_max=float(abs(kkt).max()),dual_feasibility_violation=dual_violation,rank=rank,span_residual_lower_bound=span_lower,x_min=float(x.min()),x_max=float(x.max()),negative_variables=int((x<0).sum()),above_box_variables=int((x>UP+1e-8).sum()),legs=legdetail,certificate_note='box: global convex tangent bound; cone: dual bound valid subject to listed numerical feasibility tolerance; span: numerical rank projection. Cone/span are unavailable relaxations.'))
   results.append(dict(state=states[k],fits=fits));print('PHASE_DONE',TIMES[k],[(v['family'],v['form'],round(v['relative_residual'],6)) for v in fits],flush=True)
  np.savez_compressed(O/'FIT_ARRAYS.npz',**raw)
  historical=json.loads((H/'RESULT.json').read_text())
  reused=[r for r in historical['fits'] if r['target']=='total_replacement']
  assert historical['model_sha256']==model_sha
  dump('RESULT.json',dict(states=results,reused_1p3_bounded_results=reused,reused_1p3_source=str(H/'RESULT.json'),old060_scope='Same3470model and correct target already analyzed on old058 sparse trajectory; not current121 last cycle, no cone/span/74 tests.',calls=calls,interpretation='Only these frozen(q,v) instantaneous total torques. Fits are never applied. No stand/walk or phase-handover feasibility is inferred. Root6zero does not certify contact equilibrium.'))
  unchanged=hashes=={str(p):sha(p) for p in sources};assert unchanged
  dump('EXECUTION.json',dict(calls=calls,wall_seconds=time.time()-start,source_hashes_unchanged=unchanged,exit_code=0));print('DONE',calls,flush=True)
 except Exception:
  dump('EXECUTION.json',dict(calls=calls,wall_seconds=time.time()-start,exit_code=1,error=traceback.format_exc()));raise
if __name__=='__main__':main()
