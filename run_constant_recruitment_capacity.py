"""Bounded engineering capacity diagnostic, never neural standing/control success."""
from pathlib import Path
import os,json,sys,time,traceback,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from scipy.optimize import least_squares
from unittest.mock import patch
import run_neural_muscle_closure as c
from run_command1ms_single import Observer
from run_one_second_motor_pair import INITIAL,sha,dump
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s2-constant-recruitment-capacity/20261028T001000Z')
H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
MP=Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z/MAPPING.json')

def support_metrics(b):
 t=np.asarray(b['time']);last=np.arange(len(t))>len(t)-4001;nf=np.asarray(b['nonfoot_floor_count']);ids=np.flatnonzero(nf)
 return dict(first_nonfoot_s=None if not len(ids) else float(t[ids[0]]),last50_nonfoot_fraction=float((nf[last]>0).mean()),last50_mean_com_height=float(np.asarray(b['com_position'])[last,2].mean()),final_com_height=float(b['com_position'][-1,2]),final_com_vz=float(b['com_velocity'][-1,2]),last50_mean_thorax_height=float(np.asarray(b['thorax_position'])[last,2].mean()),warnings_max=int(np.asarray(b['warnings']).max()),finite=all(np.isfinite(b[k]).all() for k in b if np.asarray(b[k]).dtype.kind in 'biuf'))

def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),Path(c.__file__),P/'run_command1ms_single.py',P/'run_quarter_second_pair.py',INITIAL,Path(c.cp.CANONICAL_XML),MP,H/'normal/BODY.npz',H/'motor_cut/BODY.npz']
 hashes={str(p):sha(p) for p in files}
 prereg=dict(scope='S2/P3 direct constant-muscle engineering capacity witness NOT neural controller or goal pass',source_hashes=hashes,canonical_sha256='65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6',fixed_state='historical normal nearest saved15ms qpos/qvel only, no pose optimization',rollout_initial='exact ORIGINAL INITIAL qpos; qvel=act=ctrl zero; constant optimized control fromt0; no activation preload',body_dt_s=.0000125,duration_s=.25,objective='residual [COMaz/g,.3*L*qacc[3]/g,.3*L*qacc[4]/g,.1*COMax/g,.1*COMay/g,sqrt(.01/nvar)*u]; COM acceleration from gravity plus total world ground reaction / total mass; L initial COM height; root angular acceleration coordinates3/4',objective_weights=dict(vertical=1.,root_roll_pitch=.3,horizontal=.1,effort=.01),bounds=[0,1],modes=['mapped74_exact_pool_ties','all90_independent_upper_capacity_only_if_first_not_substantial'],optimizer=dict(method='trf',max_nfev=8,ftol=1e-6,xtol=1e-6,gtol=1e-6,finite_difference='2-point relative diff_step1e-4',initial_variable=.05,max_forward_per_mode=1000),substantial_criterion='last50_nonfoot_fraction<.5 AND last50meanCOMheight > archivednormalmean+.1 model length units; engineering comparison only, not standing gate',limits=dict(optimization_modes=2,forward_per_mode=1000,dynamic_rollouts=2,body_steps_total=40000),forbidden=['newbaseline','onlinefeedback','posefix','externalforce','gain/site/model edits','neural steps','source audit'],interpretation='failure only rejects bounded candidates, not mechanical impossibility; all90 uses unmapped muscles as direct capacity upper bound not admissible neural mapping; canonical initial jointlimit violations retained')
 dump(O/'PREREGISTRATION.json',prereg)
 counts=dict(model_load=0,mj_forward=0,mj_step=0,neural_steps=0,optimization_forward={},rollouts=0);rf=mujoco.mj_forward;rs=mujoco.mj_step
 def fw(*a,**kw):counts['mj_forward']+=1;return rf(*a,**kw)
 def st(*a,**kw):counts['mj_step']+=1;return rs(*a,**kw)
 summaries={};error=None;begin=time.time()
 try:
  with patch.object(mujoco,'mj_forward',fw),patch.object(mujoco,'mj_step',st):
   m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125);counts['model_load']+=1;dump(O/'INITIAL_GATE.json',gate)
   assert sha(Path(c.cp.CANONICAL_XML))==prereg['canonical_sha256'];assert m.nu==90 and m.nq==49 and m.nv==48 and m.neq==0 and m.nmocap==0
   old=np.load(H/'normal/BODY.npz');cut=np.load(H/'motor_cut/BODY.npz');base=support_metrics(old);dump(O/'HISTORICAL_METRICS.json',dict(normal=base,motor_cut=support_metrics(cut)))
   i=int(np.argmin(abs(old['time']-.015)));assert abs(old['time'][i]-.015)<1e-10 and old['nonfoot_floor_count'][i]==0
   rows=json.loads(MP.read_text())['rows'];groups={}
   for r in rows:
    if r['local_indices']:groups.setdefault(tuple(r['local_indices']),[]).append(r['actuator_index'])
   mapped=list(groups.values());assert sum(map(len,mapped))==74;unmapped=[r['actuator_index'] for r in rows if not r['local_indices']];assert len(unmapped)==16
   dump(O/'AUTHORITY.json',dict(mapped_groups=[dict(local_indices=list(k),actuator_indices=v,actuators=[rows[j]['actuator'] for j in v]) for k,v in groups.items()],unmapped_indices=unmapped,unmapped_names=[rows[j]['actuator'] for j in unmapped],number_tied_variables=len(mapped)))
   di=mujoco.MjData(m);di.qpos[:]=q;mujoco.mj_forward(m,di);mass=float(m.body_mass.sum());g=float(abs(m.opt.gravity[2]));L=float((m.body_mass[:,None]*di.xipos).sum(0)[2]/mass)
   fixed=mujoco.MjData(m);fixed.qpos[:]=old['qpos'][i];fixed.qvel[:]=old['qvel'][i];fixed.xfrc_applied[:]=0;fixed.qfrc_applied[:]=0
   dump(O/'FIXED_STATE.json',dict(index=i,time=float(old['time'][i]),qpos=fixed.qpos,qvel=fixed.qvel,historical_activation=old['activation'][i],historical_command=old['command'][i],mass=mass,gravity=g,L=L,coordinate_units='project model units, absolute force calibration NOT_ESTABLISHED'))
   fixed.act[:]=old['activation'][i];fixed.ctrl[:]=old['command'][i];mujoco.mj_forward(m,fixed);ho=Observer(m,geom);ho(m,fixed)
   historical_acc=m.opt.gravity+ho.rows[-1]['ground_force_world']/mass
   historical_physical=np.array([historical_acc[2]/g,.3*L*fixed.qacc[3]/g,.3*L*fixed.qacc[4]/g,.1*historical_acc[0]/g,.1*historical_acc[1]/g])
   dump(O/'FIXED_HISTORICAL_RESPONSE.json',dict(activation=fixed.act.copy(),command=fixed.ctrl.copy(),physical_residual=historical_physical,physical_loss=float(historical_physical@historical_physical),com_acceleration=historical_acc,world_reaction=ho.rows[-1]['ground_force_world'],root_angular_qacc_native_coordinates=fixed.qacc[3:6].copy()))
   for mode,gs in [('mapped74_exact_pool_ties',mapped),('all90_independent_upper_capacity',[[j] for j in range(90)])]:
    if mode.startswith('all90') and summaries['mapped74_exact_pool_ties']['substantial_engineering_support_improvement']:break
    folder=O/mode;folder.mkdir();calls=0;history=[]
    def expand(x):
     u=np.zeros(90)
     for value,indices in zip(x,gs):u[indices]=value
     return u
    def evaluate(x,record=True):
     nonlocal calls
     if calls>=1000:raise RuntimeError('STOP optimization forward budget reached')
     calls+=1;counts['optimization_forward'][mode]=calls;u=expand(x);fixed.act[:]=np.maximum(u,m.actuator_ctrlrange[:,0]);fixed.ctrl[:]=u;fixed.qacc_warmstart[:]=0
     mujoco.mj_forward(m,fixed)
     obs=Observer(m,geom);obs(m,fixed);r=obs.rows[-1];acc=m.opt.gravity+r['ground_force_world']/mass
     physical=np.array([acc[2]/g,.3*L*fixed.qacc[3]/g,.3*L*fixed.qacc[4]/g,.1*acc[0]/g,.1*acc[1]/g]);res=np.r_[physical,np.sqrt(.01/len(gs))*x]
     if record:history.append(dict(forward=calls,loss=float(res@res),physical_residual=physical.copy(),com_acceleration=acc.copy(),root_angular_qacc=fixed.qacc[3:6].copy(),world_reaction=r['ground_force_world'].copy()))
     return res,r,acc,fixed.qfrc_actuator.copy()
    x0=np.full(len(gs),.05)
    r0,row0,acc0,f0=evaluate(x0);repeat,_,_,_=evaluate(x0,False);np.testing.assert_array_equal(r0,repeat);qsave=fixed.qpos.copy();vsave=fixed.qvel.copy()
    # Deterministic force-map check: at fixed q/v native muscle actuator generalized force is affine in activation.
    z=np.zeros(len(gs));o=np.ones(len(gs));h=np.full(len(gs),.5)
    rz,rowz,accz,fz=evaluate(z,False);_,_,_,fo=evaluate(o,False);_,_,_,fh=evaluate(h,False)
    dump(folder/'FIXED_ZERO_RESPONSE.json',dict(loss=float(rz@rz),com_acceleration=accz,world_reaction=rowz['ground_force_world']))
    # Native floor makes midpoint affine only above floor; use midpoint of effective floor and one.
    eps=m.actuator_ctrlrange[:,0].max();h[:]=(1+eps)/2;_,_,_,fh=evaluate(h,False)
    affine_error=float(abs(fh-(fz+fo)/2).max())
    # Unmapped fixed native-floor effects are constant and cancel from the affine test.
    assert affine_error<1e-7*max(1.,float(abs(fo).max()))
    assert np.array_equal(qsave,fixed.qpos) and np.array_equal(vsave,fixed.qvel);assert not fixed.xfrc_applied.any() and not fixed.qfrc_applied.any()
    assert np.array_equal(expand(z),np.zeros(90));assert np.all((expand(o)>=0)&(expand(o)<=1));assert len(r0)==5+len(gs)
    if mode.startswith('mapped'):assert not expand(o)[unmapped].any()
    dump(folder/'DETERMINISTIC_CHECKS.json',dict(passed=True,force_map_midpoint_maxerror=affine_error,fixed_qpos_qvel_unchanged=True,external_forces_zero=True,control_bounds=True,objective_weights_exact=True,ties_exact=True,unresolved_zero=mode.startswith('mapped'),note='forward-only checks before any dynamic rollout; fixed-state activation assignment is steady-state evaluation only'))
    sol=least_squares(lambda x:evaluate(x)[0],x0,bounds=(0.,1.),method='trf',max_nfev=8,diff_step=1e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    ctrl=expand(sol.x);rfina,rowf,accf,ff=evaluate(sol.x)
    np.save(folder/'CONSTANT_COMMAND.npy',ctrl);dump(folder/'OPTIMIZATION.json',dict(success=bool(sol.success),status=int(sol.status),message=sol.message,nfev=int(sol.nfev),forward_calls=calls,initial_loss=float(r0@r0),final_loss=float(rfina@rfina),command=ctrl,variables=sol.x,initial_com_acceleration=acc0,final_com_acceleration=accf,fixed_world_reaction=rowf['ground_force_world'],fixed_world_foot_reaction=rowf['ground_force_foot_world'],fixed_world_nonfoot_reaction=rowf['ground_force_nonfoot_world'],fixed_qacc=fixed.qacc,fixed_generalized_actuator_force=ff,optimizer_history=history))
    data=mujoco.MjData(m);data.qpos[:]=q;assert not data.act.any() and not data.qvel.any();data.ctrl[:]=ctrl;observer=Observer(m,geom);mujoco.mj_forward(m,data);observer(m,data);counts['rollouts']+=1
    for step in range(20000):
     data.ctrl[:]=ctrl;assert not data.xfrc_applied.any() and not data.qfrc_applied.any();mujoco.mj_step(m,data);mujoco.mj_forward(m,data);observer(m,data)
     if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
      observer.save(folder/'BODY.npz');dump(folder/'CONTACTS.json',observer.contacts);raise RuntimeError('STOP nonfinite/warning, partial state preserved')
    observer.save(folder/'BODY.npz');dump(folder/'CONTACTS.json',observer.contacts)
    b={k:np.array([r[k] for r in observer.rows]) for k in observer.rows[0]};met=support_metrics(b);met['substantial_engineering_support_improvement']=bool(met['last50_nonfoot_fraction']<.5 and met['last50_mean_com_height']>base['last50_mean_com_height']+.1)
    met['last50_foot_world_mean']=b['ground_force_foot_world'][-4000:].mean(0);met['last50_nonfoot_world_mean']=b['ground_force_nonfoot_world'][-4000:].mean(0);met['last50_foot_contact_fraction']=b['foot_contact'][-4000:].mean(0);met['activation_min']=float(b['activation'].min());met['activation_max']=float(b['activation'].max());met['command_constant_exact']=bool(np.array_equal(b['command'],np.repeat(ctrl[None,:],20001,axis=0)));met['external_force_zero']=bool(not b['external_force'].any());summaries[mode]=met;dump(folder/'METRICS.json',met);dump(O/'SUMMARY.json',summaries);print(mode,json.dumps(met,default=lambda v:v.tolist()),flush=True)
 except Exception as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0 if error is None else 1,counts=counts,error=error,runtime_s=time.time()-begin))
 dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(p))==v for p,v in hashes.items())))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('OUTPUT',O,flush=True)
 if error:raise RuntimeError(error)
if __name__=='__main__':main()
