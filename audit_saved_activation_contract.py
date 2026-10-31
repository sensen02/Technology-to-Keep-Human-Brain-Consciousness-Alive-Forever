"""No mj_step, mj_forward or neural calls: independent arithmetic of saved activation."""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np,mujoco
import engine.canonical_physics as cp
S=Path('/home/sensen/Desktop/engine修复验收/s2-one-second-motor-pair/20261027T153000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s2-activation-contract/20261027T154000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def derivative(a,u,prm):
 ac=np.clip(a,0,1);uc=np.clip(u,0,1);delta=uc-a
 tau=np.where(delta>0,prm[:,0]*(.5+1.5*ac),prm[:,1]/(.5+1.5*ac))
 return delta/np.maximum(1e-15,tau)
def main():
 old=json.loads((S/'MANIFEST_SHA256.json').read_text());assert all(sha(S/k)==v for k,v in old.items())
 m=cp.load_model();assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_EULER
 assert np.all(m.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE);assert np.all(m.actuator_dynprm[:,2]==0)
 assert np.array_equal(m.actuator_actadr,np.arange(m.nu));assert not np.any(m.actuator_actlimited);assert not np.any(m.actuator_delay)
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(m.nu)]
 fields=['dyntype','dynprm','gaintype','gainprm','biastype','biasprm','ctrlrange','ctrllimited','actrange','actlimited','actearly','actadr','actnum','forcerange','forcelimited','delay']
 dump(O/'COMPILED_PARAMETERS.json',dict(model= str(cp.CANONICAL_XML),model_sha256=sha(cp.CANONICAL_XML),mujoco_version=mujoco.__version__,integrator='Euler',dt=m.opt.timestep,disableflags=int(m.opt.disableflags),enableflags=int(m.opt.enableflags),actuators=[dict(name=names[j],**{k:getattr(m,'actuator_'+k)[j] for k in fields}) for j in range(m.nu)]))
 rows=[];report={};residuals={}
 for arm in ('normal','motor_cut'):
  z=np.load(S/arm/'BODY.npz');interval=np.load(S/arm/'INTERVALS.npz');a=z['activation'];u=z['command'][1:];effective=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);dt=m.opt.timestep
  expected=a[:-1]+dt*derivative(a[:-1],effective,m.actuator_dynprm);error=a[1:]-expected;residuals[arm]=error
  assert np.max(abs(error))<1e-12
  # Post-step record command belongs to transition ending on that row, not next transition.
  np.testing.assert_array_equal(u,np.repeat(interval['commands'],50,axis=0))
  np.testing.assert_allclose(z['time'][1:]-z['time'][:-1],dt,rtol=0,atol=1e-15)
  exceed=np.argwhere(a>1);maxrow,maxmuscle=np.unravel_index(np.argmax(a),a.shape)
  selections=[('maximum',maxrow,maxmuscle),('cut_window',150,int(np.argmax(a[150])))]
  if len(exceed):selections.append(('first_gt1',int(exceed[0,0]),int(exceed[0,1])))
  for label,centre,j in selections:
   for end in range(max(1,centre-3),min(len(a),centre+4)):
    start=end-1;k=start//50
    rows.append(dict(arm=arm,window=label,actuator=names[j],start_row=start,end_row=end,source_neural_interval=int(interval['applied_source_interval'][k]),command_interval=k,start_s=z['time'][start],end_s=z['time'][end],previous_decoded_command=None if k==0 else float(interval['decoded_next_controls'][k-1,j]),applied_interval_command=interval['commands'][k,j],written_ctrl=u[start,j],effective_clamped_ctrl=effective[start,j],pre_activation=a[start,j],independent_pre_act_dot=derivative(a[start],effective[start],m.actuator_dynprm)[j],predicted_post_activation=expected[start,j],saved_post_activation=a[end,j],residual=error[start,j],saved_post_forward_actuator_force=z['muscle_force'][end,j],force_interpretation='post-step mj_forward total active+bias; not pre-step integration force'))
  report[arm]=dict(transitions=error.size,max_abs_residual=float(np.max(abs(error))),first_gt1=None if not len(exceed) else dict(row=int(exceed[0,0]),time=float(z['time'][exceed[0,0]]),actuator=names[int(exceed[0,1])]),maximum=dict(row=int(maxrow),time=float(z['time'][maxrow]),actuator=names[maxmuscle],value=float(a[maxrow,maxmuscle])),cut_post_commands_zero=bool(not np.any(u[150:])),last_activation_range=[float(a[-1].min()),float(a[-1].max())],effective_ctrl_range=[float(effective.min()),float(effective.max())])
 with (O/'STEP_LEDGER.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(O/'ALL_RESIDUALS.npz',**residuals)
 # deterministic scalar arithmetic regression; NOT a physical simulation
 np.testing.assert_allclose(derivative(np.zeros(90),np.ones(90),m.actuator_dynprm)*dt,2.,atol=1e-15)
 assert not np.any(m.actuator_actearly)
 dump(O/'RESULT.json',dict(classification='NUMERICALLY_CONSISTENT_WITH_DECLARED_DYNAMICS',arms=report,physical_steps=0,neural_steps=0,forward_calls=0,model_compile_loads=1,independent_scalar_euler_overshoot_test=True,
 missing=['actual historical in-step act_dot and locally clamped ctrl not recorded; reconstructed from source and fully matching updates','pre-step integration actuator_force and actuator_length/velocity/gain/bias not saved; force not independently recomputed'],
 interpretation='Euler step100us with tau_act100us and factor0.5 at a=0 gives a_next near2 for u near1; no actlimit. Not a physiological validation. Internal muscle actclamp sets timescale but raw act is subtracted. Native positive control floor verified via local clamp path and all saved updates.',
 physiological_validity='NOT_ASSESSED',prior_trajectory_causal_claim='Mechanical divergence under this numerical actuator, not validated biological muscle drive'))
 dump(O/'EXECUTION.json',dict(command='MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,input_manifest_sha256=sha(S/'MANIFEST_SHA256.json'),source_archive_unchanged=True))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());dump(O/'MANIFEST_SHA256.json',{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='MANIFEST_SHA256.json'});print(json.dumps(report));print('independent 1800000 state transitions pass, no physical step or forward')
if __name__=='__main__':main()
