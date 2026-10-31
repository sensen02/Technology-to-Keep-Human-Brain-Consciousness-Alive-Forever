"""Five saved states, actual compiled muscle gain/bias and moment directions. No step."""
from pathlib import Path
import os,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_one_second_motor_pair import sha,dump
import engine.canonical_physics as cp
S=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s2-muscle-components/20261027T175000Z')
ZERO=Path('/home/sensen/Desktop/engine修复验收/s2-runtime-root-identity/20261027T165000Z/ZERO_EVENT_POOLS_11.json')
def main():
 O.mkdir(parents=True,exist_ok=True);source_manifest=json.loads((S/'MANIFEST_SHA256.json').read_text());assert all(sha(S/k)==v for k,v in source_manifest.items())
 z=np.load(S/'normal/BODY.npz');records=json.loads((S/'normal/RECORDS.json').read_text());cross=int(np.flatnonzero(z['nonfoot_floor_count']>0)[0]);indices=[0]+[int(np.argmin(abs(z['time']-v))) for v in (.01,.015,.025)]+[cross-1]
 dump(O/'PREREGISTRATION.json',dict(source_manifest_sha256=sha(S/'MANIFEST_SHA256.json'),model_sha256=sha(cp.CANONICAL_XML),code_sha256=sha(Path(__file__)),zero_evidence_sha256=sha(ZERO),indices=indices,times=[float(z['time'][i]) for i in indices],restore='same fields as audit_early_saved_forces.py:30; qpos/qvel/act/actual ctrl/time/xfrc, fresh data default qfrc_applied0',missing=['qacc_warmstart','historical solver workspace'],tolerance=dict(abs_force=1e-10,relative_force=1e-12,abs_generalized=1e-10,relative_generalized=1e-12),rationale='same double precision native force functions, no approximation; rounding-level residual expected; tolerance is reconstruction not biological acceptance'))
 m=cp.load_model();m.opt.timestep=.0000125;names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];legs=['lf','lm','lh','rf','rm','rh'];zero={r['actuator'] for r in json.loads(ZERO.read_text())};actleg=[name[:2].lower() for name in names];dofnames=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.dof_jntid[j]))+':dof'+str(j) for j in range(m.nv)]
 compiled=dict(version=mujoco.__version__,actuators=[dict(name=name,gaintype=int(m.actuator_gaintype[i]),biastype=int(m.actuator_biastype[i]),gainprm=m.actuator_gainprm[i],biasprm=m.actuator_biasprm[i],lengthrange=m.actuator_lengthrange[i],acc0=m.actuator_acc0[i],actadr=int(m.actuator_actadr[i]),actearly=bool(m.actuator_actearly[i]),ctrllimited=bool(m.actuator_ctrllimited[i]),ctrlrange=m.actuator_ctrlrange[i],forcelimited=bool(m.actuator_forcelimited[i]),forcerange=m.actuator_forcerange[i]) for i,name in enumerate(names)])
 dump(O/'COMPILED_PARAMETERS.json',compiled);ledger=[];states=[];opponents=[];moment_history=[];fig,axs=plt.subplots(6,2,figsize=(17,19),sharex=True)
 for idx in indices:
  d=mujoco.MjData(m);d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.act[:]=z['activation'][idx];d.ctrl[:]=z['command'][idx];d.time=z['time'][idx];d.xfrc_applied[:]=z['external_force'][idx];mujoco.mj_forward(m,d)
  moment=np.zeros((m.nu,m.nv))
  for i in range(m.nu):
   start=int(d.moment_rowadr[i]);n=int(d.moment_rownnz[i]);moment[i,d.moment_colind[start:start+n]]=d.actuator_moment[start:start+n]
  active=np.zeros(m.nu);bias=np.zeros(m.nu);supported=np.ones(m.nu,bool)
  for i in range(m.nu):
   if m.actuator_gaintype[i]!=mujoco.mjtGain.mjGAIN_MUSCLE or m.actuator_biastype[i]!=mujoco.mjtBias.mjBIAS_MUSCLE or m.actuator_actadr[i]<0:supported[i]=False;continue
   assert not m.actuator_actearly[i]
   gain=mujoco.mju_muscleGain(d.actuator_length[i],d.actuator_velocity[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9]);bias[i]=mujoco.mju_muscleBias(d.actuator_length[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_biasprm[i,:9]);active[i]=gain*d.act[int(m.actuator_actadr[i])]
  assert supported.all();raw=active+bias;total=raw.copy()
  for i in range(m.nu):
   if m.actuator_forcelimited[i]:total[i]=np.clip(total[i],*m.actuator_forcerange[i])
  tau_active=moment*active[:,None];tau_bias=moment*bias[:,None];tau_total=moment*total[:,None];res=total-d.actuator_force;qres=tau_total.sum(0)-d.qfrc_actuator;scale=max(float(abs(d.actuator_force).max()),np.finfo(float).tiny);qscale=max(float(abs(d.qfrc_actuator).max()),np.finfo(float).tiny)
  states.append(dict(index=idx,time=float(d.time),thorax_height=float(d.xpos[m.body('Thorax').id,2]),saved_force_max_error=float(abs(d.actuator_force-z['muscle_force'][idx]).max()),force_abs_residual=float(abs(res).max()),force_rel_residual=float(abs(res).max()/scale),generalized_abs_residual=float(abs(qres).max()),generalized_rel_residual=float(abs(qres).max()/qscale),qfrc_actuator=d.qfrc_actuator,reconstructed_qfrc=tau_total.sum(0),force_clamp_active=bool(np.any(total!=raw)),root_translation_generalized=tau_total.sum(0)[:3]))
  assert np.max(abs(res))<1e-10+1e-12*scale;assert np.max(abs(qres))<1e-10+1e-12*qscale
  window=min(max(idx-1,0)//400,49);saturated=[bool(records['records'][window-1]['decoder']['saturated'][i]) if window>0 else False for i in range(m.nu)]
  for i,name in enumerate(names):
   touched=np.flatnonzero(abs(moment[i])>1e-12);ledger.append(dict(index=idx,time=float(d.time),actuator=name,leg=actleg[i],previous_zero_event=name in zero,command=float(d.ctrl[i]),internal_control=float(np.clip(d.ctrl[i],*m.actuator_ctrlrange[i])) if m.actuator_ctrllimited[i] else float(d.ctrl[i]),activation=float(d.act[int(m.actuator_actadr[i])]),model_active_gain_force=float(active[i]),model_bias_force=float(bias[i]),total_force=float(d.actuator_force[i]),force_reconstruction_residual=float(res[i]),gain=float(active[i]/d.act[int(m.actuator_actadr[i])]) if d.act[int(m.actuator_actadr[i])]!=0 else None,recorded_decoder_saturation_at_nominal_window=saturated[i],joint_contributions=[dict(dof=int(j),joint=dofnames[j],moment=float(moment[i,j]),active_generalized=float(tau_active[i,j]),bias_generalized=float(tau_bias[i,j]),total_generalized=float(tau_total[i,j])) for j in touched]))
  pairs=[]
  for j in range(6,m.nv):
   pos=np.flatnonzero(tau_active[:,j]>1e-10);neg=np.flatnonzero(tau_active[:,j]<-1e-10)
   if len(pos) and len(neg):pairs.append(dict(dof=j,joint=dofnames[j],positive=[names[i] for i in pos],negative=[names[i] for i in neg],positive_active_sum=float(tau_active[pos,j].sum()),negative_active_sum=float(tau_active[neg,j].sum()),note='opposed active generalized directions; numerical threshold only, not functional antagonism label'))
  opponents.append(dict(index=idx,time=float(d.time),joints=pairs));moment_history.append(moment)
 for li,leg in enumerate(legs):
  group=[i for i,l in enumerate(actleg) if l==leg];ts=[s['time']*1000 for s in states]
  axs[li,0].plot(ts,[sum(r['model_active_gain_force'] for r in ledger if r['index']==idx and r['leg']==leg) for idx in indices],'o-',label='signed active gain force sum');axs[li,0].plot(ts,[sum(r['model_bias_force'] for r in ledger if r['index']==idx and r['leg']==leg) for idx in indices],'s--',label='signed bias force sum')
  # Only scalar hinge coordinates for this leg; sum positive/negative separately to avoid concealing opposed torques.
  for mode,style in [('active_generalized','-'),('bias_generalized','--')]:
   for sign in (1,-1):axs[li,1].plot(ts,[sum(c[mode] for r in ledger if r['index']==idx and r['leg']==leg for c in r['joint_contributions'] if c['dof']>=6 and c[mode]*sign>0) for idx in indices],style,marker='.',label=mode+(' positive' if sign==1 else ' negative'))
  axs[li,0].set_ylabel(leg+' / model force');axs[li,1].set_ylabel(leg+' / model torque');
  for ax in axs[li]:ax.legend(fontsize=7);ax.grid(alpha=.2);ax.set_xlabel('saved physical ms; last point pre-abdomen-contact')
 axs[0,0].set_title('Model active/bias force aggregates (not support)');axs[0,1].set_title('Signed hinge contributions, + and - kept separate');fig.suptitle('Five saved states /90muscles / model decomposition, not physiological calibration');fig.tight_layout();fig.savefig(O/'MUSCLE_COMPONENTS.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_MUSCLE_COMPONENTS.png').write_bytes((O/'MUSCLE_COMPONENTS.png').read_bytes())
 changes=[]
 for i,name in enumerate(names):
  joints=[]
  for j in range(6,m.nv):
   v=np.array([h[i,j] for h in moment_history]);nz=v[abs(v)>1e-12]
   if len(nz):joints.append(dict(dof=j,joint=dofnames[j],moments=v,sign_changes_across_sampled_states=bool(np.any(nz>0) and np.any(nz<0))))
  changes.append(dict(actuator=name,joints=joints))
 dump(O/'LEDGER_90_BY_5.json',ledger);dump(O/'STATE_RESIDUALS.json',states);dump(O/'OPPOSED_ACTIVE_DIRECTIONS.json',opponents);dump(O/'MOMENT_DIRECTIONS.json',changes)
 dump(O/'EXECUTION.json',dict(command='MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,forward=5,body_step=0,neural_step=0,preliminary_api_probe_model_load=1,all_muscles_supported=True,source_manifest_unchanged=all(sha(S/k)==v for k,v in source_manifest.items())))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps([{k:v for k,v in s.items() if k not in ('qfrc_actuator','reconstructed_qfrc')} for s in states],default=lambda x:x.tolist()))
if __name__=='__main__':main()
