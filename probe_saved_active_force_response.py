"""Fixed-state mechanical probe: +/-1% of each saturated muscle active force. No steps."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_saved_muscle_components import S,O as LEDGER,cp,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-fixed-active-response/20261027T180000Z')

def main():
 O.mkdir(parents=True,exist_ok=False)
 paths=[S/'normal/BODY.npz',S/'normal/CONTACTS.json',S/'MANIFEST_SHA256.json',LEDGER/'LEDGER_90_BY_5.json',LEDGER/'STATE_RESIDUALS.json',Path(cp.CANONICAL_XML),Path(__file__)]
 hashes={str(p):sha(p) for p in paths}
 manifest=json.loads((S/'MANIFEST_SHA256.json').read_text());assert all(sha(S/k)==h for k,h in manifest.items())
 ledger=json.loads(paths[3].read_text());rows=[r for r in ledger if abs(r['time']-.025)<1e-10];assert len(rows)==90
 chosen=[r for r in rows if r['command']>=1-1e-12];assert len(chosen)==35
 z=np.load(paths[0]);idx=int(np.argmin(abs(z['time']-.025)));assert idx==2000
 dump(O/'PREREGISTRATION.json',dict(inputs_sha256=hashes,selected_actuators=[r['actuator'] for r in chosen],index=idx,time=float(z['time'][idx]),intervention='delta_F_i = sign * 0.01 * baseline_active_force_i; delta_qfrc = baseline_actuator_moment[i,:] * delta_F_i; applied to fresh qfrc_applied only. Baseline actuator_force/bias/ctrl/act unchanged.',independent_restores=True,relative_not_equal_absolute_force=True,zero_active_rule='UNTESTABLE_NO_MINIMUM_SUBSTITUTE',solver_unchanged=True,restore_missing=['historical qacc_warmstart and solver workspace; fresh defaults used for every solve'],thresholds=dict(reconstruction_abs=1e-9,reconstruction_rel=1e-10,normal_load=1e-9,symmetry_relative=0.05,symmetry_abs_accel=1e-8),symmetry_rule='abs(delta_plus+delta_minus) > 1e-8 + .05*max(abs(delta_plus),abs(delta_minus)); numerical diagnostic not physiological threshold',calls_planned=71))
 m=cp.load_model();m.opt.timestep=.0000125;assert m.nmocap==0 and m.neq==0
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)];mass=m.body_mass.sum();calls=0
 def restore(injection=None):
  nonlocal calls
  d=mujoco.MjData(m);d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.act[:]=z['activation'][idx];d.ctrl[:]=z['command'][idx];d.time=z['time'][idx];d.xfrc_applied[:]=z['external_force'][idx]
  if injection is not None:d.qfrc_applied[:]=injection
  mujoco.mj_forward(m,d);calls+=1
  return d
 def moment(d):
  a=np.zeros((m.nu,m.nv))
  for i in range(m.nu):
   st=int(d.moment_rowadr[i]);n=int(d.moment_rownnz[i]);a[i,d.moment_colind[st:st+n]]=d.actuator_moment[st:st+n]
  return a
 def measure(d):
  # Mass-weighted body COM accelerations J*qacc + Jdot*qvel, world frame.
  com=np.zeros(3)
  for b in range(1,m.nbody):
   if m.body_mass[b]==0:continue
   J=np.zeros((3,m.nv));Jdot=np.zeros_like(J);mujoco.mj_jac(m,d,J,None,d.xipos[b],b);mujoco.mj_jacDot(m,d,Jdot,None,d.xipos[b],b);com+=m.body_mass[b]*(J@d.qacc+Jdot@d.qvel)/mass
  contacts=[];forces={leg:np.zeros(3) for leg in ['lf','lm','lh','rf','rm','rh','NONFOOT']}
  for k in range(d.ncon):
   c=d.contact[k];a,b=int(c.geom1),int(c.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);world=c.frame.reshape(3,3).T@f[:3];world=world if pa else -world;g=b if pa else a
   gn=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').lower();bn=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower();leg=next((l for l in forces if l!='NONFOOT' and (gn.startswith(l) or bn.startswith(l)) and 'tars' in gn+bn),'NONFOOT');forces[leg]+=world
   contacts.append(dict(contact_index=k,geom1=a,geom2=b,leg=leg,distance=float(c.dist),position=c.pos,frame=c.frame,efc_address=int(c.efc_address),dimension=int(c.dim),normal_force=float(f[0]),world_force_on_fly=world,normal_load_above_threshold=bool(f[0]>1e-9)))
  M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM);cart=np.zeros(m.nv)
  for b in range(1,m.nbody):mujoco.mj_applyFT(m,d,d.xfrc_applied[b,:3],d.xfrc_applied[b,3:],d.xipos[b],b,cart)
  rhs=d.qfrc_actuator+d.qfrc_passive-d.qfrc_bias+d.qfrc_constraint+d.qfrc_applied+cart;res=M@d.qacc-rhs;scale=max(float(abs(rhs).max()),1e-30)
  return dict(com_acceleration_world=com,root_generalized_acceleration=d.qacc[:6],qacc=d.qacc,ground_world_force_by_leg=forces,total_ground_world_force=sum(forces.values()),geometric_ground_contacts=contacts,total_geometric_contact_count=d.ncon,efc_count=d.nefc,efc_state=d.efc_state.copy(),efc_type=d.efc_type.copy(),efc_id=d.efc_id.copy(),efc_force=d.efc_force.copy(),qfrc_constraint=d.qfrc_constraint.copy(),equation_abs_residual=float(abs(res).max()),equation_rel_residual=float(abs(res).max()/scale),qfrc_applied=d.qfrc_applied.copy(),solver_niter=d.solver_niter.copy(),warnings=[int(w.number) for w in d.warning],whole_body_newton_residual=mass*com-(sum(forces.values())+mass*m.opt.gravity+d.xfrc_applied[:,:3].sum(0)))
 base=restore();A=moment(base);baseline=measure(base)
 assert np.allclose(base.actuator_force,z['muscle_force'][idx],atol=1e-10,rtol=1e-12)
 assert np.allclose(A.T@base.actuator_force,base.qfrc_actuator,atol=1e-10,rtol=1e-12)
 saved=json.loads((LEDGER/'STATE_RESIDUALS.json').read_text());old=next(r for r in saved if r['index']==idx);assert np.allclose(base.qfrc_actuator,old['qfrc_actuator'],atol=1e-10,rtol=1e-12)
 # Independent historical contact data retained for comparison; dynamics solver workspace was not saved.
 historical=json.loads(paths[1].read_text());dump(O/'HISTORICAL_CONTACT_SOURCE.json',historical[idx] if isinstance(historical,list) and len(historical)>idx else dict(note='source retained via hash; format not one record per BODY state',container_type=type(historical).__name__))
 dump(O/'BASELINE.json',baseline)
 results=[]
 for r in chosen:
  i=names.index(r['actuator']);active=float(r['model_active_gain_force']);gain=mujoco.mju_muscleGain(base.actuator_length[i],base.actuator_velocity[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9]);assert np.isclose(active,gain*base.act[int(m.actuator_actadr[i])],atol=1e-10,rtol=1e-12)
  item=dict(actuator=names[i],actuator_index=i,leg=names[i][:2].lower(),baseline_active_force=active,baseline_bias_force=r['model_bias_force'],baseline_command=r['command'],baseline_activation=r['activation'],moment=A[i],status='TESTED' if active!=0 else 'UNTESTABLE_ZERO_ACTIVE',sides=[])
  if active==0:results.append(item);continue
  for sign in [1,-1]:
   df=sign*.01*active;inj=A[i]*df;d=restore(inj);q=measure(d)
   assert np.array_equal(d.qpos,base.qpos) and np.array_equal(d.qvel,base.qvel) and np.array_equal(d.act,base.act) and np.array_equal(d.ctrl,base.ctrl)
   assert np.array_equal(d.actuator_force,base.actuator_force);assert np.array_equal(moment(d),A)
   virtual=base.actuator_force.copy();virtual[i]+=df
   err=float(abs((A.T@virtual-base.qfrc_actuator)-inj).max());assert err<1e-9
   q.update(sign=sign,delta_active_force=df,injection_equivalence_abs_residual=err,delta_com_vertical=float(q['com_acceleration_world'][2]-baseline['com_acceleration_world'][2]),delta_root_vertical=float(q['root_generalized_acceleration'][2]-baseline['root_generalized_acceleration'][2]),delta_ground_world_force_by_leg={l:q['ground_world_force_by_leg'][l]-baseline['ground_world_force_by_leg'][l] for l in q['ground_world_force_by_leg']},geometric_contact_list_same=[(v['geom1'],v['geom2']) for v in q['geometric_ground_contacts']]==[(v['geom1'],v['geom2']) for v in baseline['geometric_ground_contacts']],normal_load_flags_same=[v['normal_load_above_threshold'] for v in q['geometric_ground_contacts']]==[v['normal_load_above_threshold'] for v in baseline['geometric_ground_contacts']],solver_constraint_state_same=bool(np.array_equal(q['efc_state'],baseline['efc_state'])))
   item['sides'].append(q)
  p,n=[s['delta_com_vertical'] for s in item['sides']];item.update(com_vertical_symmetry_abs=abs(p+n),com_vertical_symmetry_relative=abs(p+n)/max(abs(p),abs(n),1e-30),com_vertical_nonlinearity_flag=bool(abs(p+n)>1e-8+.05*max(abs(p),abs(n))),solver_state_switch=any(not s['solver_constraint_state_same'] for s in item['sides']))
  results.append(item)
 dump(O/'ALL_RESPONSES_35.json',results)
 summary=dict(tested=sum(r['status']=='TESTED' for r in results),untestable=[r['actuator'] for r in results if r['status']!='TESTED'],plus_active_delta_com_positive=[r['actuator'] for r in results if r['sides'] and r['sides'][0]['delta_com_vertical']>1e-8],plus_active_delta_com_negative=[r['actuator'] for r in results if r['sides'] and r['sides'][0]['delta_com_vertical']<-1e-8],solver_switches=[r['actuator'] for r in results if r.get('solver_state_switch')],nonlinearity_flags=[r['actuator'] for r in results if r.get('com_vertical_nonlinearity_flag')],max_equation_abs=max([baseline['equation_abs_residual']]+[s['equation_abs_residual'] for r in results for s in r['sides']]),max_equation_rel=max([baseline['equation_rel_residual']]+[s['equation_rel_residual'] for r in results for s in r['sides']]))
 dump(O/'SUMMARY.json',summary)
 fig,axs=plt.subplots(6,1,figsize=(17,19));
 for ax,leg in zip(axs,['lf','lm','lh','rf','rm','rh']):
  rr=[r for r in results if r['leg']==leg];x=np.arange(len(rr));ax.bar(x-.17,[r['sides'][0]['delta_com_vertical'] if r['sides'] else 0 for r in rr],.34,label='+1% active force');ax.bar(x+.17,[r['sides'][1]['delta_com_vertical'] if r['sides'] else 0 for r in rr],.34,label='-1% active force');ax.set_xticks(x,[r['actuator'] for r in rr],rotation=15,ha='right',fontsize=8);ax.set_ylabel(leg+' COM az delta / model units');ax.axhline(0,color='black',lw=.6);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('25 ms fixed state / bidirectional relative force probes / NOT neural recruitment or support certification');fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(O/'BIDIRECTIONAL_RESPONSE.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_FIXED_ACTIVE_RESPONSE.png').write_bytes((O/'BIDIRECTIONAL_RESPONSE.png').read_bytes())
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_loads=1,mj_forward=calls,mj_step=0,neural_step=0,source_hashes_unchanged=True,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))
if __name__=='__main__':main()
