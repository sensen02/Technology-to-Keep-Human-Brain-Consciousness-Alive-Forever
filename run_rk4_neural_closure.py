"""Explicit research RK4 neural closure; no CPG/PD or pose correction."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,zipfile,sys,signal
import numpy as np
import mujoco
import run_neural_muscle_closure as old
import run_experiment_platform as platform
from engine.embodied.rk4_closure_contract import RK4ClosureClock
from run_s2_fixed_stimulus import dense_moment
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from engine.embodied.research_report import atomic_report as dump, guard_snapshot
def main(guard):
 out=Path('/home/sensen/Desktop/engine修复验收')/('S1-rk4-closure-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='EXPLICIT_ZERO_FLOOR_RK4_25US_SHORT_NEURAL_CLOSURE_NOT_STANDING',cases={},body_steps=0,neural_steps=0,decoder_calls=0);inputs={};saved={}
 try:
  for key in ('control','act_dyn','act_gain','act_bias'):assert getattr(mujoco,'get_mjcb_'+key)() is None
  posepath=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
  inputs=platform.input_hashes();inputs[str(posepath)]=sha(posepath);inputs[str(ROOT/'S1_RK4_CLOSURE_PROTOCOL.zh-CN.md')]=sha(ROOT/'S1_RK4_CLOSURE_PROTOCOL.zh-CN.md');inputs[str(Path(__file__))]=sha(__file__)
  m=old.cp.load_model();assert sha(old.cp.CANONICAL_XML)==old.cp.CANONICAL_XML_SHA256
  assert (m.nq,m.nv,m.na,m.nu)==(49,48,90,90);assert np.all(m.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_MUSCLE));assert np.array_equal(m.actuator_actadr,np.arange(90)) and np.all(m.actuator_actnum==1);assert not m.actuator_actlimited.any() and not m.actuator_actearly.any();assert np.array_equal(m.actuator_dynprm[:,:3],np.tile([.0001,.0004,0],(90,1)));assert not m.jnt_actgravcomp.any() and not m.jnt_actfrclimited.any();assert m.nhistory==0 and m.nplugin==0 and m.opt.disableactuator==0;assert not(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP));assert not(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))
  native=m.actuator_ctrlrange.copy();m.actuator_ctrlrange[:,0]=0;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4;m.opt.timestep=.000025;c=RK4ClosureClock()
  with np.load(posepath) as z:pose=z['projected_seated_qpos'].copy()
  joints={}
  for leg in old.LEGS:
   ids=[j for j in range(m.njnt) if (m.joint(j).name or '').lower().startswith('joint_'+leg)];tib=[j for j in ids if 'Tibia' in m.joint(j).name];assert len(ids)==7 and len(tib)==1;joints[leg]=dict(q=[int(m.jnt_qposadr[j]) for j in ids],tib_q=int(m.jnt_qposadr[tib[0]]),tib_v=int(m.jnt_dofadr[tib[0]]))
  geom_leg={}
  for g in range(m.ngeom):
   name=(m.geom(g).name or '').lower();body=(m.body(int(m.geom_bodyid[g])).name or '').lower();legs=[leg for leg in old.LEGS if 'tars' in name+body and (name.startswith(leg) or body.startswith(leg))]
   if len(legs)==1:geom_leg[g]=legs[0]
  assert set(geom_leg.values())==set(old.LEGS)
  built=old.build_neural_tier(old.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[m.actuator(i).name for i in range(m.nu)];manifest=old.build_motor_manifest(json.loads(old.POOLS.read_text()),built['tier']['root_ids'],names);groups,_=old.channel_index_map(int(built['n']),built['tier']['root_ids']);indices=np.array(sorted({int(i) for row in groups.values() for ids in row.values() for i in ids}));outside=np.ones(int(built['n']),bool);outside[indices]=False
  sources=platform.source_hashes();inputs.update(sources);lib=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));inputs[str(lib)]=sha(lib)
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in sources:z.write(p,str(Path(p).relative_to(ROOT)))
   z.write(ROOT/'S1_RK4_CLOSURE_PROTOCOL.zh-CN.md','S1_RK4_CLOSURE_PROTOCOL.zh-CN.md')
  r.update(native_ctrlrange=native.tolist(),derived_ctrlrange=m.actuator_ctrlrange.tolist(),motor_manifest=manifest,tier_root_ids=built['tier']['root_ids'],sensory_groups=groups,n_neurons=int(built['n']),environment=platform.environment_snapshot(),initial_pose=pose.tolist(),clock=dict(body=.000025,neural=.0005,command=.005),preflight_pass=True)
  fields=('qpos','qvel','act','ctrl','actuator_force','qfrc_applied','xfrc_applied')
  for arm in ('normal','repeat','replay','sensory_cut','motor_cut','output_silence'):
   d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
   enc=old.ChannelSensoryEncoder(groups,int(built['n']),old.ChannelConfig());dec=old.MotorOutput(manifest,old.MotorOutputConfig());net=old.ConductanceNetwork(int(built['n']),old.ConductanceParams(),dt_ms=.5);net.set_connectivity(built['edges']['We']*.2,built['edges']['Wi']*.2)
   pending=np.zeros(90);a={k:[] for k in fields+('time_s','warnings','controls','decoded_next','rates_hz','pool_counts','source','valid_from_s','interval_end_s','sensory_current_nA','outside_input_max','spikes_raw_packed','spikes_emitted_packed','boundary_moment','boundary_qfrc_actuator','boundary_qfrc_passive','boundary_qfrc_bias','boundary_qfrc_constraint')}
   for k in range(10):
    t=k*.005;c.validate_source(k,k-1,t);current,_=enc.encode(old.observe_quantities(m,d,pose,joints,geom_leg))
    for leg in old.LEGS:
     for ch in platform.OMIT:current[np.asarray(groups[leg][ch],int)]=0
    if arm=='sensory_cut':current[:]=0
    assert not current[outside].any();a['sensory_current_nA'].append(current[indices].copy());a['outside_input_max'].append(float(abs(current[outside]).max()) if outside.any() else 0.)
    sp=np.array([net.step(current_nA=current) for _ in range(10)],dtype=bool);r['neural_steps']+=10;em=sp.copy()
    if arm=='output_silence':em[:]=0
    applied=pending.copy()
    if arm=='motor_cut' and k>=3:applied[:]=0
    if arm=='replay':applied=saved['normal']['controls'][k].copy()
    d.ctrl[:]=applied;a['controls'].append(applied);a['source'].append(k-1);a['valid_from_s'].append(t);a['spikes_raw_packed'].append(np.packbits(sp,axis=1));a['spikes_emitted_packed'].append(np.packbits(em,axis=1))
    for i in range(200):
     assert c.command_row(k*200+i)==k;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mujoco.mj_step(m,d);r['body_steps']+=1;mujoco.mj_forward(m,d)
     if d.warning.number.any():raise RuntimeError('native warning')
     for f in fields:a[f].append(getattr(d,f).copy())
     a['time_s'].append(float(d.time));a['warnings'].append(d.warning.number.copy())
     if d.act.min()<-1e-12 or d.act.max()>1+1e-12:raise RuntimeError('activation range')
    assert abs(d.time-(k+1)*.005)<1e-10 and abs(net.t_ms/1000-(k+1)*.005)<1e-10
    pending,trace=dec.consume(em,spike_dt_s=.0005,start_s=t,end_s=t+.005,connected=not(arm=='motor_cut' and k+1>=3));r['decoder_calls']+=1
    a['decoded_next'].append(pending.copy());a['rates_hz'].append(trace['rates_hz']);a['pool_counts'].append(trace['pool_spike_counts']);a['interval_end_s'].append(float(d.time));mom=dense_moment(m,d);assert np.allclose(mom.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10);a['boundary_moment'].append(mom)
    for f in ('qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint'):a['boundary_'+f].append(getattr(d,f).copy())
   a={k:np.asarray(v) for k,v in a.items()};a['sensory_local_indices']=indices;np.savez_compressed(out/(arm+'.npz'),**a);saved[arm]=a
   entry=dict(finite=all(np.isfinite(x).all() for x in a.values()),activation_max=float(a['act'].max()),neural_spikes=int(np.unpackbits(a['spikes_raw_packed'],axis=2,count=int(built['n'])).sum()),nonzero_commands=int(np.count_nonzero(a['controls'])))
   r['cases'][arm]=entry;dump(out/'report.json',r);assert entry['finite']
   if arm=='repeat':entry['exact']=all(np.array_equal(x,saved['normal'][k]) for k,x in a.items());assert entry['exact']
   if arm=='replay':entry['body_exact']=all(np.array_equal(a[k],saved['normal'][k]) for k in fields+('time_s',));assert entry['body_exact']
   if arm in ('sensory_cut','output_silence'):assert not a['controls'].any()
   if arm=='motor_cut':assert not a['controls'][3:].any()
   if sum(f.stat().st_size for f in out.iterdir())>500_000_000:raise RuntimeError('disk budget')
   dump(out/'report.json',r)
  # New normal commands require their own activation-only reference.
  from scipy.integrate import solve_ivp
  def rhs(a,u):
   scale=.5+1.5*np.clip(a,0,1);delta=np.clip(u,0,1)-a;return delta/np.where(delta>0,.0001*scale,.0004/scale)
  v=np.zeros(90);reference=[]
  for k,u in enumerate(saved['normal']['controls']):
   sol=solve_ivp(lambda t,y:rhs(y,u),(k*.005,(k+1)*.005),v,method='DOP853',rtol=1e-12,atol=1e-14,max_step=1e-6,dense_output=True);assert sol.success;reference.append(sol.sol((k*200+np.arange(1,201))*.000025).T);v=sol.y[:,-1]
  reference=np.concatenate(reference);err=saved['normal']['act']-reference;np.savez_compressed(out/'normal-reference.npz',activation=reference);r['normal_activation_reference']=dict(max_error=float(abs(err).max()),rms_error=float(np.sqrt(np.mean(err**2))));assert abs(err).max()<=.01 and np.sqrt(np.mean(err**2))<=.001
  r['matched_effects']={arm:dict(control_max_difference=float(abs(saved[arm]['controls']-saved['normal']['controls']).max()),lf_tibia_angle_max_difference_rad=float(abs(saved[arm]['qpos'][:,13]-saved['normal']['qpos'][:,13]).max())) for arm in ('sensory_cut','motor_cut','output_silence')}
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
  for arm in ('normal','sensory_cut','motor_cut','output_silence'):
   a=saved[arm];axes[0].plot(a['time_s']*1000,a['qpos'][:,13],label=arm);axes[1].step(np.arange(10)*5,a['controls'][:,14],where='post',label=arm)
  axes[0].set_ylabel('LF tibia angle(rad)');axes[1].set_ylabel('applied control14');axes[1].set_xlabel('time(ms)');axes[0].legend();fig.suptitle('Explicit RK4 short neural closure — NOT standing/biological validation');fig.tight_layout();fig.savefig(out/'closure.png');plt.close(fig);r['status']='PASS_SHORT_CLOSURE_ENGINEERING_NOT_STANDING'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['input_hashes_before']=inputs;r['input_hashes_after']={p:sha(p) for p in inputs};r['inputs_unchanged']=r['input_hashes_before']==r['input_hashes_after'];r['memory_guard']=guard_snapshot(guard,r['status'])
  if not r['inputs_unchanged']:r['status']='FAIL_HASH_DRIFT'
  dump(out/'report.json',r)
 print(out);print(r['status']);print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':
 def timeout(*args):raise TimeoutError('120s protocol deadline')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
 with old.mg.guarded_run('neural_muscle_closure',**old.mg.BUDGETS['neural_muscle_closure']) as guard:code=main(guard)
 signal.alarm(0);raise SystemExit(code)
