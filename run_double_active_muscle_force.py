"""S2/P3 diagnostic: muscle gain capacity x2, historical bias unchanged."""
from pathlib import Path
import os,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_double_muscle_force as prior
from run_one_second_motor_pair import INITIAL,sha,dump
c=prior.c;BASE=prior.BASE
O=Path('/home/sensen/Desktop/engine修复验收/s2-active-force-double/20261027T193000Z')
TOTAL=prior.O
restore=prior.restore;components=prior.components

def scale_active(m,factor=1.):
 before=m.actuator_gainprm.copy();bias=m.actuator_biasprm.copy();changes=[]
 assert m.nu==90
 for i in range(m.nu):
  assert m.actuator_gaintype[i]==mujoco.mjtGain.mjGAIN_MUSCLE
  assert m.actuator_biastype[i]==mujoco.mjtBias.mjBIAS_MUSCLE
  f=float(before[i,2]);column=2 if f>=0 else 3
  m.actuator_gainprm[i,column]*=factor
  changes.append(dict(index=i,actuator=m.actuator(i).name,column=column,mode='explicit_force' if f>=0 else 'auto_scale_over_acc0',old_parameter=float(before[i,column]),new_parameter=float(m.actuator_gainprm[i,column]),old_effective_force=f if f>=0 else float(before[i,3]/m.actuator_acc0[i]),new_effective_force=float(m.actuator_gainprm[i,2]) if f>=0 else float(m.actuator_gainprm[i,3]/m.actuator_acc0[i])))
 np.testing.assert_array_equal(m.actuator_biasprm,bias)
 allowed=np.zeros_like(before,dtype=bool)
 for r in changes:allowed[r['index'],r['column']]=True
 np.testing.assert_array_equal(m.actuator_gainprm[~allowed],before[~allowed])
 return changes

def regression():
 z=np.load(BASE/'normal/BODY.npz');models=[c.cp.load_model() for _ in range(3)]
 scale_active(models[1],1);changes=scale_active(models[2],2);rows=[]
 for name in ['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_lengthrange','actuator_acc0']:
  np.testing.assert_array_equal(getattr(models[0],name),getattr(models[1],name))
 for name in ['actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_lengthrange','actuator_acc0','actuator_forcelimited','actuator_forcerange','tendon_stiffness','tendon_damping']:
  np.testing.assert_array_equal(getattr(models[0],name),getattr(models[2],name))
 for idx in [0,2000]:
  ds=[restore(m,z,idx) for m in models];parts=[components(m,d) for m,d in zip(models,ds)]
  for name in ['actuator_force','qfrc_actuator','actuator_moment']:np.testing.assert_array_equal(getattr(ds[0],name),getattr(ds[1],name))
  a,b=parts[0];aa,bb=parts[2];np.testing.assert_allclose(aa,2*a,atol=1e-10,rtol=1e-12);np.testing.assert_array_equal(bb,b);expected=2*a+b
  np.testing.assert_allclose(ds[2].actuator_force,expected,atol=1e-10,rtol=1e-12);np.testing.assert_array_equal(ds[0].actuator_moment,ds[2].actuator_moment)
  qforce=np.zeros(models[2].nv);mujoco.mj_mulJacTVec(models[2],ds[2],qforce,expected) if False else None
  # MuJoCo sparse actuator moment transpose multiplication, uses actual compiled transmission.
  for i in range(models[2].nu):
   adr=int(ds[2].moment_rowadr[i]);n=int(ds[2].moment_rownnz[i]);cols=ds[2].moment_colind[adr:adr+n];qforce[cols]+=ds[2].actuator_moment[adr:adr+n]*expected[i]
  np.testing.assert_allclose(ds[2].qfrc_actuator,qforce,atol=1e-10,rtol=1e-12)
  if idx==0:
   for name in ['actuator_force','qfrc_actuator','efc_force','cfrc_ext','qacc']:np.testing.assert_array_equal(getattr(ds[0],name),getattr(ds[2],name))
  rows.append(dict(index=idx,time=float(z['time'][idx]),original_active=a,original_bias=b,derived_active=aa,derived_bias=bb,force_residual=float(abs(ds[2].actuator_force-expected).max()),qfrc_residual=float(abs(ds[2].qfrc_actuator-qforce).max())))
 assert not models[2].actuator_forcelimited.any()
 dump(O/'MODEL_PARAMETER_DIFF.json',changes);dump(O/'SAVED_STATE_REGRESSION.json',dict(exit_code=0,models=3,mj_forward=6,rows=rows,initial_force_contact_acceleration_exact=True,force_limits_enabled=0,unchanged_bias_and_geometry_parameters=True))
 return ds[0],ds[2]

def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),Path(prior.__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')]
 dump(O/'PREREGISTRATION.json',dict(source_hashes={str(p):sha(p) for p in files},only_change='all90 muscle gain effective force x2; entire muscle bias unchanged',interpretation='artificial separation of active/passive capacities, not validated physiological muscle',factor=2,duration=.25,body_dt=.0000125,neural_dt=.0005,sensory_command_dt=.005,command_and_sensory_gain=1,preparation=0,rules=['saved0/25ms gain1 identity gain2 active2 bias1 total2a+b qfrc moment','zeroact initial total contact force encoding identity','early15ms perleg events command activation','all neural output saturation finite act bounds','thorax root COM/vz first nonfoot','last50ms perfoot NONFOOT world ground force and geometric contacts'],behavior='NOT_JUDGED no new short window criterion',reuse_prior_ground_forces=True))
 regression()
 original=c.cp.load_model
 def derived(*a,**kw):
  m=original(*a,**kw);scale_active(m,2);return m
 c.cp.load_model=derived
 try:m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125)
 finally:c.cp.load_model=original
 c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate)
 built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids'])
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 enc=c.ChannelSensoryEncoder(groups,built['n'],c.ChannelConfig());current,_=enc.encode(c.observe_quantities(m,d,q,joints,geom));baseline=np.load(BASE/'normal/INTERVALS.npz');np.testing.assert_array_equal(current,baseline['sensory_current_nA'][0]);dump(O/'INITIAL_ENCODING_REGRESSION.json',dict(exit_code=0,current_exact_historical=True,zero_activation=bool(np.all(d.act==0)),initial_qpos_exact=bool(np.array_equal(d.qpos,np.load(BASE/'normal/BODY.npz')['qpos'][0]))))
 obs=prior.Observer(m,geom);obs(m,d);ev={};np.random.seed(0)
 rec,tr,cmd,sp=c.episode(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts)
 hashes=json.loads((O/'PREREGISTRATION.json').read_text())['source_hashes'];assert all(sha(Path(p))==v for p,v in hashes.items())
 dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=4,mj_forward=20009,mj_step=20000,neural_step=500,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('active capacity2 bias1 250ms exit0')
if __name__=='__main__':main()
