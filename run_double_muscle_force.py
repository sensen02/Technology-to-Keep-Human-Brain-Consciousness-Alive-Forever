"""Global muscle force capacity x2 (active AND bias), one fresh historical normal arm."""
from pathlib import Path
import os,json,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_quarter_second_pair import Observer,OUT as BASE
from run_one_second_motor_pair import INITIAL,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-muscle-force-double/20261027T190000Z')
def scale_model(m,factor):
 changes=[]
 for i in range(m.nu):
  assert m.actuator_gaintype[i]==mujoco.mjtGain.mjGAIN_MUSCLE and m.actuator_biastype[i]==mujoco.mjtBias.mjBIAS_MUSCLE
  before=[];after=[]
  for arr in [m.actuator_gainprm,m.actuator_biasprm]:
   f=float(arr[i,2]);scale=float(arr[i,3]);before.append(f if f>=0 else scale/float(m.actuator_acc0[i]));arr[i,2 if f>=0 else 3]*=factor;after.append(float(arr[i,2]) if f>=0 else float(arr[i,3])/float(m.actuator_acc0[i]))
  assert before[0]==before[1];changes.append(dict(actuator=m.actuator(i).name,mode='explicit_force' if m.actuator_gainprm[i,2]>=0 else 'auto_scale_over_acc0',old_effective_force=before,new_effective_force=after))
 return changes

def restore(m,z,idx):
 d=mujoco.MjData(m);d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.act[:]=z['activation'][idx];d.ctrl[:]=z['command'][idx];d.time=z['time'][idx];d.xfrc_applied[:]=z['external_force'][idx];mujoco.mj_forward(m,d);return d

def components(m,d):
 a=[];b=[]
 for i in range(m.nu):
  a.append(mujoco.mju_muscleGain(d.actuator_length[i],d.actuator_velocity[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9])*d.act[int(m.actuator_actadr[i])]);b.append(mujoco.mju_muscleBias(d.actuator_length[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_biasprm[i,:9]))
 return np.array(a),np.array(b)
def main():
 O.mkdir(parents=True,exist_ok=False);files=[Path(__file__),Path(c.__file__),INITIAL,c.POOLS,Path(c.cp.CANONICAL_XML),Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,only_difference='all90 compiled muscle gain and bias effective force x2; active AND passive; no external compensation',formula='force>=0: force*=2; force<0: leave sentinel, scale*=2; F0=scale/acc0',duration=.25,body_dt=.0000125,neural_dt=.0005,sensory_command_dt=.005,comparisons=['commands activation active bias saturation spikes early15ms','thorax root com/vz first nonfoot','last50ms perfoot/nonfoot world force and contacts','finite warnings'],stability='NOT_JUDGED no new250ms gate'))
 z=np.load(BASE/'normal/BODY.npz');models=[c.cp.load_model() for _ in range(3)];scale_model(models[1],1);changes=scale_model(models[2],2);dump(O/'MODEL_PARAMETER_DIFF.json',changes);reg=[]
 for idx in [0,2000]:
  ds=[restore(m,z,idx) for m in models];parts=[components(m,d) for m,d in zip(models,ds)]
  for name in ['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_lengthrange']:np.testing.assert_array_equal(getattr(models[0],name),getattr(models[1],name))
  for name in ['actuator_force','qfrc_actuator','actuator_moment']:np.testing.assert_array_equal(getattr(ds[0],name),getattr(ds[1],name))
  for name in ['actuator_force','qfrc_actuator']:np.testing.assert_allclose(getattr(ds[2],name),getattr(ds[0],name)*2,atol=1e-10,rtol=1e-12)
  np.testing.assert_array_equal(ds[0].actuator_moment,ds[2].actuator_moment)
  for j in [0,1]:np.testing.assert_allclose(parts[2][j],parts[0][j]*2,atol=1e-10,rtol=1e-12)
  reg.append(dict(index=idx,baseline_active=parts[0][0],baseline_bias=parts[0][1],double_active=parts[2][0],double_bias=parts[2][1],force_residual=float(abs(ds[2].actuator_force-2*ds[0].actuator_force).max()),qfrc_residual=float(abs(ds[2].qfrc_actuator-2*ds[0].qfrc_actuator).max())))
 dump(O/'SAVED_STATE_REGRESSION.json',dict(exit_code=0,states=reg,force_limited=int(models[2].actuator_forcelimited.sum()),models=3,forwards=6))
 original=c.cp.load_model
 def derived_model(*a,**kw):
  m=original(*a,**kw);scale_model(m,2);return m
 c.cp.load_model=derived_model
 try:m,q,gate,joints,geom=c.prepare_body(np.load(INITIAL)['qpos'],body_dt_s=.0000125)
 finally:c.cp.load_model=original
 c.require_six_foot_initial_contacts(m,q,geom);dump(O/'INITIAL_GATE.json',gate);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));names=[m.actuator(i).name for i in range(m.nu)];mapping=c.build_motor_manifest(json.loads(c.POOLS.read_text()),built['tier']['root_ids'],names);dump(O/'MAPPING.json',mapping);groups,_=c.channel_index_map(built['n'],built['tier']['root_ids']);obs=Observer(m,geom);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obs(m,d);ev={};np.random.seed(0)
 rec,tr,cmd,sp=c.episode(m,q,joints,geom,built,mapping,groups,'normal',.25,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=obs,body_dt_s=.0000125)
 np.savez_compressed(O/'INTERVALS.npz',trajectory=tr,commands=cmd,spikes=sp,**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)});dump(O/'RECORDS.json',rec);obs.save(O/'BODY.npz');dump(O/'CONTACTS.json',obs.contacts);assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'EXECUTION.json',dict(command='OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl '+sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=4,mj_forward=20009,mj_step=20000,neural_step=500,failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('muscleforce x2 normal250ms exit0')
if __name__=='__main__':main()
