"""Frozen neural commands replay; single actuator ablation. NOT new neural behavior."""
from pathlib import Path
import json,hashlib,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P=ROOT/'outputs/experiment_platform/20261013_233237_7be93a1d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import mujoco
 from engine import canonical_physics as cp
 from flygym.compose.fly import musculoskeletal as mus
 from run_s2_fixed_stimulus import dense_moment
 out=Path('/home/sensen/Desktop/engine修复验收')/('B-local-replay-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',manual_research_not_product=True,new_neural_simulation=False,standing=False,step_calls=0,forward_calls=0,cases={})
 paths=[Path(__file__),ROOT/'B_LOCAL_MOTOR_REPLAY_PROTOCOL.zh-CN.md',ROOT/'run_s2_fixed_stimulus.py',Path(cp.__file__),Path(mus.__file__),P/'result.json',P/'traces.npz',P/'source_snapshot.zip',ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz']
 hashes={str(p):sha(p) for p in paths}
 try:
  old=json.loads((P/'result.json').read_text());assert sha(P/'traces.npz')==old['traces_sha256'];assert sha(P/'source_snapshot.zip')==old['source_snapshot_sha256']
  spec=mus._load_mjcf(str(cp.CANONICAL_XML));assets=[cp.CANONICAL_XML]+[Path(mesh.file).resolve() for mesh in spec.meshes];hashes.update({str(p):sha(p) for p in assets})
  r['old_asset_hashes_match']=all(old['input_hashes'].get(str(p))==sha(p) for p in assets)
  if not r['old_asset_hashes_match']:raise ValueError('old asset identity mismatch')
  # Preserve current source identity independently of historical source drift.
  r['historical_source_drift']={p:dict(recorded=h,current=sha(p) if Path(p).is_file() else None) for p,h in old['source_hashes'].items() if not Path(p).is_file() or sha(p)!=h}
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
   for i,p in enumerate(paths[:5]):z.write(p,f'{i:03d}_{p.name}')
  m=cp.load_model();native=m.actuator_ctrlrange.copy();m.actuator_ctrlrange[:,0]=0
  assert (m.nq,m.nv,m.na,m.nu)==(49,48,90,90) and m.opt.timestep==.0001
  with np.load(paths[-1],allow_pickle=False) as z:pose=z['projected_seated_qpos'].copy()
  with np.load(P/'traces.npz',allow_pickle=False) as z:commands=z['normal_controls'].copy();oldtr=z['normal_trajectory'].copy()
  assert commands.shape==(10,90)
  rows=[row for row in old['motor_manifest']['rows'] if row['actuator'].startswith('LFTibia_')];assert len(rows)==2
  meta_path=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/native_floor/metadata.json';hashes[str(meta_path)]=sha(meta_path);meta=json.loads(meta_path.read_text());lab=next(x for x in meta['dof_labels'] if x['joint']=='joint_LFTibia_pitch');vd=lab['dof'];qp=meta['jnt_qposadr'][meta['dof_jntid'][vd]]
  r.update(native_ctrlrange=native.tolist(),derived_ctrlrange=m.actuator_ctrlrange.tolist(),selected=rows,joint=dict(name=lab['joint'],qpos=qp,dof=vd),environment=dict(mujoco=mujoco.__version__,numpy=np.__version__),old_controls_sha256=hashlib.sha256(commands.tobytes()).hexdigest())
  baseline=None;endpoints={}
  fields=('qpos','qvel','act','ctrl','actuator_force','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint')
  def forward(d):
   r['forward_calls']+=1;mujoco.mj_forward(m,d)
   if np.any(d.warning.number):raise RuntimeError('native warning')
  cases=[('baseline',None),('repeat',None)]+[('cut_'+('flex' if 'flex_' in x['actuator'] else 'extensor'),x['actuator_index']) for x in rows]
  for name,target in cases:
   d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;forward(d)
   arrays={k:[] for k in fields+('moment','time_s')};ends=[];applied=commands.copy()
   if target is not None:applied[:,target]=0
   if target is not None:
    keep=np.arange(m.nu)!=target;assert np.array_equal(applied[:,keep],commands[:,keep])
   for k in range(10):
    d.ctrl[:]=applied[k]
    for sub in range(50):
     d.xfrc_applied[:]=0;r['step_calls']+=1;mujoco.mj_step(m,d);forward(d)
     moment=dense_moment(m,d)
     if not np.allclose(moment.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10):raise ValueError('force factorization')
     for f in fields:arrays[f].append(np.array(getattr(d,f),copy=True))
     arrays['moment'].append(moment);arrays['time_s'].append(d.time)
    ends.append(np.r_[d.qpos,d.qvel,d.act])
   arrays={k:np.asarray(v) for k,v in arrays.items()};arrays['applied_controls']=applied;arrays['endpoints']=np.array(ends)
   np.savez_compressed(out/(name+'.npz'),**arrays)
   if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite case')
   entry=dict(samples=500,command_cut_index=target,all_finite=True,external_push=False)
   if name=='baseline':
    baseline=arrays;entry['old_trajectory_exact']=bool(np.array_equal(arrays['endpoints'],oldtr));entry['old_trajectory_max_abs_difference']=float(np.max(np.abs(arrays['endpoints']-oldtr)));r['cases'][name]=entry
    if not entry['old_trajectory_exact']:raise RuntimeError('NOT_REPRODUCED: baseline differs from old P1; cuts NOT executed')
   elif name=='repeat':
    entry['all_arrays_exact_repeat']=all(np.array_equal(v,baseline[k]) for k,v in arrays.items());r['cases'][name]=entry
    if not entry['all_arrays_exact_repeat']:raise RuntimeError('repeat mismatch; cuts NOT executed')
   else:
    diff=arrays['qpos'][:,qp]-baseline['qpos'][:,qp];entry.update(max_abs_tibia_angle_delta_rad=float(np.max(np.abs(diff))),final_tibia_angle_delta_rad=float(diff[-1]),max_abs_tibia_velocity_delta_rad_s=float(np.max(np.abs(arrays['qvel'][:,vd]-baseline['qvel'][:,vd]))),max_abs_target_force_delta_model=float(np.max(np.abs(arrays['actuator_force'][:,target]-baseline['actuator_force'][:,target]))),other_controls_exact=True)
    r['cases'][name]=entry
   endpoints[name]=arrays['qpos'][:,qp].copy();dump(out/'report.json',r)
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,4))
  for name,a in endpoints.items():ax.plot(baseline['time_s']*1000,a,label=name)
  ax.set(xlabel='time (ms)',ylabel='LF tibia angle (rad)',title='Frozen neural command replay; isolated actuator command cuts; NOT standing');ax.legend();fig.tight_layout();fig.savefig(out/'local-cuts.png');plt.close(fig)
  r['status']='COMPLETE_LOCAL_OUTPUT_INTERVENTION'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
  if not r['hashes_unchanged']:r['status']='FAIL_HASH_DRIFT'
  dump(out/'report.json',r)
 print(out);print(r['status']);print(r.get('error',''));return 0 if r['status']=='COMPLETE_LOCAL_OUTPUT_INTERVENTION' else 1
if __name__=='__main__':raise SystemExit(main())
