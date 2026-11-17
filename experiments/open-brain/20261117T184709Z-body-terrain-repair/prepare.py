"""Derive documented engineering envelopes; verify zero-pad canonical equivalence."""
from body_model import *
import sys
import time

old=prior.body_run.Body();old.at_start(17)
jids=np.arange(1,old.m.njnt);adr=old.m.jnt_qposadr[jids]
lo=old.d.qpos[adr].copy();hi=lo.copy();sources={}
for trace in sorted((BASE/'20261117T170820Z-body-readaptation/TEACHER').glob('*/TRACE.npz')):
    meta=json.loads((trace.parent/'RESULT.json').read_text())
    if not meta['pass_']:continue
    assert sha(trace)==meta['trace_sha256']
    with np.load(trace,allow_pickle=False) as a:
        q=a['qpos'][:,adr]
        lo=np.minimum(lo,q.min(0));hi=np.maximum(hi,q.max(0))
        for i,j in enumerate(jids):
            active=np.where(old.m.actuator_trnid[old.actuators,0]==j)[0]
            if len(active):
                lo[i]=min(lo[i],a['ctrl'][:,active[0]].min());hi[i]=max(hi[i],a['ctrl'][:,active[0]].max())
    sources[str(trace)]=sha(trace)
phases=np.arange(1024)*2*np.pi/1024
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
for leg in old.motor.legs:
    q=old.motor.preprogrammed_steps.get_joint_angles(leg,phases)
    for index,spec in enumerate(old.motor.preprogrammed_steps.dofs_per_leg):
        dof=dof_spec_to_jointdof(leg,spec);a=old.dofs.index(dof);j=int(old.m.actuator_trnid[old.actuators[a],0]);i=int(np.where(jids==j)[0][0])
        lo[i]=min(lo[i],float(q[index].min()));hi[i]=max(hi[i],float(q[index].max()))
limits={}
for index,j in enumerate(jids):
    active=bool(j in old.m.actuator_trnid[old.actuators,0]);margin=.50 if active else .35
    bounds=[float(lo[index]-margin),float(hi[index]+margin)]
    limits[mujoco.mj_id2name(old.m,mujoco.mjtObj.mjOBJ_JOINT,int(j))]=dict(active=active,observed_range_rad=[float(lo[index]),float(hi[index])],
        joint_range_rad=bounds,control_range_rad=[bounds[0]+.03,bounds[1]-.03] if active else None,
        engineering_margin_rad=margin,biological_hard_range='UNKNOWN',coordinate='Original P1 NMF yaw-pitch-roll; named joints, no P2 angle transfer')
write(HERE/'JOINT_LIMITS.json',dict(type='SOURCE_LABELLED_ENGINEERING_OPERATING_ENVELOPE',joints=limits,
 source_episodes=len(sources),source_sha256=sources,statement='Finite protection ranges from same-coordinate qualified physical/official gait data, with declared engineering margin; not measured anatomical stops.'))
# Raw mesh/YAML/kinematic assets are included before any new native integration.
paths=[Path(__file__).resolve(),HERE/'body_model.py',HERE/'drive.sh',HERE/'JOINT_LIMITS.json',prior.FIRST/'FROZEN_INIT.npz']
paths += list((ROOT/'venv_body/lib/python3.12/site-packages/flygym/assets/model/neuromechfly').rglob('*.yaml'))
paths += list((ROOT/'venv_body/lib/python3.12/site-packages/flygym/assets/model/neuromechfly/meshes').rglob('*.stl'))
paths += list((ROOT/'venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/assets').glob('*'))
paths += [Path(m.__file__).resolve() for m in sys.modules.values() if getattr(m,'__file__',None) and str(ROOT) in str(m.__file__) and str(m.__file__).endswith('.py')]
write(HERE/'PREPARE_INPUTS.json',dict(source_sha256={str(p):sha(p) for p in dict.fromkeys(paths) if p.is_file()}))
a=Body(adhesion=False);b=Body(adhesion=True)
assert a.m.nbody==b.m.nbody and a.m.nq==b.m.nq and b.m.nu-a.m.nu==6
assert a.m.jnt_limited[1:].all() and len(a.actuators)==42
assert a.m.actuator_ctrllimited[a.actuators].all()
for tick in range(2800):a.apply(a.neutral)
assert not a.d.warning.number.any() and a.d.xmat[a.thorax,8]>.95
snapshot(a);a.restore(17);b.restore(17)
mass_a=np.empty((a.m.nv,a.m.nv));mass_b=np.empty_like(mass_a)
mujoco.mj_fullM(a.m,mass_a,a.d.qM);mujoco.mj_fullM(b.m,mass_b,b.d.qM)
rows=[]
for tick in range(3000):
    a.apply(a.neutral);b.apply(b.neutral,np.zeros(6))
    rows.append([np.max(np.abs(a.d.qpos-b.d.qpos)),np.max(np.abs(a.d.qvel-b.d.qvel))])
rows=np.asarray(rows);np.savez_compressed(HERE/'ZERO_ADHESION_PAIR.npz',error=rows)
result=dict(status='CANONICAL_LIMITED_BODY_PREPARED',active_limits=int(a.m.jnt_limited[a.jids].sum()),
    all_hinge_limits=int(a.m.jnt_limited[1:].sum()),actuator_target_limits=int(a.m.actuator_ctrllimited[a.actuators].sum()),
    nq=a.m.nq,nv=a.m.nv,nbody=a.m.nbody,nu_reference=a.m.nu,nu_pad=b.m.nu,
    fusion=False,mass_matrix_max_error=float(np.max(np.abs(mass_a-mass_b))),
    zero_pad_qpos_max_error=float(rows[:,0].max()),zero_pad_qvel_max_error=float(rows[:,1].max()),
    zero_pad_pass=bool(rows.max()<=1e-12 and np.max(np.abs(mass_a-mass_b))<=1e-12),
    preparation_physics_steps=8800,bodyweight_native=a.weight,nominal_pad_gain_native=GAIN,
    biological_joint_ranges='UNKNOWN_ENGINEERING_ENVELOPE_ONLY')
mujoco.mj_saveModel(a.m,str(HERE/'BODY42.mjb'),None);mujoco.mj_saveModel(b.m,str(HERE/'BODY48.mjb'),None)
assert all(sha(p)==v for p,v in json.loads((HERE/'PREPARE_INPUTS.json').read_text())['source_sha256'].items())
write(HERE/'PREPARE_RESULT.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
