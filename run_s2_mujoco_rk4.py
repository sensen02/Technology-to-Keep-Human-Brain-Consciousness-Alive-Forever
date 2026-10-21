"""Derived MuJoCo RK4 body diagnostic; no production settings change."""
from pathlib import Path
import hashlib
import json
import numpy as np
from run_s2_fixed_stimulus import dense_moment,activation_slices
from tools_s2_switched_activation import reference

ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
OUT=Path('/home/sensen/Desktop/engine修复验收/S2-mujoco-rk4-final')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    import engine.canonical_physics as cp
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'S2_MUJOCO_RK4_PROTOCOL.zh-CN.md',POSE,Path(cp.CANONICAL_XML),ROOT/'run_s2_fixed_stimulus.py',ROOT/'tools_s2_switched_activation.py',ROOT/'tools_s2_activation_reference.py',ROOT/'engine/canonical_physics.py']
    hashes={str(p):sha(p) for p in paths};assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
    pose=np.load(POSE,allow_pickle=False)['projected_seated_qpos'];records=[];trajectories={}
    for branch in ('native_floor','zero_floor'):
        floor=.0001 if branch=='native_floor' else 0.
        for case in ('pulse','decay'):
            for dt in (1e-4,5e-5,2.5e-5):
                m=cp.load_model();m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
                if branch=='zero_floor':m.actuator_ctrlrange[:,0]=0
                idx=activation_slices(m)[0];assert len(idx)==1;idx=idx[0]
                assert np.array_equal(m.actuator_dynprm[0,:3],[.0001,.0004,0.])
                d=mujoco.MjData(m);d.qpos[:]=pose
                if case=='decay':d.act[idx]=.1
                rows={k:[] for k in ('time_s','activation','qpos','qvel','actuator_force','moment','qfrc_actuator','ctrl','ncon','min_contact_dist','has_contact','warning_counts')}
                for k in range(round(.01/dt)):
                    d.ctrl[:]=0;d.ctrl[0]=.1 if case=='pulse' and k<round(.001/dt) else 0.
                    mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                    if np.any(d.warning.number):raise RuntimeError('MuJoCo warning')
                    moment=dense_moment(m,d)
                    assert np.allclose(moment.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10)
                    for key in ('qpos','qvel','actuator_force','qfrc_actuator','ctrl'):rows[key].append(getattr(d,key).copy())
                    rows['moment'].append(moment);rows['time_s'].append(float(d.time));rows['activation'].append(float(d.act[idx]));rows['ncon'].append(d.ncon)
                    rows['has_contact'].append(d.ncon>0);rows['min_contact_dist'].append(float(np.min(d.contact.dist[:d.ncon])) if d.ncon else 0.)
                    rows['warning_counts'].append(d.warning.number.copy())
                arrays={key:np.array(value) for key,value in rows.items()};assert all(np.isfinite(v).all() for v in arrays.values())
                expected=np.arange(1,len(arrays['time_s'])+1)*dt;assert np.allclose(arrays['time_s'],expected,rtol=0,atol=1e-14)
                ref=reference(expected,case,floor);arrays['reference']=ref
                name=f'{branch}_{case}_{dt}.npz';np.savez_compressed(OUT/name,**arrays);trajectories[(branch,case,dt)]=arrays
                err=float(np.max(np.abs(arrays['activation']-ref)))
                records.append(dict(branch=branch,case=case,dt_s=dt,npz=name,activation_max_error=err,screen_pass=err<=.001,integrator=int(m.opt.integrator),ncon_range=[int(np.min(arrays['ncon'])),int(np.max(arrays['ncon']))],target_actadr=idx))
    diffs=[]
    for branch in ('native_floor','zero_floor'):
        for case in ('pulse','decay'):
            for coarse,fine in ((1e-4,5e-5),(5e-5,2.5e-5)):
                a=trajectories[(branch,case,coarse)];b=trajectories[(branch,case,fine)]
                diffs.append(dict(branch=branch,case=case,coarse_dt=coarse,fine_dt=fine,qpos_max_component_difference=float(np.max(np.abs(a['qpos']-b['qpos'][1::2]))),contact_count_difference_endpoints=int(np.sum(a['ncon']!=b['ncon'][1::2]))))
    assert all(sha(p)==h for p,h in hashes.items())
    with (OUT/'report.json').open('x') as f:json.dump(dict(records=records,common_endpoint_differences=diffs,hashes=hashes,hashes_unchanged=True,biological_validation=False,body_convergence_proven=False,environment=dict(mujoco=mujoco.__version__)),f,indent=2);f.write('\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axs,('pulse','decay')):
        for dt in (1e-4,5e-5,2.5e-5):
            a=trajectories[('native_floor',case,dt)];ax.plot(a['time_s']*1000,a['activation']-a['reference'],label=f'{dt*1e6:g}us')
        ax.set_title(case+' native floor');ax.set_xlabel('Time(ms)');ax.set_ylabel('Activation minus scalar reference');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Actual MuJoCo RK4 derived diagnostic — NOT standing or biological validation')
    fig.tight_layout();fig.savefig(OUT/'mujoco-rk4-errors.png',dpi=150)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
