"""Fixed actuator0 clock refinement, engineering research only."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
OUT=Path('/home/sensen/Desktop/engine修复验收/S2-activation-clock')
DTS=(.0001,.00005,.000025,.0000125)


def common_errors(coarse,fine,ratio):
    if len(fine)!=len(coarse)*ratio:raise ValueError('clock length mismatch')
    return float(np.max(np.abs(coarse-fine[ratio-1::ratio])))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    import engine.canonical_physics as cp
    from run_s2_fixed_stimulus import activation_slices
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'S2_ACTIVATION_CLOCK_PROTOCOL.zh-CN.md',POSE,Path(cp.CANONICAL_XML),ROOT/'engine/canonical_physics.py',ROOT/'run_s2_fixed_stimulus.py']
    hashes={str(p):sha(p) for p in paths};assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
    pose=np.load(POSE,allow_pickle=False)['projected_seated_qpos'];records=[];trajectories={}
    for branch in ('native_floor','zero_floor'):
        for ctrl in (0.,.1):
            for dt in DTS:
                model=cp.load_model();model.opt.timestep=dt
                if branch=='zero_floor':model.actuator_ctrlrange[:,0]=0
                adr=activation_slices(model)[0];assert len(adr)==1;idx=adr[0]
                d=mujoco.MjData(model);d.qpos[:]=pose;d.ctrl[0]=ctrl;mujoco.mj_forward(model,d)
                initial_dot=float(d.act_dot[idx]);rows=[];times=[];forces=[];positions=[];velocities=[];derivatives=[]
                for _ in range(round(.01/dt)):
                    mujoco.mj_step(model,d);mujoco.mj_forward(model,d)
                    if np.any(d.warning.number):raise RuntimeError('MuJoCo warning')
                    rows.append(float(d.act[idx]));times.append(float(d.time));forces.append(float(d.actuator_force[0]));positions.append(d.qpos.copy());velocities.append(d.qvel.copy());derivatives.append(float(d.act_dot[idx]))
                arrays=dict(time_s=np.array(times),activation=np.array(rows),force=np.array(forces),qpos=np.array(positions),qvel=np.array(velocities),act_dot=np.array(derivatives))
                if not all(np.isfinite(a).all() for a in arrays.values()):raise RuntimeError('nonfinite')
                assert abs(d.time-.01)<1e-14 and np.all(d.ctrl[1:]==0) and d.ctrl[0]==ctrl
                name=f'{branch}_ctrl{ctrl}_dt{dt}.npz';np.savez_compressed(OUT/name,**arrays)
                trajectories[(branch,ctrl,dt)]=arrays
                records.append(dict(branch=branch,ctrl=ctrl,dt_s=dt,npz=name,actadr=idx,dyntype=int(model.actuator_dyntype[0]),dynprm=model.actuator_dynprm[0].tolist(),integrator=int(model.opt.integrator),initial_act_dot=initial_dot,first_activation=rows[0],euler_first_step_error=abs(rows[0]-dt*initial_dot),peak=max(rows),final=rows[-1]))
    differences=[]
    for branch in ('native_floor','zero_floor'):
        for ctrl in (0.,.1):
            for a,b in zip(DTS[:-1],DTS[1:]):
                x=trajectories[(branch,ctrl,a)];y=trajectories[(branch,ctrl,b)]
                differences.append(dict(branch=branch,ctrl=ctrl,coarse_dt=a,fine_dt=b,activation_max_difference=common_errors(x['activation'],y['activation'],2),qpos_max_difference=common_errors(x['qpos'],y['qpos'],2)))
    assert all(sha(p)==h for p,h in hashes.items())
    with (OUT/'report.json').open('x') as f:json.dump(dict(cases=records,common_endpoint_differences=differences,input_source_sha256=hashes,hashes_unchanged=True,biological_validation=False,convergence_proven=False),f,indent=2);f.write('\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,branch in zip(axs,('native_floor','zero_floor')):
        for dt in DTS:
            a=trajectories[(branch,.1,dt)];ax.plot(a['time_s']*1000,a['activation'],label=f'{dt*1e6:g} us')
        ax.set_title(branch+' actuator0 ctrl0.1');ax.set_xlabel('Time (ms)');ax.set_ylabel('Activation');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Fixed activation clock diagnostic — NOT biological calibration')
    fig.tight_layout();fig.savefig(OUT/'activation-clock.png',dpi=150);plt.close(fig)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
