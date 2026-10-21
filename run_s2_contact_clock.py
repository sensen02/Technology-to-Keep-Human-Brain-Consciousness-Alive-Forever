"""Fixed RK4 body clocks with typed pose differences and pair contact forces."""
from pathlib import Path
import json
import hashlib
import argparse
import sys
from datetime import datetime, timezone
import uuid
import numpy as np
from run_s2_fixed_stimulus import activation_slices, compiled_metadata
from tools_s2_switched_activation import reference

ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
OUT=Path('/home/sensen/Desktop/engine修复验收') / ('S2-contact-clock-complete-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pair_world(frame,local,g1,g2):
    """World force ON sorted pair's second geom (3.9 engine_sensor total_wrench)."""
    return tuple(sorted((g1,g2))),np.asarray(frame).reshape(3,3).T@np.asarray(local) * (1 if g1<=g2 else -1)


def dofs(m,mujoco):
    labels=[]
    for j in range(m.njnt):
        kind=int(m.jnt_type[j]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
        units=['model_length']*3+['rad']*3 if kind==0 else ['rad']*3 if kind==1 else ['model_length'] if kind==2 else ['rad']
        labels.extend(dict(dof=int(m.jnt_dofadr[j])+i,joint=name,component=i,unit=u,velocity_unit=u+'/s',constraint_force_unit='model_force' if u=='model_length' else 'model_torque') for i,u in enumerate(units))
    return sorted(labels,key=lambda r:r['dof'])


def main():
    import mujoco
    import engine.canonical_physics as cp
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    out=parser.parse_args().out.resolve();out.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'S2_CONTACT_CLOCK_PROTOCOL.zh-CN.md',POSE,Path(cp.CANONICAL_XML),ROOT/'run_s2_fixed_stimulus.py',ROOT/'engine/canonical_physics.py',ROOT/'tools_s2_switched_activation.py',ROOT/'tools_s2_activation_reference.py',ROOT/'test_s2_contact_clock.py',ROOT/'tools_verify_s2_contact_clock.py']
    hashes={str(p):sha(p) for p in paths};assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
    pose=np.load(POSE,allow_pickle=False)['projected_seated_qpos'];records=[];results=[]
    for branch in ('native_floor','zero_floor'):
        for case in ('pulse','decay'):
            archives=[]
            for dt,ratio in ((2.5e-5,1),(1.25e-5,2)):
                m=cp.load_model();m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
                if branch=='zero_floor':m.actuator_ctrlrange[:,0]=0
                indices=activation_slices(m)[0];assert len(indices)==1;idx=indices[0]
                assert np.array_equal(m.actuator_dynprm[0,:3],[.0001,.0004,0.])
                d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
                if case=='decay':d.act[idx]=.1
                initial={f'initial_{key}':getattr(d,key).copy() for key in ('qpos','qvel','act','ctrl')}
                rows={k:[] for k in ('time_s','qpos','qvel','qfrc_constraint','activation','ncon','ctrl','warning_counts','step_ctrl','step_time_s')};contacts=[]
                for k in range(round(.01/dt)):
                    d.ctrl[:]=0;d.ctrl[0]=.1 if case=='pulse' and k<round(.001/dt) else 0.
                    rows['step_ctrl'].append(d.ctrl.copy());rows['step_time_s'].append(float(d.time))
                    mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                    if np.any(d.warning.number):raise RuntimeError('warning')
                    if (k+1)%ratio:continue
                    for key in ('qpos','qvel','qfrc_constraint','ctrl'):rows[key].append(getattr(d,key).copy())
                    rows['warning_counts'].append(d.warning.number.copy())
                    rows['activation'].append(float(d.act[idx]));rows['time_s'].append(float(d.time));rows['ncon'].append(d.ncon)
                    pairs={}
                    for i in range(d.ncon):
                        c=d.contact[i];force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                        pair,world=pair_world(c.frame,force[:3],int(c.geom1),int(c.geom2));pairs[pair]=pairs.get(pair,np.zeros(3))+world
                    contacts.append(pairs)
                a={k:np.array(v) for k,v in rows.items()};assert all(np.isfinite(v).all() for v in a.values())
                expected=np.arange(1,401,dtype=np.int64)*2.5e-5
                assert np.allclose(a['time_s'],expected,rtol=0,atol=1e-14)
                a.update(initial);a['expected_time_s']=expected;a['reference']=reference(expected,case,.0001 if branch=='native_floor' else 0.)
                error=float(np.max(np.abs(a['activation']-a['reference'])))
                name=f'{branch}_{case}_{dt}.npz';np.savez_compressed(out/name,**a)
                contactname=name+'.contacts.json'
                with (out/contactname).open('x') as f:json.dump([{f'{p[0]},{p[1]}':v.tolist() for p,v in sample.items()} for sample in contacts],f)
                records.append(dict(branch=branch,case=case,dt_s=dt,npz=name,contacts=contactname,target_actadr=idx,target_actuator=0,activation_max_error=error,engineering_screen_pass=error<=.001,compiled=compiled_metadata(m,mujoco)));archives.append((a,contacts))
            a,ac=archives[0];b,bc=archives[1];delta=[];force_diffs=[];presence=0
            for i in range(400):
                v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,1.,a['qpos'][i],b['qpos'][i]);delta.append(v)
                pairs=set(ac[i])|set(bc[i]);presence+=len(set(ac[i])^set(bc[i]))
                force_diffs.append(max((float(np.linalg.norm(ac[i].get(p,np.zeros(3))-bc[i].get(p,np.zeros(3)))) for p in pairs),default=0.))
            delta=np.array(delta);label=dofs(m,mujoco)
            results.append(dict(branch=branch,case=case,dofs=[dict(v,pose_max_difference=float(np.max(np.abs(delta[:,v['dof']]))),velocity_max_difference=float(np.max(np.abs(a['qvel'][:,v['dof']]-b['qvel'][:,v['dof']]))),constraint_generalized_force_max_difference=float(np.max(np.abs(a['qfrc_constraint'][:,v['dof']]-b['qfrc_constraint'][:,v['dof']])))) for v in label],pair_presence_changes=presence,pair_force_max_norm_difference=max(force_diffs),ncon_unequal_endpoints=int(np.sum(a['ncon']!=b['ncon'])),force_difference_series=force_diffs))
    assert all(sha(p)==h for p,h in hashes.items())
    with (out/'report.json').open('x') as f:json.dump(dict(records=records,comparisons=results,source_hashes=hashes,hashes_unchanged=True,body_convergence_proven=False,biological_validation=False,standing_verified=False,engineering_tolerance=.001,control_semantics='submitted ctrl held for each full RK4 step; native clamped floor is 0.0001',environment=dict(python=sys.executable,mujoco=mujoco.__version__),output_directory=str(out)),f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for r in results:ax.plot(np.arange(1,401)*.025,r['force_difference_series'],label=r['branch']+' '+r['case'])
    ax.set_xlabel('Time(ms)');ax.set_ylabel('Max geometry-pair force difference (model units)');ax.legend();ax.grid(alpha=.2);ax.set_title('25 vs12.5us RK4 contact difference — NOT convergence proof')
    fig.tight_layout();fig.savefig(out/'contact-differences.png',dpi=150)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('dofs','force_difference_series')} for r in results],indent=2))


if __name__=='__main__':main()
