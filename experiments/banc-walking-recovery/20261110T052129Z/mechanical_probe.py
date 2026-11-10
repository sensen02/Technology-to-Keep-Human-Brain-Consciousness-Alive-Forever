"""Paired 2s derived-body mechanics diagnostics, no training or plant edits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
import mujoco as mj
import physics
import identity
from models import ActorCritic

HERE=Path(__file__).resolve().parent
OUT=HERE/'mechanical_diag'
LEGS=list(physics.LEGS)

def save(name,value):
    identity.atomic_json(OUT/name,value)

def stats(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()),p95=float(np.quantile(a,.95)),rms=float(np.sqrt(np.mean(a*a))))

def summarize(z,start,end):
    # First window includes t0 for state extrema, but initial force/speed is excluded.
    mask=(z['time_s']>max(start,0)+1e-10)&(z['time_s']<=end+1e-10)
    a=int(round(start*1000)); b=int(round(end*1000))
    state=z['com'][a:b+1]; delta=state[-1]-state[0]
    loads=z['loads'][mask]; speed=z['contact_speed'][mask]
    denominator=loads.sum(0)
    perleg={}
    for j,leg in enumerate(LEGS):
        active=z['contact_count'][mask,j]>0
        raw=z['raw_contact_speed'][mask,j][active]
        perleg[leg]=dict(load_mean_native=float(loads[:,j].mean()),contact_fraction=float(active.mean()),
            load_weighted_contact_speed_native_per_s=float((loads[:,j]*speed[:,j]).sum()/denominator[j]) if denominator[j]>0 else None,
            contact_speed_sample_stats=stats(speed[:,j]),
            raw_contact_speed_active_stats=stats(raw) if raw.size else None,
            all_pad_center_xy_speed_stats=stats(np.linalg.norm(z['pad_center_velocity'][mask,j,:,:2],axis=-1)))
    lowpass={}
    for n in (20,60):
        valid=(z[f'lp{n}_velocity_time_s']>start+1e-10)&(z[f'lp{n}_velocity_time_s']<=end+1e-10)
        lowpass[f'{n}ms_com_xy_speed']=stats(np.linalg.norm(z[f'lp{n}_com_velocity'][valid,:2],axis=1))
    return dict(window_s=[start,end],tick_end_samples=int(mask.sum()),com_start_native=state[0].tolist(),
        com_end_native=state[-1].tolist(),com_delta_native=delta.tolist(),com_xy_net_distance_native=float(np.linalg.norm(delta[:2])),
        com_z_range_native=[float(state[:,2].min()),float(state[:,2].max())],
        com_xy_velocity_stats=stats(np.linalg.norm(z['com_velocity'][mask,:2],axis=1)),
        com_z_velocity_stats=stats(z['com_velocity'][mask,2]),com_lowpass=lowpass,
        load_weighted_contact_speed_native_per_s=float((loads*speed).sum()/loads.sum()),
        foot_load_native=stats(loads.sum(1)),foot_load_over_body_weight=stats(loads.sum(1)/physics.W),
        nonfoot_peak_native=float(z['nonfoot'][mask].max()),nonfoot_nonzero_samples=int(np.count_nonzero(z['nonfoot'][mask]>0)),
        per_leg=perleg,
        actuator_joint_speed_abs_rad_s=stats(np.abs(z['actuated_qvel'][mask])),
        ctrl_abs_native=stats(np.abs(z['ctrl'][mask])))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['all','zero','policy'],default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    script=Path(__file__).read_bytes()
    (OUT/'mechanical_probe.py').write_bytes(script)
    config=dict(goal='S2/S5 mechanical settling/chattering diagnostic; not walking/frozen acceptance',
        mode=args.mode,ticks_each=2000,native_steps_each=160000,sample_dt_s=.001,decision_dt_s=.001,
        comparison='zero-rate finite-PD hold and legacy deterministic ff495 saved-best, each default adhesion vs adhesion=False',
        only_changed_physical_parameter='Core constructor adhesion boolean; all other model/actuator/body parameters unchanged',
        legacy_weight_path=str(identity.LEGACY),legacy_weight_sha256=identity.sha256(identity.LEGACY),
        windows_s=[[0.,.1],[.3,.5],[.5,2.]],additional_tail_blocks_s=[[.5,1.],[1.,1.5],[1.5,2.]],
        initial_state='fresh unchanged Core reset per case; qpos/qvel/target exact equality asserted',
        control='Core.tick(action) only, zeros(42) or model.act(core.obs(),deterministic=True); no teacher or root/pose/force writes by probe',
        policy='ActorCritic(ff495,float64).import_legacy_npz(identity.LEGACY); tanh(mean), raw logstd irrelevant to deterministic action',
        native_recorder=False,video=False,
        filtering='causal trailing 20/60 sample COM boxcar, full windows only, velocity from consecutive filtered positions / .001s',
        units={'position':'native length','velocity':'native length/s','force':'native force','angular_velocity':'rad/s','absolute_SI':'NOT_ESTABLISHED'},
        contact_speed='positive normal-load weighted horizontal world velocity; raw speed is per-leg unweighted contact mean, 0 if no contacts; independent all-3-pad center speeds also saved',
        sample_summary='forces/speeds use tick ends (start,end]; COM state includes both window endpoints; no native-step extrema claims',
        command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {HERE}/env/bin/python -B {__file__} --mode {args.mode}',
        environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        script_sha256=hashlib.sha256(script).hexdigest())
    save('config.json',config)
    # Parent supplies preregistration in this new snapshot before execution.
    if not (HERE/'PREREG.json').is_file():
        raise FileNotFoundError('Parent PREREG.json must exist before simulation')
    model=ActorCritic('ff495',dtype=torch.float64).import_legacy_npz(identity.LEGACY).eval()
    started=time.monotonic(); initial=None; manifests=[]; results={}
    modes=['zero','policy'] if args.mode=='all' else [args.mode]
    for mode in modes:
        for adhesion in (True,False):
            label=mode+('_A_adhesion' if adhesion else '_B_noadhesion')
            core=physics.Core() if adhesion else physics.Core(adhesion=False)
            manifest=identity.capture(core,dict(config,case=label),extra_paths=[__file__])
            save(label+'_identity.json',manifest); manifests.append(manifest)
            state=tuple(x.copy() for x in (core.d.qpos,core.d.qvel,core.actuator.target_rad))
            if initial is None:
                initial=state
                np.savez_compressed(OUT/'initial_state.npz',qpos=state[0],qvel=state[1],target_rad=state[2])
            else:
                assert all(np.array_equal(x,y) for x,y in zip(initial,state)), 'reset mismatch'
            def sample(m,action):
                centers=[]
                for leg in LEGS:
                    velocities=[]
                    for geom in core.pads[leg]:
                        jac=np.zeros((3,core.plant.nv))
                        mj.mj_jac(core.plant,core.d,jac,None,core.d.geom_xpos[geom],int(core.plant.geom_bodyid[geom]))
                        velocities.append(jac@core.d.qvel)
                    centers.append(velocities)
                return dict(time_s=float(core.d.time),com=core.d.subtree_com[core.th].copy(),com_velocity=core.com_velocity().copy(),
                    loads=m['loads'].copy(),contact_speed=m['speed'].copy(),raw_contact_speed=m['raw_speed'].copy(),
                    contact_count=m['contact'].copy(),nonfoot=np.asarray(m['nonfoot']),pad_center_velocity=np.asarray(centers),
                    action=action.copy(),target_rad=core.actuator.target_rad.copy(),ctrl=core.d.ctrl.copy(),
                    actuated_qvel=core.d.qvel[core.a_dof].copy(),qpos=core.d.qpos.copy(),qvel=core.d.qvel.copy())
            rows=[sample(core.measure(),np.zeros(42))]
            for k in range(2000):
                action=np.zeros(42) if mode=='zero' else model.act(core.obs(),deterministic=True)['action'][0].cpu().numpy()
                m=core.tick(action)
                rows.append(sample(m,action))
                if mode=='zero':
                    assert np.array_equal(core.actuator.target_rad,state[2])
            z={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
            assert all(np.isfinite(x).all() for x in z.values())
            assert np.allclose(z['time_s'],np.arange(2001)*.001,atol=2e-11,rtol=0)
            z['com_displacement']=z['com']-z['com'][0]
            for n in (20,60):
                cs=np.vstack([np.zeros((1,3)),np.cumsum(z['com'],axis=0)])
                filtered=(cs[n:]-cs[:-n])/n
                z[f'lp{n}_com']=filtered
                z[f'lp{n}_com_time_s']=z['time_s'][n-1:]
                z[f'lp{n}_com_velocity']=np.diff(filtered,axis=0)/.001
                z[f'lp{n}_velocity_time_s']=z['time_s'][n:]
            summary=dict(case=label,mode=mode,adhesion=adhesion,initial_state_matches=True,
                samples=2001,duration_s=float(z['time_s'][-1]),warning_count=core.d.warning.number.tolist(),
                full=summarize(z,0.,2.),windows=[summarize(z,a,b) for a,b in config['windows_s']],
                tail_blocks=[summarize(z,a,b) for a,b in config['additional_tail_blocks_s']],
                legacy_engineering=physics.summarise([(z['time_s'][i],z['com'][i],z['loads'][i],z['nonfoot'][i],z['contact_speed'][i]) for i in range(1,2001)],160000,initial_heading=core.initial_heading))
            np.savez_compressed(OUT/(label+'_1ms.npz'),**z)
            save(label+'_summary.json',summary)
            identity.verify_runtime(manifest,core)
            results[label]=summary
            print(json.dumps(dict(case=label,wall_s=time.monotonic()-started,window_weighted_speeds=[x['load_weighted_contact_speed_native_per_s'] for x in summary['windows']],tail_weighted_speeds=[x['load_weighted_contact_speed_native_per_s'] for x in summary['tail_blocks']],com_z_range=summary['full']['com_z_range_native'],nonfoot_peak=summary['full']['nonfoot_peak_native'])),flush=True)
    for manifest in manifests:
        identity.verify(manifest)
    save('summary.json',dict(config=config,results=results,wall_s=time.monotonic()-started,input_hashes_unchanged=True,
        limitation='single paired reset, derived body, finite-PD actuator; settling/chattering description only; no original acceptance or causal mechanism proof'))
    save('output_sha256.json',{str(p):identity.sha256(p) for p in OUT.iterdir() if p.is_file()})

if __name__=='__main__':
    code=1
    try:
        main(); code=0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if OUT.exists():
            save('exit_status.json',dict(exit_code=code,finished_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
