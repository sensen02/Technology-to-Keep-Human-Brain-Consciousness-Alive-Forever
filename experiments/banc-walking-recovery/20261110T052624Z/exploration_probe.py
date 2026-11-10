"""Bounded no-adhesion exploration from exact zero mean; no training or BC."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from physics import Core, W
from models import ActorCritic
from identity import capture, verify, verify_runtime, atomic_json, sha256

HERE = Path(__file__).resolve().parent
OUT = HERE / 'exploration_diag'
CONFIG = dict(seed=20261114, latent_stds=[.05,.1,.2], ticks=2000, decision_ms=5,
              architecture='ff238', critic_obs_dim=255, dtype='float64', adhesion=False,
              target_speed=0., actor_final_head='exact zeros weight and bias; no BC',
              noise='common torch float64 standard-normal array400x42 reused across conditions',
              candidate_safety=dict(com_min_native=1.2,nonfoot_max_native=0.,foot_support_mean_min=.9),
              selection='largest std satisfying candidate safety in BOTH full and0.5–2s windows',
              label='Random output perturbations around exact zero mean; finite-PD torque actuator holds body. Not walking.')


def array_digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def statistics(a, lower_s):
    mask = a['time_s'] >= lower_s-1e-10
    times, com, loads, speeds = [a[k][mask] for k in ('time_s','com','loads_native','speed_native_per_s')]
    duration = float(times[-1]-times[0])
    delta = com[-1,:2]-com[0,:2]
    progress = float(delta @ a['initial_heading'] / duration)
    weighted = (loads*speeds).sum(0)/np.maximum(loads.sum(0),1e-12)
    valid = (progress>1e-9) & (loads.sum(0)>1e-12)
    ratios = [float(v/progress) if ok else None for v,ok in zip(weighted,valid)]
    support=loads.sum(1)/W
    out=dict(samples=int(mask.sum()),first_sample_s=float(times[0]),last_sample_s=float(times[-1]),
        duration_s=duration,foot_support_mean=float(support.mean()),foot_support_min=float(support.min()),
        com_min_native=float(com[:,2].min()),com_end_native=float(com[-1,2]),
        nonfoot_max_native=float(a['nonfoot_native'][mask].max()),
        forward_signed_initial_heading_native_per_s=progress,
        lateral_signed_native_per_s=float(delta@np.array([-a['initial_heading'][1],a['initial_heading'][0]])/duration),
        loadweighted_slip_speed_per_leg_native_per_s=weighted.tolist(),
        slip_load_denominator_native=loads.sum(0).tolist(),slip_ratio_positive_progress_valid=np.asarray(valid).tolist(),
        slip_ratio_per_leg=ratios,
        action_rms=float(np.sqrt(np.mean(a['action'][mask]**2))),
        command_rate_rms_rad_s=float(np.sqrt(np.mean(a['rate_rad_s'][mask]**2))),
        target_displacement_from_reset_rms_rad=float(np.sqrt(np.mean(a['target_delta_rad'][mask]**2))),
        target_increment_per_1ms_rms_rad=float(np.sqrt(np.mean(a['target_increment_rad'][mask]**2))),
        per_leg_max_clearance_native=a['padz_native'][mask].max(0).tolist(),
        per_leg_max_clearance_change_from_initial_native=(a['padz_native'][mask]-a['initial_padz']).max(0).tolist(),
        per_leg_contact_fraction=(a['contact_count'][mask]>0).mean(0).tolist(),
        max_abs_commanded_torque_native=float(np.abs(a['torque_native'][mask]).max()),
        max_torque_fraction_of_actuator_limit=float(a['torque_limit_fraction'][mask].max()),
        max_actuator_target_bound_violation_rad=float(a['bound_violation_rad'][mask].max()),
        actuator_target_clip_fraction=float(a['target_clip_fraction'][mask].mean()),
        max_abs_policy_mean=float(np.abs(a['policy_mean'][mask]).max()))
    out['candidate_safety'] = dict(com=out['com_min_native']>=1.2,
                                  nonfoot=out['nonfoot_max_native']==0.,
                                  support=out['foot_support_mean']>=.9)
    out['candidate_safe']=bool(all(out['candidate_safety'].values()))
    return out


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    OUT.mkdir(exist_ok=False)
    core=Core(adhesion=False)
    model=ActorCritic('ff238',critic_obs_dim=255,dtype=torch.float64)
    with torch.no_grad():
        model.actor.layers[-1].weight.zero_()
        model.actor.layers[-1].bias.zero_()
    model.eval()
    generator=torch.Generator(device='cpu').manual_seed(CONFIG['seed'])
    noise=torch.randn((400,42),dtype=torch.float64,generator=generator)
    random_obs=torch.randn((23,238),dtype=torch.float64,generator=generator)
    with torch.no_grad():
        random_means=model.actor(random_obs)
    assert torch.count_nonzero(random_means)==0
    common_noise=noise.numpy().copy()
    model_initial={name:value.detach().numpy().copy() for name,value in model.state_dict().items()}
    initial_actor_digest=hashlib.sha256(b''.join(model_initial[k].tobytes() for k in sorted(model_initial) if k.startswith('actor.'))).hexdigest()
    initial_qpos,initial_qvel=core.d.qpos.copy(),core.d.qvel.copy()
    initial_target=core.actuator.target_rad.copy()
    initial_padz=core.measure()['padz'].copy()
    config=dict(CONFIG,script_path=str(Path(__file__).resolve()),script_sha256=sha256(__file__),
                common_noise_sha256=array_digest(common_noise),initial_actor_sha256=initial_actor_digest,
                initial_qpos_sha256=array_digest(initial_qpos),initial_qvel_sha256=array_digest(initial_qvel),
                randomized_observation_mean_max_abs=float(random_means.abs().max()))
    atomic_json(OUT/'CONFIG.json',config)
    manifest=capture(core,config,extra_paths=[Path(__file__).resolve(),OUT/'CONFIG.json'])
    atomic_json(OUT/'IDENTITY_BEFORE.json',manifest)
    np.savez_compressed(OUT/'COMMON_NOISE_AND_INITIAL.npz',noise=common_noise,
                        qpos=initial_qpos,qvel=initial_qvel,target_rad=initial_target,
                        padz=initial_padz,randomized_observations=random_obs.numpy(),
                        randomized_means=random_means.numpy(),**{'model_'+k:v for k,v in model_initial.items()})
    results=[]
    started=time.perf_counter()
    for std in CONFIG['latent_stds']:
        core.reset()
        assert np.array_equal(core.d.qpos,initial_qpos) and np.array_equal(core.d.qvel,initial_qvel)
        previous=np.zeros(42)
        rows=[]
        with torch.no_grad():
            model.raw_logstd.fill_(np.log(std))
        assert np.allclose(model.logstd().exp().detach().numpy(),std,rtol=0,atol=1e-16)
        for decision in range(400):
            obs=core.obs238(previous,0.)
            with torch.no_grad():
                mean=model.actor(torch.as_tensor(obs[None],dtype=torch.float64))[0]
                assert torch.count_nonzero(mean)==0
                latent=mean+model.logstd().exp()*noise[decision]
                action=latent.tanh().numpy().copy()
            for sub in range(5):
                old_target=core.actuator.target_rad.copy()
                m=core.tick(action)
                target=core.actuator.target_rad.copy()
                bounds=core.actuator.target_bounds_rad
                rows.append(dict(time_s=float(core.d.time),com=core.d.subtree_com[core.th].copy(),
                    loads_native=m['loads'].copy(),nonfoot_native=float(m['nonfoot']),
                    speed_native_per_s=m['speed'].copy(),padz_native=m['padz'].copy(),
                    contact_count=m['contact'].copy(),action=action.copy(),
                    rate_rad_s=core.requested_rate.copy(),target_rad=target,
                    target_delta_rad=target-initial_target,target_increment_rad=target-old_target,
                    torque_native=core.d.ctrl.copy(),
                    torque_limit_fraction=float(np.max(np.abs(core.d.ctrl)/core.actuator.torque_limit)),
                    bound_violation_rad=float(max(0.,np.max(bounds[:,0]-target),np.max(target-bounds[:,1]))),
                    target_clip_fraction=float(core.actuator.angle_clip_mask.mean()),
                    policy_mean=mean.numpy().copy()))
            previous=action.copy()
        arrays={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
        arrays.update(initial_heading=core.initial_heading.copy(),initial_padz=initial_padz)
        path=OUT/f'std_{std:.2f}.npz'
        np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as saved:
            assert saved['time_s'].shape==(2000,)
            assert saved['action'].shape==(2000,42)
            assert np.allclose(np.diff(saved['time_s']),.001,rtol=0,atol=1e-12)
            assert np.isclose(saved['time_s'][-1],2.,rtol=0,atol=1e-10)
            assert all(np.isfinite(saved[key]).all() for key in saved.files)
            assert np.count_nonzero(saved['policy_mean'])==0
            assert np.allclose(saved['action'][::5],np.tanh(std*common_noise),rtol=0,atol=1e-15)
            for key,value in arrays.items():
                assert np.array_equal(saved[key],value)
        full=statistics(arrays,0.)
        late=statistics(arrays,.5)
        condition=dict(latent_std=std,full=full,late_0p5_to_2s=late,
                       candidate_safe_both=full['candidate_safe'] and late['candidate_safe'],
                       trajectory_path=str(path),trajectory_sha256=sha256(path),
                       actual_ticks=2000,native_steps=160000,native_recording=False,
                       initial_qpos_sha256=array_digest(initial_qpos),common_noise_sha256=array_digest(common_noise))
        results.append(condition)
        atomic_json(OUT/'PROGRESS.json',dict(completed_conditions=results))
        print(json.dumps(condition),flush=True)
        verify_runtime(manifest,core)
    verify(manifest)
    after=capture(core,config,extra_paths=[Path(__file__).resolve(),OUT/'CONFIG.json'])
    atomic_json(OUT/'IDENTITY_AFTER.json',after)
    assert manifest['identity_sha256']==after['identity_sha256']
    safe=[r['latent_std'] for r in results if r['candidate_safe_both']]
    summary=dict(config=config,results=results,selected_largest_candidate_safe_std=max(safe) if safe else None,
        identity_sha256=manifest['identity_sha256'],identity_before_after_equal=True,
        source_hashes={str(Path(__file__).resolve()):sha256(__file__)},
        output_hashes={str(p):sha256(p) for p in sorted(OUT.glob('*')) if p.is_file()},
        elapsed_s=time.perf_counter()-started,status='completed',
        caveat='One common-noise seed,1ms endpoint diagnostics only; candidate safety is not native peak/original walking acceptance. Exact zero policy mean; random perturbations and finite-PD actuator, no neural/BC/training claim.')
    atomic_json(OUT/'SUMMARY.json',summary)
    loaded=json.loads((OUT/'SUMMARY.json').read_text())
    assert loaded['identity_before_after_equal'] and len(loaded['results'])==3
    for path,digest in loaded['output_hashes'].items():
        assert sha256(path)==digest
    print(json.dumps(dict(status='complete',selected_std=summary['selected_largest_candidate_safe_std'],
                          elapsed_s=summary['elapsed_s'],summary=str(OUT/'SUMMARY.json'))),flush=True)

if __name__=='__main__':
    main()
