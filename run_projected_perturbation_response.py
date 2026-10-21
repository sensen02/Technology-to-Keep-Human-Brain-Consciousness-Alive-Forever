"""Predeclared matched physical perturbation, not standing/recovery acceptance.
手工研究原型，不是成品. No background, gains searched, PD or invented edges.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
import json
import time
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as loop
from engine.embodied.declared_pose_projection import project_scalar_ranges
from engine.embodied.initial_pose_contract import model_limit_report
from engine.embodied.source_snapshot import build_source_snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/projected_perturbation_response'
OMIT = ('hairplate_angle', 'campaniform_strain', 'bristle_touch')
CLOCKS = [('base', .0001, .0005), ('quarter', .000025, .000125)]


def first_difference(a, b, times):
    if len(a) != len(b) or len(a) != len(times):
        raise ValueError('matched arrays/times required')
    return next((float(t) for x, y, t in zip(a, b, times) if not np.array_equal(x, y)), None)


def pose_metrics(model, trajectory, initial_pose):
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = initial_pose
    mujoco.mj_forward(model, data)
    thorax = model.body('Thorax').id
    initial_thorax = data.xpos[thorax].copy()
    initial_com = data.subtree_com[0].copy()
    rows = []
    for row in trajectory:
        data.qpos[:] = row[:model.nq]
        data.qvel[:] = row[model.nq:model.nq+model.nv]
        data.act[:] = row[model.nq+model.nv:]
        mujoco.mj_forward(model, data)
        limit = model_limit_report(model, data.qpos)
        q = data.qpos[3:7]
        q0 = initial_pose[3:7]
        cosine = abs(float(np.dot(q, q0)/(np.linalg.norm(q)*np.linalg.norm(q0))))
        facts = loop.cp.stance_facts(model, data, raise_on_fail=False).as_dict()
        rows.append(dict(thorax_displacement_model=(data.xpos[thorax]-initial_thorax).tolist(),
            com_displacement_model=(data.subtree_com[0]-initial_com).tolist(),
            root_rotation_from_initial_rad=float(2*np.arccos(np.clip(cosine, 0, 1))),
            joint_range_violation_count=len(limit['violations']),
            max_range_excess=float(max((v['outside_by'] for v in limit['violations']), default=0)),
            n_floor_contacts=facts['n_floor_contact'], n_nonfoot_floor_contacts=facts['n_body_floor_contact']))
    return rows


def main():
    import mujoco
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start = time.monotonic()
    dest = OUT/time.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=False)
    model, historical, _, joints, geoms = loop.prepare_body()
    pose, projection = project_scalar_ranges(model, historical)
    d = mujoco.MjData(model)
    pose[2] = loop.cp.seat_root(model, d, pose)
    d.qpos[:] = pose
    mujoco.mj_forward(model, d)
    initial_gate = loop.cp.stance_facts(model, d).as_dict()
    model.actuator_ctrlrange[:, 0] = 0
    weight = float(model.body_mass.sum()*abs(model.opt.gravity[2]))
    push = dict(start_s=.001, end_s=.002, force_model=[.1*weight, 0., 0.],
                weight_fraction=.1, total_weight_model=weight)
    built = loop.build_neural_tier(loop.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    manifest = loop.build_motor_manifest(json.loads(loop.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = loop.channel_index_map(int(built['n']), built['tier']['root_ids'])
    arrays = dict(projected_pose=pose, tier_root_ids=np.asarray(built['tier']['root_ids']))
    episodes, comparisons, checks = {}, {}, {}
    for clock, dt_body, dt_neural in CLOCKS:
        model.opt.timestep = dt_body
        for label, arm, perturb in [('unperturbed', 'normal', False), ('feedback', 'normal', True),
                                   ('replay', 'replay', True), ('sensory_cut', 'sensory_cut', True),
                                   ('motor_cut', 'motor_cut', True)]:
            name = clock+'_'+label
            replay = arrays.get(clock+'_unperturbed_controls') if label == 'replay' else None
            forces = []
            actual_step = mujoco.mj_step
            thorax = model.body('Thorax').id
            def witnessed_step(m, data):
                forces.append(data.xfrc_applied[thorax, :3].copy())
                actual_step(m, data)
            with patch.object(loop, 'DT_BODY', dt_body), patch.object(loop, 'DT_NEURAL', dt_neural), patch.object(mujoco, 'mj_step', witnessed_step):
                result, trajectory, controls, spikes = loop.episode(model, pose, joints, geoms,
                    built, manifest, groups, arm, .1, replay=replay,
                    external_touch=False, disabled_channels=OMIT, push=push if perturb else None)
            observed_forces = np.asarray(forces)
            expected = np.zeros_like(observed_forces)
            if perturb:
                expected[round(.001/dt_body):round(.002/dt_body)] = push['force_model']
            np.testing.assert_array_equal(observed_forces, expected)
            checks[name+'_force_window_exact'] = True
            checks[name+'_impulse_exact'] = bool(np.allclose(observed_forces.sum(axis=0)*dt_body,
                np.asarray(push['force_model'])*.001 if perturb else np.zeros(3), rtol=1e-12, atol=1e-15))
            result.update(body_dt_s=dt_body, neural_dt_s=dt_neural, experiment=name,
                          motor_spike_total=sum(r['motor_spikes'] for r in result['records']),
                          body_metrics=pose_metrics(model, trajectory, pose))
            episodes[name] = result
            for key, value in [('trajectory', trajectory), ('controls', controls), ('spike_counts', spikes), ('applied_force', observed_forces)]:
                arrays[name+'_'+key] = value
            print(name, 'motor', result['motor_spike_total'], 'escape', result['escape_time_s'],
                  'height', result['final_height_model'], flush=True)
        baseline = episodes[clock+'_unperturbed']
        feedback = episodes[clock+'_feedback']
        replay = episodes[clock+'_replay']
        times = [r['time_start_s'] for r in feedback['records']]
        end_times = [r['time_end_s'] for r in feedback['records']]
        input_b = [r['actual_network_input_by_channel'] for r in baseline['records']]
        input_f = [r['actual_network_input_by_channel'] for r in feedback['records']]
        first_input = next((float(t) for x,y,t in zip(input_b,input_f,times) if x != y), None)
        comparisons[clock] = dict(
            perturb_vs_unperturbed_first_input_start_s=first_input,
            perturb_vs_unperturbed_first_motor_count_interval_start_s=first_difference(
                [r['motor_spikes'] for r in baseline['records']], [r['motor_spikes'] for r in feedback['records']], times),
            feedback_vs_unperturbed_first_applied_control_start_s=first_difference(
                arrays[clock+'_unperturbed_controls'], arrays[clock+'_feedback_controls'], times),
            feedback_vs_replay_first_body_sample_end_s=first_difference(
                arrays[clock+'_feedback_trajectory'][:, :model.nq+model.nv], arrays[clock+'_replay_trajectory'][:, :model.nq+model.nv], end_times),
            feedback_minus_replay_final_height_model=feedback['final_height_model']-replay['final_height_model'],
            feedback_minus_replay_final_com_displacement_model=(np.asarray(feedback['body_metrics'][-1]['com_displacement_model'])-np.asarray(replay['body_metrics'][-1]['com_displacement_model'])).tolist(),
            feedback_minus_replay_final_rotation_rad=feedback['body_metrics'][-1]['root_rotation_from_initial_rad']-replay['body_metrics'][-1]['root_rotation_from_initial_rad'],
            interpretation='Descriptive perturbation response, not recovery: no stable initial equilibrium or converged trajectory established.')
        checks[clock+'_replay_uses_unperturbed_commands'] = bool(np.array_equal(arrays[clock+'_replay_controls'], arrays[clock+'_unperturbed_controls']))
        checks[clock+'_sensory_cut_zero_spikes'] = not bool(arrays[clock+'_sensory_cut_spike_counts'].any())
    checks['frozen_xml_hash'] = loop.cp.sha256_file(loop.cp.CANONICAL_XML) == loop.cp.CANONICAL_XML_SHA256
    assert all(checks.values()), checks
    np.savez_compressed(dest/'perturbation_traces.npz', **arrays)
    (dest/'neural_identity.json').write_text(json.dumps(dict(tier_root_ids=list(map(str,built['tier']['root_ids'])), sensory_groups=groups, motor_manifest=manifest), indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, (clock,_,_) in enumerate(CLOCKS):
        for label in ('unperturbed','feedback','replay','sensory_cut','motor_cut'):
            s=episodes[clock+'_'+label]
            times=[r['time_end_s']*1000 for r in s['records']]
            axes[0,col].plot(times,[r['height_mm_declared'] for r in s['records']],label=label)
            axes[1,col].plot(times,[np.linalg.norm(r['com_displacement_model'][:2]) for r in s['body_metrics']],label=label)
        axes[0,col].set_title(clock);axes[0,col].legend(fontsize=8)
        axes[1,col].set_xlabel('Time (ms)')
    axes[0,0].set_ylabel('Thorax height (model mm)'); axes[1,0].set_ylabel('Horizontal COM displacement (model mm)')
    fig.suptitle('Manual prototype — matched 1–2ms physical push; NOT recovery acceptance')
    fig.tight_layout();fig.savefig(dest/'perturbation_response.png',dpi=150);plt.close(fig)
    sources={}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(x in ('venv','venv_body','vendor') for x in Path(p).parts):
            sources[str(Path(p).resolve())]=loop.cp.sha256_file(p)
    sources[str(Path(__file__).resolve())]=loop.cp.sha256_file(__file__)
    build_source_snapshot(sources,dest/'source_snapshot.zip')
    report=dict(label='手工研究原型，不是成品',status='MEASURED_PERTURBATION_RESPONSE_NOT_RECOVERY',
        protocol_path=str(HERE/'AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md'), protocol_sha256=loop.cp.sha256_file(HERE/'AUTONOMOUS_NEXT_PROTOCOL.zh-CN.md'),
        initial_gate=initial_gate, projection=projection, push=push, clocks=CLOCKS,
        episodes=episodes,comparisons=comparisons,checks=checks,source_hashes=sources,
        limits=['Body and neural time refinements are not yet converged.', 'Zero-floor compiled derivative, frozen XML unchanged.',
                'Claw/hook signs/gains, motor pooling, receptor signs remain uncalibrated assumptions.',
                'Replay is same physical push with baseline commands, not a neural drive bypass in the qualifying feedback arm.',
                'Input/control/body comparisons are 5ms sampled; first motor count difference is not first individual spike difference.',
                'Initial scalar limits/contact gate passed but not an equilibrium or sourced biological standing posture.'],wall_s=time.monotonic()-start)
    (dest/'perturbation_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('OUTPUT',dest,flush=True)
    return dest


if __name__=='__main__':
    with loop.mg.guarded_run('projected_perturbation_response',**loop.mg.BUDGETS['neural_muscle_closure']):
        main()
