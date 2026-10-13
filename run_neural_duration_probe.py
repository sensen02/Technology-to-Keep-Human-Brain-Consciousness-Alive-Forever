"""Bounded, hand-built 200 ms NORMAL-only durability probe; no search or tuning.
Observers wrap the frozen episode without changing its numerical operations.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import run_neural_pose_comparison as pose_runner
closure = pose_runner.closure
model_limit_report = pose_runner.model_limit_report
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
DURATION = .2
BODY_FIELDS = pose_runner.BODY_FIELDS


class UnsafeState(RuntimeError):
    pass


def enforce_safe(finite, warnings, angular, translational):
    if not finite:
        raise UnsafeState('nonfinite live state after step')
    if any(warnings):
        raise UnsafeState('MuJoCo warning after step')
    # Gross numerical-safety bounds, NOT standing acceptance thresholds.
    if angular > 1e6 or translational > 1e6:
        raise UnsafeState('gross qvel safety bound exceeded')


def run_arm(model, pose, joints, geom_leg, built, manifest, groups):
    import mujoco
    snapshots, velocities, activations, controls_live = [], [], [], []
    metrics, finite_rows, warning_rows, neural_finite, raw_spikes = [], [], [], [], []
    neural_health, decoder_health, initial = {}, {}, {}
    motor = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    floor, feet = closure.cp.floor_geom_id(model), set(closure.cp.tarsus_geom_ids(model))
    forward, step = mujoco.mj_forward, mujoco.mj_step
    network, decoder = closure.ConductanceNetwork, closure.MotorOutput
    # Free-root translation versus root rotation + all scalar hinge velocities.
    translation_dofs = [0, 1, 2]
    angular_dofs = list(range(3, model.nv))

    def check_arrays(obj, health):
        values = {name: bool(np.isfinite(value).all()) for name, value in vars(obj).items()
                  if isinstance(value, np.ndarray)}
        for name, value in values.items():
            health[name] = health.get(name, True) and value
        return all(values.values())

    class ObservedNetwork(network):
        def step(self, *args, **kwargs):
            sp = super().step(*args, **kwargs)
            finite = check_arrays(self, neural_health)
            neural_finite.append(finite)
            raw_spikes.append(sp.copy())
            if not finite:
                raise UnsafeState('nonfinite actual neural state after neural step')
            return sp

    class ObservedDecoder(decoder):
        def consume(self, *args, **kwargs):
            result = super().consume(*args, **kwargs)
            if not check_arrays(self, decoder_health):
                raise UnsafeState('nonfinite decoder state')
            return result

    def observed_step(m, d):
        step(m, d)
        # Check immediately AFTER mj_step, before forward could mask/reset anything.
        enforce_safe(all(np.isfinite(getattr(d, n)).all() for n in BODY_FIELDS),
                     [int(w.number) for w in d.warning],
                     float(np.max(np.abs(d.qvel[angular_dofs]))),
                     float(np.max(np.abs(d.qvel[translation_dofs]))))

    def observed_forward(m, d):
        forward(m, d)
        report = model_limit_report(m, d.qpos)
        if not snapshots:
            initial.update(ranges=report, contact_gate=closure.cp.stance_facts(m, d).as_dict(),
                           qvel=d.qvel.tolist(), act=d.act.tolist(), ctrl=d.ctrl.tolist())
        violations = report['violations']
        nonfoot = sum(floor in (int(c.geom1), int(c.geom2)) and
                      (int(c.geom2) if int(c.geom1) == floor else int(c.geom1)) not in feet
                      for c in d.contact)
        snapshots.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        activations.append(d.act.copy())
        controls_live.append(d.ctrl.copy())
        metrics.append([len(violations), max((v['outside_by'] for v in violations), default=0.), nonfoot,
                        np.max(np.abs(d.qvel[angular_dofs])), np.max(np.abs(d.qvel[translation_dofs]))])
        finite_rows.append([bool(np.isfinite(getattr(d, n)).all()) for n in BODY_FIELDS])
        warning_rows.append([int(w.number) for w in d.warning])
        enforce_safe(all(finite_rows[-1]), warning_rows[-1], metrics[-1][3], metrics[-1][4])

    aborted = None
    with patch.object(mujoco, 'mj_step', observed_step), patch.object(mujoco, 'mj_forward', observed_forward), \
         patch.object(closure, 'ConductanceNetwork', ObservedNetwork), patch.object(closure, 'MotorOutput', ObservedDecoder):
        try:
            summary, trajectory, controls, spikes = closure.episode(
                model, pose, joints, geom_leg, built, manifest, groups, 'normal', DURATION)
        except (UnsafeState, FloatingPointError) as exc:
            aborted = str(exc)
            n = (len(snapshots)-1)//50
            trajectory = np.asarray([np.r_[snapshots[i], velocities[i], activations[i]] for i in range(50, n*50+1, 50)])
            controls = np.asarray([controls_live[i] for i in range(1, n*50+1, 50)])
            spikes = np.asarray(raw_spikes[:n*10]).reshape(n, 10, int(built['n'])).sum(axis=1)
            summary = dict(arm='normal', records=[])
    qpos, qvel = np.asarray(snapshots), np.asarray(velocities)
    scratch, heights = mujoco.MjData(model), []
    thorax = model.body('Thorax').id
    for q in qpos:
        scratch.qpos[:] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        heights.append([q[2], scratch.xpos[thorax, 2], scratch.subtree_com[0, 2]])
    heights, metrics = np.asarray(heights), np.asarray(metrics)
    def onset(mask):
        indices = np.flatnonzero(mask)
        return float(indices[0]*closure.DT_BODY) if len(indices) else None
    fall = onset(heights[:, 1] < .9*heights[0, 1])
    nonfoot = onset(metrics[:, 2] > 0)
    all_finite = bool(np.all(finite_rows) and all(neural_health.values()) and all(decoder_health.values()))
    standing = dict(completed_200ms=aborted is None and len(qpos)==2001,
                    no_bodyfall=fall is None, no_nonfoot_contacts=nonfoot is None,
                    all_steps_within_strict_limits=not bool(metrics[:, 0].any()),
                    all_live_states_finite=all_finite, no_mujoco_warnings=not bool(np.any(warning_rows)))
    total_counts, motor_counts = spikes.sum(axis=1), spikes[:, motor].sum(axis=1)
    def count_summary(counts):
        return dict(per_5ms=counts.tolist(), first_50ms=int(counts[:10].sum()),
                    final_50ms=int(counts[-10:].sum()), final_5ms=int(counts[-1]) if len(counts) else None,
                    minimum_5ms=int(counts.min()) if len(counts) else None,
                    recovery_after_minimum=bool(len(counts) and counts[-1] > counts.min()))
    summary.update(initial=initial, duration_s=DURATION, completed_body_steps=len(qpos)-1,
                   execution_status='unsafe_abort' if aborted else 'completed', abort_reason=aborted,
                   escape_time_s=fall, nonfoot_contact_onset_s=nonfoot,
                   max_nonfoot_floor_contacts=int(metrics[:,2].max()),
                   max_limit_violation_count=int(metrics[:,0].max()), max_limit_excess_rad=float(metrics[:,1].max()),
                   max_angular_qvel_rad_s=float(metrics[:,3].max()), max_translational_qvel_mm_s=float(metrics[:,4].max()),
                   endpoint_heights_mm=dict(zip(('root','thorax','whole_body_com'), heights[-1].tolist())),
                   standing_checks=standing, standing_status='PASS' if all(standing.values()) else 'FAIL',
                   neural_spike_counts=dict(all_neurons=count_summary(total_counts), motor_neurons=count_summary(motor_counts)),
                   health=dict(all_live_states_finite=all_finite, neural_live_arrays=neural_health,
                               decoder_live_arrays=decoder_health, mujoco_warning_max=np.max(warning_rows,axis=0).tolist()))
    initial_trajectory = np.r_[qpos[0], qvel[0], activations[0]]
    return summary, dict(trajectory=trajectory, trajectory_with_initial=np.vstack([initial_trajectory,trajectory]),
                        controls=controls, spike_counts=spikes, body_qpos=qpos, body_qvel=qvel,
                        body_metrics=metrics, heights_mm=heights, body_finite=np.asarray(finite_rows),
                        body_warning_counts=np.asarray(warning_rows), neural_state_finite=np.asarray(neural_finite),
                        neural_raw_spike_counts=np.asarray(raw_spikes).sum(axis=1),
                        body_time_s=np.arange(len(qpos))*closure.DT_BODY)


def main(guard_report):
    import mujoco
    started = time.time()
    current = json.loads((OUT/'current_run.json').read_text())
    bundle = OUT/current['path']
    current_doc = json.loads((bundle/'closure_results.json').read_text())
    files = dict(current_doc['provenance']['code_hashes'])
    for path in (Path(__file__), Path(pose_runner.__file__), Path(model_limit_report.__code__.co_filename)):
        files[str(path)] = closure.cp.sha256_file(path)
    for path, expected in files.items():
        assert closure.cp.sha256_file(path) == expected, path
    original, repaired = pose_runner.source_poses()
    model, prepared, _, joints, geom_leg = closure.prepare_body()
    assert np.array_equal(original, prepared)
    native = pose_runner.native_metadata(model,joints,geom_leg)
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),built['tier']['root_ids'],names)
    groups, _ = closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    episodes, arrays, checks = {}, dict(original_qpos=original,repaired_qpos=repaired), {}
    old_path = OUT/'neural_pose_comparison.npz'
    with np.load(old_path) as prior:
        for label, pose in (('original',original),('repaired',repaired)):
            key = label+'_normal'
            summary, result = run_arm(model,pose,joints,geom_leg,built,manifest,groups)
            episodes[key] = summary
            arrays.update({key+'_'+name:value for name,value in result.items()})
            for name in ('trajectory','controls','spike_counts'):
                checks[key+'_'+name+'_50ms_exact'] = bool(np.array_equal(result[name][:10],prior[key+'_'+name]))
            # Prior trajectory excludes t=0: augment it explicitly for the 11-row contract.
            zero_initial = np.r_[prior[key+'_body_qpos'][0],np.zeros(model.nv+model.na)]
            checks[key+'_11_trajectory_rows_exact'] = bool(np.array_equal(result['trajectory_with_initial'][:11],np.vstack([zero_initial,prior[key+'_trajectory']])))
            checks[key+'_501_body_qpos_exact'] = bool(np.array_equal(result['body_qpos'][:501],prior[key+'_body_qpos']))
            print(key,summary['standing_status'],summary['endpoint_heights_mm'],flush=True)
    checks['native_model_unchanged'] = native == pose_runner.native_metadata(model,joints,geom_leg)
    checks['source_hashes_unchanged'] = all(closure.cp.sha256_file(path)==value for path,value in files.items())
    checks['canonical_xml_unchanged'] = closure.cp.sha256_file(closure.cp.CANONICAL_XML)==closure.cp.CANONICAL_XML_SHA256
    output = OUT/'neural_duration_probe.npz'
    np.savez_compressed(output,**arrays)
    guard = guard_report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):
        guard['previous_soft_limit_mb']='unlimited'
    doc = dict(schema_version=1,status='PASS' if all(e['standing_status']=='PASS' for e in episodes.values()) and all(checks.values()) else 'FAIL',
               status_meaning='Strict standing durability, not interface execution; bodyfall/contact/limit violation is failure.',
               execution_status='completed' if all(e['execution_status']=='completed' for e in episodes.values()) else 'unsafe_abort',
               duration_s=DURATION, episodes=episodes, checks=checks,
               schema=dict(body_metrics_columns=['strict_limit_count','max_limit_excess_rad','nonfoot_floor_contacts','max_abs_angular_qvel_rad_s','max_abs_translational_qvel_mm_s'],
                           heights_mm_columns=['root','thorax','whole_body_com'],body_finite_columns=list(BODY_FIELDS),
                           body_sampling='Initial + every post-step forward, 0.1ms; immediate post-mj_step safety check also performed',
                           trajectory='40 post-command endpoints; trajectory_with_initial has 41 rows',
                           neural_sampling='400 actual neural steps at 0.5ms; spike_counts 40 x neuron, per 5ms',
                           angular_qvel_dofs=list(range(3,model.nv)),translational_qvel_dofs=[0,1,2]),
               provenance=dict(current_bundle=str(bundle),frozen_source_hashes=files,
                               numerical_prefix_artifact=str(old_path),numerical_prefix_sha256=closure.cp.sha256_file(old_path),
                               prior_provenance_not_required_current=True,
                               canonical_xml_sha256=closure.cp.sha256_file(closure.cp.CANONICAL_XML),
                               native_model_and_mapping=native,source_activation_used=False,
                               initialization='Each saved qpos; zero qvel/act/ctrl; each pose own proprioceptive reference',
                               gain_or_reference_changes='None except pose-derived reference; no tuning/search',
                               npz_sha256=closure.cp.sha256_file(output)),
               limitations=['Manual experimental implementation, not a finished controller.',
                            'Native positive ctrl floor unchanged; not strict neural-only physical actuation.',
                            '200ms is bounded durability evidence, not indefinite stability.',
                            'Recovery refers only to final spike count above its interval minimum, not postural recovery.'],
               memory_guard=guard,wall_s=time.time()-started)
    (OUT/'neural_duration_probe.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    assert all(checks.values()), checks
    print('COMPLETED strict standing:',doc['status'],'wall_s',doc['wall_s'],flush=True)


if __name__ == '__main__':
    def deadline(*_):
        raise TimeoutError('60-second process deadline; no other jobs touched')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(60)
    with closure.mg.guarded_run('neural_duration_probe',**closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
