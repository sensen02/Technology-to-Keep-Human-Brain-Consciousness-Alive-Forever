"""Bounded physical perturbation experiment using the frozen closure runner unchanged.
No parameter search, CPG, body/reference change, or standing-benefit inference.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import run_neural_muscle_closure as clo
from engine.embodied.closure_bundle import validate_bundle

EXPECTED_RUN = 'f10c9354-8b21-4faa-b48c-80fee6a5c157'
OUT = clo.OUT
ARMS = ('baseline', 'push_normal', 'push_replay', 'push_sensory_cut')
CONTROL_ATOL = 1e-12
MECHANICAL_ATOL = 1e-9  # model length units; reporting resolution, not biological significance


def sha(path):
    return clo.cp.sha256_file(Path(path))


def digest_arrays(arrays):
    h = hashlib.sha256()
    for x in arrays:
        a = np.ascontiguousarray(x)
        h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def first_difference(a, b, times, atol=0):
    delta = np.abs(np.asarray(a, float)-np.asarray(b, float))
    mask = np.any(delta.reshape(len(delta), -1) > atol, axis=1)
    ids = np.flatnonzero(mask)
    return dict(atol=atol, rtol=0, max_abs=float(delta.max()),
                first_index=int(ids[0]) if len(ids) else None,
                first_time_s=float(times[ids[0]]) if len(ids) else None)


@contextmanager
def witness(model, motor):
    """Only copy simulator inputs/state; call original step exactly once, unmodified."""
    import mujoco
    original_body = mujoco.mj_step
    original_neural = clo.ConductanceNetwork.step
    rows = {k: [] for k in ('state', 'time_s', 'xfrc', 'ctrl', 'motor_raster', 'currents')}
    def body(m, d, *args, **kwargs):
        assert m is model
        rows['state'].append(np.r_[d.qpos.copy(), d.qvel.copy(), d.act.copy()])
        rows['time_s'].append(float(d.time))
        rows['xfrc'].append(d.xfrc_applied.copy())
        rows['ctrl'].append(d.ctrl.copy())
        rows['data'] = d
        return original_body(m, d, *args, **kwargs)
    def neural(net, *args, **kwargs):
        if not rows['motor_raster']:
            rows['initial_neural_sha256'] = digest_arrays([net.v, net.ge, net.gi,
                net.adaptation, net.refractory_until_ms, net.spike_count, net._ring,
                np.array([net._ring_i, net.steps, net.t_ms])])
        if net.steps % 10 == 0:
            rows['currents'].append(np.asarray(kwargs['current_nA']).copy())
        spikes = original_neural(net, *args, **kwargs)
        rows['motor_raster'].append(spikes[motor].copy())
        return spikes
    mujoco.mj_step = body
    clo.ConductanceNetwork.step = neural
    try:
        yield rows
    finally:
        mujoco.mj_step = original_body
        clo.ConductanceNetwork.step = original_neural


def capture(model, args, arm, motor, push=None, replay=None):
    import mujoco
    with witness(model, motor) as w:
        summary, trajectory, controls, counts = clo.episode(model, *args, arm, .05,
            push=push, replay=replay)
    d = w.pop('data')
    w['state'].append(np.r_[d.qpos.copy(), d.qvel.copy(), d.act.copy()])
    w['time_s'].append(float(d.time))
    neural_hash = w.pop('initial_neural_sha256')
    arrays = {k: np.asarray(v) for k,v in w.items()}
    arrays.update(trajectory=trajectory, controls=controls, spike_counts=counts)
    scratch = mujoco.MjData(model)
    thorax = model.body('Thorax').id
    root, com = [], []
    for state in arrays['state']:
        scratch.qpos[:] = state[:model.nq]
        # Pure geometry on separate MjData, never forward the live integration state.
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        root.append(scratch.xpos[thorax].copy())
        com.append((scratch.xipos*model.body_mass[:, None]).sum(axis=0)/model.body_mass.sum())
    arrays.update(root_xyz=np.asarray(root), com_xyz=np.asarray(com))
    assert len(arrays['state']) == 501
    assert all(np.isfinite(x).all() for x in arrays.values())
    assert not any(any(r['mujoco_warning_counts']) for r in summary['records'])
    return summary, arrays, neural_hash


def verify_force(arrays, model, push):
    thorax = model.body('Thorax').id
    expected = np.zeros_like(arrays['xfrc'])
    if push:
        expected[10:20, thorax, :3] = push['force_model']
    np.testing.assert_array_equal(arrays['xfrc'], expected)
    np.testing.assert_allclose(arrays['time_s'], np.arange(501)*clo.DT_BODY, rtol=0, atol=1e-14)
    impulse = arrays['xfrc'][:, :, :3].sum(axis=(0,1))*clo.DT_BODY
    expected_impulse = np.asarray(push['force_model'])*.001 if push else np.zeros(3)
    np.testing.assert_allclose(impulse, expected_impulse, rtol=1e-14, atol=0)
    return dict(exact_xfrc_array=True, active_step_indices=np.flatnonzero(np.any(expected != 0, axis=(1,2))).tolist(),
                force_model=push['force_model'] if push else [0.,0.,0.],
                impulse_model=impulse.tolist(), expected_impulse_model=expected_impulse.tolist())


def main(guard_report=None):
    started = time.monotonic()
    bundle = validate_bundle(OUT)
    assert bundle['run_id'] == EXPECTED_RUN
    source = Path(bundle['path'])
    result = json.loads((source/'closure_results.json').read_text())
    identity = json.loads((source/'neural_identity.json').read_text())
    manifest = json.loads((source/'motor_manifest.json').read_text())['measured']
    assert result['measured']['external_perturbation'] is None
    assert result['measured']['duration_s'] == .05
    hashes = result['provenance']['code_hashes']
    assert all(sha(p) == h for p,h in hashes.items())
    inputs = {str(p): sha(p) for p in (clo.POOLS, clo.LOOKUP, clo.HERE/'data/flywire/banc_connectome.npz',
                                     clo.HERE/'data/banc888/banc_888_meta.feather')}
    for p, key in zip(inputs, ('pool_result_sha256','sensory_lookup_sha256','connectome_sha256','source_feather_sha256')):
        assert inputs[p] == result['provenance'][key]
    model, seated, gate, joints, geom_leg = clo.prepare_body()
    built = clo.build_neural_tier(clo.TierConfig(background_mean_nA=0, background_sd_nA=0))
    assert list(map(int, built['tier']['root_ids'])) == identity['tier_root_ids']
    groups = identity['sensory_groups']
    assert clo.channel_index_map(int(built['n']), built['tier']['root_ids'])[0] == groups
    motor = np.asarray(sorted({i for r in manifest['rows'] for i in r['local_indices']}))
    weight = float(model.body_mass.sum()*abs(model.opt.gravity[2]))
    push = dict(start_s=.001, end_s=.002, weight_fraction=.1, total_weight_model=weight,
                force_model=[.1*weight, 0., 0.])
    args = (seated, joints, geom_leg, built, manifest, groups)
    summaries, arrays, neural_hashes, forces = {}, {}, {}, {}
    for label, arm in zip(ARMS, ('normal','normal','replay','sensory_cut')):
        summaries[label], arrays[label], neural_hashes[label] = capture(model, args, arm, motor,
            push=None if label == 'baseline' else push,
            replay=arrays['baseline']['controls'] if label == 'push_replay' else None)
        forces[label] = verify_force(arrays[label], model, None if label == 'baseline' else push)
        print(label, 'escape_s', summaries[label]['escape_time_s'], flush=True)
    baseline = arrays['baseline']; normal = arrays['push_normal']; replay = arrays['push_replay']
    reference_checks = {}
    with np.load(source/'closure_traces.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(seated, z['seated_qpos'])
        for old in ('normal', 'normal_repeat'):
            for suffix in ('trajectory','controls','spike_counts'):
                np.testing.assert_array_equal(baseline[suffix], z[old+'_'+suffix])
                reference_checks[old+'_'+suffix] = True
    for label in ('baseline','push_normal','push_replay'):
        np.testing.assert_array_equal(arrays[label]['state'][:11], baseline['state'][:11])
        np.testing.assert_array_equal(arrays[label]['motor_raster'][:10], baseline['motor_raster'][:10])
        np.testing.assert_array_equal(arrays[label]['currents'][0], baseline['currents'][0])
    assert len(set(neural_hashes.values())) == 1
    np.testing.assert_array_equal(replay['controls'], baseline['controls'])
    for label in ARMS:
        np.testing.assert_array_equal(arrays[label]['ctrl'], np.repeat(arrays[label]['controls'], 50, axis=0))
        for k,r in enumerate(summaries[label]['records']):
            assert r['applied_source_interval'] == k-1
            assert abs(r['decoder']['valid_from_s'] - (k+1)*clo.DT_COMMAND) < 1e-14
    for label in ARMS[2:]:
        np.testing.assert_array_equal(arrays[label]['xfrc'], normal['xfrc'])
    timing = {}
    for name, a, b, ts in (
        ('sensory_current',normal['currents'],baseline['currents'],np.arange(10)*.005),
        ('motor_spike',normal['motor_raster'],baseline['motor_raster'],np.arange(1,101)*.0005),
        ('applied_command',normal['controls'],replay['controls'],np.arange(10)*.005)):
        timing[name] = dict(exact=first_difference(a,b,ts), above_1e_12=first_difference(a,b,ts,CONTROL_ATOL))
    ci = timing['applied_command']['above_1e_12']['first_index']
    timing['first_changed_command_source'] = None if ci is None else dict(
        source_interval=ci-1, source_start_s=(ci-1)*.005, source_end_s=ci*.005,
        command_valid_from_s=ci*.005, applied_start_s=ci*.005)
    if ci is not None:
        assert timing['motor_spike']['exact']['first_time_s'] <= ci*.005
        assert timing['sensory_current']['above_1e_12']['first_time_s'] >= .001
    comparisons = {}
    for left,right in (('push_normal','baseline'), ('push_normal','push_replay'), ('push_normal','push_sensory_cut')):
        comparisons[left+'_minus_'+right] = {}
        for field in ('root_xyz','com_xyz'):
            a,b = arrays[left][field],arrays[right][field]
            comparisons[left+'_minus_'+right][field] = dict(
                exact=first_difference(a,b,normal['time_s']),
                above_length_tolerance=first_difference(a,b,normal['time_s'],MECHANICAL_ATOL),
                endpoint_delta_model=(a[-1]-b[-1]).tolist())
    mechanics = {label:dict(root_initial_model=a['root_xyz'][0].tolist(),root_final_model=a['root_xyz'][-1].tolist(),
        com_initial_model=a['com_xyz'][0].tolist(),com_final_model=a['com_xyz'][-1].tolist(),
        root_min_z_model=float(a['root_xyz'][:,2].min()),com_min_z_model=float(a['com_xyz'][:,2].min()),
        root_displacement_model=(a['root_xyz'][-1]-a['root_xyz'][0]).tolist(),
        com_displacement_model=(a['com_xyz'][-1]-a['com_xyz'][0]).tolist(),
        escape_time_s=summaries[label]['escape_time_s']) for label,a in arrays.items()}
    assert validate_bundle(OUT) == bundle
    assert all(sha(p)==h for p,h in {**hashes,**inputs}.items())
    traces = OUT/'early_feedback_probe.npz'
    flat = {label+'_'+k:v for label,a in arrays.items() for k,v in a.items()}
    flat['motor_local_indices'] = motor
    np.savez_compressed(traces, **flat)
    guard = dict(guard_report('completed')) if guard_report else None
    if guard and not np.isfinite(guard['previous_soft_limit_mb']):
        guard['previous_soft_limit_mb'] = 'unlimited'
    report = dict(schema_version=1,status='PASS',
        status_meaning='Source identity, exact physical scheduling and replay invariants passed; not useful control acceptance',
        source_bundle=bundle, source_manifest_sha256=sha(source/'manifest.json'),
        provenance=dict(code=str(Path(__file__).resolve()),code_sha256=sha(__file__),source_code_hashes=hashes,inputs=inputs),
        protocol=dict(duration_s=.05,arms=list(ARMS),body_dt_s=.0001,neural_dt_s=.0005,command_dt_s=.005,
            push=push,force_target='Thorax world +x, no applied torque',
            unchanged_native_physics=True,unchanged_neural_and_translation_parameters=True,
            external_touch='Original runner artificial LF touch at 5–15ms retained; cut zeros all input. Strict bristle motor path disconnected.',
            replay_source='BASELINE no-push applied command trace, not pushed normal trace',
            witness='All 500 actual mj_step inputs, 501 physical states at 0..50ms; original step called once unchanged',
            root='Thorax body origin; equals free root position',
            com='Mass-weighted all body inertial centers; scratch qpos kinematics/comPos only',
            clock='Neural interval integrated before same-index body interval using sensory sample at interval START; decoded output valid only next interval',
            tolerances=dict(control_absolute=CONTROL_ATOL,length_model_absolute=MECHANICAL_ATOL,relative=0,
                meaning='Predeclared reporting resolution, not biological significance; exact bit differences reported separately')),
        invariants=dict(reference_arrays_exact=reference_checks,initial_neural_sha256=neural_hashes,
            prepush_body_substeps_0_through_10_exact_baseline_normal_replay=True,
            prepush_neural_first_interval_exact_normal_replay=True,
            sensory_cut_not_required_to_match_prepush_neural_or_body=True,
            replay_controls_equal_baseline=True,all_three_push_xfrc_identical=True,
            finite_all_arrays=True,zero_mujoco_warnings=True,source_files_unchanged=True),
        force_witness=forces,command_causality=timing,mechanical_outcomes=mechanics,comparisons=comparisons,
        interpretation=dict(commands_respond_above_tolerance=ci is not None,
            classification='state_contingent_response' if ci is not None else 'no_command_response_above_declared_tolerance',
            usefulness='not_identified; report net mechanical outcomes only; one perturbation/initial state, no dose search',
            endpoint_timing='Escape is only sampled at body-step ends; no earlier continuous-time crossing claimed'),
        limitations=['Native positive control floor preserved, strict neural-only actuation not established.',
            'Channel transduction and neural parameters unvalidated; no biological control inference.',
            'Sensory-cut removes all inputs from time zero: not an isolated post-push intervention.',
            '50ms single sample cannot establish corrective benefit, standing, robustness or statistical significance.',
            'Model length units declared mm; force/mass SI calibration unresolved.'],
        traces=dict(path=str(traces),sha256=sha(traces),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in flat.items()}),
        memory_guard=guard,wall_s=time.monotonic()-started)
    assert report['wall_s'] < 60
    (OUT/'early_feedback_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(wall_s=report['wall_s'],timing=timing,mechanics=mechanics,comparisons=comparisons),indent=2))
    return report


if __name__ == '__main__':
    with clo.mg.guarded_run('early_feedback_probe', **clo.mg.BUDGETS['neural_muscle_closure']) as guard:
        main(guard)
