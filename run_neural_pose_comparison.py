"""Explicit 2x2 pose comparison; observational instrumentation of frozen closure episode.
Hand-built experiment, not a standing controller. Never loads source activation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import run_neural_muscle_closure as closure
from engine.embodied.initial_pose_contract import model_limit_report

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
PRIOR = OUT / 'runs/f10c9354-8b21-4faa-b48c-80fee6a5c157'
SOURCE = HERE / 'outputs/workspace_recheck/pose_support_search.json'
EXPECTED_CHANGED = [2, 8, 11, 23, 36, 42]
BODY_FIELDS = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'act', 'act_dot', 'ctrl',
               'actuator_force', 'qfrc_actuator', 'qfrc_constraint', 'qfrc_passive',
               'qfrc_bias', 'qfrc_applied', 'xfrc_applied', 'efc_force')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_poses():
    with np.load(PRIOR / 'closure_traces.npz') as saved:
        original = saved['seated_qpos'].copy()
    rows = json.loads(SOURCE.read_text())['rows']
    row, = [r for r in rows if r['label'] == 'baseline_exact_source']
    repaired = np.asarray(row['qpos'], dtype=float)  # deliberately ONLY qpos
    if np.flatnonzero(original != repaired).tolist() != EXPECTED_CHANGED:
        raise AssertionError('unexpected pose modifications')
    return original, repaired


def native_metadata(model, joints, geom_leg):
    fields = ('jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_limited', 'jnt_range',
              'jnt_solref', 'jnt_solimp', 'actuator_ctrlrange', 'actuator_ctrllimited',
              'actuator_gainprm', 'actuator_biasprm', 'geom_friction', 'geom_solref', 'geom_solimp')
    return dict(array_hashes={n: digest(getattr(model, n)) for n in fields},
                joints=joints, geom_leg={str(k): v for k, v in geom_leg.items()},
                mapping_sha256=json_hash(dict(joints=joints, geom_leg=geom_leg)),
                timestep_s=float(model.opt.timestep), gravity=model.opt.gravity.tolist())


def passive_constraints(model, d):
    import mujoco
    rows = []
    for kind in np.unique(d.efc_type):
        mask = np.asarray(d.efc_type) == kind
        force = np.where(mask, d.efc_force, 0.)
        generalized = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, d, generalized, force)
        rows.append(dict(type_id=int(kind), type_name=mujoco.mjtConstraint(int(kind)).name,
                         row_count=int(mask.sum()), row_force_model=d.efc_force[mask].tolist(),
                         generalized_force_model=generalized.tolist()))
    return dict(nefc=int(d.nefc), by_type=rows, qacc=d.qacc.tolist(),
                qfrc_constraint=d.qfrc_constraint.tolist(),
                note='Initial passive mj_forward with qvel/act/ctrl=0; native positive ctrl floor intact. Constraint row forces and generalized components reported separately; no mixed torque/force ratio.')


def run_arm(model, pose, joints, geom_leg, built, manifest, groups, arm):
    import mujoco
    snapshots, metrics, warnings, finite_rows = [], [], [], []
    initial = {}
    raw_counts = []
    raw_motor_first = None
    neural_health = {}
    decoder_health = {}
    motor = sorted({i for r in manifest['rows'] for i in r['local_indices']})
    floor = closure.cp.floor_geom_id(model)
    feet = set(closure.cp.tarsus_geom_ids(model))
    real_forward = mujoco.mj_forward
    real_network = closure.ConductanceNetwork
    real_decoder = closure.MotorOutput

    def check_arrays(obj, health):
        for name, value in vars(obj).items():
            if isinstance(value, np.ndarray):
                health[name] = health.get(name, True) and bool(np.isfinite(value).all())

    class ObservedNetwork(real_network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            check_arrays(self, neural_health)

        def step(self, *args, **kwargs):
            nonlocal raw_motor_first
            sp = super().step(*args, **kwargs)
            check_arrays(self, neural_health)
            raw_counts.append(int(sp.sum()))
            if raw_motor_first is None and sp[motor].any():
                raw_motor_first = self.t_ms / 1000
            return sp

    class ObservedDecoder(real_decoder):
        def consume(self, *args, **kwargs):
            result = super().consume(*args, **kwargs)
            check_arrays(self, decoder_health)
            return result

    def observed_forward(m, d):
        real_forward(m, d)  # exactly the original single forward, never extra on live d
        if not snapshots:
            initial.update(contact_gate=closure.cp.stance_facts(m, d).as_dict(),
                           ranges=model_limit_report(m, d.qpos),
                           passive_constraints=passive_constraints(m, d),
                           qvel=d.qvel.tolist(), act=d.act.tolist(), ctrl=d.ctrl.tolist())
        report = model_limit_report(m, d.qpos)
        violations = report['violations']
        nonfoot = sum(floor in (int(c.geom1), int(c.geom2)) and
                      (int(c.geom2) if int(c.geom1) == floor else int(c.geom1)) not in feet
                      for c in d.contact)
        snapshots.append(d.qpos.copy())
        metrics.append([len(violations), max((v['outside_by'] for v in violations), default=0.), nonfoot])
        finite_rows.append([bool(np.isfinite(getattr(d, n)).all()) for n in BODY_FIELDS])
        warnings.append([int(w.number) for w in d.warning])

    with patch.object(mujoco, 'mj_forward', observed_forward), \
         patch.object(closure, 'ConductanceNetwork', ObservedNetwork), \
         patch.object(closure, 'MotorOutput', ObservedDecoder):
        summary, trajectory, controls, spikes = closure.episode(
            model, pose, joints, geom_leg, built, manifest, groups, arm, .05)
    qpos = np.asarray(snapshots)
    if len(qpos) != 501:
        raise AssertionError('must capture initial and exactly after each of 500 body steps')
    # All geometry recomputation is on SCRATCH data after the actual episode has finished.
    scratch = mujoco.MjData(model)
    heights = []
    thorax = model.body('Thorax').id
    for q in qpos:
        scratch.qpos[:] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        massweighted_com = np.sum(model.body_mass[:, None]*scratch.xipos, axis=0)/model.body_mass.sum()
        np.testing.assert_allclose(scratch.subtree_com[0], massweighted_com, rtol=1e-13, atol=1e-13)
        heights.append([q[2], scratch.xpos[thorax, 2], scratch.subtree_com[0, 2]])
    heights = np.asarray(heights)
    active_intervals = np.flatnonzero(np.any(controls != 0, axis=1))
    emitted_first = raw_motor_first if arm != 'spike_output_silence' else None
    finite_rows = np.asarray(finite_rows, dtype=bool)
    warnings = np.asarray(warnings)
    summary.update(initial=initial, first_raw_motor_spike_s=raw_motor_first,
                   first_emitted_motor_spike_s=emitted_first,
                   first_applied_nonzero_control_s=float(active_intervals[0]*closure.DT_COMMAND) if len(active_intervals) else None,
                   control_nonzero_entries=int(np.count_nonzero(controls)),
                   control_sum=float(controls.sum()), control_max=float(controls.max()),
                   total_raw_spikes=int(sum(raw_counts)), total_emitted_spikes=int(spikes.sum()),
                   total_emitted_motor_spikes=int(spikes[:, motor].sum()),
                   endpoint_step=500, endpoint_nominal_s=.05,
                   endpoint_heights_mm=dict(zip(('root', 'thorax', 'whole_body_com'), heights[-1].tolist())),
                   max_limit_violation_count=int(np.max(np.asarray(metrics)[:, 0])),
                   max_limit_excess_rad=float(np.max(np.asarray(metrics)[:, 1])),
                   max_nonfoot_floor_contacts=int(np.max(np.asarray(metrics)[:, 2])),
                   health=dict(body_fields=dict(zip(BODY_FIELDS, finite_rows.all(axis=0).tolist())),
                               neural_live_arrays=neural_health, decoder_live_arrays=decoder_health,
                               mujoco_warning_max=warnings.max(axis=0).tolist(),
                               all_live_states_finite=bool(finite_rows.all() and all(neural_health.values()) and all(decoder_health.values()))))
    return summary, dict(trajectory=trajectory, controls=controls, spike_counts=spikes,
                         body_qpos=qpos, body_metrics=np.asarray(metrics), heights_mm=heights,
                         body_finite=finite_rows, body_warning_counts=warnings,
                         neural_raw_spike_counts=np.asarray(raw_counts))


def main(guard_report):
    import mujoco
    started = time.time()
    prior_doc = json.loads((PRIOR / 'closure_results.json').read_text())
    frozen_files = prior_doc['provenance']['code_hashes']
    for path, expected in frozen_files.items():
        if closure.cp.sha256_file(path) != expected:
            raise AssertionError(f'frozen source changed: {path}')
    original, repaired = source_poses()
    model, prepared, _, joints, geom_leg = closure.prepare_body()
    assert np.array_equal(original, prepared)
    assert len(model_limit_report(model, original)['violations']) == 5
    assert model_limit_report(model, repaired)['within_declared_scalar_ranges']
    native = native_metadata(model, joints, geom_leg)
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    prior_manifest = json.loads((PRIOR / 'motor_manifest.json').read_text())['measured']
    assert manifest == prior_manifest
    identity = json.loads((PRIOR / 'neural_identity.json').read_text())
    assert [int(x) for x in built['tier']['root_ids']] == identity['tier_root_ids']
    assert {leg: {ch: [int(i) for i in ids] for ch, ids in row.items()} for leg, row in groups.items()} == identity['sensory_groups']
    episodes, arrays, checks = {}, dict(original_qpos=original, repaired_qpos=repaired,
                                        body_step=np.arange(501), body_time_s=np.arange(501)*closure.DT_BODY), {}
    with np.load(PRIOR / 'closure_traces.npz') as prior:
        for pose_label, pose in (('original', original), ('repaired', repaired)):
            for arm in ('normal', 'spike_output_silence'):
                label = pose_label + '_' + arm
                summary, result = run_arm(model, pose, joints, geom_leg, built, manifest, groups, arm)
                summary['pose_label'] = pose_label
                episodes[label] = summary
                arrays.update({label+'_'+k: v for k, v in result.items()})
                if pose_label == 'original':
                    for key in ('trajectory', 'controls', 'spike_counts'):
                        checks[label+'_'+key+'_prior_exact'] = bool(np.array_equal(result[key], prior[arm+'_'+key]))
                print(label, 'escape', summary['escape_time_s'], 'heights', summary['endpoint_heights_mm'], flush=True)
    assert all(checks.values()), checks
    assert native_metadata(model, joints, geom_leg) == native
    for path, expected in frozen_files.items():
        assert closure.cp.sha256_file(path) == expected
    assert closure.cp.sha256_file(closure.cp.CANONICAL_XML) == closure.cp.CANONICAL_XML_SHA256
    checks.update(frozen_source_hashes_unchanged=True, native_metadata_unchanged=True,
                  mapping_reused=True, prior_motor_manifest_exact=True, prior_neural_identity_exact=True,
                  initial_repaired_zero_range_violations=True, pose_changes_restricted_to_five_hinges_and_root=True)
    output = OUT / 'neural_pose_comparison.npz'
    np.savez_compressed(output, **arrays)
    doc = dict(schema_version=1, status='PASS', execution_status='completed', duration_s=.05,
               body_steps=500, checks=checks, episodes=episodes,
               schema=dict(body_metrics_columns=['limit_violation_count', 'max_excess_rad', 'nonfoot_floor_contact_count'],
                           heights_mm_columns=['root', 'thorax', 'whole_body_com'], body_finite_columns=list(BODY_FIELDS),
                           body_sampling='0..500 inclusive, after each original mj_step + mj_forward; 0.1 ms spacing',
                           trajectory='original episode concatenated qpos/qvel/act at 5ms endpoints',
                           neural_raw_spike_counts='100 neural steps, raw network counts before output silence'),
               provenance=dict(prior_bundle=str(PRIOR), prior_npz_sha256=closure.cp.sha256_file(PRIOR/'closure_traces.npz'),
                               repaired_source=str(SOURCE), repaired_source_sha256=closure.cp.sha256_file(SOURCE),
                               repaired_row='baseline_exact_source', source_activation_used=False,
                               initialization='Both poses: qvel=act=ctrl=0. Original is unmodified round37 (5 limit violations); repaired is explicit saved five-hinge range repair plus reseated root (0 initial violations).',
                               original_qpos_sha256=digest(original), repaired_qpos_sha256=digest(repaired),
                               changed_qpos_addresses=EXPECTED_CHANGED,
                               proprioceptive_reference='Each pose is its own seated reference: this explicitly changes the pose-derived multi-joint proprioceptive reference; tibia reference is unchanged.',
                               gain_or_reference_changes='No gain or reference changes except pose-derived seated proprioceptive reference. No CPG, PD, optimized activation, bias, external push or stabilizer.',
                               identical_external_touch='LF bristle +0.3 nA at command intervals starting 5 and 10 ms, exactly as prior experiment',
                               native_positive_ctrl_floor_preserved=True, native_model_and_mapping=native,
                               canonical_xml_sha256=closure.cp.sha256_file(closure.cp.CANONICAL_XML),
                               frozen_source_hashes=frozen_files, runner_sha256=closure.cp.sha256_file(Path(__file__)),
                               same_neural_config=prior_doc['provenance'], npz_sha256=closure.cp.sha256_file(output),
                               live_body_untouched_by_observer=True, heights_recomputed_on_scratch_from_saved_qpos=True),
               limitations=['Manual experimental implementation, not a finished autonomous controller.',
                            '50ms paired comparison cannot establish long-term standing stability or autonomy.',
                            'Pose repair also changes geometry/contact/root seating and proprioceptive reference; cannot isolate limit-force causation.',
                            'Native positive ctrl floor is retained; zero command is not zero physical muscle drive.',
                            'Constraint generalized forces mix translation and rotation coordinates; no combined torque/force ratio is calculated.'],
               wall_s=time.time()-started, memory_guard=guard_report('completed'))
    if not np.isfinite(doc['memory_guard']['previous_soft_limit_mb']):
        doc['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / 'neural_pose_comparison.json').write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n')
    print('COMPLETED', json.dumps(checks), flush=True)


if __name__ == '__main__':
    with closure.mg.guarded_run('neural_pose_comparison', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
