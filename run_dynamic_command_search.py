"""Bounded constant-command offline mechanical diagnostics; never neural success."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import json
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import run_dynamic_support_audit as dyn
s, cp, mg = dyn.s, dyn.cp, dyn.mg
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/workspace_recheck/dynamic_command_search.json'
CROSS = HERE / 'outputs/workspace_recheck/pose_command_cross.json'
AMPLITUDE = .02


def orientation_angle(q, reference):
    q, reference = np.asarray(q, dtype=float), np.asarray(reference, dtype=float)
    q, reference = q/np.linalg.norm(q), reference/np.linalg.norm(reference)
    # atan2 is stable near identity and invariant to quaternion sign.
    w, v = q[0], q[1:]
    r, u = reference[0], reference[1:]
    relative_vector = r*v - w*u - np.cross(v, u)
    return float(2*np.arctan2(np.linalg.norm(relative_vector), abs(w*r + v@u)))


def score_terms(position, velocity, orientation, joints, length=1., gravity=9810.):
    """Position: RMS Euclidean displacement over 1..T ms, all axes; others terminal."""
    position, joints = np.asarray(position), np.asarray(joints)
    displacement = position[1:] - position[0]
    terms = dict(COM_position_RMS_over_L=float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))/length),
        final_COM_speed_over_sqrt_gL=float(np.linalg.norm(velocity[-1])/np.sqrt(gravity*length)),
        final_root_geodesic_over_1rad=orientation_angle(orientation[-1], orientation[0]),
        final_joint_RMS_over_1rad=float(np.sqrt(np.mean((joints[-1]-joints[0])**2))))
    return terms, float(sum(v*v for v in terms.values()))


def source_rows(doc, model, covered):
    a, b = dyn.select_rows(doc)
    if a['label'] != 'baseline_exact_source':
        raise ValueError('only exact nonlowered baseline pose permitted')
    frozen = cp.sha256_file(cp.CANONICAL_XML)
    if frozen != cp.CANONICAL_XML_SHA256 or any(doc['provenance'][k] != frozen for k in ('xml_sha256_before', 'xml_sha256_after')):
        raise ValueError('source XML freeze mismatch')
    if doc['actuator_order'] != [model.actuator(i).name for i in range(model.nu)]:
        raise ValueError('source actuator order mismatch')
    if not np.array_equal(doc['native_control_ranges'], model.actuator_ctrlrange):
        raise ValueError('source native bounds mismatch')
    if set(doc['covered_indices']) != set(covered) or len(covered) != 74:
        raise ValueError('source covered74 mismatch')
    excluded = np.setdiff1d(np.arange(model.nu), covered)
    for row in (a, b):
        activation = np.asarray(row['activation'])
        if not np.isfinite(activation).all() or np.any(activation < model.actuator_ctrlrange[:, 0]) or np.any(activation > model.actuator_ctrlrange[:, 1]):
            raise ValueError('saved source command violates original bounds')
        if not np.array_equal(activation[excluded], model.actuator_ctrlrange[excluded, 0]):
            raise ValueError('excluded source muscles not at native floors')
    s.validate_limits(model, np.asarray(a['qpos']))
    return a, b


def candidates(a, b, covered, bounds):
    a, b, covered, bounds = np.asarray(a), np.asarray(b), np.asarray(covered), np.asarray(bounds)
    raw = [('A_baseline', a.copy(), dict(kind='saved_A')),
           ('B_lowered_command_replayed', b.copy(), dict(kind='saved_B_command_only'))]
    for fraction in (.25, .5, .75):
        raw.append((f'blend_{fraction:g}', (1-fraction)*a+fraction*b, dict(kind='linear_blend', fraction_B=fraction)))
    # Three fixed sign patterns in covered manifest order, each plus and minus: no RNG or feedback.
    patterns = [np.ones(len(covered)), np.where(np.arange(len(covered)) % 2 == 0, 1., -1.),
                np.where((np.arange(len(covered))//2) % 2 == 0, 1., -1.)]
    for i, pattern in enumerate(patterns):
        for sign in (-1, 1):
            requested = a.copy()
            requested[covered] += sign*AMPLITUDE*pattern
            raw.append((f'perturb_pattern{i+1}_{"plus" if sign > 0 else "minus"}', requested,
                        dict(kind='bounded_perturbation', pattern=i+1, sign=sign, amplitude=AMPLITUDE)))
    result = []
    for label, requested, recipe in raw:
        activation = np.clip(requested, bounds[:, 0], bounds[:, 1])
        changes = [dict(index=int(i), requested=float(requested[i]), clipped=float(activation[i]),
                        change=float(activation[i]-requested[i])) for i in np.flatnonzero(activation != requested)]
        result.append(dict(label=label, recipe=recipe, requested_activation=requested.tolist(),
            requested_delta_from_A=(requested-a).tolist(), activation=activation.tolist(),
            actual_delta_from_A=(activation-a).tolist(), clipping_changes=changes,
            activation_sha256=dyn.audit.digest_array(activation)))
    return result


def integrate(model, qpos, activation, duration=.05, deadline=float('inf')):
    """Native step then forward exactly as crossed AA; read-only COM Jqvel at 1ms."""
    if not np.isclose(model.opt.timestep, .0001, rtol=0, atol=1e-15):
        raise ValueError('requires frozen 0.1ms native timestep')
    ev = s.FixedPose(model, qpos)
    activation = np.asarray(activation)
    ev.acceleration(activation)
    d = ev.data
    floor, feet = cp.floor_geom_id(model), set(cp.tarsus_geom_ids(model))
    covered, _, _ = s.coverage(model)
    excluded = np.setdiff1d(np.arange(model.nu), covered)
    trace = {k: [] for k in ('time_s', 'COM_position_mm', 'COM_velocity_mm_s', 'root_position_mm',
                             'root_quaternion', 'root_geodesic_rad', 'joint_angles_rad', 'joint_displacement_rad')}
    native = {k: [] for k in ('time_s', 'root_height_mm', 'contact_count', 'nonfoot_floor_contacts')}
    warnings = np.zeros(len(d.warning), dtype=int)
    finite, reset, unchanged, bounds_ok, floors_ok = True, False, True, True, True
    def health():
        nonlocal finite, unchanged, bounds_ok, floors_ok, warnings
        warnings = np.maximum(warnings, [w.number for w in d.warning])
        finite &= all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.act, d.ctrl, d.qacc, d.xipos))
        unchanged &= np.array_equal(d.ctrl, activation)
        bounds_ok &= bool(np.all(d.ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(d.ctrl <= model.actuator_ctrlrange[:, 1])
                          and np.all(d.act[ev.addresses] >= model.actuator_ctrlrange[:, 0]) and np.all(d.act[ev.addresses] <= model.actuator_ctrlrange[:, 1]))
        floors_ok &= np.array_equal(d.ctrl[excluded], model.actuator_ctrlrange[excluded, 0])
    def count_body():
        return sum(floor in (c.geom1, c.geom2) and (c.geom2 if c.geom1 == floor else c.geom1) not in feet for c in d.contact[:d.ncon])
    body_max = count_body()
    def record_native():
        for key, value in [('time_s', float(d.time)), ('root_height_mm', float(d.qpos[2])),
                           ('contact_count', int(d.ncon)), ('nonfoot_floor_contacts', int(count_body()))]:
            native[key].append(value)
    def sample():
        mass = model.body_mass[1:].sum()
        pos = (model.body_mass[1:, None]*d.xipos[1:]).sum(axis=0)/mass
        vel = dyn.com_jacobian(model, d) @ d.qvel
        values = dict(time_s=float(d.time), COM_position_mm=pos.tolist(), COM_velocity_mm_s=vel.tolist(),
            root_position_mm=d.qpos[:3].tolist(), root_quaternion=d.qpos[3:7].tolist(),
            root_geodesic_rad=orientation_angle(d.qpos[3:7], np.asarray(qpos)[3:7]),
            joint_angles_rad=d.qpos[7:].tolist(), joint_displacement_rad=(d.qpos[7:]-np.asarray(qpos)[7:]).tolist())
        for key, value in values.items():
            trace[key].append(value)
    health(); record_native(); sample()
    checkpoints = {}
    for step in range(int(round(duration/model.opt.timestep))):
        if time.monotonic() >= deadline:
            raise TimeoutError('bounded command search deadline exceeded')
        before = float(d.time)
        mujoco.mj_step(model, d)
        health(); body_max = max(body_max, count_body())
        reset |= d.time <= before
        mujoco.mj_forward(model, d)
        health(); body_max = max(body_max, count_body())
        record_native()
        if (step+1) % 10 == 0:
            sample()
        if step+1 == 500:
            checkpoints['50ms'] = dict(qpos=d.qpos.tolist(), activation=d.act.tolist())
        if not finite or reset:
            break
    completed = len(native['time_s'])-1 == round(duration/model.opt.timestep)
    reasons = [name for name, ok in [('finite', finite), ('no_automatic_reset', not reset), ('no_warnings', not warnings.sum()),
        ('no_nonfoot_floor_contacts_allsteps', body_max == 0), ('original_activation_bounds_allsteps', bounds_ok),
        ('excluded_native_floors', floors_ok), ('constant_command', unchanged), ('completed_duration', completed)] if not ok]
    metrics = None
    if finite and len(trace['time_s']) > 1:
        terms, score = score_terms(trace['COM_position_mm'], trace['COM_velocity_mm_s'], trace['root_quaternion'],
            trace['joint_angles_rad'], gravity=float(np.linalg.norm(model.opt.gravity)))
        p = np.asarray(trace['COM_position_mm']); v = np.asarray(trace['COM_velocity_mm_s'])
        metrics = dict(score=score, dimensionless_terms=terms,
            squared_terms={k: v*v for k, v in terms.items()}, final_COM_displacement_mm=(p[-1]-p[0]).tolist(),
            COM_displacement_RMS_by_axis_mm=np.sqrt(np.mean((p[1:]-p[0])**2, axis=0)).tolist(),
            final_COM_velocity_mm_s=v[-1].tolist(), final_COM_vz_mm_s=float(v[-1, 2]),
            signed_COM_vertical_drop_mm=float(p[0, 2]-p[-1, 2]),
            signed_root_vertical_drop_mm=float(native['root_height_mm'][0]-native['root_height_mm'][-1]),
            final_root_geodesic_rad=terms['final_root_geodesic_over_1rad'],
            final_joint_displacement_rad=trace['joint_displacement_rad'][-1])
    return dict(trace=trace, native_trace=native, metrics=metrics, eligible=not reasons, ineligibility_reasons=reasons,
        finite=bool(finite), automatic_reset_detected=bool(reset), warning_counts_by_index=warnings.tolist(),
        warning_count=int(warnings.sum()), command_unchanged=bool(unchanged), original_bounds_allsteps=bool(bounds_ok),
        excluded_native_floors=bool(floors_ok), max_nonfoot_floor_contacts=int(body_max), steps=len(native['time_s'])-1,
        activation_initialization_writes=1, final_qpos=d.qpos.tolist(), final_activation=d.act.tolist(), checkpoints=checkpoints)


def choose(rows):
    eligible = [r for r in rows if r['eligible'] and r['metrics'] is not None]
    return min(eligible, key=lambda r: r['metrics']['score']) if eligible else None


def prefix_check(short, long):
    checks = {group+'.'+key: dyn.bit_identical(values, long[group][key][:len(values)])
              for group in ('trace', 'native_trace') for key, values in short[group].items()}
    checks.update(final50_qpos=dyn.bit_identical(short['final_qpos'], long['checkpoints']['50ms']['qpos']),
                  final50_activation=dyn.bit_identical(short['final_activation'], long['checkpoints']['50ms']['activation']))
    return dict(all_bit_identical=all(checks.values()), fields=checks)


def main(report):
    start = time.monotonic(); deadline = start+55.
    sources = [dyn.SOURCE, CROSS, Path(s.__file__), Path(dyn.__file__), Path(dyn.audit.__file__), Path(cp.__file__), s.POOLS, s.MANIFEST]
    hashes = {str(p): cp.sha256_file(p) for p in sources}
    frozen = cp.sha256_file(cp.CANONICAL_XML)
    model = cp.load_model()
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('requires free root then hinges')
    covered, mapping, names = s.coverage(model)
    if not mapping['runtime_validated']:
        raise ValueError('requires saved validated coverage mapping')
    doc = json.loads(dyn.SOURCE.read_text()); a, b = source_rows(doc, model, covered)
    rows = candidates(a['activation'], b['activation'], covered, model.actuator_ctrlrange)
    for row in rows:
        row.update(integrate(model, a['qpos'], row['activation'], deadline=deadline))
        print(row['label'], 'eligible', row['eligible'], 'score', row['metrics']['score'] if row['metrics'] else None, flush=True)
    saved = next(r for r in json.loads(CROSS.read_text())['trajectories'] if r['label'] == 'PA_CA')
    cross_checks = dict(initial_qpos=dyn.bit_identical(a['qpos'], saved['initial_qpos']),
        initial_activation=dyn.bit_identical(a['activation'], saved['initial_activation']),
        final_qpos=dyn.bit_identical(rows[0]['final_qpos'], saved['final_qpos']),
        final_activation=dyn.bit_identical(rows[0]['final_activation'], saved['final_activation']))
    cross_checks.update({k: dyn.bit_identical(rows[0]['native_trace'][k], saved['trace'][k]) for k in saved['trace']})
    if not all(cross_checks.values()):
        raise AssertionError('baseline differs from saved crossed AA')
    winner = choose(rows)
    validations = []
    # Independent fresh MjData even when winner happens to be baseline A.
    selected = [('original_baseline_A', rows[0])] + ([('selected_winner', winner)] if winner is not None else [])
    for role, row in selected:
        replay = integrate(model, a['qpos'], row['activation'], duration=.1, deadline=deadline)
        prefix = prefix_check(row, replay)
        if not prefix['all_bit_identical']:
            raise AssertionError('fresh100ms first50ms differs')
        p = np.asarray(replay['trace']['COM_position_mm']); root = np.asarray(replay['trace']['root_position_mm'])
        validations.append(dict(role=role, candidate_label=row['label'], first50ms_identity=prefix,
            loses_validity_by100ms=bool(row['eligible'] and not replay['eligible']),
            validity_at50ms=row['eligible'], validity_at100ms=replay['eligible'],
            drift_50_to100=dict(COM_displacement_mm=(p[-1]-p[50]).tolist(), root_displacement_mm=(root[-1]-root[50]).tolist(),
                root_geodesic_rad=orientation_angle(replay['trace']['root_quaternion'][-1], replay['trace']['root_quaternion'][50]),
                joint_RMS_displacement_rad=float(np.sqrt(np.mean((np.asarray(replay['trace']['joint_angles_rad'][-1])-replay['trace']['joint_angles_rad'][50])**2)))),
            **replay))
    after = cp.sha256_file(cp.CANONICAL_XML)
    if after != frozen or any(cp.sha256_file(Path(p)) != h for p, h in hashes.items()):
        raise AssertionError('immutable source changed')
    guard = report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):
        guard['previous_soft_limit_mb'] = 'unlimited'
    out = dict(schema_version=1, status='COMPLETED_OFFLINE_DIAGNOSTIC', candidates=rows,
        selection=dict(winner_label=winner['label'] if winner else None, eligible_count=sum(r['eligible'] for r in rows),
            winner_score=winner['metrics']['score'] if winner else None,
            rule='minimum declared equal-weight sum squares among eligible50ms ONLY; no eligible means no winner; deterministic order breaks ties',
            selection_frozen_before100ms=True), validation100ms=validations,
        baseline_crossed_AA_identity=dict(all_bit_identical=all(cross_checks.values()), fields=cross_checks),
        initial_pose=dict(label=a['label'], qpos=a['qpos'], qpos_sha256=a['qpos_sha256'], lowered=False, offsets_applied=False),
        actuator_order=names, covered_indices=covered.tolist(), excluded_indices=np.setdiff1d(np.arange(model.nu), covered).tolist(),
        native_control_ranges=model.actuator_ctrlrange.tolist(), mapping=mapping,
        scales=dict(L_mm=1., gravity_mm_s2=float(np.linalg.norm(model.opt.gravity)),
            velocity_mm_s=float(np.sqrt(np.linalg.norm(model.opt.gravity))), root_orientation_rad=1., joint_angle_rad=1.),
        method=dict(candidate_count=11, search_duration_s=.05, validation_duration_s=.1, native_dt_s=float(model.opt.timestep), sample_interval_s=.001,
            score='sum of four equally weighted squared dimensionless terms: RMS Euclidean COM displacement from initial over 1..50ms / L; norm(final COM velocity)/sqrt(gL); terminal root quaternion geodesic /1rad; terminal hinge displacement RMS /1rad',
            position_penalty='all xyz directions squared, including upward bounce; RMS over time of Euclidean norm, not RMS over coordinates',
            COM='mass weighted xipos; mass weighted mj_jacBodyCom @ current nonzero qvel, sampled 1ms; no acceleration needed',
            command='constant ctrl and initial act=ctrl once; no controller or time-varying command',
            integration='fresh FixedPose initialization; mj_step then mj_forward identical prior reference; read-only COM observations',
            eligibility='500 native steps, no nonfoot-floor contacts at initialization/each step before AND after forward, no warning/reset/nonfinite, unchanged command, original activation bounds and excluded floors',
            perturbation='3 manifest-order sign patterns (all+, alternating +-, repeating ++--), paired +/- amplitude0.02 on covered74 only; explicit clipping ledger; no hidden offsets',
            execution='serial, BLAS/OMP/MKL/NUMEXPR single thread, shared memory guard, deadline55s and total<60s; no100ms feedback or tuning'),
        provenance=dict(canonical_xml=str(cp.CANONICAL_XML), xml_sha256_before=frozen, xml_sha256_after=after,
            expected_frozen_sha256=cp.CANONICAL_XML_SHA256, code_sha256=cp.sha256_file(Path(__file__)),
            test_code_sha256=cp.sha256_file(HERE/'test_dynamic_command_search.py'), source_hashes=hashes,
            command_A_source=a['label'], command_A_sha256=a['activation_sha256'], command_B_source=b['label'], command_B_sha256=b['activation_sha256'],
            interpreter=sys.executable, mujoco_version=mujoco.__version__), memory_guard=guard, wall_s=time.monotonic()-start,
        limits=['Manually implemented offline mechanical diagnostics, not a finished product or neural success.',
            'Per-muscle independent covered74 commands relax shared neural pool recruitment constraints; optimized commands need not be neurally realizable.',
            'Finite11 candidate search and100ms holdout do not prove standing, robustness, gait, long-term stability or infeasibility.',
            'Source A/B were optimized statically elsewhere; initial activation is changed together with constant command; XML frozen.',
            'Dynamic act is additionally required to stay inside the original ctrl interval as an experimental eligibility gate; this is NOT a native actlimited constraint (native actlimited is false).'])
    encoded = json.dumps(out, indent=2, allow_nan=False)+'\n'
    if time.monotonic()-start >= 60:
        raise TimeoutError('total command experiment >=60s')
    OUT.write_text(encoded)
    print('Wrote', OUT, 'wall_s', time.monotonic()-start, 'selection', out['selection'], flush=True)
    return out


if __name__ == '__main__':
    with mg.guarded_run('dynamic_command_search', **mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
