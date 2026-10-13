"""Finite pose/activation mechanical experiment; not a controller or infeasibility proof."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
import run_support_tradeoff as tradeoff

audit = tradeoff.audit
s = audit.search
cp = s.cp
OUT = s.HERE / 'outputs/workspace_recheck/pose_support_search.json'
TOTAL_SECONDS = 80.
MAX_NFEV = 60
HINGE_BOUND_RAD = .03
ROOT_BOUND_MM = .01


def residual(qacc, com_j, gravity):
    """One immutable dimensionless objective for seeds, all poses, and ranking."""
    return tradeoff.residual(qacc, com_j, gravity, 1.)


def pose_specs():
    specs = [('baseline_exact_source', {}, 0.)]
    # Equal/opposite coordinate signs are experimental patterns, not anatomical symmetry.
    for group in ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch'):
        for pattern in ('same', 'opposite'):
            for sign in (-1, 1):
                delta = {f'joint_{side}{leg}{group}': sign * HINGE_BOUND_RAD *
                         (-1 if side == 'R' and pattern == 'opposite' else 1)
                         for side in ('L', 'R') for leg in ('F', 'M', 'H')}
                specs.append((f'{group}_{pattern}_{"plus" if sign > 0 else "minus"}', delta, 0.))
    for dz in (-.01, -.005, .005, .01):
        specs.append((f'root_height_{dz:+.3f}mm', {}, dz))
    return specs


def candidate_pose(model, base, delta, dz):
    if abs(dz) > ROOT_BOUND_MM or any(abs(v) > HINGE_BOUND_RAD for v in delta.values()):
        raise ValueError('declared pose bounds exceeded')
    q = np.array(base, dtype=float, copy=True)
    for name, value in delta.items():
        j = model.joint(name).id
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('only hinge offsets permitted')
        q[model.jnt_qposadr[j]] += value
    q[2] += cp.mm_to_model(dz)
    return q  # Never clip joint limits or reseat root.


def gate(model, ev):
    facts = cp.stance_facts(model, ev.data, raise_on_fail=False).as_dict()
    floor = cp.floor_geom_id(model)
    feet = set(cp.tarsus_geom_ids(model))
    distances = {model.geom(g).name: float(cp.model_to_mm(mujoco.mj_geomDistance(
        model, ev.data, g, floor, 1e3, None))) for g in feet}
    penetration = max(0., -min(distances.values()))
    facts.update(foot_floor_distances_mm=distances, max_abs_foot_penetration_mm=penetration)
    failures = []
    try:
        cp.stance_facts(model, ev.data)
    except cp.StanceGateError as exc:
        failures.append(str(exc))
    try:
        s.validate_limits(model, ev.qpos)
    except ValueError as exc:
        failures.append(str(exc))
    if penetration > .06 + 1e-12:
        failures.append('foot penetration exceeds engineering absolute 0.06 mm bound')
    if facts['n_body_floor_contact']:
        failures.append('nonfoot floor contact forbidden')
    return facts, failures


def measure(model, qpos, activation, label, gravity, angular_reference=None):
    # Independent fresh evaluator, independent COM Jacobian and force audit.
    snap = audit.snapshot(model, qpos, activation, label)
    if not all(c['within_tolerance'] for c in snap['checks'].values()):
        raise RuntimeError('independent force balance check failed')
    q = np.asarray(snap['raw_vectors']['qacc'])
    com = float(snap['COM_acceleration_mm_s2'][2] / gravity)
    full = s.dimensionless(q, gravity)
    angular = float(np.sqrt(np.mean(q[3:]**2)))
    contact = snap['contact']
    row = dict(activation=np.asarray(activation).tolist(), activation_sha256=audit.digest_array(activation),
        full_rms=float(np.sqrt(np.mean(full**2))), angular_rms_rad_s2=angular,
        angular_dimensionless_rms=float(np.sqrt(np.mean(full[3:]**2))), COM_z_over_g=com,
        support=snap['support_over_weight'], objective=float(full @ full + com**2),
        contact_count=contact['total_contact_count'], foot_contact_count=contact['named_foot_contact_count'],
        distinct_foot_count=contact['distinct_named_foot_geom_count'], force_audit=snap)
    if angular_reference is not None:
        row.update(angular_ratio=angular/angular_reference,
            angular_screen_pass=bool(angular <= 1.1*angular_reference),
            screen_pass=bool(angular <= 1.1*angular_reference and abs(com) <= .01))
    return row


def optimize(ev, covered, initial, com_j, gravity, deadline):
    start = time.monotonic()
    calls = ev.calls
    best_a = initial.copy()
    r = residual(ev.acceleration(best_a), com_j, gravity)
    seed_cost = best_cost = float(r @ r)
    class Budget(Exception): pass
    def fun(x):
        nonlocal best_a, best_cost
        if time.monotonic() >= deadline:
            raise Budget()
        a = ev.command(covered, x)
        r = residual(ev.acceleration(a), com_j, gravity)
        cost = float(r @ r)
        if cost < best_cost:
            best_cost, best_a = cost, a.copy()
        return r
    try:
        sol = least_squares(fun, initial[covered], bounds=(ev.floor[covered], ev.upper[covered]),
            max_nfev=MAX_NFEV, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        solver = dict(status=int(sol.status), success=bool(sol.success), message=str(sol.message),
            nfev=int(sol.nfev), njev=int(sol.njev), optimality=float(sol.optimality))
    except Budget:
        solver = dict(status='TIME_BUDGET', success=False, message='Retained best seed/iterate', nfev=None, njev=None)
    except (ValueError, FloatingPointError) as exc:
        solver = dict(status='OPTIMIZER_FAILED', success=False, message=str(exc), nfev=None, njev=None)
    solver.update(actual_forward_evaluations=ev.calls-calls, wall_s=time.monotonic()-start,
        retained_best_including_seed=True, seed_objective=seed_cost, best_evaluated_objective=best_cost)
    return best_a, solver


def main(report):
    start = time.monotonic()
    # Reserve 10 seconds for independent audits, <=3 integrations and serialization.
    search_deadline = start + 70.
    before = cp.sha256_file(cp.CANONICAL_XML)
    if before != cp.CANONICAL_XML_SHA256:
        raise ValueError('canonical freeze hash mismatch')
    source_hash = cp.sha256_file(audit.SOURCE)
    doc = json.loads(audit.SOURCE.read_text())
    if doc['provenance']['xml_sha256_before'] != before or doc['provenance']['xml_sha256_after'] != before:
        raise ValueError('source canonical hash mismatch')
    source = doc['best']['covered74']
    if source['label'] != 'round37_explicit_limit_repair':
        raise ValueError('requires saved repaired round37 best74')
    model = cp.load_model()
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('requires free root then hinges')
    covered, mapping, names = s.coverage(model)
    if not mapping['runtime_validated']:
        raise ValueError('runtime manifest required')
    np.testing.assert_array_equal(model.actuator_ctrlrange, doc['native_control_ranges'])
    assert names == doc['actuator_order']
    excluded = np.setdiff1d(np.arange(model.nu), covered)
    base = np.array(source['qpos'])
    original = np.array(source['activation'])
    np.testing.assert_array_equal(original[excluded], model.actuator_ctrlrange[excluded, 0])
    if np.any(original < model.actuator_ctrlrange[:, 0]) or np.any(original > model.actuator_ctrlrange[:, 1]):
        raise ValueError('source activation out of bounds')
    g = float(np.linalg.norm(model.opt.gravity))
    baseline = measure(model, base, original, 'baseline_original_saved_activation', g)
    baseline.update(qpos=base.tolist(), qpos_sha256=audit.digest_array(base))
    angular_reference = baseline['angular_rms_rad_s2']
    joints = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(1, model.njnt)}
    rows = []
    specs = pose_specs()
    for index, (label, delta, dz) in enumerate(specs):
        row_start = time.monotonic()
        qpos = candidate_pose(model, base, delta, dz)
        row = dict(label=label, requested_joint_deltas_rad=delta, requested_root_height_delta_mm=dz,
            actual_joint_deltas_rad={n: float(qpos[a]-base[a]) for n, a in joints.items()},
            root_height_delta_mm=float(cp.model_to_mm(qpos[2]-base[2])),
            qpos=qpos.tolist(), qpos_sha256=audit.digest_array(qpos), seated=False,
            offsets_relative_keyframe_rad={n: float(qpos[a]-model.key_qpos[0, a]) for n, a in joints.items()})
        rows.append(row)
        ev = s.FixedPose(model, qpos)
        ev.acceleration(original)
        facts, failures = gate(model, ev)
        row['gate'] = facts
        if failures:
            row.update(status='REJECTED_GATE_OR_LIMIT', reject_reasons=failures, wall_s=time.monotonic()-row_start)
            print(label, row['status'], failures, flush=True)
            continue
        _, weighted = audit.com_acceleration(model, ev.data)
        cj = weighted / model.body_mass.sum()
        remaining = max(0., search_deadline-time.monotonic())
        deadline = min(search_deadline, time.monotonic()+remaining/(len(specs)-index))
        activation, solver = optimize(ev, covered, original, cj, g, deadline)
        row.update(measure(model, qpos, activation, label, g, angular_reference))
        row.update(status='EVALUATED', solver=solver, wall_s=time.monotonic()-row_start,
            seed_activation_sha256=audit.digest_array(original),
            max_abs_foot_penetration_mm=facts['max_abs_foot_penetration_mm'])
        np.testing.assert_array_equal(activation[excluded], model.actuator_ctrlrange[excluded, 0])
        print(label, 'objective', row['objective'], 'COM/g', row['COM_z_over_g'],
              'angular ratio', row['angular_ratio'], 'solver', solver['status'], flush=True)
    valid = sorted([r for r in rows if r['status'] == 'EVALUATED'], key=lambda r: r['objective'])
    if not valid:
        raise RuntimeError('even source baseline rejected; inspect gates')
    angular_rows = [r for r in valid if r['angular_screen_pass']]
    best = valid[0]
    angular_best = angular_rows[0] if angular_rows else None
    selection = [('baseline_original_saved_activation', base, original, 'original unchanged source')]
    selection.append((best['label'], best['qpos'], best['activation'], 'minimum common objective regardless screen'))
    optimized_baseline = next(r for r in valid if r['label'] == 'baseline_exact_source')
    if optimized_baseline['label'] != best['label']:
        selection.append((optimized_baseline['label'], optimized_baseline['qpos'], optimized_baseline['activation'],
                          'optimized exact-pose baseline control; isolates geometry from activation optimization'))
    integrations = []
    for label, qpos, activation, reason in selection:
        if time.monotonic()-start >= TOTAL_SECONDS-1:
            raise TimeoutError('total budget exhausted before integration')
        integrations.append(dict(label=label, selection=reason, qpos_sha256=audit.digest_array(qpos),
            activation_sha256=audit.digest_array(activation), result=s.integrate(model, qpos, activation)))
    after = cp.sha256_file(cp.CANONICAL_XML)
    assert before == after and cp.sha256_file(audit.SOURCE) == source_hash
    optimized_baseline = next(r for r in valid if r['label'] == 'baseline_exact_source')
    result = dict(status='COMPLETED_BOUNDED_POSE_SEARCH', baseline_original=baseline, rows=rows,
        ranked_labels=[r['label'] for r in valid], best_objective_label=best['label'],
        best_angular_screen_label=angular_best['label'] if angular_best else None,
        geometry_comparison=dict(optimized_baseline_objective=optimized_baseline['objective'],
            best_objective=best['objective'], improvement_over_optimized_baseline=optimized_baseline['objective']-best['objective']),
        integrations=integrations, gravity_mm_s2=g, mapping=mapping, actuator_order=names,
        covered_indices=covered.tolist(), excluded_indices=excluded.tolist(), native_control_ranges=model.actuator_ctrlrange.tolist(),
        method=dict(objective='sum((translation qacc/g, all angular qacc*L/g)^2) + (COM_z/g)^2',
            objective_units='dimensionless sum of squares, identical across every pose and seed', weight=1., L_mm=1.,
            angular_reference_rad_s2=angular_reference, angular_limit_rad_s2=1.1*angular_reference,
            screen='Exploratory engineering: abs(COM_z/g)<=0.01 AND angular RMS<=1.1 original saved activation FixedPose baseline; same absolute reference for every pose',
            angular_screen_selection='minimum common objective among angular-only screen passers, not COM-only ranking',
            integration_selection='Parent clarification: original baseline, best objective, optimized exact-pose baseline (deduplicated); prioritize geometry dynamic control over optional angular-screen integration; maximum three',
            hinge_bound_rad=HINGE_BOUND_RAD, root_height_bound_mm=ROOT_BOUND_MM,
            pose_count=len(specs), hinge_candidate_count=12, root_height_candidates_mm=[-.01, -.005, .005, .01],
            bilateral_patterns='Same/opposite coordinate signs on all three L/R leg pairs; experimental only, NOT assumed anatomical symmetry; Trochanter_pitch is proximal femur angle group.',
            baseline='Exact source best74 repaired round37 qpos and original activation; separate baseline_exact_source row optimizes activation with no geometry change',
            pose_handling='No reseating, no silent clipping, strict range rejection; root offsets are physical position changes, never contact parameter changes',
            gate='cp.stance_facts plus strict joint ranges, no nonfoot floor contacts, abs foot penetration<=0.06mm',
            static_state='qvel=0; FixedPose clears all state and qacc_warmstart every eval; COM mass Jacobian recomputed at every pose',
            seed='Identical original saved covered74 activation for every pose; excluded muscles retain own native floor',
            max_nfev=MAX_NFEV, total_budget_s=TOTAL_SECONDS, optimization_deadline_s=70.,
            budget_allocation='Remaining optimization seconds divided equally over remaining candidates; rejected poses consume no optimizer budget',
            execution='serial; BLAS1; shared guarded_run; no controller or neural bypass; up to three independent 50ms integrations with initial act set once and never rewritten'),
        provenance=dict(source=str(audit.SOURCE), source_sha256=source_hash, source_sha256_after=cp.sha256_file(audit.SOURCE),
            source_label=source['label'], canonical_xml=str(cp.CANONICAL_XML), xml_sha256_before=before,
            xml_sha256_after=after, code_sha256=cp.sha256_file(Path(__file__)), interpreter=sys.executable,
            mujoco_version=mujoco.__version__, original_qpos_sha256=audit.digest_array(base), original_activation_sha256=audit.digest_array(original)),
        memory_guard=report('completed'), wall_s=time.monotonic()-start,
        limits=['Hand-built manually implemented mechanical research prototype, not a finished product.',
            'Finite local pose search: negative results do not prove infeasibility.',
            'Independent covered muscle activations relax neural pool constraints; no neural success or bypass claimed.',
            '50ms constant-command integration is not standing, robustness, gait or long-term stability proof.',
            'Native canonical contact/solver/model parameters unchanged; L and gates are engineering choices, not biological calibration.'])
    if result['wall_s'] > TOTAL_SECONDS:
        raise TimeoutError('total experiment exceeded 80s')
    if not np.isfinite(result['memory_guard']['previous_soft_limit_mb']):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Wrote', OUT, 'wall_s', result['wall_s'], flush=True)
    return result


if __name__ == '__main__':
    with s.mg.guarded_run('pose_support_search', **s.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
