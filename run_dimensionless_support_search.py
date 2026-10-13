"""Bounded engineering static-support search, not a neural controller or infeasibility proof.

Only this new entry point is changed. Canonical XML/contact/solver parameters stay native.
L=1 mm is an explicitly chosen engineering scale, not an anatomical measurement.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Avoid large BLAS thread pools in this serial, shared-memory-guarded experiment.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import engine.canonical_physics as cp
import engine.memory_guard as mg

HERE = Path(__file__).resolve().parent
POSE = HERE / 'outputs/stance_margin_round37/stance_for_reflex.json'
POOLS = HERE / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
MANIFEST = HERE / 'outputs/neural_muscle_closure/motor_manifest.json'
OUT = HERE / 'outputs/workspace_recheck/dimensionless_support_search.json'


def dimensionless(qacc, gravity, length=1.0):
    """Free-root translation first, all remaining coordinates angular; same length unit for g,L."""
    if not (np.isfinite(gravity) and gravity > 0 and np.isfinite(length) and length > 0):
        raise ValueError('positive finite gravity and characteristic length required')
    r = np.array(qacc, dtype=float, copy=True)
    r[:3] /= gravity
    r[3:] *= length / gravity
    return r


def metrics(qacc, gravity):
    def stats(x):
        return dict(max_abs=float(np.max(np.abs(x))), rms=float(np.sqrt(np.mean(x*x))))
    q = np.asarray(qacc)
    sensitivity = {}
    for length in (.5, 1., 2.):
        r = dimensionless(q, gravity, length)
        sensitivity[str(length)] = dict(all=stats(r), root_translation=stats(r[:3]),
            root_angular=stats(r[3:6]), joint_angular=stats(r[6:]), per_dof=r.tolist())
    return dict(root_translation_mm_s2=q[:3].tolist(), root_angular_rad_s2=q[3:6].tolist(),
                joint_angular_rad_s2=q[6:].tolist(), signed_root_z_mm_s2=float(q[2]),
                signed_root_z_over_g=float(q[2]/gravity), sensitivity_L_mm=sensitivity)


def initialize(model, offsets=None):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for name, value in (offsets or {}).items():
        j = model.joint(name).id
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f'offset is not a hinge: {name}')
        data.qpos[model.jnt_qposadr[j]] += value
    # Initialization identity is preserved even if the native keyframe violates a limit.
    # The search validates/rejects it explicitly rather than silently repairing the asset.
    return data


def validate_limits(model, qpos):
    import mujoco
    failures = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                        mujoco.mjtJoint.mjJNT_SLIDE):
            q = qpos[model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            if not lo - 1e-12 <= q <= hi + 1e-12:
                failures.append(dict(joint=model.joint(j).name, value=float(q), bounds=[float(lo), float(hi)]))
    if failures:
        raise ValueError('joint limit failures: ' + json.dumps(failures))


class FixedPose:
    def __init__(self, model, qpos):
        import mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos = np.array(qpos, copy=True)
        self.muscles = np.flatnonzero(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
        if len(self.muscles) != 90 or model.nu != 90:
            raise ValueError('this protocol requires exactly 90 muscle actuators, no other actuators')
        if not np.all(model.actuator_ctrllimited) or not np.all(model.actuator_actnum == 1):
            raise ValueError('bounded one-state muscle actuators required')
        self.addresses = model.actuator_actadr[self.muscles]
        self.floor = model.actuator_ctrlrange[:, 0].copy()
        self.upper = model.actuator_ctrlrange[:, 1].copy()
        self.calls = 0

    def command(self, free, x):
        a = self.floor.copy()  # ALL excluded muscles remain at their own native floor
        a[free] = x
        return a

    def acceleration(self, activation):
        import mujoco
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)  # also clears applied forces/time and unrelated history
        d.qpos[:] = self.qpos
        d.qvel[:] = 0
        d.ctrl[:] = activation
        d.act[self.addresses] = activation[self.muscles]
        d.qacc_warmstart[:] = 0  # explicitly cleared on EVERY evaluation
        mujoco.mj_forward(m, d)
        self.calls += 1
        if not np.isfinite(d.qacc).all():
            raise FloatingPointError('nonfinite acceleration')
        return d.qacc.copy()


def coverage(model):
    names = [model.actuator(i).name for i in range(model.nu)]
    pools = json.loads(POOLS.read_text())['measured']
    declared = [r['actuator'] for r in pools['resolved']]
    source_names = declared + [r['actuator'] for r in pools['unresolved']]
    if len(set(source_names)) != 90 or set(source_names) != set(names) or len(declared) != 74:
        raise ValueError('structural annotation does not match 90/74 body identities')
    report = dict(structural_count=74, runtime_validated=False, pools_path=str(POOLS),
                  pools_sha256=cp.sha256_file(POOLS))
    if MANIFEST.exists():
        doc = json.loads(MANIFEST.read_text())
        rows = doc['measured']['rows']
        if len(rows) != 90 or {r['actuator'] for r in rows} != set(names):
            raise ValueError('runtime manifest identity mismatch')
        runtime = []
        for r in rows:
            if names[r['actuator_index']] != r['actuator']:
                raise ValueError('runtime actuator index mismatch')
            idx = r['local_indices']
            if len(set(idx)) != len(idx) or any(i < 0 or i >= doc['measured']['n_neurons'] for i in idx):
                raise ValueError('invalid runtime neuron indices')
            if idx:
                runtime.append(r['actuator'])
        if set(runtime) != set(declared) or len(runtime) != 74:
            raise ValueError('runtime usable set differs from annotated covered74')
        report.update(runtime_validated=True, runtime_count=len(runtime), manifest_path=str(MANIFEST),
                      manifest_sha256=cp.sha256_file(MANIFEST),
                      caveat='Validates saved closure tier mapping, not current neural firing or independent recruitment.')
    else:
        report['caveat'] = 'No closure manifest available: covered74 is structural only, NOT verified runtime coverage.'
    return np.array([names.index(n) for n in declared]), report, names


def solve(ev, free, initial, gravity, max_nfev, deadline):
    from scipy.optimize import least_squares
    start_calls = ev.calls
    best_a = initial.copy()
    best_q = ev.acceleration(best_a)
    best_cost = float(np.dot(dimensionless(best_q, gravity), dimensionless(best_q, gravity)))
    seed_cost = best_cost
    class TimeBudget(Exception):
        pass
    def residual(x):
        nonlocal best_a, best_q, best_cost
        if time.monotonic() >= deadline:
            raise TimeBudget()
        a = ev.command(free, x)
        q = ev.acceleration(a)
        r = dimensionless(q, gravity)
        cost = float(r @ r)
        if cost < best_cost:
            best_a, best_q, best_cost = a.copy(), q.copy(), cost
        return r
    try:
        sol = least_squares(residual, initial[free], bounds=(ev.floor[free], ev.upper[free]),
                            max_nfev=max_nfev, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        status = dict(status=int(sol.status), message=str(sol.message), success=bool(sol.success),
                      budget_limited=bool(sol.status == 0),
                      nfev=int(sol.nfev), njev=int(sol.njev), optimality=float(sol.optimality))
    except TimeBudget:
        status = dict(status='time_budget', message='Best evaluated feasible command retained',
                      success=False, nfev=None, njev=None)
    status['actual_forward_evaluations'] = ev.calls - start_calls
    return dict(activation=best_a.tolist(), residual=metrics(best_q, gravity),
                dimensionless_sum_squares=best_cost, initial_sum_squares=seed_cost,
                retained_best_including_seed=True, solver=status)


def integrate(model, qpos, activation):
    """Independent MjData, steady initial act once; thereafter ONLY constant ctrl, never overwrite act."""
    import mujoco
    ev = FixedPose(model, qpos)
    ev.acceleration(np.asarray(activation))
    d = ev.data
    heights = [float(d.qpos[2])]
    times = [float(d.time)]
    contacts = [int(d.ncon)]
    body_contacts = []
    warning_counts = np.zeros(len(d.warning), dtype=int)
    nsteps = int(np.ceil(.050 / model.opt.timestep))
    finite = True
    reset_detected = False
    for _ in range(nsteps):
        before = float(d.time)
        mujoco.mj_step(model, d)
        warning_counts = np.maximum(warning_counts, np.array([w.number for w in d.warning]))
        finite = bool(all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.act, d.qacc)))
        if d.time <= before:
            reset_detected = True
        mujoco.mj_forward(model, d)
        heights.append(float(d.qpos[2]))
        times.append(float(d.time))
        contacts.append(int(d.ncon))
        floor = cp.floor_geom_id(model)
        feet = set(cp.tarsus_geom_ids(model))
        body_contacts.append(sum(floor in (c.geom1, c.geom2) and
                                 (c.geom2 if c.geom1 == floor else c.geom1) not in feet
                                 for c in d.contact[:d.ncon]))
        if not finite or reset_detected:
            break
    return dict(requested_duration_s=.050, actual_time_s=float(d.time), steps=len(heights)-1,
                finite=finite, automatic_reset_detected=reset_detected,
                warning_counts_by_index=warning_counts.tolist(), warning_count=int(warning_counts.sum()),
                initial_root_height_mm=heights[0], final_root_height_mm=heights[-1],
                min_root_height_mm=min(heights), max_root_height_mm=max(heights),
                contact_count_initial=contacts[0], contact_count_final=contacts[-1],
                contact_count_min=min(contacts), contact_count_max=max(contacts),
                max_nonfoot_floor_contacts=max(body_contacts, default=0),
                trace=dict(time_s=times, root_height_mm=heights, contact_count=contacts),
                final_qpos=d.qpos.tolist(), final_activation=d.act.tolist(),
                activation_initialization='act=ctrl once; no act writes during mj_step loop',
                interpretation='Open-loop mechanical check only; NOT neural success or stability proof.')


def main(guard_report):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-nfev', type=int, default=100)
    parser.add_argument('--search-seconds', type=float, default=125.)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.max_nfev < 1 or not 0 < args.search_seconds <= 150:
        parser.error('max-nfev >=1 and 0 < search-seconds <=150 required')
    start = time.monotonic()
    before = cp.sha256_file(cp.CANONICAL_XML)
    if before != cp.CANONICAL_XML_SHA256:
        raise ValueError('canonical freeze hash mismatch before run')
    model = cp.load_model()
    # This coordinate scaling is valid only for one free root followed by hinge joints.
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('unexpected generalized-coordinate layout')
    gravity = float(np.linalg.norm(model.opt.gravity))
    covered, mapping, names = coverage(model)
    base = json.loads(POSE.read_text())['joint_offsets']
    rng = np.random.default_rng(20261011)
    samples = [('round37_seated', base, True), ('actual_keyframe0', {}, False)]
    repairs = {}
    centers = []
    for label, original in [('round37', base), ('keyframe', {n: 0. for n in base})]:
        center, corrections = {}, {}
        for n in base:
            j = model.joint(n).id
            key = model.key_qpos[0, model.jnt_qposadr[j]]
            requested = key + original.get(n, 0.)
            lo, hi = model.jnt_range[j]
            corrected = float(np.clip(requested, lo, hi)) if model.jnt_limited[j] else requested
            center[n] = corrected - key
            if corrected != requested:
                corrections[n] = dict(requested_rad=float(requested), corrected_rad=corrected,
                                      correction_rad=float(corrected-requested))
        centers.append(center)
        candidate_label = label + '_explicit_limit_repair'
        repairs[candidate_label] = corrections
        samples.append((candidate_label, center, True))
    for k in range(8):
        center = centers[0 if k < 4 else 1]
        offsets = {}
        for n, v in center.items():
            j = model.joint(n).id
            key = model.key_qpos[0, model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            low, high = max(-.035, lo-key-v), min(.035, hi-key-v)
            offsets[n] = float(v + rng.uniform(low, high))
        samples.append((f'small_offset_{k:02d}', offsets, True))
    rows = []
    deadline = start + args.search_seconds
    for label, offsets, seat in samples:
        row = dict(label=label, offsets_relative_keyframe_rad=offsets, seated=seat)
        rows.append(row)
        row['explicit_limit_repairs'] = repairs.get(label, {})
        if time.monotonic() >= deadline:
            row.update(status='NOT_RUN_TIME_BUDGET')
            continue
        try:
            data = initialize(model, offsets)
            row['initial_qpos_before_seating'] = data.qpos.tolist()
            if seat:
                data.qpos[2] = cp.seat_root(model, data, data.qpos.copy())
            validate_limits(model, data.qpos)
            ev = FixedPose(model, data.qpos)
            baseline = ev.acceleration(ev.floor)
            row['qpos'] = data.qpos.tolist()
            row['gate'] = cp.stance_facts(model, ev.data).as_dict()
        except (cp.StanceGateError, ValueError) as exc:
            row.update(status='GATE_OR_LIMIT_FAILED', error=str(exc))
            print(label, row['status'], str(exc), flush=True)
            continue
        row.update(status='EVALUATED', baseline_native_floor=metrics(baseline, gravity))
        # A moderate seed escapes near-floor finite-difference degeneracy; retain floor too.
        seed = ev.command(covered, .05 * np.ones(len(covered)))
        floor_cost = float(np.sum(dimensionless(baseline, gravity)**2))
        c74 = solve(ev, covered, seed, gravity, args.max_nfev, deadline)
        if floor_cost < c74['dimensionless_sum_squares']:
            c74.update(activation=ev.floor.tolist(), residual=metrics(baseline, gravity),
                       dimensionless_sum_squares=floor_cost, retained_floor_baseline=True)
        row['covered74'] = c74
        row['all90'] = solve(ev, ev.muscles, np.array(c74['activation']), gravity, args.max_nfev, deadline)
        row['nested_best_nonworsening'] = row['all90']['dimensionless_sum_squares'] <= c74['dimensionless_sum_squares'] + 1e-12
        print(label, 'covered74 RMS', c74['residual']['sensitivity_L_mm']['1.0']['all']['rms'],
              'all90 RMS', row['all90']['residual']['sensitivity_L_mm']['1.0']['all']['rms'], flush=True)
    valid = [r for r in rows if r['status'] == 'EVALUATED']
    best = {}
    for condition in ('covered74', 'all90'):
        if valid:
            winner = min(valid, key=lambda r: r[condition]['dimensionless_sum_squares'])
            best[condition] = dict(label=winner['label'], qpos=winner['qpos'], **winner[condition],
                                  integration_50ms=integrate(model, winner['qpos'], winner[condition]['activation']))
    after = cp.sha256_file(cp.CANONICAL_XML)
    result = dict(status='COMPLETED_BOUNDED_SEARCH' if len(valid) else 'NO_GATED_CANDIDATE',
        status_meaning='Execution status only; local optimization is not an existence or infeasibility proof.',
        method=dict(residual='translation qacc/g; ALL angular qacc*L/g', characteristic_length_mm=1.,
                    scale_status='declared engineering choice, not biological calibration',
                    sensitivity='L=0.5,1,2 mm post-hoc rescaling of each result, NOT reoptimization',
                    pose_bounds='Original round37 and actual keyframe retained/rejected honestly; two separate explicit limit-repaired seated baselines (corrections recorded); eight +/-0.035 rad uniform perturbations within limits around those repaired centers (four each). Canonical XML never repaired.',
                    static_state='qvel=0, steady act=ctrl; all excluded muscles native per-actuator floor',
                    keyframe='explicit mj_resetDataKeyframe(0); actual keyframe baseline not reseated',
                    solver='scipy least_squares; best evaluated feasible seed/iterate retained; all90 starts best74',
                    max_nfev=args.max_nfev, search_budget_s=args.search_seconds, seed=20261011),
        gravity_mm_s2=gravity, mapping=mapping, actuator_order=names,
        native_control_ranges=model.actuator_ctrlrange.tolist(), rows=rows, best=best,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        canonical_xml=str(cp.CANONICAL_XML), xml_sha256_before=before,
                        xml_sha256_after=after, canonical_xml_untouched=before == after,
                        pose_path=str(POSE), pose_sha256=cp.sha256_file(POSE), mujoco_version=mujoco.__version__),
        memory_guard=guard_report('completed'), wall_s=time.monotonic()-start,
        limits=['Finite local search only; nonzero residual cannot prove infeasibility of this or any other pose.',
                'Independent activation per muscle is a mechanical relaxation: shared neural pools need not supply independent commands.',
                'Saved runtime mapping is not neural activity or a trained controller; NOT neural success.',
                '50 ms open-loop integration is not robust standing, recovery, gait or long-term stability.',
                'MuJoCo joint limits are soft: rejecting out-of-range baselines is a declared strict search gate, not proof the asset is invalid.',
                'Native contact compliance, passive forces, joint stiffness and uncertain mass calibration remain unchanged.'])
    if before != after:
        result['status'] = 'FAIL_CANONICAL_HASH_CHANGED'
    # memory_guard records an unlimited previous RLIMIT_AS as inf; JSON records that as text.
    if not np.isfinite(result['memory_guard']['previous_soft_limit_mb']):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Wrote', args.output.resolve(), 'wall_s', result['wall_s'], flush=True)


if __name__ == '__main__':
    with mg.guarded_run('dimensionless_support_search', **mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
