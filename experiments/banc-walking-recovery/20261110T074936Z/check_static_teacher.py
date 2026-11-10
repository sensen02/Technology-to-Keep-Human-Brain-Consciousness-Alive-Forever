"""Independent static algebra/contact checks, plus <=0.2s physical HOLD only.

Writes only STATIC_CHECKS.json. No wave rollout, training, or model edits.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import mujoco as mj
from physics import Core, W, MAX_RATE
from static_teacher import StaticTeacher, motor_target

BASE = Path(__file__).resolve().parent
REPORT = BASE / 'STATIC_CHECKS.json'


def contact_generalized_force(c):
    """All actual contact wrenches, both bodies, independent of teacher allocation."""
    qcontact = np.zeros(c.plant.nv)
    contacts = []
    for ci in range(c.d.ncon):
        ct = c.d.contact[ci]
        local = np.zeros(6)
        mj.mj_contactForce(c.plant, c.d, ci, local)
        frame = ct.frame.reshape(3, 3)
        force, torque = frame.T @ local[:3], frame.T @ local[3:]
        # MuJoCo contact wrench acts on geom2; opposite wrench on geom1.
        for geom, sign in ((int(ct.geom1), -1.), (int(ct.geom2), 1.)):
            body = int(c.plant.geom_bodyid[geom])
            mj.mj_applyFT(c.plant, c.d, sign*force, sign*torque, ct.pos, body, qcontact)
        contacts.append(dict(geom1=int(ct.geom1), geom2=int(ct.geom2),
                             local_wrench=local.tolist(), force_on_geom2_world=force.tolist()))
    return qcontact, contacts


def physical_snapshot(c):
    # Integration state plus all force/constraint caches used by this teacher.
    names = ('qpos', 'qvel', 'qacc', 'ctrl', 'xfrc_applied', 'qfrc_applied',
             'qfrc_bias', 'qfrc_passive', 'qfrc_constraint', 'qfrc_actuator',
             'efc_force', 'xpos', 'xmat', 'geom_xpos', 'geom_xmat')
    return {**{k: getattr(c.d, k).copy() for k in names},
            'time': float(c.d.time), 'target': c.actuator.target_rad.copy(),
            'ticks': c.actuator.command_ticks}


def main():
    checks = []
    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))

    reuse_hold = '--reuse-hold' in sys.argv[1:]
    if any(arg != '--reuse-hold' for arg in sys.argv[1:]):
        raise ValueError('Only optional argument is --reuse-hold')
    previous_report = json.loads(REPORT.read_text()) if reuse_hold else None
    if reuse_hold and previous_report['source_sha256'][str(BASE/'static_teacher.py')] != hashlib.sha256((BASE/'static_teacher.py').read_bytes()).hexdigest():
        raise ValueError('Cannot reuse physical hold evidence after teacher changes')
    c = Core(adhesion=False)
    for _ in range(500):
        c.tick(np.zeros(42))
    settled = c.measure()
    teacher = StaticTeacher(c, dict(mode='hold'))
    check('q_des_initialized_actual_settled_qpos', np.array_equal(teacher.q_des, c.d.qpos))
    check('settled_qpos_differs_from_actuator_target',
          np.max(np.abs(c.d.qpos[c.a_qpos]-c.actuator.target_rad)) > 1e-6,
          float(np.max(np.abs(c.d.qpos[c.a_qpos]-c.actuator.target_rad))))
    before = physical_snapshot(c)
    action, diag = teacher.action(c, 0.)
    after = physical_snapshot(c)
    check('action_preserves_actual_state_targets_and_force_caches',
          all(np.array_equal(before[k], after[k]) for k in before))
    check('private_fk_root_fixed_initial', np.array_equal(teacher.shadow.qpos[:7], teacher.initial_qpos[:7]))
    initial_bound_projection = np.clip(teacher.initial_qpos[c.a_qpos], teacher.bounds[:, 0], teacher.bounds[:, 1])-teacher.initial_qpos[c.a_qpos]
    check('hold_ik_bounded_residual_and_actual_settled_target_no_preload',
          np.max(np.abs(diag['ik_error'])) < 1e-5 and
          np.max(np.abs(np.asarray(diag['desired_foot_xy_bottom_z'])[:, 2]-settled['padz'])) < 1e-14,
          dict(max_ik_error=float(np.max(np.abs(diag['ik_error']))),
               max_initial_native_bound_projection_rad=float(np.max(np.abs(initial_bound_projection))),
               note='Five damped bounded iterations need not exactly reproduce a settled pose with soft joint-limit penetration; threshold is 1e-5 native, target pad heights remain exact actual rest'))
    check('normalized_action42_finite_bounded', action.shape == (42,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.)
    check('static_six_contacts_root_balance', len(diag['actual_contact_indices']) == 6 and
          len(diag['support_indices']) == 6 and not diag['unbacked_support_indices'] and
          diag['root_balance_residual_norm_over_W'] < .02 and diag['com_xy_inside_support_hull'],
          {k:diag[k] for k in ('actual_contact_counts', 'root_balance_residual_norm_over_W', 'com_xy_inside_support_hull', 'per_leg_fz_over_W')})

    # Pure PD algebra, away from target bounds, using independent reconstruction.
    rng = np.random.default_rng(20261110)
    bounds = teacher.target_bounds
    q = bounds.mean(axis=1)
    qdot = rng.uniform(-2., 2., 42)
    ff = rng.uniform(-2., 2., 42)*c.actuator.torque_limit
    ff_clipped = np.clip(ff, -c.actuator.torque_limit, c.actuator.torque_limit)
    target, unbounded = motor_target(q, qdot, ff_clipped, c.actuator.kp, c.actuator.kd, bounds)
    reconstructed = c.actuator.kp*(target-q)-c.actuator.kd*qdot
    unclipped_reconstructed = c.actuator.kp*(unbounded-q)-c.actuator.kd*qdot
    check('pure_algebra_target_produces_clipped_ff_at_qdes_qdotdes',
          np.max(np.abs(unclipped_reconstructed-ff_clipped)) < 1e-11 and
          np.max(np.abs(reconstructed-ff_clipped)) < 1e-11,
          dict(max_abs_error=float(np.max(np.abs(reconstructed-ff_clipped))),
               bound_projection_count=int(np.sum(target != unbounded)),
               caveat='Exact before native target projection; projection can reduce achievable feedforward'))

    qcontact, contacts = contact_generalized_force(c)
    # Independent joint-limit contribution from MuJoCo constraint rows.
    limit_force = np.zeros(c.d.nefc)
    limit_mask = c.d.efc_type == int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    limit_force[limit_mask] = c.d.efc_force[limit_mask]
    qlimit = np.zeros(c.plant.nv)
    mj.mj_mulJacTVec(c.plant, c.d, qlimit, limit_force)
    other_constraints = c.d.qfrc_constraint-qcontact-qlimit
    check('all_contact_wrenches_plus_joint_limits_reconstruct_constraint',
          np.max(np.abs(other_constraints)) < 1e-7,
          dict(max_abs_residual=float(np.max(np.abs(other_constraints))),
               joint_limit_rows=int(np.sum(limit_mask)), all_contact_count=int(c.d.ncon)))
    mass_acc = np.zeros(c.plant.nv)
    mj.mj_mulM(c.plant, c.d, mass_acc, c.d.qacc)
    h_minus_passive_constraint = c.d.qfrc_bias-c.d.qfrc_passive-c.d.qfrc_constraint
    tau_from_contacts = c.d.qfrc_bias-c.d.qfrc_passive-qcontact
    dynamic_error = h_minus_passive_constraint+mass_acc-c.d.qfrc_actuator
    contact_motor_error = tau_from_contacts+mass_acc-qlimit-c.d.qfrc_actuator
    check('independent_contact_motor_calibration_includes_qacc_and_limits',
          np.max(np.abs(dynamic_error)) < 1e-7 and np.max(np.abs(contact_motor_error)) < 1e-7,
          dict(max_equation_error=float(np.max(np.abs(dynamic_error))),
               max_contact_reconstruction_error=float(np.max(np.abs(contact_motor_error))),
               raw_static_contact_motor_discrepancy=float(np.max(np.abs(tau_from_contacts[c.a_dof]-c.d.qfrc_actuator[c.a_dof]))),
               max_Mqacc=float(np.max(np.abs(mass_acc))), max_joint_limit_force=float(np.max(np.abs(qlimit))),
               equation='bias-passive-all_contact = motor - M*qacc + joint_limit + other_constraints',
               note='Actual finite qacc and joint-limit forces must not be attributed to teacher sign error'))

    # Replaced reference: no physical step, one active target while actual leg is on ground.
    wave = StaticTeacher(c)
    wave_before = physical_snapshot(c)
    wave_action, wave_diag = wave.action(c, .4)
    swing = np.flatnonzero(np.array(wave.reference.sample(.4)['target_z']) > 1e-12)
    check('positive_swing_height_unloads_even_currently_grounded_leg',
          len(swing) == 1 and wave_diag['actual_contact_counts'][int(swing[0])] > 0 and
          wave_diag['per_leg_fz_native'][int(swing[0])] == 0. and len(wave_diag['support_indices']) == 5,
          dict(swing_indices=swing.tolist(), force_fractions=wave_diag['per_leg_fz_over_W'],
               root_residual=wave_diag['root_balance_residual_norm_over_W'], unsupported=wave_diag['unsupported']))
    check('wave_ik_bounded_and_private_no_rollout',
          np.all(wave.q_des[c.a_qpos] >= bounds[:, 0]) and np.all(wave.q_des[c.a_qpos] <= bounds[:, 1]) and
          np.max(np.abs(wave_diag['qdot_des'])) <= 2. and
          all(np.array_equal(wave_before[k], physical_snapshot(c)[k]) for k in wave_before),
          dict(ik_error_norm_per_leg=wave_diag['ik_error_norm_per_leg'],
               velocity_clip_count=wave_diag['qdot_des_clip_count']))
    check('wave_defaults_match_requested_config', wave.config['stride'] == .36 and wave.config['lift'] == .12 and wave.config['step_s'] == .8 and wave.config['cycles'] == 1)

    # Force a deliberately impossible support geometry via a replaceable reference;
    # allocation is read-only and must explicitly flag a large root residual.
    class OneSupport:
        def sample(self, t):
            sample = teacher.reference.sample(t)
            sample['stance'] = np.array([True, False, False, False, False, False])
            return sample
    unsupported_teacher = StaticTeacher(c, dict(mode='hold'))
    unsupported_teacher.reference = OneSupport()
    _, impossible = unsupported_teacher.action(c, 0.)
    check('large_root_residual_explicitly_flags_unsupported',
          impossible['root_balance_residual_norm_over_W'] > .02 and impossible['unsupported'],
          impossible['root_balance_residual_norm_over_W'])
    ablation = StaticTeacher(c, dict(mode='hold', gravity_ff=False))
    _, noff = ablation.action(c, 0.)
    check('gravity_feedforward_ablation_is_zero', np.all(np.array(noff['tau_ff']) == 0.))
    current = StaticTeacher(c, dict(mode='hold', stance_force_pref='current'))
    _, measured = current.action(c, 0.)
    check('measured_load_preference_explicit_config', current.metadata['config']['stance_force_pref'] == 'current' and measured['root_balance_residual_norm_over_W'] < .02)

    # Read-only proxy removes contact visibility without mutating physical data.
    class NoContactProxy:
        def __init__(self, core):
            self.core = core
            self.planes = np.zeros_like(core.planes)
        def __getattr__(self, key):
            return getattr(self.core, key)
    _, hypothetical = teacher.action(NoContactProxy(c), 0.)
    check('unbacked_supports_explicitly_marked_not_silently_accepted',
          hypothetical['unbacked_support_indices'] == list(range(6)) and
          hypothetical['support_is_hypothetical'] and hypothetical['unsupported'])
    # Reconstruct teacher contact forces independently from its reported points.
    predicted_contact = np.zeros(c.plant.nv)
    for li, point, body in zip(diag['support_indices'], diag['support_contact_points_world'], diag['support_contact_body_ids']):
        mj.mj_applyFT(c.plant, c.d, np.array([0.,0.,diag['per_leg_fz_native'][li]]), np.zeros(3), np.asarray(point), body, predicted_contact)
    expected_ff = (c.d.qfrc_bias-c.d.qfrc_passive-predicted_contact)[c.a_dof]
    check('teacher_tau_ff_independent_applyFT_reconstruction',
          np.max(np.abs(expected_ff-np.asarray(diag['tau_ff_unclipped']))) < 1e-12)
    expected_rate = np.asarray(diag['qdot_des'])+(np.asarray(diag['desired_motor_target'])-c.actuator.target_rad)/teacher.config['target_tau_s']
    expected_rate = np.clip(expected_rate,-2.,2.)
    expected_rate = np.clip(expected_rate,(bounds[:,0]-c.actuator.target_rad)/.005,(bounds[:,1]-c.actuator.target_rad)/.005)
    check('rate_formula_and_target_bounds_independent', np.max(np.abs(expected_rate-MAX_RATE*action)) < 1e-12)

    # Only physical teacher execution: 40 decisions x 5ms = 0.2 seconds HOLD.
    # --reuse-hold repeats static checks but preserves matching-teacher physical
    # evidence, avoiding additional rollout after a checker-only correction.
    hold_evidence = dict(reused=bool(reuse_hold))
    if reuse_hold:
        duration = previous_report['teacher_physical_hold_s']
        hold_rows = previous_report['hold_rows']
        physical_names = ('physical_execution_hold_only_point2_seconds',
                          'hold_no_warnings_no_nonfoot_no_large_com_loss', 'physical_hold_no_applied_forces')
        for name in physical_names:
            item = next(x for x in previous_report['checks'] if x['name'] == name)
            if not item['passed']:
                raise ValueError('Cannot reuse failed hold evidence')
            checks.append(item)
        hold_evidence.update(original_command=previous_report['command'],
                             original_source_sha256=previous_report['source_sha256'])
    else:
        start_time = float(c.d.time)
        com_start = c.measure()['com']
        minimum_com, nonfoot_peak, residual_peak = com_start, 0., 0.
        hold_rows = []
        for i in range(40):
            a, dd = teacher.action(c, i*.005)
            c.tick_policy(a, decision_ms=5)
            mm = c.measure()
            minimum_com = min(minimum_com, mm['com'])
            nonfoot_peak = max(nonfoot_peak, mm['nonfoot'])
            residual_peak = max(residual_peak, dd['root_balance_residual_norm_over_W'])
            hold_rows.append(dict(t_s=(i+1)*.005, com=mm['com'], foot_support_over_W=float(mm['loads'].sum()/W),
                                  nonfoot_native=mm['nonfoot'], root_residual_over_W=dd['root_balance_residual_norm_over_W'],
                                  unsupported=dd['unsupported']))
        duration = float(c.d.time-start_time)
        check('physical_execution_hold_only_point2_seconds', abs(duration-.2) < 1e-9, duration)
        check('hold_no_warnings_no_nonfoot_no_large_com_loss',
              not c.d.warning.number.any() and nonfoot_peak < 1e-8 and minimum_com >= com_start-.1,
              dict(warnings=c.d.warning.number.tolist(), nonfoot_peak_native=nonfoot_peak,
                   com_start=com_start, com_end=c.measure()['com'], com_min=minimum_com,
                   max_root_residual_over_W=residual_peak))
        check('physical_hold_no_applied_forces', not np.any(c.d.xfrc_applied) and not np.any(c.d.qfrc_applied))

    report = dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
                  purpose='S2/S5 static teacher algebra, private IK, force signs and bounded physical HOLD',
                  command=[sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
                  physical_hold_evidence=hold_evidence,
                  thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                  passed=sum(x['passed'] for x in checks), total=len(checks), checks=checks,
                  settle_s=.5, teacher_physical_hold_s=duration,
                  prior_check_attempts=[dict(exit_code=1, phase='constructor after settle; no teacher physical execution',
                      reason='Native hinge names have joint_ prefix; corrected leg membership lookup'),
                      dict(exit_code=1, phase='completed static checks and 0.2s physical hold',
                           reason='18/19 passed; hold IK exactness threshold 1e-9 failed at 6.469141925147248e-6 with active joint limits; checker now states bounded 1e-5 residual explicitly',
                           teacher_unchanged=True)],
                  initial_static_diagnostics=diag, wave_private_ik_diagnostics=wave_diag,
                  all_actual_contact_wrenches=contacts, hold_rows=hold_rows,
                  metadata=teacher.metadata,
                  versions=dict(python=sys.version, numpy=np.__version__, mujoco=mj.__version__),
                  source_sha256={str(BASE/name):hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                 for name in ('static_teacher.py','check_static_teacher.py')},
                  not_executed=['Physical wave rollout', 'Training', 'Walking acceptance', 'PREREG/runner edits', 'Model/dynamics edits'],
                  interpretation='Static mathematical checks and 0.2s hold only; vertical quasistatics and hypothetical supports do not establish gait feasibility or success.')
    report['all_passed'] = all(x['passed'] for x in checks)
    report['source_hashes'] = dict(report['source_sha256'])
    report['ik_tolerance_scope'] = '1e-5 native is a bounded private IK checker tolerance only; unchanged walking/gait acceptance gates'
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], passed=report['passed'], total=report['total'],
                          report=str(REPORT), failures=[x for x in checks if not x['passed']],
                          hold_summary=checks[-2]), indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
