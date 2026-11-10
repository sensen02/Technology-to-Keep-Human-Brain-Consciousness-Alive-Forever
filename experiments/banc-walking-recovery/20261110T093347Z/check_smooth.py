"""S2/S5 conditional teacher checks; no mj_step, training, or gait rollout."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import sys

import numpy as np

from physics import Core, W
from static_teacher import StaticTeacher, motor_target
from wave_reference import WaveReference, DEFAULT_REFERENCE

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '20261110T074936Z'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    baseline_wave = load('_baseline_wave_smooth_check', BASE / 'wave_reference.py')
    baseline_teacher = load('_baseline_teacher_smooth_check', BASE / 'static_teacher.py')
    sources = [HERE / p for p in ('static_teacher.py', 'wave_reference.py', 'check_smooth.py', 'physics.py', 'identity.py')]
    sources += sorted((HERE / 'sources').glob('*.py'))
    sources += [BASE / 'wave_reference.py', BASE / 'static_teacher.py', DEFAULT_REFERENCE]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checks, measurements = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)

    c = Core(adhesion=False)
    teacher = StaticTeacher(c, dict(cycles=2))
    ref = teacher.reference
    old_ref = baseline_wave.WaveReference(teacher.rest, stride_native=.36, lift_native=.12, step_s=.8, cycles=2)
    check('measured_shapes_exactly_inherited', np.array_equal(ref.shapes, old_ref.shapes))
    check('measured_segments_indices_and_provenance_exactly_inherited', ref.segments == old_ref.segments)
    check('measured_source_hash_inherited', ref.metadata['source_sha256'] == old_ref.metadata['source_sha256'])
    check('order_geometry_config_preserved', ref.order == old_ref.order and ref.stride_native == .36 and ref.lift_native == .12 and ref.step_s == .8)
    check('teacher_only_adaptation_explicit', ref.metadata['active_swing_time_adaptation']['teacher_only'])
    reconstruction_error, source_clock_error, fd_error = 0., 0., 0.
    # Avoid internal piecewise-linear interpolation knots when checking derivatives.
    for i in range(6):
        active = ref.order_indices[i]
        for u in np.linspace(.013, .987, 79):
            t = (i+u)*ref.step_s
            sample = ref.sample(t)
            p = 10*u**3-15*u**4+6*u**5
            shape = np.array([np.interp(p, np.linspace(0, 1, 201), ref.shapes[active, :, k]) for k in range(3)])
            xy = teacher.rest[:, :2].copy()
            xy[:, 0] -= (i+u)*.36/6
            xy[list(ref.order_indices[:i]), 0] += .36
            xy[active] += .36*shape[:2]
            z = np.zeros(6)
            z[active] = .12*shape[2]
            reconstruction_error = max(reconstruction_error, np.max(np.abs(sample['target_xy']-xy)), np.max(np.abs(sample['target_z']-z)))
            sr = ref.segments[ref.order[i]]['source_time_range_s']
            source_clock_error = max(source_clock_error, abs(sample['source_time_s']-(sr[0]+p*(sr[1]-sr[0]))))
            h = 1e-7
            if abs(p*200-round(p*200)) > 1e-4:
                a, b = ref.sample(t-h), ref.sample(t+h)
                fd_error = max(fd_error, np.max(np.abs((b['target_xy']-a['target_xy'])/(2*h)-sample['velocity_xy'])), np.max(np.abs((b['target_z']-a['target_z'])/(2*h)-sample['velocity_z'])))
            # Nonactive stance displacement and derivative reflect constant body advance.
            inactive = [j for j in range(6) if j != active]
            check(f'constant_stance_velocity_phase_{i}', np.allclose(sample['velocity_xy'][inactive, 0], -.36/(6*.8), atol=1e-14))
    check('measured_shape_reconstruction_under_time_warp', reconstruction_error < 2e-14)
    check('source_clock_uses_same_warp', source_clock_error < 2e-14)
    check('active_phase_velocity_finite_difference', fd_error < 2e-7)
    measurements.update(reconstruction_max_abs=reconstruction_error, source_clock_max_abs_s=source_clock_error, velocity_fd_max_abs=fd_error)
    seam_positions, seam_velocity = [], []
    for boundary in range(1, 7):
        t = boundary*.8
        left, center, right = ref.sample(t-1e-7), ref.sample(t), ref.sample(t+1e-7)
        v = np.zeros((6, 3)); v[:, 0] = -.36/(6*.8)
        seam_positions.append(float(max(np.max(np.abs(left['target_xy']-center['target_xy'])), np.max(np.abs(right['target_xy']-center['target_xy'])), np.max(np.abs(left['target_z'])), np.max(np.abs(right['target_z'])))))
        seam_velocity.append(float(max(np.max(np.abs(np.column_stack([s['velocity_xy'], s['velocity_z']])-v)) for s in (left, center, right))))
        check(f'C1_internal_boundary_{boundary}', seam_positions[-1] < 1e-8 and seam_velocity[-1] < 1e-9)
        # Decimal phase times can divide to k +/- 1 ulp; compare sampled
        # endpoint geometry at roundoff tolerance, while shape arrays above
        # and their assigned endpoints below are checked bit-for-bit.
        check(f'endpoint_geometry_at_roundoff_{boundary}', np.allclose(center['target_xy'], old_ref.sample(t)['target_xy'], atol=2e-15, rtol=0) and np.allclose(center['target_z'], old_ref.sample(t)['target_z'], atol=2e-15, rtol=0))
    measurements.update(seam_position_max_abs=seam_positions, seam_velocity_max_abs=seam_velocity)
    check('measured_shape_endpoints_exact', np.array_equal(ref.shapes[:, 0], np.zeros((6, 3))) and np.array_equal(ref.shapes[:, -1], np.tile([1., 0., 0.], (6, 1))))
    check('cycle_returns_rest_xy_at_roundoff', np.allclose(ref.sample(4.8)['target_xy'], teacher.rest[:, :2], atol=2e-15, rtol=0))
    check('finite_end_stop_explicit', 'finite-end stop is not C1' in ref.metadata['derivatives'])

    old = baseline_teacher.StaticTeacher(c, dict(mode='hold'))
    all_stance = np.ones(6, bool)
    old_q, old_diag = old._allocation(c, all_stance)
    teacher._contact_activation = np.ones(6)
    q, diag = teacher._allocation(c, all_stance)
    check('all_one_allocation_exact_baseline', np.array_equal(q, old_q) and np.array_equal(diag['per_leg_fz_over_W'], old_diag['per_leg_fz_over_W']))
    endpoint_errors, cap_violation, residual_reconstruction = [], 0., 0.
    all_root_residuals = []
    for li in range(6):
        endpoint = {}
        for alpha in (0., 1e-8, 1e-5, .001, .1, .5, .9, .999, 1.-1e-5, 1.-1e-8, 1.):
            activation = np.ones(6); activation[li] = alpha
            teacher._contact_activation = activation
            stance = np.ones(6, bool); stance[li] = alpha == 1.
            planned_q, d = teacher._allocation(c, stance)
            fractions = np.asarray(d['per_leg_fz_over_W'])
            caps = np.asarray(d['force_fraction_upper_bounds'])
            cap_violation = max(cap_violation, float(np.max(fractions-caps)), float(np.max(-fractions)))
            residual_reconstruction = max(residual_reconstruction, float(np.max(np.abs(planned_q[:6]/W-(c.d.qfrc_bias-c.d.qfrc_passive)[:6]/W-d['root_balance_residual_over_W']))))
            all_root_residuals.append(d['root_balance_residual_norm_over_W'])
            check(f'hypothetical_support_flags_leg_{li}', d['support_is_hypothetical'] == bool(d['unbacked_support_indices']) and not d['force_solver_is_physics_validation'])
            check(f'unsupported_residual_flag_leg_{li}', d['root_balance_residual_norm_over_W'] <= .02 or d['unsupported'])
            endpoint[alpha] = fractions
        endpoint_errors.append([float(np.max(np.abs(endpoint[0.]-endpoint[1e-8]))), float(np.max(np.abs(endpoint[1.]-endpoint[1.-1e-8])))])
        check(f'both_force_endpoints_continuous_leg_{li}', max(endpoint_errors[-1]) < 1e-7)
        stance = np.ones(6, bool); stance[li] = False
        _, old_five = old._allocation(c, stance)
        check(f'zero_activation_matches_five_leg_fit_{li}', np.allclose(endpoint[0.], old_five['per_leg_fz_over_W'], atol=1e-7, rtol=0))
        check(f'zero_activation_exact_zero_force_{li}', endpoint[0.][li] == 0.)
    check('force_fraction_bounds', cap_violation < 1e-12)
    check('root_balance_uses_actual_positions_and_all_six_jacobians', residual_reconstruction < 1e-14)
    measurements.update(force_endpoint_max_abs=endpoint_errors, force_cap_violation=cap_violation, root_residual_reconstruction_max_abs=residual_reconstruction, no_step_root_residual_range=[min(all_root_residuals), max(all_root_residuals)], actual_contact_counts=diag['actual_contact_counts'], unbacked_support_indices=diag['unbacked_support_indices'])

    teacher._contact_activation = np.zeros(6)
    zero_q, zero_diag = teacher._allocation(c, np.zeros(6, bool))
    check('all_zero_caps_removed_without_BVLS_zero_width', np.array_equal(zero_q, np.zeros(c.plant.nv)) and zero_diag['support_indices'] == [] and zero_diag['unsupported'])
    check('no_step_unbacked_support_remains_hypothetical', diag['support_is_hypothetical'] and diag['unsupported'] and bool(diag['unbacked_support_indices']))
    state_names = ('qpos', 'qvel', 'qacc', 'ctrl', 'xfrc_applied', 'qfrc_applied', 'act', 'qacc_warmstart')
    state = {name: getattr(c.d, name).copy() for name in state_names}
    target = c.actuator.target_rad.copy()
    initial_time = float(c.d.time)
    action_finite = True
    activation_error = 0.
    partial_air_logged = False
    for t in (0., .005, .04, .08, .2, .4, .72, .795, .8, .805):
        action, d = teacher.action(c, t)
        z = teacher.reference.sample(t)['target_z']
        h = np.clip(z/.03, 0., 1.)
        activation_error = max(activation_error, float(np.max(np.abs(np.asarray(d['contact_activation'])-(1-h*h*(3-2*h))))))
        partial_air_logged |= bool(d['reference_air_support_indices'])
        action_finite &= np.isfinite(action).all() and np.max(np.abs(action)) <= 1.+1e-14
    check('action_preserves_all_physical_state_arrays', all(np.array_equal(getattr(c.d, name), value) for name, value in state.items()))
    check('action_preserves_actuator_targets', np.array_equal(target, c.actuator.target_rad))
    check('no_physics_steps_or_elapsed_time', float(c.d.time) == initial_time == 0.)
    check('finite_bounded_actions', action_finite)
    check('activation_from_reference_height_exact', activation_error < 1e-14)
    check('near_air_hypothetical_planned_force_logged', partial_air_logged)
    check('physics_validation_source_unchanged', inspect.getsource(StaticTeacher._validate_physics) == inspect.getsource(baseline_teacher.StaticTeacher._validate_physics))
    check('PD_inverse_source_unchanged', inspect.getsource(motor_target) == inspect.getsource(baseline_teacher.motor_target))
    check('IK_source_unchanged', inspect.getsource(StaticTeacher._ik) == inspect.getsource(baseline_teacher.StaticTeacher._ik))
    check('physics_file_exact_baseline', (HERE/'physics.py').read_bytes() == (BASE/'physics.py').read_bytes())
    check('source_helpers_exact_baseline', all(p.read_bytes() == (BASE/'sources'/p.name).read_bytes() for p in (HERE/'sources').glob('*.py')))
    # Strict constructor configuration rejection before any simulation step.
    for value in (0., -.03, float('inf'), float('nan')):
        try:
            StaticTeacher(c, dict(contact_transition_height=value))
        except ValueError:
            rejected = True
        else:
            rejected = False
        check('reject_transition_height_'+str(value), rejected)
    try:
        StaticTeacher(c, dict(unknown_smoothing_option=True))
    except ValueError:
        check('unknown_config_rejected', True)
    else:
        check('unknown_config_rejected', False)
    check('source_hashes_unchanged_during_checks', hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    result = dict(all_passed=all(checks.values()), checks=checks, measurements=measurements, source_hashes=hashes,
                  physical_steps=0, simulated_hold_s=0., status='PREPARED_CONDITIONAL_EXPERIMENT',
                  limitations=['No physical walking rollout or neural training performed', 'Positive optimized near-air support is hypothetical; actual contact and root residual flags remain authoritative', 'Internal measured interpolation knots and finite-reference stop are not globally C1', 'Native units; absolute SI NOT_ESTABLISHED'],
                  command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+str(HERE/'env/bin/python')+' -B '+str(HERE/'check_smooth.py'),
                  environment=dict(python=sys.executable, numpy=np.__version__, thread_limits={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    (HERE/'SMOOTH_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(all_passed=result['all_passed'], checks=len(checks), failed=[k for k,v in checks.items() if not v], measurements=measurements), indent=2))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
