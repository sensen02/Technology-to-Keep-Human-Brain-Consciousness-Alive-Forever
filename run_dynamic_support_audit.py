"""Scratch-only dynamic audit of two saved optimized states; no causal/standing claim."""
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
import run_support_force_audit as audit
s, cp, mg = audit.search, audit.cp, audit.mg
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'outputs/workspace_recheck/pose_support_search.json'
OUT = HERE / 'outputs/workspace_recheck/dynamic_support_audit.json'
LABELS = ('baseline_exact_source', 'root_height_-0.010mm')


def select_rows(doc):
    if doc['best_objective_label'] != LABELS[1]:
        raise ValueError('saved best objective is not the requested root height row')
    rows = []
    for label in LABELS:
        matches = [r for r in doc['rows'] if r['label'] == label]
        if len(matches) != 1 or matches[0]['status'] != 'EVALUATED':
            raise ValueError('missing/ambiguous evaluated source')
        row = matches[0]
        for field in ('qpos', 'activation'):
            if audit.digest_array(row[field]) != row[field + '_sha256']:
                raise ValueError('source array hash mismatch')
        rows.append(row)
    return rows


def com_jacobian(m, d):
    weighted = np.zeros((3, m.nv))
    jac = np.zeros_like(weighted)
    for b in range(1, m.nbody):
        if m.body_mass[b]:
            mujoco.mj_jacBodyCom(m, d, jac, None, b)
            weighted += m.body_mass[b] * jac
    return weighted / m.body_mass[1:].sum()


def moving_com(m, d, eps=None):
    """World COM: mass-weighted local-body COM Jacobians; central directional Jdot.

    mj_integratePos handles quaternion tangent coordinates. Only kinematics/comPos
    are needed on scratch data; never replace qacc by a static forward evaluation.
    """
    mass = float(m.body_mass[1:].sum())
    position = (m.body_mass[1:, None] * d.xipos[1:]).sum(axis=0) / mass
    jac = com_jacobian(m, d)
    if eps is None:
        eps = 1e-5 / max(1., float(np.max(np.abs(d.qvel))))
    scratch = mujoco.MjData(m)
    def directional(h):
        js = []
        for sign in (-1., 1.):
            scratch.qpos[:] = d.qpos
            mujoco.mj_integratePos(m, scratch.qpos, d.qvel, sign*h)
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            js.append(com_jacobian(m, scratch))
        return ((js[1]-js[0])/(2*h)) @ d.qvel
    coarse, fine = directional(eps), directional(eps/2)
    acc = jac @ d.qacc + fine
    scale = max(float(np.linalg.norm(m.opt.gravity)), float(np.linalg.norm(acc)), 1.)
    check = audit.error_check(coarse, fine, 1e-7*scale, rtol=1e-6)
    return dict(position=position, velocity=jac @ d.qvel, acceleration=acc,
                J_qacc=jac @ d.qacc, Jdot_qvel=fine, epsilon_s=eps,
                convergence=check)


def muscle_decomposition(m, d):
    """Actuation-only scratch counterfactual at identical qpos/qvel/length/velocity.

    ctrl is NEVER changed. Scratch act=0 measures passive bias even if native
    command floor is positive; midpoint tests force affinity, not a trajectory.
    """
    scratch = mujoco.MjData(m)
    mujoco.mj_copyData(scratch, m, d)
    adr = m.actuator_actadr
    total = d.actuator_force.copy()
    activation = d.act[adr].copy()
    scratch.act[adr] = 0
    mujoco.mj_fwdActuation(m, scratch)
    passive = scratch.actuator_force.copy()
    scratch.act[adr] = activation/2
    mujoco.mj_fwdActuation(m, scratch)
    midpoint = scratch.actuator_force.copy()
    gain = np.array([mujoco.mju_muscleGain(d.actuator_length[i], d.actuator_velocity[i],
        m.actuator_lengthrange[i], m.actuator_acc0[i], m.actuator_gainprm[i, :9]) for i in range(m.nu)])
    bias = np.array([mujoco.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i],
        m.actuator_acc0[i], m.actuator_biasprm[i, :9]) for i in range(m.nu)])
    active = total-passive
    checks = dict(midpoint_affine=audit.error_check(midpoint, (total+passive)/2, 1e-10),
        active_gain_times_act=audit.error_check(active, gain*activation, 1e-10),
        passive_bias=audit.error_check(passive, bias, 1e-10))
    for key in ('qpos', 'qvel', 'ctrl', 'actuator_length', 'actuator_velocity'):
        np.testing.assert_array_equal(getattr(scratch, key), getattr(d, key))
    return dict(actuator_length=d.actuator_length.tolist(), actuator_velocity=d.actuator_velocity.tolist(),
        activation=activation.tolist(), total_force=total.tolist(), passive_force=passive.tolist(),
        active_force=active.tolist(), active_gain=gain.tolist(), passive_bias=bias.tolist(),
        scalar_force_l2={k: float(np.linalg.norm(v)) for k, v in [('total', total), ('active', active), ('passive', passive)]},
        same_physical_state=True, checks=checks)


def observe(m, original):
    # All potentially computing API calls operate on scratch, never original.
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, original)
    com = moving_com(m, d)
    mass = float(m.body_mass[1:].sum())
    weight = mass * np.linalg.norm(m.opt.gravity)
    floor, feet = cp.floor_geom_id(m), set(cp.tarsus_geom_ids(m))
    contact = audit.contact_forces(m, d, floor, feet)
    named = {}
    for g in sorted(feet):
        rows = [r for r in contact['rows'] if r['other_geom'] == m.geom(g).name]
        named[m.geom(g).name] = dict(contact_count=len(rows), force_world=np.sum(
            [r['force_world'] for r in rows], axis=0).tolist() if rows else [0., 0., 0.])
    j = audit.constraint_jacobian(m, d)
    noncontact = d.efc_type < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
    noncontact_force = (j[noncontact].T @ d.efc_force[noncontact])[:3]
    limits = []
    for r in np.flatnonzero(d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)):
        joint = int(d.efc_id[r])
        limits.append(dict(row_index=int(r), joint_id=joint, joint=m.joint(joint).name,
            efc_position=float(d.efc_pos[r]), efc_force=float(d.efc_force[r]),
            coordinate_unit='rad', force_unit=audit.TORQUE_UNIT,
            root_force_world=(j[r, :3]*d.efc_force[r]).tolist()))
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
        raise ValueError('protocol requires zero applied forces')
    terms = dict(passive_root_force=d.qfrc_passive[:3], actuator_root_force=d.qfrc_actuator[:3],
        noncontact_constraint_root_force=noncontact_force, applied_root_force=d.qfrc_applied[:3],
        applied_cartesian_force=d.xfrc_applied[1:, :3].sum(axis=0))
    other = sum(terms.values())
    rhs = mass*m.opt.gravity + np.array(contact['force_world']) + other
    # Finite-difference roundoff tolerance: 1e-6 weight absolute plus 1e-7 component relative.
    balance = audit.error_check(mass*com['acceleration'], rhs, 1e-6*weight, rtol=1e-7)
    return dict(time_s=float(d.time), root_z_mm=float(d.qpos[2]), qvel_l2=float(np.linalg.norm(d.qvel)),
        support_over_weight=float(np.dot(contact['force_world'], -m.opt.gravity/np.linalg.norm(m.opt.gravity))/weight),
        COM_position_mm=com['position'].tolist(), COM_velocity_mm_s=com['velocity'].tolist(),
        COM_acceleration_mm_s2=com['acceleration'].tolist(), J_qacc_mm_s2=com['J_qacc'].tolist(),
        Jdot_qvel_mm_s2=com['Jdot_qvel'].tolist(), jacobian_epsilon_s=com['epsilon_s'],
        mass_times_COM_acceleration=(mass*com['acceleration']).tolist(),
        gravity_contact_other_sum=rhs.tolist(), other_external_force_world=other.tolist(),
        other_terms={k: v.tolist() for k, v in terms.items()}, contact=contact, named_feet=named,
        joint_limits=limits, joint_limit_row_count=len(limits),
        joint_limit_torque_l2=float(np.linalg.norm([r['efc_force'] for r in limits])),
        muscles=muscle_decomposition(m, d),
        checks=dict(COM_Newton_balance=balance, Jdot_convergence=com['convergence']))


def integrate_observed(model, qpos, activation, deadline=float('inf')):
    """Matches search.integrate step/forward ordering exactly; observation is scratch-only."""
    ev = s.FixedPose(model, qpos)
    ev.acceleration(np.asarray(activation))
    d = ev.data
    heights, times, contacts = [float(d.qpos[2])], [float(d.time)], [int(d.ncon)]
    samples = [observe(model, d)]
    body_contacts = []
    warning_counts = np.zeros(len(d.warning), dtype=int)
    nsteps = int(np.ceil(.050 / model.opt.timestep))
    if nsteps != 500 or not np.isclose(model.opt.timestep, .0001, rtol=0, atol=1e-15):
        raise ValueError('requires frozen native 0.1ms timestep and 500 steps')
    finite, reset_detected, command_unchanged = True, False, True
    for step in range(nsteps):
        if time.monotonic() > deadline:
            raise TimeoutError('dynamic audit exceeded 90-second budget')
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
            (c.geom2 if c.geom1 == floor else c.geom1) not in feet for c in d.contact[:d.ncon]))
        command_unchanged &= np.array_equal(d.ctrl, activation)
        if (step+1) % 10 == 0:
            samples.append(observe(model, d))
        if not finite or reset_detected:
            break
    return dict(samples=samples, steps=len(heights)-1, actual_time_s=float(d.time), finite=finite,
        automatic_reset_detected=reset_detected, warning_counts_by_index=warning_counts.tolist(),
        warning_count=int(warning_counts.sum()), final_warning_counts=[int(w.number) for w in d.warning],
        command_unchanged=bool(command_unchanged), activation_initialization_writes=1,
        trace=dict(time_s=times, root_height_mm=heights, contact_count=contacts),
        final_qpos=d.qpos.tolist(), final_activation=d.act.tolist(), max_nonfoot_floor_contacts=max(body_contacts, default=0))


def bit_identical(a, b):
    x, y = np.asarray(a, dtype='<f8'), np.asarray(b, dtype='<f8')
    return x.shape == y.shape and x.tobytes() == y.tobytes()


def run_audit(report):
    start = time.monotonic()
    frozen = cp.sha256_file(cp.CANONICAL_XML)
    if frozen != cp.CANONICAL_XML_SHA256:
        raise ValueError('canonical freeze mismatch')
    source_hash = cp.sha256_file(SOURCE)
    doc = json.loads(SOURCE.read_text())
    for key in ('xml_sha256_before', 'xml_sha256_after'):
        if doc['provenance'][key] != frozen:
            raise ValueError('source XML mismatch')
    m = cp.load_model()
    if m.opt.density != 0 or m.opt.viscosity != 0 or np.any(m.body_gravcomp):
        raise ValueError('requires no fluid/gravcomp')
    if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(m.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('requires free root and hinge joints')
    if len(set(m.body_rootid[m.body_mass > 0])) != 1 or np.any(m.dof_armature[:3]):
        raise ValueError('requires one body tree and no root armature')
    np.testing.assert_array_equal(m.actuator_ctrlrange, doc['native_control_ranges'])
    assert doc['actuator_order'] == [m.actuator(i).name for i in range(m.nu)]
    trajectories = []
    for row in select_rows(doc):
        observed = integrate_observed(m, row['qpos'], row['activation'], start+90)
        reference = s.integrate(m, row['qpos'], row['activation'])
        identity = {k: bit_identical(observed[k], reference[k]) for k in ('final_qpos', 'final_activation')}
        identity.update({k: bit_identical(observed['trace'][k], reference['trace'][k]) for k in reference['trace']})
        identity['all_bit_identical'] = all(identity.values())
        trajectory = dict(label=row['label'], source_row_index=doc['rows'].index(row),
            initial_qpos=row['qpos'], initial_activation=row['activation'],
            qpos_sha256=row['qpos_sha256'], activation_sha256=row['activation_sha256'],
            observation_identity=identity, reference=reference, **observed)
        trajectory['summary_at_ms'] = [{k: sample[k] for k in ('time_s', 'root_z_mm', 'support_over_weight',
            'COM_position_mm', 'COM_velocity_mm_s', 'COM_acceleration_mm_s2', 'joint_limit_row_count')} |
            dict(muscle_scalar_force_l2=sample['muscles']['scalar_force_l2'],
                 named_foot_count=sample['contact']['distinct_named_foot_geom_count'])
            for sample in observed['samples'] if any(abs(sample['time_s']-t/1000) < 1e-10 for t in (0, 10, 25, 50))]
        trajectories.append(trajectory)
    checks_ok = all(c['within_tolerance'] for t in trajectories for sample in t['samples']
        for c in [*sample['checks'].values(), *sample['muscles']['checks'].values()])
    healthy = all(t['finite'] and not t['automatic_reset_detected'] and not t['warning_count'] and
        t['command_unchanged'] and t['steps'] == 500 and t['observation_identity']['all_bit_identical'] for t in trajectories)
    after = cp.sha256_file(cp.CANONICAL_XML)
    source_after = cp.sha256_file(SOURCE)
    passed = checks_ok and healthy and frozen == after and source_hash == source_after
    result = dict(schema_version=1, status='MEASURED_WITHIN_DECLARED_TOLERANCES' if passed else 'REQUIRES_INVESTIGATION',
        trajectories=trajectories, actuator_order=doc['actuator_order'],
        units=dict(position='mm model length', velocity='mm/s', acceleration='mm/s^2',
            force=audit.FORCE_UNIT, joint_limit_torque=audit.TORQUE_UNIT,
            muscle_length='model tendon length (mm)', muscle_velocity='mm/s',
            muscle_scalar_force_l2='separate same-unit scalar muscle force norm; NOT external support',
            mass='model_mass_unit; physical calibration not established'),
        method=dict(duration_s=.050, native_dt_s=float(m.opt.timestep), steps=500, sample_interval_s=.001,
            comparison='Optimized fixed-pose baseline row versus best-objective root-height row; own saved qpos and activation unchanged',
            COM='sum(m*xipos)/M; sum(m*J_bodyCOM*qvel)/M; J_COM*qacc + central directional Jdot_COM*qvel via mj_integratePos, eps and eps/2',
            finite_difference='eps=1e-5/max(1,max(abs(qvel))) seconds; convergence atol=1e-7*max(g,|a|,1), rtol=1e-6',
            Newton_tolerance='component abs error <= 1e-6*weight + 1e-7*max(abs(lhs),abs(rhs)); measured independent COM kinematics vs external force; FD roundoff allowed',
            muscles='Scratch act=0, unchanged qpos/qvel/ctrl and length/velocity; total-passive active, every sample checked midpoint and installed mju_muscleGain/Bias',
            integration='Exact search.integrate mj_step then mj_forward ordering; initial act=ctrl once only; duplicate direct search.integrate reference',
            observation='Every computing diagnostic runs on copied MjData; trajectory never touched by diagnostics',
            execution='Serial single-thread environment; mg.guarded_run; 90-second integration deadline; no XML/command modification'),
        environment=dict(density=float(m.opt.density), viscosity=float(m.opt.viscosity), wind=m.opt.wind.tolist(),
            gravity_mm_s2=m.opt.gravity.tolist(), mass_model_units=float(m.body_mass[1:].sum()),
            weight_force=float(m.body_mass[1:].sum()*np.linalg.norm(m.opt.gravity)), applied_forces='asserted exactly zero at every sample'),
        provenance=dict(source=str(SOURCE), source_sha256=source_hash, source_sha256_after=source_after,
            canonical_xml=str(cp.CANONICAL_XML), xml_sha256_before=frozen, xml_sha256_after=after,
            expected_frozen_sha256=cp.CANONICAL_XML_SHA256, code_sha256=cp.sha256_file(Path(__file__)),
            dependency_code_sha256={str(Path(mod.__file__)): cp.sha256_file(Path(mod.__file__)) for mod in (s, audit, cp)},
            interpreter=sys.executable, mujoco_version=mujoco.__version__),
        wall_s=time.monotonic()-start, memory_guard=report('completed'),
        limits=['Manually implemented mechanical research prototype, not finished product.',
            'Local association, not causal proof; two trajectories differ in both optimized commands and pose.',
            '50ms constant-command check is not standing, robustness, neural success or long-term stability proof.',
            'Scalar tendon forces are internal forces and are never summed as external support; no mixed generalized ratios.'])
    if not np.isfinite(result['memory_guard']['previous_soft_limit_mb']):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    if result['wall_s'] >= 100:
        raise TimeoutError('total audit >=100 seconds')
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Wrote', OUT, result['status'], 'wall_s', result['wall_s'], flush=True)
    for t in trajectories:
        print(t['label'], json.dumps(t['summary_at_ms']), flush=True)
    if not passed:
        raise RuntimeError('Measured audit failed: output retained honestly for investigation')
    return result


if __name__ == '__main__':
    with mg.guarded_run('dynamic_support_audit', **mg.BUDGETS['neural_muscle_closure']) as guard_report:
        run_audit(guard_report)
