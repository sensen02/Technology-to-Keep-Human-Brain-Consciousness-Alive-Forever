"""Read-only, same-pose force audit of saved commands; not a standing proof."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import run_dimensionless_support_search as search
cp, mg = search.cp, search.mg
HERE = Path(__file__).resolve().parent
SOURCE = search.OUT
OUT = HERE / 'outputs/workspace_recheck/support_force_audit.json'
FORCE_UNIT = 'model_mass_unit * mm / s^2'
TORQUE_UNIT = 'model_mass_unit * mm^2 / s^2'


def digest_array(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def force_on_nonfloor(frame, local_force, geom1, geom2, floor):
    """mj_contactForce is force on geom2; frame stores world axes as rows."""
    if (geom1 == floor) == (geom2 == floor):
        raise ValueError('exactly one floor geom required')
    return (1. if geom1 == floor else -1.) * np.asarray(frame).reshape(3, 3).T @ np.asarray(local_force)[:3]


def contact_forces(m, d, floor, feet=()):
    total = np.zeros(3)
    rows = []
    for i, c in enumerate(d.contact[:d.ncon]):
        if floor not in (c.geom1, c.geom2):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, local)
        world = force_on_nonfloor(c.frame, local, c.geom1, c.geom2, floor)
        other = c.geom2 if c.geom1 == floor else c.geom1
        total += world
        rows.append(dict(contact_index=i, geom1=int(c.geom1), geom2=int(c.geom2),
                         other_geom=m.geom(other).name, is_named_foot=other in feet,
                         efc_address=int(c.efc_address), distance_mm=float(c.dist),
                         frame_rows_world=c.frame.reshape(3, 3).tolist(),
                         local_force_torque=local.tolist(), force_world=world.tolist()))
    foot_names = sorted({r['other_geom'] for r in rows if r['is_named_foot']})
    return dict(force_world=total.tolist(), total_contact_count=int(d.ncon),
                floor_contact_count=len(rows), named_foot_contact_count=sum(r['is_named_foot'] for r in rows),
                distinct_named_foot_geom_count=len(foot_names), distinct_named_foot_geoms=foot_names,
                nonfoot_floor_contact_count=sum(not r['is_named_foot'] for r in rows), rows=rows)


def com_acceleration(m, d):
    if np.any(d.qvel != 0):
        raise ValueError('Jdot*qvel omitted: this audit requires exactly qvel=0')
    weighted = np.zeros((3, m.nv))
    for b in range(1, m.nbody):
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jac, None, b)
        weighted += m.body_mass[b] * jac
    mass = float(m.body_mass[1:].sum())
    return weighted @ d.qacc / mass, weighted


def constraint_jacobian(m, d):
    if not mujoco.mj_isSparse(m):
        return np.asarray(d.efc_J).reshape(d.nefc, m.nv).copy()
    j = np.zeros((d.nefc, m.nv))
    for r in range(d.nefc):
        a, n = int(d.efc_J_rowadr[r]), int(d.efc_J_rownnz[r])
        j[r, d.efc_J_colind[a:a+n]] = d.efc_J[a:a+n]
    return j


def coordinates(m):
    out = []
    for v in range(m.nv):
        j = int(m.dof_jntid[v])
        offset = v-int(m.jnt_dofadr[j])
        translation = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE or (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and offset < 3)
        out.append(dict(dof=v, joint=m.joint(j).name, component=offset,
                        generalized_force_unit=FORCE_UNIT if translation else TORQUE_UNIT,
                        acceleration_unit='mm/s^2' if translation else 'rad/s^2'))
    return out


def error_check(actual, expected, atol, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    error = np.abs(a-b)
    bound = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    return dict(max_abs_error=float(error.max(initial=0)), abs_error=error.tolist(),
                atol=atol, rtol=rtol, component_tolerance=bound.tolist(), within_tolerance=bool(np.all(error <= bound)))


def convention_probe():
    # Separate SI sphere/plane model, not a modification or ablation of the canonical model.
    xml = '''<mujoco><option gravity="1 2 -9.81"/><worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body pos="0 0 .099"><freejoint/><geom name="sphere" type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    floor = m.geom('floor').id
    contacts = contact_forces(m, d, floor)
    a, _ = com_acceleration(m, d)
    contact = np.array(contacts['force_world'])
    expected = float(m.body_mass.sum()) * (a-m.opt.gravity)
    check = error_check(contact, expected, 1e-10)
    return dict(model='independent 1 kg sphere at z=.099m, radius=.1m, plane, qvel=0',
                units='N; acceleration m/s^2', contact=contacts, expected_force_from_COM=expected.tolist(),
                force_balance=check, upward_force=bool(contact[2] > 0),
                convention='force on geom2; + if floor geom1, - if floor geom2; frame.T @ local force')


def selected_rows(doc):
    rows = []
    for condition in ('covered74', 'all90'):
        best = doc['best'][condition]
        matches = [r for r in doc['rows'] if r['label'] == best['label']]
        if len(matches) != 1:
            raise ValueError('best pose row missing or ambiguous')
        row = matches[0]
        np.testing.assert_array_equal(row['qpos'], best['qpos'])
        np.testing.assert_array_equal(row[condition]['activation'], best['activation'])
        if not any(r['label'] == row['label'] for r in rows):
            rows.append(row)
    if len(rows) != 2 or digest_array(rows[0]['qpos']) == digest_array(rows[1]['qpos']):
        raise ValueError('expected two distinct saved best poses')
    return rows


def snapshot(m, qpos, activation, condition):
    ev = search.FixedPose(m, qpos)
    ev.acceleration(np.asarray(activation))
    d = ev.data
    contact = contact_forces(m, d, cp.floor_geom_id(m), set(cp.tarsus_geom_ids(m)))
    a, weighted = com_acceleration(m, d)
    mass = float(m.body_mass.sum())
    j = constraint_jacobian(m, d)
    decomposition = {}
    noncontact_root = np.zeros(3)
    reconstructed = np.zeros(m.nv)
    for typ in range(int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC) + 1):
        mask = d.efc_type == typ
        q = j[mask].T @ d.efc_force[mask]
        name = mujoco.mjtConstraint(int(typ)).name
        decomposition[name] = dict(row_count=int(mask.sum()), nonzero_force_row_count=int(np.count_nonzero(d.efc_force[mask])), efc_force=d.efc_force[mask].tolist(), generalized_force=q.tolist(),
                                   root_force_world=q[:3].tolist(), root_torque=q[3:6].tolist(), joint_torques=q[6:].tolist())
        reconstructed += q
        if int(typ) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS):
            noncontact_root += q[:3]
    gravity = mass * m.opt.gravity
    # For a single free-root tree, virtual uniform translation selects the first three DOFs.
    # Internal muscle/passive/limit terms cancel here; retain measured values rather than assume.
    other = d.qfrc_passive[:3] + d.qfrc_actuator[:3] + d.qfrc_applied[:3] + d.xfrc_applied[1:, :3].sum(axis=0) + noncontact_root
    rhs = gravity + np.array(contact['force_world']) + other
    force_tol = 1e-8 * max(float(np.linalg.norm(gravity)), 1.)
    balance = error_check(mass*a, rhs, force_tol)
    mqacc = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, mqacc, d.qacc)
    generalized_rhs = d.qfrc_passive + d.qfrc_actuator + d.qfrc_applied + d.qfrc_constraint - d.qfrc_bias
    # Applied Cartesian forces are reset to zero; assert before using this RHS.
    if np.any(d.xfrc_applied):
        raise ValueError('unexpected Cartesian applied force')
    checks = dict(COM_force_balance=balance,
                  contact_force_vs_constraint_root=error_check(contact['force_world'], reconstructed[:3]-noncontact_root, force_tol),
                  reconstructed_constraint_root_force=error_check(reconstructed[:3], d.qfrc_constraint[:3], 1e-9),
                  reconstructed_constraint_torques=error_check(reconstructed[3:], d.qfrc_constraint[3:], 1e-9),
                  generalized_equation_root_force=error_check(mqacc[:3], generalized_rhs[:3], force_tol),
                  generalized_equation_torques=error_check(mqacc[3:], generalized_rhs[3:], 1e-8),
                  COM_J_root_mass_identity=error_check(weighted[:, :3], np.eye(3)*mass, 1e-12))
    return dict(condition=condition, qpos=d.qpos.tolist(), qpos_sha256=digest_array(d.qpos),
                activation=np.asarray(activation).tolist(), qvel_zero=bool(np.all(d.qvel == 0)),
                contact=contact, mass_model_units=mass, weight_force=float(np.linalg.norm(gravity)),
                support_over_weight=float(np.dot(contact['force_world'], -m.opt.gravity/np.linalg.norm(m.opt.gravity))/np.linalg.norm(gravity)),
                COM_acceleration_mm_s2=a.tolist(), root_translation_qacc_mm_s2=d.qacc[:3].tolist(),
                mass_times_COM_acceleration=(mass*a).tolist(), gravity_force=gravity.tolist(),
                other_external_force_world=other.tolist(), gravity_contact_other_sum=rhs.tolist(),
                other_terms=dict(passive_root_force=d.qfrc_passive[:3].tolist(), actuator_root_force=d.qfrc_actuator[:3].tolist(),
                                 applied_generalized_root_force=d.qfrc_applied[:3].tolist(),
                                 applied_cartesian_force_sum=d.xfrc_applied[1:, :3].sum(axis=0).tolist(),
                                 noncontact_constraint_root_force=noncontact_root.tolist()),
                constraint_storage='sparse' if mujoco.mj_isSparse(m) else 'dense',
                constraint_by_type=decomposition, raw_vectors={key: getattr(d, key).tolist() for key in
                    ('qacc', 'qfrc_passive', 'qfrc_actuator', 'qfrc_bias', 'qfrc_constraint', 'qfrc_applied', 'xfrc_applied')},
                checks=checks)


def run_audit(guard_report):
    start = time.monotonic()
    before = cp.sha256_file(cp.CANONICAL_XML)
    if before != cp.CANONICAL_XML_SHA256:
        raise ValueError('frozen XML hash mismatch')
    source_hash = cp.sha256_file(SOURCE)
    doc = json.loads(SOURCE.read_text())
    if doc['provenance']['xml_sha256_before'] != before or doc['provenance']['xml_sha256_after'] != before:
        raise ValueError('source used different canonical XML')
    m = cp.load_model()
    if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(m.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('requires one free root plus hinges')
    if len(set(m.body_rootid[m.body_mass > 0])) != 1 or np.any(m.dof_armature[:3]):
        raise ValueError('requires single body tree and no root translation armature')
    np.testing.assert_array_equal(m.actuator_ctrlrange, doc['native_control_ranges'])
    assert [m.actuator(i).name for i in range(m.nu)] == doc['actuator_order']
    covered, mapping, _ = search.coverage(m)
    excluded = np.setdiff1d(np.arange(m.nu), covered)
    poses = []
    for row in selected_rows(doc):
        if time.monotonic()-start > 90:
            raise TimeoutError('audit budget exceeded')
        commands = {'native_floor': m.actuator_ctrlrange[:, 0].copy(),
                    'covered74': np.array(row['covered74']['activation']), 'all90': np.array(row['all90']['activation'])}
        np.testing.assert_array_equal(commands['covered74'][excluded], commands['native_floor'][excluded])
        for a in commands.values():
            if np.any(a < m.actuator_ctrlrange[:, 0]) or np.any(a > m.actuator_ctrlrange[:, 1]):
                raise ValueError('saved activation out of native bounds')
        snapshots = [snapshot(m, row['qpos'], a, name) for name, a in commands.items()]
        poses.append(dict(label=row['label'], source_row_index=doc['rows'].index(row),
                          selected_as_best_for=[k for k, v in doc['best'].items() if v['label'] == row['label']],
                          qpos_sha256=digest_array(row['qpos']), snapshots=snapshots))
    probe = convention_probe()
    after = cp.sha256_file(cp.CANONICAL_XML)
    checks_ok = all(c['within_tolerance'] for p in poses for s in p['snapshots'] for c in s['checks'].values())
    result = dict(status='MEASURED_WITHIN_DECLARED_TOLERANCES' if checks_ok and probe['force_balance']['within_tolerance'] and probe['upward_force'] and before == after else 'RESIDUAL_REQUIRES_INVESTIGATION',
        poses=poses, contact_convention_probe=probe, coordinate_units=coordinates(m),
        units=dict(force=FORCE_UNIT, torque=TORQUE_UNIT, mass='model_mass_unit; physical mass calibration not established',
                   support_over_weight='dimensionless; independent of mass-unit interpretation', contact_local='first 3 force; last 3 torque'),
        model_environment=dict(viscosity=float(m.opt.viscosity), density=float(m.opt.density), wind=m.opt.wind.tolist(),
                               gravity_mm_s2=m.opt.gravity.tolist(), body_gravcomp=m.body_gravcomp.tolist(),
                               root_translation_armature=m.dof_armature[:3].tolist(),
                               note='Fluid effects, gravcomp and springs enter qfrc_passive. Applied arrays reset every snapshot; raw values retained.'),
        method=dict(state='Saved qpos unchanged, no reseating; FixedPose resets MjData every call, qvel=0, act=ctrl, warmstart=0',
                    COM='sum(body_mass * mj_jacBodyCom @ qacc)/sum(body_mass); Jdot*qvel=0',
                    ground='mj_contactForce; contact.frame.T; sign chosen from floor geom1/geom2',
                    constraints='group efc rows by type, J.T @ efc_force, dense/sparse supported',
                    execution='serial, single-thread environment, shared mg.guarded_run; six forwards, no integration/optimization'),
        provenance=dict(source=str(SOURCE), source_sha256=source_hash, source_sha256_after=cp.sha256_file(SOURCE),
                        canonical_xml=str(cp.CANONICAL_XML), xml_sha256_before=before, xml_sha256_after=after,
                        expected_frozen_sha256=cp.CANONICAL_XML_SHA256, mujoco_version=mujoco.__version__,
                        code_sha256=cp.sha256_file(Path(__file__)), interpreter=sys.executable), mapping=mapping,
        wall_s=time.monotonic()-start, memory_guard=guard_report('completed'),
        limits=['Hand-built research prototype, not standing proof or finished product.',
                'Instantaneous qvel=0 snapshots only; no robustness, trajectory or neural success claim.',
                'Independent internal joint-limit forces have no net root translational force but may affect contacts indirectly; no snapshot causality claim.',
                'No model modifications, ablations, neural bypass or mixed force/torque norm ratios.',
                'Only saved best poses and their own saved commands audited, not a new search.'])
    if not np.isfinite(result['memory_guard']['previous_soft_limit_mb']):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Wrote', OUT, result['status'], 'wall_s', result['wall_s'], flush=True)
    for p in poses:
        for s in p['snapshots']:
            print(p['label'], s['condition'], 'support/weight', s['support_over_weight'],
                  'COM az', s['COM_acceleration_mm_s2'][2], 'balance_error', s['checks']['COM_force_balance']['max_abs_error'], flush=True)
    if result['status'] == 'RESIDUAL_REQUIRES_INVESTIGATION':
        raise RuntimeError('Measured residual outside tolerance: inspect output before interpreting')
    return result


if __name__ == '__main__':
    with mg.guarded_run('support_force_audit', **mg.BUDGETS['neural_muscle_closure']) as report:
        run_audit(report)
