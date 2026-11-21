"""Offline force balance at the two archived poses, with native constraint rows."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import linprog

from run_body import HERE, LEGS, MODEL, load_pose, moment_matrix


def constraint_columns(model, data):
    assert model.opt.cone == mj.mjtCone.mjCONE_PYRAMIDAL
    jac = np.zeros((data.nefc, model.nv))
    if mj.mj_isSparse(model):
        for row, start in enumerate(data.efc_J_rowadr):
            end = start + data.efc_J_rownnz[row]
            jac[row, data.efc_J_colind[start:end]] = data.efc_J[start:end]
    else:
        jac[:] = data.efc_J.reshape(jac.shape)
    error = float(np.max(np.abs(jac.T @ data.efc_force - data.qfrc_constraint)))
    assert error < 1e-8
    allowed = []
    bounds = []
    ids = []
    foot_contacts = set()
    limit_names = []
    for row, role in enumerate(data.efc_type):
        item = int(data.efc_id[row])
        if role in (mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS, mj.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL):
            geoms = [model.geom(int(geom)) for geom in data.contact[item].geom]
            for floor, other in (geoms, geoms[::-1]):
                if floor.type[0] == mj.mjtGeom.mjGEOM_PLANE:
                    if other.name[:2] not in LEGS or 'Tarsus' not in other.name:
                        break
                    foot_contacts.add(item)
                    allowed.append(jac[row])
                    bounds.append((0., None))
                    ids.append(row)
                    break
            else:
                allowed.append(jac[row])
                bounds.append((0., None))
                ids.append(row)
        elif role == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            limit_names.append(model.joint(item).name)
            allowed.append(jac[row])
            bounds.append((0., None))
            ids.append(row)
        else:
            raise ValueError(f'Unknown native constraint at row {row}, type {role}')
    return np.asarray(allowed).T, bounds, ids, len(foot_contacts), limit_names, error


def permissions(rows):
    mapped = [row for row in rows if row['local_indices']]
    groups = {}
    for row in mapped:
        key = tuple(row['local_indices'])
        groups.setdefault(key, []).append(row['actuator_index'])
    ident = np.eye(len(rows))
    shared = np.column_stack([ident[:, indices].sum(axis=1) for indices in groups.values()])
    return dict(all_muscles=ident, mapped_only=ident[:, [r['actuator_index'] for r in mapped]],
                shared_pool_hypothesis=shared)


def main(out):
    model = mj.MjModel.from_xml_path(str(MODEL))
    wiring = json.loads((out / 'WIRING.json').read_text())
    variants = permissions(wiring['motor_manifest']['rows'])
    with np.load(HERE / 'body/archived_pose.npz') as artifact:
        poses = dict(keyframe=model.key_qpos[0].copy(), archived_six_feet=artifact['qpos'])
    weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
    report = {}
    for label, pose in poses.items():
        data = load_pose(model, pose)
        reaction, bounds, row_ids, n_contacts, limits, reaction_error = constraint_columns(model, data)
        baseline = data.qfrc_bias - data.qfrc_passive - data.qfrc_actuator
        moments = moment_matrix(model, data)
        passive_force = data.actuator_force.copy()
        data.act[:] = 1.
        mj.mj_fwdActuation(model, data)
        muscle = moments.T * (data.actuator_force - passive_force)
        data.act[:] = np.linspace(0.04, 0.96, model.na)
        mj.mj_fwdActuation(model, data)
        error = float(np.max(np.abs(data.qfrc_actuator - (moments.T @ passive_force + muscle @ data.act))))
        assert error < 1e-8
        report[label] = dict(foot_contacts=n_contacts, native_limit_joints=limits,
                             native_constraint_projection_error=reaction_error,
                             actuator_projection_error=error, variants={})
        for arm, group in variants.items():
            matrix = np.column_stack((muscle @ group, reaction))
            matrix_scaled = matrix / weight
            baseline_scaled = baseline / weight
            constraints = np.column_stack((np.vstack((matrix_scaled, -matrix_scaled)), -np.ones(2 * model.nv)))
            result = linprog(np.r_[np.zeros(matrix.shape[1]), 1.], A_ub=constraints,
                             b_ub=np.r_[baseline_scaled, -baseline_scaled],
                             bounds=[(0., 1.)] * group.shape[1] + bounds + [(0., None)], method='highs')
            if not result.success:
                raise RuntimeError(result.message)
            residual = matrix @ result.x[:-1] - baseline
            peak = float(np.max(abs(residual)))
            assert abs(peak / weight - result.fun) < 1e-8
            report[label]['variants'][arm] = dict(activation_variables=group.shape[1],
                max_normalized_residual=peak / weight, max_native_residual=peak, balance=peak <= 1e-8)
            np.savez_compressed(out / f'static_{label}_{arm}.npz', matrix=matrix, baseline=baseline,
                                solution=result.x, residual=residual, muscle_groups=group,
                                native_constraint_rows=row_ids)
    (out / 'STATIC.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
