"""v4: build the tarsus extension with the ANATOMICALLY derived joint axis.

Frame mapping (declared, and the only non-trivial step):
  In the source dataset each leg has a segment chain (ThCx, CxTr, TrFe, FeTi, TiTar, Tar) and a
  condyle pair per joint. A segment-relative basis is built from the source geometry:
      u = tibia direction (FeTi -> TiTar)
      n = unit normal of the plane spanned by the femur direction (TrFe -> FeTi) and u
      w = u x n
  The anatomical axis is expressed in that basis as coefficients (a_u, a_n, a_w). The SAME
  coefficients are then applied to the corresponding basis built from this model's own rest
  geometry, and the result is converted into the tarsus body frame. So the axis direction is
  anatomical; only the frame correspondence is an assumption, and it is stated.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import csv, json, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
import run_tarsus_dof_structure as S
DS = HERE / 'data/kinematic_leg_model'
CSV = DS / 'KinematicModel_InitalPosture_Coords3D-2.csv'
OUT = HERE / 'outputs/model_branches/tarsus_dof'
LEG_PREFIX = {'LF': 'L1', 'LM': 'L2', 'LH': 'L3', 'RF': 'R1', 'RM': 'R2', 'RH': 'R3'}
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(v):
    return v / np.linalg.norm(v)


def main(report):
    import mujoco
    pts = {}
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            name = row.get(',x') or list(row.values())[0]
            pts[name] = np.array([float(row['x']), float(row['y']), float(row['z'])])

    base = mujoco.MjModel.from_xml_path(str(S.zero_xml()))
    d = mujoco.MjData(base)
    d.qpos[:] = base.key_qpos[0]
    d.qvel[:] = 0
    mujoco.mj_forward(base, d)

    axes, mapping = {}, {}
    for leg, pref in LEG_PREFIX.items():
        # --- source frame coefficients
        u_s = unit(pts[f'{pref}-TiTar'] - pts[f'{pref}-FeTi'])
        femur_s = unit(pts[f'{pref}-FeTi'] - pts[f'{pref}-TrFe'])
        n_s = unit(np.cross(femur_s, u_s))
        w_s = np.cross(u_s, n_s)
        a_s = unit(pts[f'{pref}-TiTarV2'] - pts[f'{pref}-TiTarV1'])
        coeffs = np.array([a_s @ u_s, a_s @ n_s, a_s @ w_s])
        # --- our model's corresponding basis, from rest geometry
        tib = base.body(f'{leg}Tibia').id
        tar = base.body(f'{leg}Tarsus1').id
        fem = base.body(f'{leg}Femur').id
        u_o = unit(d.xpos[tar] - d.xpos[tib])
        # NOTE: only the front legs have a Trochanter body in this model, so the femur segment
        # direction is taken as femur -> tibia, which is available on all six legs.
        femur_o = unit(d.xpos[tib] - d.xpos[fem])
        n_o = unit(np.cross(femur_o, u_o))
        w_o = np.cross(u_o, n_o)
        axis_world = coeffs[0] * u_o + coeffs[1] * n_o + coeffs[2] * w_o
        axis_world = unit(axis_world)
        jid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, f'joint_{leg}Tibia_pitch')
        tib_axis_world = unit(d.xmat[tib].reshape(3, 3) @ np.asarray(base.jnt_axis[jid]))
        axes[leg] = (d.xmat[tar].reshape(3, 3).T @ axis_world).tolist()
        mapping[leg] = dict(
            source_basis_coefficients=[round(float(c), 6) for c in coeffs],
            source_axis_body_coords=[round(float(x), 6) for x in a_s],
            angle_to_tibia_in_source_deg=round(float(np.degrees(
                np.arccos(np.clip(a_s @ u_s, -1, 1)))), 2),
            model_axis_angle_to_tibia_deg=round(float(np.degrees(
                np.arccos(np.clip(axis_world @ u_o, -1, 1)))), 2),
            angle_to_existing_tibia_joint_axis_deg=round(float(np.degrees(
                np.arccos(np.clip(axis_world @ tib_axis_world, -1, 1)))), 2),
            axis_local=[round(float(x), 8) for x in axes[leg]])

    params = S.muscle_params(Path(closure.cp.CANONICAL_XML))
    text, _ = S.build_xml(S.zero_xml(), params, 2.5e-5, axes)
    xml = OUT / 'fruitfly_tarsus_dof_v4.xml'
    xml.write_text(text)
    model = mujoco.MjModel.from_xml_path(str(xml))
    doc = dict(status='V4_ANATOMICAL_AXIS_BUILT',
               label='manual labelled extension using an anatomically derived axis; not a behaviour result',
               method='anatomical axis expressed in a segment-relative basis (tibia direction, femur-derived plane '
                      'normal, cross product) in the source dataset, then the same coefficients applied to the same '
                      'basis built from this model rest geometry, then converted to the tarsus body frame',
               source=dict(csv=str(CSV), csv_sha256=sha(CSV),
                           note='condyle pairs define the hinge axis; dataset doi:10.7910/DVN/XWREDA'),
               per_leg=mapping,
               dimensions=dict(nq=int(model.nq), nv=int(model.nv), nu=int(model.nu)),
               declared_assumption='the frame correspondence between the source dataset and this model is assumed to '
                                   'be captured by the segment-relative basis; it is not proven',
               provenance=dict(base_branch_xml=str(S.zero_xml()), branch_xml=str(xml), branch_sha256=sha(xml)),
               memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(doc['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        doc['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / 'tarsus_dof_v4_axis.json').write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {kk: v[kk] for kk in ('angle_to_tibia_in_source_deg', 'model_axis_angle_to_tibia_deg',
                                               'angle_to_existing_tibia_joint_axis_deg')} for k, v in mapping.items()},
                     indent=2))
    print('built', xml)
    return doc


if __name__ == '__main__':
    with closure.mg.guarded_run('tarsus_dof_v4_axis', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
