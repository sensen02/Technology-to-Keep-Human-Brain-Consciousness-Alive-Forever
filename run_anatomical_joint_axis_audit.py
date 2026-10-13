"""Anatomical joint-axis audit from the published kinematic leg model dataset.

Read-only. The dataset (Haustein et al. 2024, Front Bioeng Biotechnol 12:1357598; Harvard Dataverse
doi:10.7910/DVN/XWREDA) provides, per leg, the joint centre keypoint and the two condyle vertex
positions for every leg joint. For a hinge, the rotation axis runs through the condyles, so the
condyle pair gives an anatomically derived axis direction rather than a model convention.

The comparison that matters is frame-invariant: the ANGLE between the joint axis and the segment
direction, measured in the dataset, against the same angle in this project's model.
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
DS = HERE / 'data/kinematic_leg_model'
CSV = DS / 'KinematicModel_InitalPosture_Coords3D-2.csv'
OUT = HERE / 'outputs/model_branches/anatomical_joint_axis'
LEG_PREFIX = {'LF': 'L1', 'LM': 'L2', 'LH': 'L3', 'RF': 'R1', 'RM': 'R2', 'RH': 'R3'}
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_points():
    pts = {}
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            name = row.get(',x') or list(row.values())[0]
            pts[name] = np.array([float(row['x']), float(row['y']), float(row['z'])])
    return pts


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def main(report):
    import mujoco
    pts = load_points()
    per_leg = {}
    for leg, pref in LEG_PREFIX.items():
        tibia_dir = pts[f'{pref}-TiTar'] - pts[f'{pref}-FeTi']          # tibia, proximal->distal
        tarsus_dir = pts[f'{pref}-Tar'] - pts[f'{pref}-TiTar']          # tarsus, proximal->distal
        axis = pts[f'{pref}-TiTarV2'] - pts[f'{pref}-TiTarV1']          # condyle-to-condyle
        femur_dir = pts[f'{pref}-FeTi'] - pts[f'{pref}-TrFe']
        fe_axis = pts[f'{pref}-FeTiV2'] - pts[f'{pref}-FeTiV1']
        per_leg[leg] = dict(
            axis_unit=[round(float(x), 6) for x in axis / np.linalg.norm(axis)],
            condyle_separation_mm=round(float(np.linalg.norm(axis)), 6),
            angle_axis_to_tibia_deg=round(angle(axis, tibia_dir), 2),
            angle_axis_to_tarsus_deg=round(angle(axis, tarsus_dir), 2),
            angle_axis_to_femur_deg=round(angle(axis, femur_dir), 2),
            angle_femur_tibia_axis_deg=round(angle(fe_axis, tibia_dir), 2),
            tibia_length_mm=round(float(np.linalg.norm(tibia_dir)), 4),
            tarsus_length_mm=round(float(np.linalg.norm(tarsus_dir)), 4))
    # our model: angle between the added tarsus hinge axis and the tibia segment direction
    model = mujoco.MjModel.from_xml_path(str(HERE / 'outputs/model_branches/tarsus_dof/fruitfly_tarsus_dof.xml'))
    d = mujoco.MjData(model)
    d.qpos[:] = model.key_qpos[0]
    d.qvel[:] = 0
    mujoco.mj_forward(model, d)
    our = {}
    for leg in LEG_PREFIX:
        tib_body = model.body(f'{leg}Tibia').id
        tar_body = model.body(f'{leg}Tarsus1').id
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint_{leg}Tarsus1_pitch')
        axis_world = d.xmat[tar_body].reshape(3, 3) @ np.asarray(model.jnt_axis[jid])
        seg_world = d.xpos[tar_body] - d.xpos[tib_body]
        our[leg] = dict(angle_axis_to_tibia_segment_deg=round(angle(axis_world, seg_world), 2),
                        tibia_segment_length_mm=round(float(np.linalg.norm(seg_world)), 4))
    ds_angles = [per_leg[l]['angle_axis_to_tibia_deg'] for l in per_leg]
    our_angles = [our[l]['angle_axis_to_tibia_segment_deg'] for l in our]
    result = dict(
        status='MEASURED_ANATOMICAL_JOINT_AXIS',
        label='manual read-only derivation from a published dataset; no model was changed by this audit',
        question='What is the anatomically derived tibia-tarsus rotation axis, and how far is this project\'s '
                 'convention-derived axis from it?',
        source=dict(dataset='Haustein, Blanke, Bockemuhl, Buschges (2024) Front Bioeng Biotechnol 12:1357598; '
                            'Harvard Dataverse doi:10.7910/DVN/XWREDA',
                    csv=str(CSV), csv_sha256=sha(CSV),
                    method='for a hinge the rotation axis runs through the two condyles, which the dataset gives '
                           'per leg as the V1/V2 vertex pair; the axis is their difference'),
        dataset_per_leg=per_leg,
        our_model_per_leg=our,
        comparison=dict(
            dataset_angle_axis_to_tibia_deg=dict(mean=round(float(np.mean(ds_angles)), 2),
                                                 min=round(float(np.min(ds_angles)), 2),
                                                 max=round(float(np.max(ds_angles)), 2)),
            our_angle_axis_to_tibia_deg=dict(mean=round(float(np.mean(our_angles)), 2),
                                             min=round(float(np.min(our_angles)), 2),
                                             max=round(float(np.max(our_angles)), 2)),
            difference_deg=round(float(abs(np.mean(ds_angles) - np.mean(our_angles))), 2)),
        findings=['the anatomically derived tibia-tarsus axis is strongly oblique rather than perpendicular to the '
                  'leg plane, and it varies between legs',
                  'the axis this project derived by convention (parallel to the existing tibia pitch axis) sits at a '
                  'measurably different angle to the tibia segment than the anatomical one',
                  'the dataset also provides a femur-tibia axis, so the same derivation is available for a future '
                  'femur degree of freedom',
                  'joint ranges of motion remain unavailable as measurements: the source states its angle constraints '
                  'were empirically adjusted because the true ROMs are unknown'],
        not_licensed=['using the dataset axis without mapping the dataset body frame onto this model\'s body frames',
                      'treating any published range as a measured range of motion',
                      'any muscle attachment point'],
        limitations=['The dataset is a kinematic model of one female fly; per-leg axes vary and are reported individually.',
                     'The frame mapping is not solved here: only the frame-invariant axis-to-segment angle is compared.',
                     'No model was modified and no episode was run in this audit.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'anatomical_joint_axis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(per_leg={k: {kk: vv for kk, vv in v.items() if 'angle' in kk or 'axis_unit' in kk}
                                 for k, v in per_leg.items()},
                          comparison=result['comparison'], our=our), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('anatomical_joint_axis', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
