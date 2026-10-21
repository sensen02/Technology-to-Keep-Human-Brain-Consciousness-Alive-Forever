"""Explicit derivative pose, never a silent repair of frozen experimental data.
Coordinate-wise nearest scalar-range projection; does not optimize standing.
"""
import numpy as np
from engine.embodied.initial_pose_contract import model_limit_report


def project_scalar_ranges(model, qpos):
    original = np.asarray(qpos, dtype=float)
    if original.shape != (model.nq,) or not np.isfinite(original).all():
        raise ValueError('finite model-sized qpos required')
    result = original.copy()
    changes = []
    for j in range(model.njnt):
        if not model.jnt_limited[j] or model.jnt_type[j] not in (2, 3):
            continue
        address = int(model.jnt_qposadr[j])
        lo, hi = map(float, model.jnt_range[j])
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError('invalid scalar range')
        value = float(np.clip(original[address], lo, hi))
        if value != original[address]:
            changes.append(dict(joint=model.joint(j).name, qpos_address=address,
                                original=float(original[address]), projected=value,
                                delta=value-float(original[address])))
            result[address] = value
    report = model_limit_report(model, result)
    if not report['within_declared_scalar_ranges']:
        raise AssertionError(report)
    return result, dict(rule='Nearest coordinate-wise projection onto native scalar limits, then separately seat root height; no dynamics or stability objective',
                        changes=changes, scalar_limit_report=report,
                        frozen_pose_modified=False, derivative_pose_modified=bool(changes))
