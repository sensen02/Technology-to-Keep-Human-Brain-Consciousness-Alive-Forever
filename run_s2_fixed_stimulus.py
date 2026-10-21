"""S2 fixed engineering stimulus; hand-built research, NOT biological calibration.

Sequential case archives bound memory to one case plus its same-branch zero
reference (~7 MB of dense moments for the canonical layout). No neural controller,
pose/gain search, or state overwrites during integration. CLI execution is explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
DT = 0.0001
STEPS = 100
STIMULUS = 0.1
RTOL = ATOL = 1e-10
FIELDS = ('qpos', 'qvel', 'act', 'actuator_force', 'qfrc_actuator',
          'qfrc_passive', 'qfrc_bias', 'qfrc_constraint', 'ctrl')


def dense_moment(model, data):
    """Decode MuJoCo dense or row-compressed actuator x DOF moment layouts.

    Sparse metadata takes precedence for flat buffers, even if buffer size
    happens to equal nu*nv. This avoids mistaking padded sparse data for dense.
    """
    nu, nv = int(model.nu), int(model.nv)
    values = np.asarray(data.actuator_moment, dtype=float)
    if values.shape == (nu, nv):
        return values.copy()
    if values.ndim != 1:
        raise ValueError('unsupported actuator_moment shape')
    sparse = ('moment_rownnz', 'moment_rowadr', 'moment_colind')
    if all(hasattr(data, k) for k in sparse):
        counts, addresses, columns = [np.asarray(getattr(data, k)) for k in sparse]
        if counts.shape != (nu,) or addresses.shape != (nu,) or columns.ndim != 1:
            raise ValueError('unsupported sparse moment metadata')
        if any(x.dtype.kind not in 'iu' for x in (counts, addresses, columns)):
            raise ValueError('sparse indices must be integers')
        result = np.zeros((nu, nv), dtype=float)
        for row, (count, address) in enumerate(zip(counts, addresses)):
            count, address = int(count), int(address)
            if count < 0 or address < 0 or address+count > len(values) or address+count > len(columns):
                raise ValueError('sparse moment row out of bounds')
            col = columns[address:address+count]
            if np.any(col < 0) or np.any(col >= nv) or len(np.unique(col)) != count:
                raise ValueError('invalid or duplicate sparse moment columns')
            result[row, col] = values[address:address+count]
        return result
    if any(hasattr(data, k) for k in sparse):
        raise ValueError('incomplete sparse moment metadata')
    if values.size == nu*nv:
        return values.reshape(nu, nv).copy()
    raise ValueError('unsupported flat moment without sparse metadata')


def activation_slices(model):
    """Map actuators to zero, one, or multiple compiled activation states."""
    adr = np.asarray(model.actuator_actadr)
    num = np.asarray(model.actuator_actnum)
    if adr.shape != (model.nu,) or num.shape != (model.nu,):
        raise ValueError('unsupported activation address layout')
    occupied = []
    result = []
    for a, n in zip(adr, num):
        a, n = int(a), int(n)
        if n < 0 or (n == 0 and a != -1) or (n > 0 and (a < 0 or a+n > model.na)):
            raise ValueError('invalid actadr/actnum')
        indices = list(range(a, a+n)) if n else []
        occupied.extend(indices)
        result.append(indices)
    if sorted(occupied) != list(range(model.na)):
        raise ValueError('overlapping or unaccounted activation states')
    return result


def factorization(moment, force, qfrc):
    calculated = np.asarray(moment).T @ np.asarray(force)
    residual = calculated - np.asarray(qfrc)
    return calculated, residual, bool(np.allclose(calculated, qfrc, rtol=RTOL, atol=ATOL))


def first_difference(a, b, times):
    a, b, times = np.asarray(a), np.asarray(b), np.asarray(times)
    if a.shape != b.shape or len(a) != len(times):
        raise ValueError('matched arrays/times required')
    changed = np.any((a != b).reshape(len(times), -1), axis=1) if a.size else np.zeros(len(times), bool)
    indices = np.flatnonzero(changed)
    return float(times[indices[0]]) if len(indices) else None


def validate_controls(model):
    ranges = np.asarray(model.actuator_ctrlrange)
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if ranges.shape != (model.nu, 2) or not np.isfinite(ranges).all():
        raise ValueError('invalid compiled control ranges')
    if np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError('inverted control range')
    if np.any(limited & ((ranges[:, 0] > STIMULUS) | (ranges[:, 1] < STIMULUS))):
        raise ValueError('fixed 0.1 stimulus unsupported; protocol refused, no amplitude adjustment')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def provenance(cp, loop):
    from flygym.compose.fly import musculoskeletal
    source_paths = [Path(__file__), HERE/'test_s2_fixed_stimulus.py',
        Path(loop.__file__), Path(loop.r39.__file__), Path(cp.__file__),
        HERE/'engine/embodied/declared_pose_projection.py',
        HERE/'engine/embodied/initial_pose_contract.py',
        HERE/'engine/embodied/source_snapshot.py', Path(musculoskeletal.__file__)]
    inputs = [cp.CANONICAL_XML, loop.r39.ROUND37_POSE,
              HERE/'S2_FIXED_STIMULUS_PROTOCOL.zh-CN.md']
    spec = musculoskeletal._load_mjcf(str(cp.CANONICAL_XML))
    assets = sorted({str(Path(mesh.file).resolve()) for mesh in spec.meshes})
    source = {str(p.resolve()): sha(p) for p in source_paths}
    inputs = {str(Path(p).resolve()): sha(p) for p in inputs+assets}
    return source, inputs


def compiled_metadata(model, mujoco):
    keys = ('actuator_ctrlrange', 'actuator_ctrllimited', 'actuator_forcerange',
        'actuator_forcelimited', 'actuator_actrange', 'actuator_actlimited',
        'actuator_actadr', 'actuator_actnum', 'actuator_gaintype', 'actuator_gainprm',
        'actuator_biastype', 'actuator_biasprm', 'actuator_dyntype', 'actuator_dynprm',
        'actuator_trntype', 'actuator_trnid', 'actuator_gear', 'actuator_lengthrange',
        'actuator_acc0', 'actuator_actearly', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr',
        'jnt_limited', 'jnt_range', 'jnt_actfrclimited', 'jnt_actfrcrange', 'dof_jntid')
    result = {k: np.asarray(getattr(model, k)).tolist() for k in keys}
    result.update(nq=int(model.nq), nv=int(model.nv), nu=int(model.nu), na=int(model.na),
        timestep_s=float(model.opt.timestep), integrator=int(model.opt.integrator),
        disableflags=int(model.opt.disableflags), enableflags=int(model.opt.enableflags),
        actuator_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)],
        activation_indices=activation_slices(model))
    labels = []
    for d in range(model.nv):
        j = int(model.dof_jntid[d]); kind = int(model.jnt_type[j])
        offset = d-int(model.jnt_dofadr[j])
        unit = 'model_force' if kind == 2 or (kind == 0 and offset < 3) else 'model_torque'
        labels.append(dict(dof=d, joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), unit=unit))
    result['dof_labels'] = labels
    return result


def run_case(model, pose, target, mujoco):
    """A fresh data object; mj_forward after mj_step aligns ALL endstep fields.

    mj_forward refreshes forces at integrated qpos/qvel/act; it does not replace
    these states. No qpos/act assignments appear inside the stepping loop.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    initial = {k: np.asarray(getattr(data, k)).copy() for k in FIELDS}
    initial['moment'] = dense_moment(model, data)
    initial['qfrc_factorized'], initial['factorization_residual'], initial['factorization_ok'] = factorization(
        initial['moment'], initial['actuator_force'], initial['qfrc_actuator'])
    if target is not None:
        data.ctrl[target] = STIMULUS
    submitted_ctrl = data.ctrl.copy()
    rows = {k: [] for k in FIELDS+('moment', 'time_s', 'qfrc_factorized', 'factorization_residual', 'factorization_ok', 'warning_counts')}
    error = None
    try:
        if np.any(np.asarray(data.warning.number)):
            raise RuntimeError('MuJoCo warning in initial state')
        if not all(np.isfinite(v).all() for v in initial.values()):
            raise RuntimeError('nonfinite initial state')
        for _ in range(STEPS):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            moment = dense_moment(model, data)
            calculated, residual, ok = factorization(moment, data.actuator_force, data.qfrc_actuator)
            for k in FIELDS:
                rows[k].append(np.asarray(getattr(data, k)).copy())
            rows['moment'].append(moment)
            rows['time_s'].append(float(data.time))
            rows['qfrc_factorized'].append(calculated)
            rows['factorization_residual'].append(residual)
            rows['factorization_ok'].append(ok)
            rows['warning_counts'].append(np.asarray(data.warning.number).copy())
            if not all(np.isfinite(rows[k][-1]).all() for k in rows):
                raise RuntimeError('nonfinite endstep sample')
            if np.any(rows['warning_counts'][-1]):
                raise RuntimeError('MuJoCo warning')
            if not ok:
                raise RuntimeError('engineering factorization check failed')
        if not np.isclose(data.time, DT*STEPS, rtol=0, atol=1e-14):
            raise RuntimeError('end time mismatch')
        if not np.array_equal(data.ctrl, submitted_ctrl):
            raise RuntimeError('submitted control changed')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update({'initial_'+k: v for k, v in initial.items()})
    arrays['initial_time_s'] = np.asarray(0.)
    return arrays, error


def summarize_case(arrays, baseline, target, indices):
    result = dict(samples=len(arrays['time_s']),
        max_abs_factorization_residual=float(np.max(np.abs(arrays['factorization_residual']), initial=0)),
        factorization_allclose=bool(arrays['factorization_ok'].all()),
        peak_abs_activation=float(np.max(np.abs(arrays['act']), initial=0)),
        peak_abs_actuator_force=float(np.max(np.abs(arrays['actuator_force']), initial=0)),
        peak_abs_qfrc_actuator=float(np.max(np.abs(arrays['qfrc_actuator']), initial=0)))
    if target is not None:
        act = arrays['act'][:, indices[target]]
        zero_act = baseline['act'][:, indices[target]]
        result.update(target=target, target_activation_indices=indices[target],
            target_peak_abs_activation=float(np.max(np.abs(act), initial=0)),
            target_peak_abs_force=float(np.max(np.abs(arrays['actuator_force'][:, target]), initial=0)),
            target_peak_abs_force_delta_vs_zero=float(np.max(np.abs(arrays['actuator_force'][:, target]-baseline['actuator_force'][:, target]), initial=0)),
            first_force_difference_s=first_difference(arrays['actuator_force'][:, target], baseline['actuator_force'][:, target], arrays['time_s']),
            first_activation_difference_s=first_difference(act, zero_act, arrays['time_s']),
            first_qfrc_difference_s=first_difference(arrays['qfrc_actuator'], baseline['qfrc_actuator'], arrays['time_s']),
            first_qpos_difference_s=first_difference(arrays['qpos'], baseline['qpos'], arrays['time_s']))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent', type=Path, default=HERE/'outputs/s2_fixed_stimulus')
    args = parser.parse_args(argv)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import engine.canonical_physics as cp
    import run_neural_muscle_closure as loop
    from engine.embodied.declared_pose_projection import project_scalar_ranges
    from engine.embodied.source_snapshot import build_source_snapshot
    dest = args.output_parent.resolve()/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12])
    dest.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', protocol='S2_FIXED_STIMULUS_PROTOCOL.zh-CN.md v1',
        manual_research_not_product=True, biological_calibration=False,
        standing_or_neural_control_claim=False, dt_s=DT, steps=STEPS, stimulus=STIMULUS,
        numerical_tolerance=dict(rtol=RTOL, atol=ATOL, scope='engineering identity only'),
        sample_semantics='endstep state then mj_forward forces at that state',
        latency_semantics='first exact nonzero grid difference, NOT biological latency',
        units='model force/DOF generalized force; no established biological SI force calibration',
        memory_plan='sequential per-case compressed NPZ; only zero reference and current case retained',
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, mujoco=mujoco.__version__), branches={})
    source = inputs = None
    try:
        source, inputs = provenance(cp, loop)
        report.update(source_hashes_before=source, input_hashes_before=inputs)
        build_source_snapshot(source, dest/'source_snapshot.zip')
        if sha(cp.CANONICAL_XML) != cp.CANONICAL_XML_SHA256:
            raise RuntimeError('canonical XML identity mismatch')
        model, historical, historical_gate, _, _ = loop.prepare_body()
        pose, projection = project_scalar_ranges(model, historical)
        scratch = mujoco.MjData(model)
        pose[2] = cp.seat_root(model, scratch, pose)
        gate = cp.stance_facts(model, scratch).as_dict()
        report.update(historical_gate=historical_gate, projection=projection, projected_seated_gate=gate)
        np.savez_compressed(dest/'initial_pose.npz', historical_qpos=historical, projected_seated_qpos=pose)
        del model, scratch
        for branch in ('native_floor', 'zero_floor'):
            model = cp.load_model()  # independent compiled model for EACH branch
            if not np.isclose(model.opt.timestep, DT, rtol=0, atol=1e-15):
                raise ValueError('compiled timestep not 0.0001s; no clock adjustment')
            validate_controls(model)
            native_ranges = model.actuator_ctrlrange.copy()
            if branch == 'zero_floor':
                model.actuator_ctrlrange[:, 0] = 0
            metadata = compiled_metadata(model, mujoco)
            metadata['native_ctrlrange'] = native_ranges.tolist()
            scratch = mujoco.MjData(model); scratch.qpos[:] = pose
            mujoco.mj_forward(model, scratch)
            moment = dense_moment(model, scratch)
            metadata['initial_static_moment_nonzero_dofs'] = [
                [dict(dof=int(d), value=float(moment[i, d]), direction=int(np.sign(moment[i, d])))
                 for d in np.flatnonzero(moment[i])] for i in range(model.nu)]
            folder = dest/branch; folder.mkdir()
            np.savez_compressed(folder/'initial_compiled.npz', qpos=pose,
                qvel=scratch.qvel, act=scratch.act, ctrl=scratch.ctrl, moment=moment,
                **{k: np.asarray(getattr(model, k)) for k in metadata if hasattr(model, k) and isinstance(getattr(model, k), np.ndarray)})
            dump(folder/'metadata.json', metadata)
            report['branches'][branch] = dict(metadata=str(folder/'metadata.json'), cases={})
            zero = None
            for target in [None, None]+list(range(model.nu)):
                label = ('zero_0' if zero is None else 'zero_1') if target is None else f'actuator_{target:03d}'
                arrays, error = run_case(model, pose, target, mujoco)
                np.savez_compressed(folder/(label+'.npz'), **arrays)
                entry = dict(npz=str(folder/(label+'.npz')), error=error)
                report['branches'][branch]['cases'][label] = entry
                if error:
                    raise RuntimeError(branch+'/'+label+': '+error)
                entry.update(summarize_case(arrays, zero, target, metadata['activation_indices']))
                if target is None:
                    entry['zero_submission_not_passive_claim'] = True
                    if zero is None:
                        zero = arrays
                    else:
                        exact = all(np.array_equal(arrays[k], zero[k]) for k in arrays)
                        entry['exact_repeat'] = exact
                        if not exact:
                            raise RuntimeError('zero repeat not exact')
                dump(dest/'report.json', report)
            del zero, arrays, model, scratch
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', failure=type(exc).__name__+': '+str(exc))
    finally:
        if source is not None:
            report['source_hashes_after'] = {p: sha(p) if Path(p).is_file() else None for p in source}
            report['input_hashes_after'] = {p: sha(p) if Path(p).is_file() else None for p in inputs}
            report['hashes_unchanged'] = (report['source_hashes_after'] == source and report['input_hashes_after'] == inputs)
            if not report['hashes_unchanged']:
                report.update(status='FAIL', hash_failure='source/input drift')
        dump(dest/'report.json', report)
    print(str(dest), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
