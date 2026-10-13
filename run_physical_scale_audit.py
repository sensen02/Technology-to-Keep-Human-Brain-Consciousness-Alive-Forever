"""Physical scale audit: mass, weight and muscle force in model units.

Read-only. Compiles the frozen canonical and zero-floor models, reports total
mass, weight, actuator force capacity and observed forces from saved traces.
This measures model-unit ratios; it does NOT resolve SI calibration.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
import run_zero_floor_branch as zero
OUT = HERE / 'outputs/model_branches/physical_scale'


def sentinel(value):
    """Non-finite model values are recorded with an explicit label, never altered."""
    if isinstance(value, dict):
        return {k: sentinel(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sentinel(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return 'unlimited_or_undefined_in_model'
    return value


def main(report):
    import mujoco
    canonical = closure.cp.load_model()
    branch = mujoco.MjModel.from_xml_path(str(zero.XML))
    rows = {}
    for name, m in (('canonical', canonical), ('zero_floor', branch)):
        mass = m.body_mass.copy()
        total = float(mass.sum())
        weight = total * abs(float(m.opt.gravity[2]))
        gear = m.actuator_gear[:, 0].copy() if m.nu else np.zeros(0)
        rows[name] = dict(
            nbody=int(m.nbody), nu=int(m.nu), nq=int(m.nq), nv=int(m.nv),
            timestep_s=float(m.opt.timestep), gravity=list(map(float, m.opt.gravity)),
            total_mass_model_units=total,
            min_body_mass=float(mass.min()), max_body_mass=float(mass.max()),
            body_mass_zero_count=int(np.count_nonzero(mass == 0)),
            weight_force_model_units=weight,
            gear_unique=bool(np.allclose(gear, gear[0])) if gear.size else None,
            gear_value=float(gear[0]) if gear.size else None,
            force_range=list(map(float, m.actuator_forcerange.ravel())),
            ctrl_range_bounds=[float(m.actuator_ctrlrange.min()), float(m.actuator_ctrlrange.max())],
            dynprm_first_actuator=[float(x) for x in m.actuator_dynprm[0, :3]],
            gainprm_first_actuator=[float(x) for x in m.actuator_gainprm[0, :3]],
            biasprm_first_actuator=[float(x) for x in m.actuator_biasprm[0, :3]],
            number_of_zero_mass_bodies=int(np.count_nonzero(mass == 0)),
        )
    # Observed actuator forces from the saved canonical and zero-floor episodes.
    observed = {}
    with np.load(HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c/closure_traces.npz') as z:
        for key in [k for k in z.files if k.endswith('body_actuator_force')]:
            v = z[key]
            observed['canonical/' + key] = dict(max_abs=float(np.abs(v).max()), max_sum_per_step=float(v.sum(axis=1).max()))
    with np.load(HERE / 'outputs/model_branches/zero_floor/zero_floor_traces.npz') as z:
        for key in [k for k in z.files if k.endswith('body_actuator_force')]:
            v = z[key]
            observed['zero_floor/' + key] = dict(max_abs=float(np.abs(v).max()), max_sum_per_step=float(v.sum(axis=1).max()))
    weight = rows['canonical']['weight_force_model_units']
    ratio = {k: dict(weight_ratio_of_max_sum=float(v['max_sum_per_step'] / weight),
                     weight_ratio_of_max_single=float(v['max_abs'] / weight))
             for k, v in observed.items()}
    result = dict(
        status='MEASURED_MODEL_UNITS_CALIBRATION_STILL_UNRESOLVED',
        label='manual read-only physical-scale audit, not a calibrated biomechanical model',
        models=rows, observed_actuator_forces=observed,
        force_to_weight_ratios=ratio,
        interpretation=[
            'All numbers are model units as MuJoCo sees them; the model was compiled with lengths in mm and gravity 9801 mm/s^2.',
            'If masses were meant as kilograms while forces were scaled for milligram-scale animals, absolute force values are not physically interpretable even though trajectories are deterministic.',
            'The force-to-weight ratios below are dimensionless model-unit ratios; they are reported to expose implausibility, not to calibrate.',
        ],
        limitations=['Single compilation per model, no integration, no episode.',
                     'Does not determine which mass unit the original author intended.',
                     'Saved-force comparison uses previously published episodes, not a new run.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'physical_scale_results.json').write_text(json.dumps(sentinel(result), indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2)[:4000])
    return result


if __name__ == '__main__':
    START = time.time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('physical_scale_audit', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
