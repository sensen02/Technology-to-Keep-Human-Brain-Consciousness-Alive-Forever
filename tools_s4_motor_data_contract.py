"""Structural admission for motor events and muscle observations; no parameter fitting."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from tools_s4_paired_data_contract import validate as validate_common, load, MAX_POINTS


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate(data):
    if not isinstance(data, dict):
        return dict(contract_valid=False, biological_validation=False, calibration_executed=False,
                    errors=[dict(path='/', message='object required')])
    # Reuse v1 source, split, exact root, kinematics and neural clock rules.
    # Only the internal structural adapter supplies the sensory channel constant;
    # it does not claim a motor cell is a claw neuron.
    adapted = copy.deepcopy(data)
    adapted['scope'] = 'sensory_claw_hook'
    adapted['units'] = {k: v for k, v in data.get('units', {}).items()} if isinstance(data.get('units'), dict) else {}
    trials = adapted.get('trials')
    if isinstance(trials, list):
        for trial in trials:
            if isinstance(trial, dict):
                trial['channel'] = 'claw_position'
    result = validate_common(adapted)
    errors = result['errors']
    def require(ok, path, message):
        if not ok:
            errors.append(dict(path=path, message=message))
        return bool(ok)
    def text(obj, key, path):
        return require(isinstance(obj.get(key), str) and bool(obj[key].strip()), path+'/'+key, 'nonempty string required')
    require(data.get('scope') == 'motor_muscle', '/scope', 'motor_muscle scope required')
    total = 0
    records = data.get('trials', [])
    if not isinstance(records, list):
        records = []
    for i, trial in enumerate(records[:1000]):
        path = f'/trials/{i}'
        if not isinstance(trial, dict):
            continue
        for key in ('muscle_bundle_id', 'recruitment_evidence', 'pool_coverage'):
            text(trial, key, path)
        require(trial.get('pool_coverage') in ('complete', 'partial', 'unknown'), path+'/pool_coverage', 'explicit coverage required')
        # Single identified cell per record; multi-cell pools require separate records.
        obs = trial.get('observation')
        if not require(isinstance(obs, dict), path+'/observation', 'muscle observation object required'):
            continue
        for key in ('measurement_model', 'model_revision', 'calibration_evidence'):
            text(obs, key, path+'/observation')
        allowed = {'emg': ('V',), 'calcium': ('deltaF/F',),
                   'activation_estimate': ('1',), 'force': ('N',), 'torque': ('N*m',)}
        kind = obs.get('kind')
        unit = obs.get('unit')
        require(isinstance(kind, str) and kind in allowed, path+'/observation/kind', 'unsupported observation kind')
        if isinstance(kind, str) and kind in allowed:
            require(unit in allowed[kind], path+'/observation/unit', 'canonical observation unit required')
        for key in ('time_s', 'values'):
            values = obs.get(key)
            ok = isinstance(values, list) and len(values) <= MAX_POINTS and all(finite(v) for v in values)
            require(ok, path+'/observation/'+key, 'bounded finite numeric array required')
            if isinstance(values, list):
                total += len(values)
        times, values = obs.get('time_s'), obs.get('values')
        if isinstance(times, list) and all(finite(v) for v in times):
            require(len(times) >= 2 and all(b > a for a, b in zip(times, times[1:])),
                    path+'/observation/time_s', 'at least two strictly increasing observation times required')
            if isinstance(values, list):
                require(len(values) == len(times), path+'/observation/values', 'observation sample count mismatch')
            body = trial.get('time_s')
            if isinstance(body, list) and len(body) >= 2 and all(finite(v) for v in body):
                require(all(body[0] <= t <= body[-1] for t in times), path+'/observation/time_s', 'observations outside body window')
        # Observation times are already in the common body clock; mapping must be explicit.
        text(obs, 'clock_mapping_evidence', path+'/observation')
        require(finite(obs.get('max_sync_error_s')) and obs['max_sync_error_s'] >= 0,
                path+'/observation/max_sync_error_s', 'nonnegative finite measured/declared synchronization error required')
        require(finite(obs.get('sensor_delay_s')) and obs['sensor_delay_s'] >= 0,
                path+'/observation/sensor_delay_s', 'nonnegative sensor delay required; not a muscle time constant')
        if kind == 'activation_estimate' and isinstance(values, list) and all(finite(v) for v in values):
            require(all(0 <= v <= 1 for v in values), path+'/observation/values', 'normalized activation estimates must be in [0,1]')
    # Include shared body/neural arrays in the aggregate budget, not just muscle samples.
    for trial in records[:1000]:
        if isinstance(trial, dict):
            total += sum(len(trial[k]) for k in ('time_s', 'angle_rad', 'velocity_rad_s', 'spike_time_neural_s') if isinstance(trial.get(k), list))
    require(total <= MAX_POINTS, '/trials', 'aggregate point budget exceeded')
    result.update(contract_valid=not errors, scope='motor_muscle structural admission only',
                  calibration_executed=False, biological_validation=False)
    result['limitations'] += ['EMG/calcium are observations, not automatically muscle activation',
                              'Source identity, recruitment and observation models are declarations, not verified',
                              'No force/torque inferred from other modalities; no parameter fit performed']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        data, digest = load(args.manifest)
        result = validate(data)
        result.update(manifest=str(args.manifest.resolve()), manifest_sha256=digest)
    except (ValueError, TypeError, OSError) as error:
        result = dict(contract_valid=False, biological_validation=False, calibration_executed=False,
                      errors=[dict(path='/', message=str(error))])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('STRUCTURAL_PASS' if result['contract_valid'] else 'REJECTED')
    raise SystemExit(0 if result['contract_valid'] else 1)


if __name__ == '__main__':
    main()
