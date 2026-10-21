"""Read-only P1 evidence audit, not simulation, authenticity or biological validation.

Only the installed lightweight source_snapshot utility is imported by file path.
Archived code is never extracted/imported/executed. Reports must be newly created
outside the evidence directory and may not overwrite recorded sources or inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ARMS = ('normal', 'normal_repeat', 'replay_baseline', 'feedback_push',
        'replay_push', 'sensory_cut', 'motor_cut', 'output_silence')


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_utility():
    path = Path(__file__).resolve().parent / 'engine/embodied/source_snapshot.py'
    spec = importlib.util.spec_from_file_location('_p1_local_source_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def close(a, b, message):
    require(np.shape(a) == np.shape(b) and
            np.allclose(a, b, rtol=1e-12, atol=1e-12), message)


def indices(values, n, name):
    require(isinstance(values, list) and all(type(i) is int and 0 <= i < n for i in values)
            and len(set(values)) == len(values), 'invalid indices: ' + name)
    return values


def validate_arrays(result, arrays):
    """Recompute causal, current, event and filter invariants from recorded evidence."""
    require(set(result['episodes']) == set(ARMS), 'eight expected arms required')
    metadata = result['evidence_schema']
    require(metadata['version'] == 1, 'unsupported evidence schema')
    require(metadata['sensory_local_indices_key'] == 'sensory_local_indices', 'sensory identity key')
    require(metadata['replay_source_arms'] == {'replay_baseline': 'normal', 'replay_push': 'normal'}, 'replay sources')
    db, dn, dc = (float(result['clocks'][k]) for k in ('dt_body_s', 'dt_neural_s', 'dt_command_s'))
    require(all(math.isfinite(x) and x > 0 for x in (db, dn, dc)), 'invalid clocks')
    steps = round(dc / dn)
    count = round(result['duration_s'] / dc)
    require(steps > 0 and count > 0 and abs(steps * dn - dc) < 1e-12
            and abs(round(dc/db)*db-dc) < 1e-12
            and abs(count*dc-result['duration_s']) < 1e-12, 'unaligned clocks')
    manifest = result['motor_manifest']
    n, nu = manifest['n_neurons'], manifest['n_actuators']
    roots = result['tier_root_ids']
    require(len(roots) == n and len(set(map(str, roots))) == n, 'tier identity mismatch')
    require(len(manifest['rows']) == nu, 'motor manifest shape')
    motors = set()
    pools = []
    for ai, row in enumerate(manifest['rows']):
        require(row['actuator_index'] == ai, 'actuator ordering')
        pool = indices(row['local_indices'], n, 'motor pool')
        if 'tier_root_ids' in row:
            require(list(map(str, row['tier_root_ids'])) == [str(roots[i]) for i in pool],
                    'motor root identity mismatch')
        pools.append(pool)
        motors.update(pool)
    groups = result['sensory_groups']
    union = set()
    for leg, channels in groups.items():
        for channel, group in channels.items():
            union.update(indices(group, n, leg + '/' + channel))
    selected = arrays['sensory_local_indices']
    require(selected.dtype.kind in 'iu' and selected.ndim == 1
            and selected.tolist() == sorted(union), 'sensory local identity mismatch')
    columns = {int(i): j for j, i in enumerate(selected)}
    config = metadata['motor_translation']
    tau, full, maximum = (float(config[k]) for k in
                          ('filter_tau_s', 'rate_for_full_control_hz', 'max_control'))
    require(all(math.isfinite(x) and x > 0 for x in (tau, full, maximum)) and maximum <= 1,
            'invalid decoder configuration')
    decay = math.exp(-dn / tau)
    for key, array in arrays.items():
        require(array.dtype.kind in 'biuf' and np.isfinite(array).all(), 'nonfinite/non-numeric: ' + key)
    require(arrays['projected_pose'].ndim == 1, 'pose shape')
    require(arrays['native_control_ranges'].shape == (nu, 2), 'native ranges shape')
    for arm in ARMS:
        def a(kind):
            return arrays[arm + '_' + kind]
        expected = {'controls': (count, nu), 'decoded_next_controls': (count, nu),
                    'spike_counts': (count, n), 'sensory_current_nA': (count, len(selected)),
                    'outside_sensory_input_max_abs_nA': (count,), 'interval_start_s': (count,),
                    'interval_end_s': (count,), 'applied_source_interval': (count,)}
        for kind, shape in expected.items():
            require(a(kind).shape == shape, arm + ' shape: ' + kind)
        require(a('activation').shape == (count, metadata['body_na']), arm + ' activation shape')
        require(a('trajectory').shape == (count, metadata['body_nq'] + metadata['body_nv'] + metadata['body_na']), arm + ' trajectory shape')
        require(metadata['body_nu'] == nu, 'body/manifest actuator mismatch')
        close(result['episodes'][arm]['probe_spike_dt_s'], dn, arm + ' probe clock')
        if a('activation').shape[1]:
            require(np.array_equal(a('trajectory')[:, -a('activation').shape[1]:], a('activation')),
                    arm + ' activation trajectory mismatch')
        close(a('interval_start_s'), np.arange(count)*dc, arm + ' start clocks')
        close(a('interval_end_s'), (np.arange(count)+1)*dc, arm + ' end clocks')
        require(a('applied_source_interval').dtype.kind in 'iu' and
                np.array_equal(a('applied_source_interval'), np.arange(count)-1), arm + ' causal source indices')
        counts = a('spike_counts')
        require(np.all(counts == np.floor(counts)) and np.all((counts >= 0) & (counts <= steps)), arm + ' invalid spike counts')
        require(not a('outside_sensory_input_max_abs_nA').any(), arm + ' input outside sensory groups')
        require(np.all((a('controls') >= 0) & (a('controls') <= maximum)), arm + ' controls range')
        records = result['episodes'][arm]['records']
        require(len(records) == count, arm + ' record count')
        probe = indices(result['episodes'][arm]['probe_local_indices'], n, 'probe')
        require(motors.issubset(probe), arm + ' motor raster incomplete')
        rates = np.zeros(nu)
        for k, record in enumerate(records):
            prefix = arm + ' interval ' + str(k) + ': '
            close(record['time_start_s'], k*dc, prefix + 'record start')
            close(record['time_end_s'], (k+1)*dc, prefix + 'record end')
            require(record['applied_source_interval'] == k-1, prefix + 'record source')
            close(record['source_completed_by_s'], k*dc, prefix + 'completed clock')
            replay = arm.startswith('replay_')
            require(record['command_origin'] == ('offline_normal_replay' if replay else
                    'previous_completed_neural_interval'), prefix + 'command origin')
            wanted = (arrays['normal_controls'][k] if replay else
                      np.zeros(nu) if k == 0 or (arm == 'motor_cut' and k*dc >= .015-1e-12)
                      else a('decoded_next_controls')[k-1])
            require(np.array_equal(a('controls')[k], wanted), prefix + 'command causality')
            for leg, channels in groups.items():
                for channel, group in channels.items():
                    values = a('sensory_current_nA')[k, [columns[i] for i in group]]
                    stats = record['actual_network_input_by_channel'][leg][channel]
                    require(stats['count'] == len(group), prefix + 'channel count')
                    for name, value in (('sum_nA', values.sum()), ('min_nA', values.min() if len(values) else 0),
                                        ('max_nA', values.max() if len(values) else 0)):
                        close(stats[name], value, prefix + 'current ' + leg + '/' + channel + '/' + name)
                    if channel in record['disabled_channels'] or arm == 'sensory_cut':
                        require(not values.any(), prefix + 'cut channel nonzero')
            rasters = []
            for kind in ('raw', 'emitted'):
                events = record['probe_' + kind + '_spike_events_step_column']
                raster = np.zeros((steps, len(probe)), dtype=bool)
                seen = set()
                for event in events:
                    require(isinstance(event, list) and len(event) == 2 and all(type(x) is int for x in event)
                            and 0 <= event[0] < steps and 0 <= event[1] < len(probe)
                            and tuple(event) not in seen, prefix + 'invalid/duplicate raster event')
                    seen.add(tuple(event))
                    raster[tuple(event)] = True
                rasters.append(raster)
            raw, emitted = rasters
            require(not emitted.any() if arm == 'output_silence' else np.array_equal(raw, emitted),
                    prefix + 'raw/emitted intervention mismatch')
            require(np.array_equal(emitted.sum(axis=0), counts[k, probe]), prefix + 'raster spike counts')
            require(record['total_network_spikes'] == int(counts[k].sum()), prefix + 'total counts')
            require(record['motor_spikes'] == int(counts[k, sorted(motors)].sum()), prefix + 'motor counts')
            if arm == 'output_silence':
                require(not counts[k].any(), prefix + 'silenced output counts')
            connected = not (arm == 'motor_cut' and (k+1)*dc >= .015-1e-12)
            trace = record['decoder']
            require(trace['connected'] is connected and trace['source_interval'] == k, prefix + 'decoder source/connection')
            for name, value in (('source_start_s', k*dc), ('source_end_s', (k+1)*dc), ('valid_from_s', (k+1)*dc)):
                close(trace[name], value, prefix + 'decoder clock')
            position = {idx: col for col, idx in enumerate(probe)}
            pool_counts = [int(emitted[:, [position[i] for i in pool]].sum()) for pool in pools]
            require(trace['pool_spike_counts'] == pool_counts, prefix + 'decoder pool counts')
            if connected:
                for sample in emitted:
                    instantaneous = np.array([sample[[position[i] for i in pool]].mean()/dn if pool else 0 for pool in pools])
                    rates = decay*rates + (1-decay)*instantaneous
            else:
                rates[:] = 0
            close(trace['rates_hz'], rates, prefix + 'decoder reconstructed rates')
            controls = np.clip(rates/full, 0, maximum)
            close(a('decoded_next_controls')[k], controls, prefix + 'decoder reconstructed control')
            require(trace['saturated'] == (controls >= maximum).tolist(), prefix + 'decoder saturation')
    require(result['episodes']['normal']['records'] == result['episodes']['normal_repeat']['records'],
            'repeat exact interval records including raw events')
    for kind in ('trajectory', 'controls', 'spike_counts', 'sensory_current_nA',
                 'outside_sensory_input_max_abs_nA', 'interval_start_s', 'interval_end_s',
                 'applied_source_interval', 'decoded_next_controls', 'activation'):
        require(np.array_equal(arrays['normal_'+kind], arrays['normal_repeat_'+kind]), 'repeat exact: ' + kind)
    for kind in ('trajectory', 'controls', 'activation'):
        require(np.array_equal(arrays['normal_'+kind], arrays['replay_baseline_'+kind]), 'replay baseline: ' + kind)
    require(not arrays['sensory_cut_sensory_current_nA'].any(), 'sensory cut input')
    for arm in ('sensory_cut', 'output_silence'):
        require(not arrays[arm+'_controls'].any(), arm + ' nonzero commands')
    require(not arrays['motor_cut_controls'][np.arange(count)*dc >= .015-1e-12].any(), 'motor cut commands')
    return {'arms': len(ARMS), 'intervals_per_arm': count, 'tier_neurons': n,
            'recorded_sensory_neurons': len(selected), 'motor_raster_neurons': len(motors),
            'decoder_reconstructed': True}


def verify_bundle(directory, *, check_live_sources=False, check_live_inputs=False):
    directory = Path(directory).resolve(strict=True)
    require(directory.is_dir(), 'explicit result directory required')
    result_path = directory / 'result.json'
    with result_path.open(encoding='utf-8') as stream:
        result = json.load(stream, object_pairs_hook=unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(EvidenceError('nonfinite JSON')))
    require(sha256(directory/'traces.npz') == result['traces_sha256'], 'traces SHA-256 mismatch')
    require(sha256(directory/'source_snapshot.zip') == result['source_snapshot_sha256'], 'source ZIP SHA-256 mismatch')
    archived = snapshot_utility().verify_source_snapshot(directory/'source_snapshot.zip', result['source_hashes'])
    for enabled, key in ((check_live_sources, 'source_hashes'), (check_live_inputs, 'input_hashes')):
        if enabled:
            for path, digest in result[key].items():
                require(Path(path).is_absolute() and sha256(path) == digest, 'live hash mismatch: ' + path)
    with np.load(directory/'traces.npz', allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    checks = validate_arrays(result, arrays)
    return dict(status='PASS', result_directory=str(directory), result_sha256=sha256(result_path),
                source_records=len(archived['files']), live_sources_checked=check_live_sources,
                live_inputs_checked=check_live_inputs, checks=checks,
                limits=['Self-consistent hashes are not signed authenticity.',
                        'No archived code or simulation executed; no historical numeric reproduction.',
                        'Motor raw/emitted rasters and declared decoder reconstructed; unprobed raw neural events not available.',
                        'Actual currents checked against groups/statistics, not independently rederived from body sensor model.',
                        'No independent physics integration, environment reproduction, convergence or biological validation.',
                        'Does not certify standing, walking, recovery or consciousness.'])


def write_report(path, report, directory, result=None):
    target = Path(path).resolve()
    bundle = Path(directory).resolve()
    require(not target.is_relative_to(bundle), 'report must be outside evidence bundle')
    protected = set()
    if result:
        protected = {Path(p).resolve() for key in ('source_hashes', 'input_hashes') for p in result.get(key, {})}
    require(target not in protected, 'report may not overwrite a source/input')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    parser.add_argument('--check-live-sources', action='store_true')
    parser.add_argument('--check-live-inputs', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_bundle(args.result_directory, check_live_sources=args.check_live_sources,
                               check_live_inputs=args.check_live_inputs)
        if args.report:
            with (args.result_directory/'result.json').open() as stream:
                result = json.load(stream)
            write_report(args.report, report, args.result_directory, result)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        print(json.dumps({'status': 'FAIL', 'error': str(error)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
