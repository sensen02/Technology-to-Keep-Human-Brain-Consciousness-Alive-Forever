"""Lightweight saved-artifact audit; no simulation, gain changes, or network build."""
from pathlib import Path
import argparse
import json
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / 'outputs/model_branches/canonical_paired_baseline/20261012_235335'
DEFAULT_OUTPUT = HERE / 'outputs/model_branches/decoder_saturation_audit/audit.json'


def saturation_summary(rates, times, threshold):
    rates = np.asarray(rates, float)
    saturated = rates >= threshold
    hits = np.flatnonzero(saturated)
    return dict(sample_count=int(rates.size), saturated_count=int(hits.size),
                saturated_fraction=float(saturated.mean()) if rates.size else 0.0,
                first_saturation_s=float(times[hits[0]]) if hits.size else None,
                peak_rate_hz=float(rates.max()) if rates.size else 0.0)


def audit(results, identity, arrays):
    config = results['provenance']['motor_config']
    threshold = config['rate_for_full_control_hz'] * config['max_control']
    rows = identity['motor_manifest']['rows']
    pools = {}
    for row in rows:
        if row['local_indices']:
            pools.setdefault(tuple(row['local_indices']), []).append(row['actuator_index'])
    pool_ids = {key: f'pool_{j:03d}' for j, key in enumerate(pools)}
    motor = sorted({i for key in pools for i in key})
    episodes = {}
    for name, episode in results['episodes'].items():
        records = episode['records']
        rates = np.asarray([r['decoder']['rates_hz'] for r in records], float)
        controls = arrays[name + '_controls']
        spikes = arrays[name + '_spike_counts']
        times = [r['decoder']['valid_from_s'] for r in records]
        applied_times = [r['time_start_s'] for r in records]
        if rates.shape != controls.shape or rates.shape[1] != len(rows):
            raise ValueError('rate/control/manifest shape mismatch')
        state = np.zeros(len(rows))
        decay = np.exp(-episode['probe_spike_dt_s'] / config['filter_tau_s'])
        subrates, subtimes = [], []
        count_check = True
        max_error = 0.0
        for record in records:
            dt = episode['probe_spike_dt_s']
            steps = round((record['decoder']['source_end_s'] - record['time_start_s']) / dt)
            probe = episode['probe_local_indices']
            if not set(motor).issubset(probe):
                raise ValueError('motor raster incomplete')
            events = record['probe_emitted_spike_events_step_column']
            if len(events) != len({tuple(e) for e in events}):
                raise ValueError('duplicate spike events')
            raster = np.zeros((steps, len(identity['tier_root_ids'])), bool)
            for step, col in events:
                if not (0 <= step < steps and 0 <= col < len(probe)):
                    raise ValueError('event outside raster')
                raster[step, probe[col]] = True
            counts = [int(raster[:, row['local_indices']].sum()) for row in rows]
            count_check &= counts == record['decoder']['pool_spike_counts']
            for step, sample in enumerate(raster):
                if record['decoder']['connected']:
                    instant = np.asarray([sample[row['local_indices']].mean() / dt
                                          if row['local_indices'] else 0.0 for row in rows])
                    state = decay * state + (1 - decay) * instant
                else:
                    state[:] = 0
                subrates.append(state.copy())
                subtimes.append(record['time_start_s'] + (step + 1) * dt)
            max_error = max(max_error, float(np.max(np.abs(state - record['decoder']['rates_hz']))))
        subrates = np.asarray(subrates)
        per_actuator = []
        for j, row in enumerate(rows):
            key = tuple(row['local_indices'])
            aliases = [rows[i]['actuator'] for i in pools.get(key, [])]
            entry = dict(actuator=row['actuator'], actuator_index=j, status=row['status'],
                         pool_id=pool_ids.get(key), aliases=aliases,
                         present_cell_count=len(key), declared_cell_count=len(row['pool_root_ids']),
                         pool_root_ids=row['pool_root_ids'], local_indices=list(key),
                         pool_spike_counts=[r['decoder']['pool_spike_counts'][j] for r in records],
                         saturation_mask=(rates[:, j] >= threshold).tolist(),
                         rates_hz=rates[:, j].tolist(),
                         unclipped_normalized_demand=(rates[:, j] / config['rate_for_full_control_hz']).tolist(),
                         clipping_excess=np.maximum(rates[:, j] / config['rate_for_full_control_hz'] - config['max_control'], 0).tolist(),
                         decoded=saturation_summary(rates[:, j], times, threshold),
                         substep=saturation_summary(subrates[:, j], subtimes, threshold),
                         applied=saturation_summary(controls[:, j], applied_times, config['max_control']),
                         alias_rates_exact=all(np.array_equal(rates[:, j], rates[:, i]) for i in pools.get(key, [])))
            entry['applied']['peak_control'] = entry['applied'].pop('peak_rate_hz')
            per_actuator.append(entry)
        unique_columns = [indices[0] for indices in pools.values()]
        episodes[name] = dict(network_spike_total=int(spikes.sum()),
                              unique_motor_spike_total=int(spikes[:, motor].sum()),
                              actuator_pool_spike_total=sum(sum(r['decoder']['pool_spike_counts']) for r in records),
                              decoder_saturated_samples=int((rates >= threshold).sum()),
                              decoder_sample_count=int(rates.size),
                              unique_pool_saturated_samples=int((rates[:, unique_columns] >= threshold).sum()),
                              unique_pool_sample_count=int(rates[:, unique_columns].size),
                              applied_saturated_samples=int((controls >= config['max_control']).sum()),
                              saturated_actuator_count=int((rates >= threshold).any(axis=0).sum()),
                              rate_reconstruction_max_error_hz=max_error,
                              reconstructed_pool_counts_exact=bool(count_check),
                              valid_from_s=times, applied_interval_start_s=applied_times,
                              actuators=per_actuator)
    comparisons = []
    if 'zero_floor_body' in episodes:
        for name in episodes:
            if name == 'zero_floor_body':
                continue
            base = 'zero_floor_body'
            comparisons.append(dict(reference=base, comparison=name,
                network_spike_delta=episodes[name]['network_spike_total'] - episodes[base]['network_spike_total'],
                motor_spike_delta=episodes[name]['unique_motor_spike_total'] - episodes[base]['unique_motor_spike_total'],
                motor_interval_counts_exact=bool(np.array_equal(arrays[name+'_spike_counts'][:, motor], arrays[base+'_spike_counts'][:, motor])),
                controls_exact=bool(np.array_equal(arrays[name+'_controls'], arrays[base+'_controls'])),
                body_trajectory_exact=bool(np.array_equal(arrays[name+'_trajectory'], arrays[base+'_trajectory']))))
    return dict(status='SAVED_ARTIFACT_DIAGNOSTIC_NOT_STANDING_ACCEPTANCE', config=config,
                unique_motor_cells=len(motor), unique_nonempty_pools=len(pools),
                assumptions=['Causal EMA over mean PRESENT-cell binary spikes; no biological calibration.',
                             'Endpoint saturation differs from substep crossings and lagged applied commands.',
                             'Last decoded command is not applied within the saved episode.',
                             'Disconnected decoder clears rates despite ongoing motor spikes.',
                             'Shared aliases duplicate actuator commands, not independent neurons.',
                             'Applied controls are submitted commands; native positive floor is not reconstructed physical actuation.'],
                episodes=episodes, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-results', type=Path, default=DEFAULT_SOURCE/'paired_results.json')
    parser.add_argument('--paired-traces', type=Path, default=DEFAULT_SOURCE/'paired_traces.npz')
    parser.add_argument('--neural-identity', type=Path, default=DEFAULT_SOURCE/'neural_identity.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    results = json.loads(args.paired_results.read_text())
    identity = json.loads(args.neural_identity.read_text())
    with np.load(args.paired_traces, allow_pickle=False) as arrays:
        report = audit(results, identity, arrays)
    report['sources'] = {key: str(getattr(args, key).resolve()) for key in
                         ('paired_results', 'paired_traces', 'neural_identity')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output.resolve())
    for name, episode in report['episodes'].items():
        print(name, 'network/motor', episode['network_spike_total'], episode['unique_motor_spike_total'],
              'saturated', episode['decoder_saturated_samples'], 'reconstruction_error', episode['rate_reconstruction_max_error_hz'])
    return report


if __name__ == '__main__':
    main()
