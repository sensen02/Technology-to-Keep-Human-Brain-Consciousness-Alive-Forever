"""Execute preregistered engineering probes, saving every completed attempt."""
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from module import LocalCircuit, SOURCE, HERE


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    if (HERE / 'VALIDATION.json').exists():
        raise RuntimeError('refuse to overwrite completed experiment evidence')
    prereg_hash = sha(HERE / 'PREREGISTRATION.json')
    c = LocalCircuit(total_conductance_uS=.02)
    lookup = {int(g): i for i, g in enumerate(c.cache_indices)}
    with (SOURCE / 'SENSORY_CHANNEL_CANDIDATES.csv').open() as f:
        rows = list(csv.DictReader(f))
    qualified, coverage = set(), {}
    for channel in sorted(set(r['channel'] for r in rows)):
        ids = np.array(sorted({lookup[int(r['cache_index_0based'])] for r in rows if r['channel'] == channel and int(r['cache_index_0based']) in lookup}), dtype=int)
        ids = np.intersect1d(ids, c.allowed_sensory_local)
        w = c.net.We[:, ids] + c.net.Wi[:, ids]
        coverage[channel] = {'allowed_cells': len(ids), 'strict_source_cells': int(np.count_nonzero(w.getnnz(axis=0))), 'strict_outgoing_edges': w.nnz}
        if any(token in channel for token in ('claw', 'hook', 'hairplate', 'campaniform')):
            qualified.update(ids.tolist())
    physical_ids = np.array(sorted(qualified), dtype=np.int64)
    np.savez_compressed(HERE / 'PHYSICAL_QUALIFIED_PORTS.npz', local_indices=physical_ids, root_ids=c.root_ids[physical_ids])
    dump(HERE / 'PORT_COVERAGE.json', coverage)
    # Saved full currents are exact playback inputs for both timestep values.
    sensory = np.zeros((40, 20000)); descending = sensory.copy()
    sensory[5:25, c.allowed_sensory_local] = .05
    descending[5:25, c.descending_local] = .05
    np.savez_compressed(HERE / 'RECORDED_INPUT.npz', sensory_current_nA=sensory, descending_current_nA=descending, duration_ms=np.full(40, 5.))
    results, actual_steps = {}, 0

    def run(name, gain, s, d, dt=.5, no_edges=False):
        nonlocal actual_steps
        circuit = LocalCircuit(total_conductance_uS=gain, dt_ms=dt)
        if no_edges:
            # Numerical software control ONLY; never an anatomical surgical claim.
            circuit.net.We.data[:] = 0.
            circuit.net.Wi.data[:] = 0.
        out, mn_counts, all_counts, vrange = [], [], [], []
        before_steps = circuit.net.steps
        for si, di in zip(s, d):
            out.append(circuit.advance(si, di, 5.))
            mn_counts.append(circuit.net.spike_count[circuit.mn_local].copy())
            all_counts.append(int(circuit.net.spike_count.sum()))
            vrange.append([float(circuit.net.v.min()), float(circuit.net.v.max())])
        actual_steps += circuit.net.steps - before_steps
        count = circuit.net.spike_count[circuit.mn_local]
        out = np.array(out)
        state = {'v': circuit.net.v.copy(), 'ge': circuit.net.ge.copy(), 'gi': circuit.net.gi.copy(), 'adaptation': circuit.net.adaptation.copy(), 'refractory_until_ms': circuit.net.refractory_until_ms.copy(), 'ring': circuit.net._ring.copy(), 'filtered_rates_hz': circuit.filtered_rates_hz.copy(), 'spike_count': circuit.net.spike_count.copy()}
        summary = {'total_conductance_uS': gain, 'dt_ms': dt, 'neural_steps': circuit.net.steps, 'neural_duration_ms': circuit.net.t_ms,
                   'all_spikes': int(circuit.net.spike_count.sum()), 'mn_spikes': int(count.sum()), 'mn_recruited': int(np.count_nonzero(count)),
                   'mn_mean_rate_hz': float(count.mean()/ .2), 'mn_max_cell_mean_rate_hz': float(count.max()/ .2),
                   'mn_filtered_peak_hz': float(out.max()), 'voltage_min_mV': float(np.min(vrange)), 'voltage_max_mV': float(np.max(vrange)),
                   'finite': all(np.isfinite(v).all() for v in state.values()), 'no_edges_software_control': no_edges,
                   'per_leg_mn_spikes': {leg: int(count[circuit.mn_leg == leg].sum()) for leg in np.unique(circuit.mn_leg)}}
        summary['selection_rule_pass'] = bool(summary['mn_recruited'] >= 16 and 0 < summary['mn_mean_rate_hz'] < 100 and summary['mn_max_cell_mean_rate_hz'] < 200 and summary['finite'] and summary['voltage_min_mV'] >= -90 and summary['voltage_max_mV'] <= -49)
        np.savez_compressed(HERE / (name + '.npz'), mn_filtered_hz=out, mn_spike_counts=np.array(mn_counts), all_spike_count=np.array(all_counts), voltage_range_mV=np.array(vrange), **state)
        dump(HERE / (name + '.json'), summary)
        results[name] = summary
        dump(HERE / 'RUN_PROGRESS.json', {'completed': results, 'actual_neural_steps': actual_steps, 'physical_steps': 0, 'training_updates': 0})
        print(name, json.dumps(summary), flush=True)
        return circuit, out, state

    chosen = None
    for gain in (.02, .005, .08):
        run('calibration_' + str(gain), gain, sensory, descending)
        if results['calibration_' + str(gain)]['selection_rule_pass']:
            chosen = gain
            break
    gain = .02 if chosen is None else chosen
    dump(HERE / 'CHOSEN_CONFIG.json', {'total_conductance_uS': gain, 'dt_ms': .5, 'filter_tau_ms': 20., 'delay_ms': 2., 'status': 'ENGINEERING_PROPAGATION_CALIBRATED' if chosen else 'CALIBRATION_FAILED', 'selection_rule': 'first passing .02,.005,.08 per PREREGISTRATION.json', 'preregistration_sha256': prereg_hash, 'physical_feedback_status': 'pending restricted probe; broad engineering stimulus not physical feedback', 'physiology': 'illustrative uncalibrated'})
    zero = np.zeros_like(sensory)
    run('sensory_only', gain, sensory, zero)
    run('descending_only', gain, zero, descending)
    quiet, _, _ = run('no_input', gain, zero, zero)
    assert quiet.net.spike_count.sum() == 0
    noedge, _, _ = run('no_edges_software_control', gain, sensory, descending, no_edges=True)
    assert not np.any(noedge.net.spike_count[noedge.mn_local])
    _, replay_out, replay_state = run('recorded_replay', gain, sensory, descending)
    with np.load(HERE / ('calibration_' + str(gain) + '.npz')) as ref:
        replay_equal = bool(np.array_equal(replay_out, ref['mn_filtered_hz']) and all(np.array_equal(v, ref[k]) for k, v in replay_state.items()))
    assert replay_equal
    _, fine_out, _ = run('timestep_025', gain, sensory, descending, dt=.25)
    base = results['recorded_replay']; fine = results['timestep_025']
    relative = abs(base['mn_mean_rate_hz'] - fine['mn_mean_rate_hz']) / max(base['mn_mean_rate_hz'], 1e-12)
    sensitivity = {'aggregate_rate_relative_difference': relative, 'filtered_MN_rms_Hz': float(np.sqrt(np.mean((replay_out-fine_out)**2))), 'recruited_count_difference': abs(base['mn_recruited']-fine['mn_recruited']), 'engineering_sensitivity_pass': bool(relative <= .25 and abs(base['mn_recruited']-fine['mn_recruited']) <= .25*323), 'not_convergence_proof': True}
    constant_s = np.broadcast_to(sensory[5], sensory.shape).copy()
    constant_d = np.broadcast_to(descending[5], descending.shape).copy()
    run('constant_combined', gain, constant_s, constant_d)
    physical_s = np.zeros_like(sensory); physical_s[5:25, physical_ids] = .05
    np.savez_compressed(HERE / 'RECORDED_PHYSICAL_INPUT.npz', sensory_current_nA=physical_s, descending_current_nA=zero, duration_ms=np.full(40,5.))
    run('physical_qualified_zeroDN', gain, physical_s, zero)
    run('physical_qualified_withDN', gain, physical_s, descending)
    dump(HERE / 'VALIDATION.json', {'status': 'EXECUTED', 'preregistration_sha256': prereg_hash, 'module_sha256': sha(HERE/'module.py'), 'results': results, 'chosen_gain_uS': gain, 'calibration_pass': chosen is not None, 'exact_replay': replay_equal, 'timestep_sensitivity': sensitivity, 'physical_qualified_count':len(physical_ids), 'port_coverage': coverage, 'actual_neural_steps_this_script': actual_steps, 'earlier_API_smoke_neural_steps': 10, 'physical_steps':0, 'training_updates':0, 'wall_seconds':time.monotonic()-start})

if __name__ == '__main__':
    main()
