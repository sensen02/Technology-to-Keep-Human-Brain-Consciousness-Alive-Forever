"""Matched-state propagation through actual runtime BANC edges; no body control.
手工研究原型，不是成品. Diagnostic constant sensory pulse, not natural movement.
"""
import json
from pathlib import Path
import numpy as np
import run_flybody_loop as loop
from run_sensory_tuning_annotation_audit import sha, exact_groups

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/flybody_loop'


def main():
    pointer = HERE/'outputs/neural_muscle_closure/current_run.json'
    p = json.loads(pointer.read_text())
    bundle = pointer.parent/p['path']
    manifest_path = bundle/'manifest.json'
    m = json.loads(manifest_path.read_text())
    identity_path = bundle/'neural_identity.json'
    assert sha(manifest_path) == p['manifest_sha256']
    assert sha(identity_path) == m['files']['neural_identity.json']['sha256']
    identity = json.loads(identity_path.read_text())
    lookup_path = HERE/'outputs/neural_rebuild/sensory_channel_lookup.json'
    cells = exact_groups(identity, json.loads(lookup_path.read_text()))
    built = loop.closure.build_neural_tier(loop.closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    roots = list(map(str, built['tier']['root_ids']))
    assert roots == list(map(str, identity['tier_root_ids'])), 'Rebuilt tier identity differs'
    n = int(built['n'])
    sensory = np.array(sorted(c['tier_index'] for c in cells), int)
    motor = np.array(sorted({i for g in identity['motor_pool_manifest']['rows'] for i in g['local_indices']}), int) if 'motor_pool_manifest' in identity else None
    if motor is None:
        import mujoco
        model = loop.cp.load_model()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        pools = json.loads(loop.closure.POOLS.read_text())
        manifest = loop.closure.build_motor_manifest(pools, built['tier']['root_ids'], names)
        motor = np.array(sorted({i for g in manifest['rows'] for i in g['local_indices']}), int)
    assert len(motor) > 0
    nonsensory = np.setdiff1d(np.arange(n), sensory)
    groups = identity['sensory_groups']
    input_path = OUT/'flybody_input_transfer_audit.json'
    inputs = json.loads(input_path.read_text())
    # Reuse measured currents; refuse synthetic root membership or unexplained current changes.
    assert inputs['sensory_roots_verified'] == len(cells)
    selected = [x for x in inputs['probes'] if x['leg'] == 'lf']
    arms = [('zero_input', None)] + [(x['probe'], x) for x in selected]
    records = []
    dt_ms, steps, pulse_steps = .5, 200, 40
    for name, probe in arms:
        net = loop.ConductanceNetwork(n, loop.ConductanceParams(), dt_ms=dt_ms)
        net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
        current = np.zeros(n)
        target = np.array([], int)
        if probe:
            target = np.array(groups['lf'][probe['channel']], int)
            assert [roots[int(i)] for i in target] == list(map(str, probe['target_root_ids']))
            current[target] = probe['measured_delta_current_nA_per_cell']
        trace = []
        motor_steps, spike_events = [], []
        fired_counts = np.zeros(n, dtype=int)
        peak_v = net.v[motor].copy()
        peak_ge = np.zeros(len(motor))
        peak_gi = np.zeros(len(motor))
        totals = {'all': 0, 'sensory': 0, 'nonsensory': 0, 'motor': 0, 'target': 0}
        for k in range(steps):
            # Read the exact pending ring slot without consuming it; core step stays frozen.
            arrivals = net._ring[net._ring_i].copy()
            arriving_exc = (net.We @ arrivals)[motor]
            arriving_inh = (net.Wi @ arrivals)[motor]
            sp = net.step(current_nA=current if k < pulse_steps else np.zeros(n))
            spike_events.extend({'root_id': roots[int(i)], 'tier_index': int(i), 'emission_end_ms': (k+1)*dt_ms}
                                for i in np.flatnonzero(sp))
            motor_steps.append({'time_start_ms': k*dt_ms, 'time_end_ms': (k+1)*dt_ms,
                                'voltage_mV': net.v[motor].tolist(),
                                'arriving_exc_uS': arriving_exc.tolist(),
                                'arriving_inh_uS': arriving_inh.tolist()})
            fired_counts += sp
            peak_v = np.maximum(peak_v, net.v[motor])
            peak_ge = np.maximum(peak_ge, net.ge[motor])
            peak_gi = np.maximum(peak_gi, net.gi[motor])
            counts = {'all': int(sp.sum()), 'sensory': int(sp[sensory].sum()),
                      'nonsensory': int(sp[nonsensory].sum()), 'motor': int(sp[motor].sum()),
                      'target': int(sp[target].sum())}
            for key in totals:
                totals[key] += counts[key]
            trace.append({'time_ms': (k+1)*dt_ms, **counts,
                          'motor_mean_voltage_mV': float(net.v[motor].mean()),
                          'nonsensory_mean_voltage_mV': float(net.v[nonsensory].mean())})
        from tools_connectome_path_diagnostics import diagnose
        diagnostics = diagnose(net, target, motor, fired_counts, roots, sensory, peak_v, peak_ge, peak_gi)
        top = np.argsort(-peak_v, kind='stable')[:5]
        focus = []
        for j in top:
            root = roots[int(motor[j])]
            focus.append({'root_id': root, 'tier_index': int(motor[j]),
                          'trace': [{**{k: row[k] for k in ('time_start_ms', 'time_end_ms')},
                                     'voltage_mV': row['voltage_mV'][j],
                                     'arriving_exc_uS': row['arriving_exc_uS'][j],
                                     'arriving_inh_uS': row['arriving_inh_uS'][j]} for row in motor_steps],
                          'incoming_fired_edges': [e for e in diagnostics['fired_source_to_motor_edges'] if e['target_root_id'] == root]})
        records.append({'arm': name, 'focus_motor_cells': focus, 'spike_events': spike_events,
                        'diagnostics': diagnostics, 'target_cells': len(target),
                        'current_nA_per_cell': float(current[target][0]) if len(target) else 0.,
                        'spike_totals': totals, 'trace': trace})
    result = {'label': '手工研究原型，不是成品', 'status': 'MEASURED_REAL_CONNECTOME_PAIRED_INPUT_NOT_BODY_CLOSURE',
              'n_neurons': n, 'sensory_cells': len(sensory), 'motor_pool_cells': len(motor),
              'dt_ms': dt_ms, 'duration_ms': steps*dt_ms, 'pulse_ms': pulse_steps*dt_ms,
              'edge_scale': .2, 'arms': records,
              'limits': ['Constant 20ms diagnostic LF sensory pulse; not natural behavior or cell-specific tuning.',
                         'Same zero-background initial state and actual rebuilt edges in all arms.',
                         'Motor spikes are observed only; no motor current, body actuator or decoder is used.',
                         'Single LF probe per modality; no claim of standing, causal stabilization or all-leg generality.'],
              'sha256': {str(x): sha(x) for x in [identity_path, manifest_path, lookup_path, input_path, Path(__file__)]}}
    (OUT/'flybody_connectome_event_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for r in records:
        t = [x['time_ms'] for x in r['trace']]
        for ax, key in zip(axes, ['target', 'nonsensory', 'motor']):
            ax.plot(t, np.cumsum([x[key] for x in r['trace']]), label=r['arm'])
            ax.set_ylabel('Cumulative '+key+' spikes')
    axes[0].legend()
    axes[-1].set_xlabel('Time (ms); diagnostic input applied 0–20 ms')
    fig.suptitle('Actual BANC connectome: matched-state sensory input probes\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; no body control')
    fig.tight_layout()
    fig.savefig(OUT/'flybody_connectome_path_audit.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in records:
        values = sorted(x['peak_post_step_voltage_mV'] for x in r['diagnostics']['motor_cells'])
        ax.plot(values, label=r['arm'])
    ax.axhline(loop.ConductanceParams().threshold_mV, color='red', linestyle='--', label='Spike threshold')
    ax.set_xlabel('Motor pool cell rank (sorted independently within each arm)')
    ax.set_ylabel('Peak post-step voltage (mV; includes initial rest)')
    ax.set_title('Motor subthreshold response; 100 ms diagnostic sensory pulse experiment\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'flybody_motor_subthreshold.png', dpi=150)
    plt.close(fig)
    print(json.dumps({r['arm']: r['spike_totals'] for r in records}, indent=2), flush=True)


if __name__ == '__main__':
    with loop.mg.guarded_run('flybody_connectome_transfer_audit', need_mb=2200., cap_mb=4608.):
        main()
