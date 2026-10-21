"""P2/S1 saved-event analysis only: manual research prototype, not a product.
No neural/body simulation and no import or execution of model source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import uuid
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE/'outputs/model_branches/p1_fixed_input_clock_replay/20261013_234617_bd6f9aa2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(path, expected):
    """Verify byte integrity against pinned result; not signed authenticity."""
    with zipfile.ZipFile(path) as z:
        records = json.loads(z.read('manifest.json'))['files']
        if {r['path']: r['sha256'] for r in records} != expected:
            raise ValueError('snapshot manifest differs from pinned source hashes')
        for row in records:
            if hashlib.sha256(z.read(row['id'])).hexdigest() != row['sha256']:
                raise ValueError('source tampering: '+row['path'])
        return {r['path']: z.read(r['id']).decode('utf-8') for r in records}


def validate_events(times, neurons, n, duration_ms):
    t, ids = np.asarray(times), np.asarray(neurons)
    if t.ndim != 1 or ids.shape != t.shape or ids.dtype.kind not in 'iu':
        raise ValueError('invalid event arrays')
    if not np.isfinite(t).all() or np.any(t <= 0) or np.any(t > duration_ms):
        raise ValueError('events outside (0, duration]')
    if np.any(ids < 0) or np.any(ids >= n):
        raise ValueError('event neuron outside tier')
    # Repeated timestamp across different roots is legal; same root/time is not
    # a binary source event. Fail rather than silently drop or double count it.
    if len(set(zip(t.tolist(), ids.tolist()))) != len(t):
        raise ValueError('duplicate root/timestamp event')
    return t, ids


def sampled(times, neurons, manifest, dt_ms, config, duration_ms=50):
    if not math.isfinite(dt_ms) or dt_ms <= 0 or not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError('positive finite grid/duration required')
    if any(not math.isfinite(v) or v <= 0 for v in config.values()):
        raise ValueError('positive finite decoder parameters required')
    t, ids = validate_events(times, neurons, manifest['n_neurons'], duration_ms)
    steps = round(duration_ms/dt_ms)
    if abs(steps*dt_ms-duration_ms) > 1e-10 or abs(round(5/dt_ms)*dt_ms-5) > 1e-10:
        raise ValueError('unaligned grid')
    # Inclusive right endpoints (s-dt,s]; searchsorted handles exact boundaries
    # without rounding events to nearest sample or early consumption.
    ends = np.arange(1, steps+1)*dt_ms
    bins = np.searchsorted(ends, t, side='left')
    nu = manifest['n_actuators']
    counts = np.zeros((steps, nu))
    for ai, row in enumerate(manifest['rows']):
        group = np.asarray(row['local_indices'], dtype=int)
        if len(set(group.tolist())) != len(group) or np.any(group < 0) or np.any(group >= manifest['n_neurons']):
            raise ValueError('invalid pool membership')
        if len(group):
            np.add.at(counts[:, ai], bins[np.isin(ids, group)], 1/len(group))
    dt = dt_ms/1000
    decay = math.exp(-dt/config['filter_tau_s'])
    rates = np.zeros(nu)
    saved = []
    for k in range(steps):
        rates = decay*rates + (1-decay)*(counts[k]/dt)
        if (k+1) % round(5/dt_ms) == 0:
            saved.append(rates.copy())
    r = np.asarray(saved)
    return r, np.clip(r/config['rate_for_full_control_hz'], 0, config['max_control'])


def continuous(times, neurons, manifest, config, duration_ms=50):
    t, ids = validate_events(times, neurons, manifest['n_neurons'], duration_ms)
    endpoints = np.arange(5, duration_ms+1e-10, 5)
    r = np.zeros((len(endpoints), manifest['n_actuators']))
    tau = config['filter_tau_s']
    for ai, row in enumerate(manifest['rows']):
        group = row['local_indices']
        if not group:
            continue
        events = t[np.isin(ids, group)]/1000
        for k, end in enumerate(endpoints/1000):
            causal = events[events <= end]
            r[k, ai] = np.exp(-(end-causal)/tau).sum()/(len(group)*tau)
    return r, np.clip(r/config['rate_for_full_control_hz'], 0, config['max_control'])


def difference(a, b):
    d = np.abs(a-b)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), l1=float(d.sum()))


def first_commands(a, b, manifest, roots):
    results = []
    for ai, row in enumerate(manifest['rows']):
        ks = np.flatnonzero(abs(a[:, ai]-b[:, ai]) > 1e-12)
        if len(ks):
            k = int(ks[0])
            results.append(dict(actuator=row['actuator'], actuator_index=ai,
                time_ms=(k+1)*5, base=float(a[k, ai]), other=float(b[k, ai]),
                local_indices=row['local_indices'], tier_root_ids=[str(roots[i]) for i in row['local_indices']]))
    return sorted(results, key=lambda r: (r['time_ms'], r['actuator_index']))


def divergences(ta, ia, tb, ib, roots):
    records = []
    for i in np.union1d(ia, ib):
        a, b = np.sort(ta[ia == i]), np.sort(tb[ib == i])
        m = min(len(a), len(b))
        mismatch = np.flatnonzero(a[:m] != b[:m])
        if len(mismatch) or len(a) != len(b):
            j = int(mismatch[0]) if len(mismatch) else m
            av = float(a[j]) if j < len(a) else None
            bv = float(b[j]) if j < len(b) else None
            records.append(dict(root_id=str(roots[i]), local_index=int(i),
                base_count=len(a), other_count=len(b), first_event_ordinal=j,
                base_timestamp_ms=av, other_timestamp_ms=bv,
                earliest_divergence_ms=min(v for v in (av,bv) if v is not None),
                interpretation='ordinal timestamp mismatch may be quantization; not automatically integration error'))
    return sorted(records, key=lambda r: (r['earliest_divergence_ms'], r['local_index']))


def run(replay, output_root):
    replay = Path(replay).resolve()
    result = json.loads((replay/'result.json').read_text())
    if sha(replay/'traces.npz') != result['traces_sha256'] or sha(replay/'source_snapshot.zip') != result['source_snapshot_sha256']:
        raise ValueError('replay bundle digest mismatch')
    source = verify_snapshot(replay/'source_snapshot.zip', result['source_hashes'])
    p1dir = Path(result['bundle'])
    for name in ('result.json','traces.npz','source_snapshot.zip'):
        p = p1dir/name
        if sha(p) != result['input_hashes'][str(p)]:
            raise ValueError('P1 input tampering')
    p1 = json.loads((p1dir/'result.json').read_text())
    verify_snapshot(p1dir/'source_snapshot.zip', p1['source_hashes'])
    manifest, config = p1['motor_manifest'], p1['evidence_schema']['motor_translation']
    motor_source = source[str(HERE/'engine/embodied/motor_output.py')]
    if 'self.rates = decay*self.rates + (1-decay)*instantaneous' not in motor_source:
        raise ValueError('unsupported archived recurrence')
    with np.load(replay/'traces.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(p1dir/'traces.npz', allow_pickle=False) as z:
        p1arrays = {k: z[k] for k in z.files}
    roots = arrays['tier_root_ids']
    if roots.tolist() != p1['tier_root_ids']:
        raise ValueError('tier root mapping mismatch')
    outputs, checks, summaries, ds, cmd = {}, {}, {}, {}, {}
    for label, dt in [('base',.5), ('half',.25), ('quarter',.125)]:
        times, neurons = arrays[label+'_event_time_ms'], arrays[label+'_event_neuron']
        if not np.array_equal(times, (arrays[label+'_event_step']+1)*dt):
            raise ValueError('saved event step/time mismatch')
        r, c = sampled(times, neurons, manifest, dt, config)
        checks[label] = dict(rates_max_abs=float(abs(r-arrays[label+'_motor_rates_hz']).max()),
            commands_exact=bool(np.array_equal(c, arrays[label+'_decoded_next_controls'])))
        if checks[label]['rates_max_abs'] > 1e-10 or not np.allclose(c, arrays[label+'_decoded_next_controls'], atol=1e-12, rtol=0):
            raise ValueError('independent saved decoder reconstruction failed')
        for grid in (.5,.25,.125):
            outputs[f'{label}_{grid}'] = sampled(times, neurons, manifest, grid, config)
        outputs[label+'_continuous'] = continuous(times, neurons, manifest, config)
        if label != 'base':
            ds[label] = divergences(arrays['base_event_time_ms'], arrays['base_event_neuron'], times, neurons, roots)
            cmd[label] = first_commands(outputs['base_0.5'][1], c, manifest, roots)
    checks['base_commands_match_p1'] = bool(np.allclose(outputs['base_0.5'][1], p1arrays['normal_decoded_next_controls'], atol=1e-12, rtol=0))
    if not checks['base_commands_match_p1']:
        raise ValueError('P1 command mismatch')
    for label, dt in [('half',.25), ('quarter',.125)]:
        summaries[label] = {}
        pairs = {'same_base_events_sampling_only':('base_0.5',f'base_{dt}'),
                 'different_events_same_refined_grid':(f'base_{dt}',f'{label}_{dt}'),
                 'observed_combined':('base_0.5',f'{label}_{dt}'),
                 'different_events_continuous':('base_continuous',label+'_continuous')}
        for name, (a,b) in pairs.items():
            summaries[label][name] = dict(rates_hz=difference(outputs[a][0], outputs[b][0]), controls=difference(outputs[a][1], outputs[b][1]))
    for dt in (.5,.25,.125):
        summaries['base_grid_'+str(dt)+'_vs_continuous'] = dict(rates_hz=difference(outputs[f'base_{dt}'][0],outputs['base_continuous'][0]), controls=difference(outputs[f'base_{dt}'][1],outputs['base_continuous'][1]))
    dest = Path(output_root)/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    flat = {k+'_rates_hz': v[0] for k,v in outputs.items()}
    flat.update({k+'_controls': v[1] for k,v in outputs.items()})
    np.savez_compressed(dest/'sampling.npz', **flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    x = np.arange(5,51,5)
    for dt in (.5,.25,.125):
        for ai, quantity in enumerate(('rate','control')):
            axes[ai,0].plot(x, np.max(abs(outputs[f'base_{dt}'][ai]-outputs['base_continuous'][ai]),axis=1),label=str(dt)+'ms same events')
    for label,dt in [('half',.25),('quarter',.125)]:
        for ai in (0,1):
            axes[ai,1].plot(x,np.max(abs(outputs[f'{label}_{dt}'][ai]-outputs[f'base_{dt}'][ai]),axis=1),label=label+' events vs base, same grid')
    for ai in (0,1):
        for bi in (0,1):
            axes[ai,bi].legend(fontsize=8); axes[ai,bi].set_xlabel('Boundary time (ms)')
            axes[ai,bi].set_ylabel('Max pool '+('rate difference (Hz)' if ai==0 else 'control difference'))
    fig.suptitle('P2 saved events only: sampling effect vs event-stream difference; NOT convergence')
    fig.tight_layout(); fig.savefig(dest/'sampling.png',dpi=150); plt.close(fig)
    inputs = [replay/n for n in ('result.json','traces.npz','source_snapshot.zip')]+[p1dir/n for n in ('result.json','traces.npz','source_snapshot.zip')]
    report = dict(status='PASS',label='Manual research implementation; not a finished product', output_directory=str(dest),
        checks=checks, comparisons=summaries, per_root_divergences=ds, first_per_pool_command_differences=cmd,
        manifest=manifest, motor_translation=config, input_hashes={str(p):sha(p) for p in inputs},
        tool_sha256=sha(__file__), sampling_sha256=sha(dest/'sampling.npz'), plot_sha256=sha(dest/'sampling.png'), environment=dict(executable=sys.executable,numpy=np.__version__),
        formulas=dict(discrete='r[k]=exp(-dt/tau)*r[k-1]+(1-exp(-dt/tau))*sum_events_in_bin(1/N_present)/dt',
                      continuous='r(T)=sum_{t_e<=T} exp(-(T-t_e)/tau)/(N_present*tau); zero initial memory; impulses right-continuous',
                      controls='clip(r/100Hz,0,1)'),
        assumptions=['Inclusive end bins (start,end]; t=0 rejected; duplicates same root/time rejected; multiple distinct events within bin counted with multiplicity.',
                     'Each event has population weight 1/N_present; missing pools zero; shared pools remain shared.',
                     'Continuous tau is same 5ms tissue-facing decoder tau, not neural conductance tau or measured biological calibration.',
                     'Full factorial events x decoder grid saved at common 5ms boundaries; sampling original binary grid reconstructed without model execution.',
                     'Command clipping nonlinear; max/L1 differences are not uniquely additive causal decomposition. Alternative grid/path changes attribution.',
                     'Timestamp ordinal shift is quantization-compatible, not automatically error; count changes do not prove which neural mechanism caused them.',
                     'No simulations, source/model edits, history overwrite, threshold/gain changes or biological/behavior convergence claims.'],
        next_minimum_diagnostic='Inspect saved first-divergent-root incoming paths, current membership and delay/refractory quantization in archived source; voltage/conductance trace needed to distinguish threshold crossing from propagation timing, not available in saved events.')
    with (dest/'result.json').open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(dict(output=str(dest),checks=checks,comparisons=summaries),indent=2))
    return dest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,default=DEFAULT)
    parser.add_argument('--output-root',type=Path,default=HERE/'outputs/p2_event_sampling')
    args=parser.parse_args()
    run(args.replay,args.output_root)
