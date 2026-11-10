"""S5/P2 numerics_shift: is the pointwise dt difference dominated by discrete-spike grid phase?

One fixed neural configuration (dt = 0.015625 ms, backward_euler, 64 sub-steps per 1 ms command,
200 ms = 12800 steps, 12000 cells, same transferred initial state, same RH .375 once, same causal
1 ms commands, same dt-aware 5 ms pool filter, unchanged readout). Exactly one thing changes per
run: the whole 200 x 12000 injected-current stream is shifted in time by N sub-steps.
Run 0 (N = 0) is the control and must reproduce the sealed numerics_followup
backward_euler_0p015625ms 200 ms result bitwise.

Shift implementation (identical for every run, declared in PREREGISTRATION.json):
the injected-current stream is defined on the 1 ms command grid and held constant inside a 1 ms
block, i.e. at global neural sub-step s (0..12799) the sealed stage applies currents[s // 64].
The shift delays the stream by N sub-step indices and holds the first sample for the leading N
sub-steps:

    k       = max(s - N, 0)          # delayed source sub-step, first sample held
    block_s = k // 64                # 1 ms source block whose held value is applied at s
    applied_current(s) = currents[block_s]

For N = 0 this reduces to currents[s // 64], i.e. bitwise the sealed application. For 0 < N < 64
the first source block covers s = 0..N+63 and every later 1 ms hold boundary is delayed by exactly
N sub-steps (boundaries at N+64, N+128, ... instead of 64, 128, ...), so a shift of N sub-steps is
exactly a grid-phase offset of the injected stream, 0.125 ms when N = 8. Nothing else changes: the
command grid stays on the original 1 ms boundaries and remains causal.

Never writes into the sealed numerics / numerics_followup / sensory stages.
Manual research diagnostic, not a product; no timestep and no model is selected.
"""
import sys, json, pickle, copy, hashlib, time, traceback
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]
from engine.neural_cond import ConductanceNetwork, ConductanceParams

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift')
SRC = Path(__file__).resolve().parent
NUM = OUT.parent / 'numerics'
FOL = OUT.parent / 'numerics_followup'

DT_MS = 0.015625                 # the finest already-tested grid; one single dt for every run
SUBSTEPS = 64                    # neural sub-steps per 1 ms command
N_MS = 200
N_STEP = N_MS * SUBSTEPS         # 12800
WINDOWS_MS = [1, 5, 10, 20, 50, 100, 200]
METHOD = 'backward_euler'
RH_GAIN = 0.375
FILTER_TAU_MS = 5.0
FIELDS = ['v', 'ge', 'gi', 'adaptation', 'refractory_until_ms', 'spike_count']

# Runs: control plus four declared sub-step shifts of the injected-current stream.
SHIFTS = [0, 1, 2, 4, 8]         # 0, 0.015625, 0.03125, 0.0625, 0.125 ms
PREFIX_STEPS = 96                # exact full-state prefix probe length (2 x 96 sub-steps, cheap)

# Sealed reference: numerics_followup supplement, backward_euler, 200 ms, pool_rates nRMS,
# fine pair 0.03125 vs 0.015625 ms. Asserted against FOL/SUMMARY.json at prereg time.
D_DT = 0.287739
D_DT_COARSE = 0.292466
SEALED_DT_FLOOR = 0.25           # the "sealed dt difference stays >= 0.25" clause
PHASE_FRACTION = 0.5             # PHASE_EXPLAINS_DT if D_shift(0.125ms) >= 0.5 * D_DT
INTEGRATOR_FRACTION = 0.1        # INTEGRATOR_EXPLAINS_DT if D_shift(0.125ms) <= 0.1 * D_DT
ENGINEERING_THRESHOLD = 0.15     # the engineering nRMS tolerance used by the sealed gate
SPLIT_THRESHOLD = 0.10           # "pool within 10 %" used for the aggregate comparison

SEALED_FOL_RUNS = ['backward_euler_0p015625ms', 'backward_euler_0p03125ms', 'backward_euler_0p0625ms']

_MEMO = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def js(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=default) + '\n')


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def stats():
    """Full statistics of the sealed FOLLOWUP 200 ms runs, for the fixed 5 ms filter/readout check."""
    return js(FOL / 'SUMMARY.json')


def warm_cache():
    if 'cache' not in _MEMO:
        with (NUM / 'CONFIGS/dt_0p5ms/WARM_CACHE.pkl').open('rb') as f:
            _MEMO['cache'] = pickle.load(f)
    return _MEMO['cache']


def sealed_currents():
    """The same 200x12000 float64 actual sensory currents used by every sealed stage."""
    if 'currents' not in _MEMO:
        with np.load(NUM / 'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz') as z:
            cur = z['sensory_current'].copy()
        assert cur.shape == (N_MS, 12000) and cur.dtype == np.float64
        _MEMO['currents'] = cur
    return _MEMO['currents']


def identities():
    if 'ident' not in _MEMO:
        ident = js(NUM / 'TRANSLATOR_IDENTITIES.json')
        pools = [np.asarray(p) for p in ident['pools']]
        poolmat = sparse.csr_matrix(
            (np.concatenate([np.full(len(p), 1 / len(p)) for p in pools]),
             (np.repeat(np.arange(66), [len(p) for p in pools]), np.concatenate(pools))),
            shape=(66, 12000))
        _MEMO['ident'] = (ident, poolmat)
    return _MEMO['ident']


def transfer(old, dt, method):
    """Identical state transfer to numerics_followup, used only to cross-check the sealed pickle."""
    ratio = old.dt / dt
    assert abs(ratio - round(ratio)) < 1e-12
    new = ConductanceNetwork(old.n, old.p, dt, method, old.threshold_offset_mV)
    new.set_connectivity(old.We, old.Wi)
    for k in FIELDS:
        setattr(new, k, getattr(old, k).copy())
    new.t_ms = old.t_ms
    new.steps = round(old.t_ms / dt)
    assert new.steps * dt == old.t_ms
    for k in range(len(old._ring)):
        idx = round(k * old.dt / dt)
        assert idx < len(new._ring)
        new._ring[idx] = old._ring[(old._ring_i + k) % len(old._ring)]
    new._ring_i = 0
    return new


_SEALED_INITIAL = FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl'


def load_sealed_initial():
    """The transferred initial state, loaded from the sealed numerics_followup pickle itself."""
    with _SEALED_INITIAL.open('rb') as f:
        return pickle.load(f)


def rebuilt_initial():
    """Independent rebuild (warm cache -> transfer -> RH .375 once) for cross-checking the pickle."""
    c = warm_cache()[1.3]
    old = c['net']
    net = transfer(old, DT_MS, METHOD)
    assert net.We.nnz == 180604 and net.Wi.nnz == 148811
    ident, _ = identities()
    rows = np.repeat(np.arange(net.n), np.diff(net.We.indptr))
    mask = np.isin(rows, ident['RH_cells'])
    net.We.data[mask] *= RH_GAIN
    assert np.array_equal(net.We.data[~mask], old.We.data[~mask])
    assert np.array_equal(net.We.data[mask], old.We.data[mask] * RH_GAIN)
    return net, c['rates'].copy()


def net_exact(a, b):
    out = {k: bool(np.array_equal(getattr(a, k), getattr(b, k))) for k in FIELDS}
    out['t_ms'] = a.t_ms == b.t_ms
    out['steps'] = a.steps == b.steps
    out['dt'] = a.dt == b.dt
    out['voltage_method'] = a.voltage_method == b.voltage_method
    out['ring'] = bool(np.array_equal(a._ring, b._ring))
    out['ring_i'] = a._ring_i == b._ring_i
    out['threshold_offset_mV'] = bool(np.array_equal(a.threshold_offset_mV, b.threshold_offset_mV))
    for key in ['We', 'Wi']:
        x, y = getattr(a, key), getattr(b, key)
        out[key] = bool(x.shape == y.shape and np.array_equal(x.indices, y.indices)
                        and np.array_equal(x.indptr, y.indptr) and np.array_equal(x.data, y.data))
    return out


def block_index(shift):
    """Delayed 1 ms source block applied at each neural sub-step (see module docstring)."""
    s = np.arange(N_STEP)
    k = np.maximum(s - shift, 0)
    blk = (k // SUBSTEPS).astype(np.int16)
    assert blk.min() == 0 and blk.max() == N_MS - 1
    return blk


def prefix_block_index(shift, n=PREFIX_STEPS):
    s = np.arange(n)
    return ((np.maximum(s - shift, 0)) // SUBSTEPS).astype(int)


def label(shift):
    return 'shift_%03d_substeps' % shift


def metrics(a, b):
    """Sealed normalization: RMS(a-b)/max(RMS(b),1e-9) with absolute and reference RMS."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    absolute = rms(a - b)
    ref = rms(b)
    return dict(absolute_RMS=absolute, reference_RMS=ref, normalized_RMS=absolute / max(ref, 1e-9))


def spike_streams(packed, chunk=512):
    for a in range(0, packed.shape[0], chunk):
        b = min(a + chunk, packed.shape[0])
        yield a, np.unpackbits(packed[a:b], axis=1, count=12000)


def spike_difference(zp, zc):
    """Per-neural-step discrete difference statistics against the control packed spikes."""
    per_step = np.zeros(N_STEP, np.int64)
    extra = np.zeros(N_STEP, np.int64)      # spikes present only in the shifted run
    missing = np.zeros(N_STEP, np.int64)    # spikes present only in the control
    for a, sp in spike_streams(zp['spikes12000_packed']):
        b = a + sp.shape[0]
        c = np.unpackbits(zc['spikes12000_packed'][a:b], axis=1, count=12000)
        sp = sp.astype(bool)
        c = c.astype(bool)
        per_step[a:b] = (sp != c).sum(1)
        extra[a:b] = (sp & ~c).sum(1)
        missing[a:b] = (~sp & c).sum(1)
    return dict(per_step=per_step, extra=extra, missing=missing)


def window_mean_aggregate(x, y):
    """Aggregate metric: pool rates averaged over the whole window, compared across the 66 pools."""
    mx = np.asarray(x, float).mean(0)
    my = np.asarray(y, float).mean(0)
    m = metrics(mx, my)
    den = np.maximum(np.abs(my), 1e-12)
    rel = np.abs(mx - my) / den
    return dict(definition='per-pool time mean over the window, then nRMS across the 66 pools '
                          'using the sealed formula RMS(a-b)/max(RMS(b),1e-9)',
                **m,
                pearson_across_pools=float(np.corrcoef(mx, my)[0, 1]),
                pools_within_10pct=int(np.sum(rel <= SPLIT_THRESHOLD)),
                pools_total=int(rel.size),
                max_pool_relative_difference=float(rel.max()),
                median_pool_relative_difference=float(np.median(rel)))


def sealed_aggregate_anchors():
    """The same aggregate metric computed on the sealed dt pairs (read only, inputs not results)."""
    if 'agg' in _MEMO:
        return _MEMO['agg']
    pr = {}
    for name in SEALED_FOL_RUNS:
        with np.load(FOL / 'RUNS' / name / 'NEURAL.npz') as z:
            pr[name] = z['pool_rates'].copy()
    out = {}
    for tag, fine, coarse in [('fine_0p03125_vs_0p015625', 'backward_euler_0p03125ms', 'backward_euler_0p015625ms'),
                              ('coarse_0p0625_vs_0p03125', 'backward_euler_0p0625ms', 'backward_euler_0p03125ms')]:
        out[tag] = dict(fine=coarse, coarse=fine, window_mean=window_mean_aggregate(pr[fine], pr[coarse]))
    _MEMO['agg'] = out
    return out


def sealed_dt_baseline():
    s = js(FOL / 'SUMMARY.json')
    sup = s['supplement']['backward_euler']
    fine = {k: sup['windows']['200']['metrics'][k]['fine']['normalized_RMS']
            for k in ['pool_rates', 'raw_command', 'clipped_command']}
    coarse = {k: sup['windows']['200']['metrics'][k]['coarse']['normalized_RMS']
              for k in ['pool_rates', 'raw_command', 'clipped_command']}
    for k, v in [('pool_rates', D_DT), ('raw_command', 0.290435), ('clipped_command', 0.307135)]:
        assert abs(fine[k] - v) < 5e-7, (k, fine[k], v)
    assert abs(coarse['pool_rates'] - D_DT_COARSE) < 5e-7
    matched = s['matched_methods']['0.015625']['200']
    return dict(source=str(FOL / 'SUMMARY.json'),
                fine_pair_200ms=fine, coarse_pair_200ms=coarse,
                same_dt_method_pair_200ms={k: matched[k] for k in ['pool_rates', 'raw_command', 'clipped_command']},
                same_dt_method_note='frozen_exact vs backward_euler at the SAME dt=0.015625 ms, sealed '
                                    'read-only reference for the size of an integrator change at fixed dt')


# --------------------------------------------------------------------------------------- runs

def run_case(shift):
    folder = OUT / 'RUNS' / label(shift)
    folder.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    ini = load_sealed_initial()
    initial_net = copy.deepcopy(ini['net'])
    net = copy.deepcopy(ini['net'])
    rates = np.asarray(ini['rates'], float).copy()
    initial = rates.copy()
    ident, poolmat = identities()
    matrix = np.asarray(ident['old_matrix'])
    limits = np.asarray(ident['limits'])
    currents = sealed_currents()
    bidx = block_index(shift)
    decay = np.exp(-DT_MS / FILTER_TAU_MS)
    packed, pool, raw, clip, sources, counts, clock = [], [], [], [], [], [], []
    states = {k: [] for k in FIELDS}
    for m in range(N_MS):
        sources.append(rates.copy())
        cmd = limits / 100 * (matrix @ rates)
        raw.append(cmd)
        clip.append(np.clip(cmd, -limits, limits))
        cnt = np.zeros(12000, np.int32)
        for j in range(SUBSTEPS):
            s = m * SUBSTEPS + j
            sp = net.step(currents[bidx[s]])
            packed.append(np.packbits(sp))
            cnt += sp
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / (DT_MS / 1000)
        pool.append(rates.copy())
        counts.append(cnt)
        clock.append([net.t_ms, net.steps])
        for k in FIELDS:
            states[k].append(getattr(net, k).copy())
    with (folder / 'INITIAL_STATE.pkl').open('wb') as f:
        pickle.dump(dict(net=initial_net, rates=initial.copy()), f, protocol=5)
    with (folder / 'FINAL_STATE.pkl').open('wb') as f:
        pickle.dump(dict(net=net, rates=rates), f, protocol=5)
    z = dict(spikes12000_packed=np.asarray(packed), pool_rates=np.asarray(pool),
             raw_command=np.asarray(raw), clipped_command=np.asarray(clip),
             command_source_rates=np.asarray(sources), counts12000=np.asarray(counts),
             initial_pool_rates=initial, sensory_current=currents,
             applied_block_index=bidx, shift_substeps=np.int64(shift),
             shift_ms=np.float64(shift * DT_MS),
             neural_clock=np.asarray(clock),
             neural_dt_ms=np.float64(DT_MS), time_s=1.3 + np.arange(1, N_MS + 1) / 1000,
             command_time_s=1.3 + np.arange(N_MS) / 1000, **states)
    np.savez_compressed(folder / 'NEURAL.npz', **z)
    result = dict(label=label(shift), shift_substeps=int(shift), shift_ms=float(shift * DT_MS),
                  dt_ms=DT_MS, method=METHOD, substeps_per_command=SUBSTEPS, neural_steps=N_STEP,
                  neural_elapsed_ms=float(net.t_ms - 1100.0), wall_s=time.monotonic() - t0,
                  folder=str(folder), physical_runs=0, new_warm_runs=0,
                  applied_shift=dict(rule='k=max(s-N,0); block=k//64; applied_current(s)=sensory_current[block]',
                                     substeps=int(shift), milliseconds=float(shift * DT_MS),
                                     first_sample_held=bool(shift > 0),
                                     first_substep_with_different_source=(None if shift == 0 else 64),
                                     first_substep_with_different_source_reason=(
                                         'for 0 < N < 64 the control applies block 1 at s=64 while the delayed '
                                         'stream still applies block 0'),
                                     sensory_current_within_1ms_block_constant=True),
                  total_spikes=int(np.asarray(counts).sum()),
                  mean_hz=float(np.asarray(counts).sum() / 12000 / .2),
                  sensory_current_stream_unchanged=True)
    dump(folder / 'RESULT.json', result)
    print('CASE', result['label'], round(result['wall_s'], 3), result['total_spikes'], flush=True)
    return result, z


def control_check(z):
    """Control must reproduce the sealed numerics_followup dt=0.015625 backward_euler 200 ms bitwise."""
    sealed = arrays(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz')
    keys = ['spikes12000_packed', 'pool_rates', 'raw_command', 'clipped_command', 'command_source_rates',
            'counts12000', 'sensory_current', 'initial_pool_rates', 'neural_dt_ms', 'neural_clock',
            'time_s', 'command_time_s'] + FIELDS
    exact = {k: bool(np.array_equal(np.asarray(z[k]), np.asarray(sealed[k]))) for k in keys}
    s_ini = load_sealed_initial()
    with (FOL / 'RUNS/backward_euler_0p015625ms/FINAL_STATE.pkl').open('rb') as f:
        s_fin = pickle.load(f)
    with (OUT / 'RUNS' / label(0) / 'INITIAL_STATE.pkl').open('rb') as f:
        m_ini = pickle.load(f)
    with (OUT / 'RUNS' / label(0) / 'FINAL_STATE.pkl').open('rb') as f:
        m_fin = pickle.load(f)
    initial = dict(net=net_exact(m_ini['net'], s_ini['net']), rates=bool(np.array_equal(m_ini['rates'], s_ini['rates'])))
    final = dict(net=net_exact(m_fin['net'], s_fin['net']), rates=bool(np.array_equal(m_fin['rates'], s_fin['rates'])))
    rec = dict(sealed_neural_npz=str(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz'),
               arrays=exact, initial_state=initial, final_state=final)
    rec['all_exact'] = all(exact.values()) and all(initial['net'].values()) and initial['rates'] \
        and all(final['net'].values()) and final['rates']
    dump(OUT / 'CONTROL_BITWISE_CHECK.json', rec)
    if not rec['all_exact']:
        raise RuntimeError('CONTROL_NOT_BITWISE:' + json.dumps({
            'arrays': {k: v for k, v in exact.items() if not v},
            'initial_state': {k: v for k, v in initial['net'].items() if not v},
            'initial_rates': initial['rates'],
            'final_state': {k: v for k, v in final['net'].items() if not v},
            'final_rates': final['rates']}))
    return rec


def prefix_probe(shifts):
    """Exact full-state determination of the first sub-step at which the continuous state differs.

    Two networks (control stream and shifted stream) advance side by side from the same transferred
    initial state; every sub-step compares ALL continuous state arrays with exact float64 equality.
    """
    out = {}
    current = sealed_currents()
    for shift in shifts:
        if shift == 0:
            continue
        ini = load_sealed_initial()
        a = copy.deepcopy(ini['net'])
        b = copy.deepcopy(ini['net'])
        control_blk = prefix_block_index(0)
        shifted_blk = prefix_block_index(shift)
        first_state = None
        first_any = None
        detail = None
        executed = 0
        for s in range(PREFIX_STEPS):
            sa = a.step(current[control_blk[s]])
            sb = b.step(current[shifted_blk[s]])
            executed = s + 1
            eq = {k: bool(np.array_equal(getattr(a, k), getattr(b, k))) for k in FIELDS}
            state_eq = all(eq.values())
            spike_eq = bool(np.array_equal(sa, sb))
            if first_any is None and not (state_eq and spike_eq):
                first_any = s
            if first_state is None and not state_eq:
                first_state = s
                detail = dict(step=s, t_ms=float(b.t_ms),
                              fields_equal=eq,
                              differing_entries={k: int(np.count_nonzero(np.asarray(getattr(b, k)) != np.asarray(getattr(a, k))))
                                                 for k in FIELDS},
                              max_abs_difference={k: float(np.max(np.abs(np.asarray(getattr(b, k), dtype=float)
                                                                         - np.asarray(getattr(a, k), dtype=float))))
                                                  for k in FIELDS},
                              control_applied_block=int(control_blk[s]),
                              shifted_applied_block=int(shifted_blk[s]),
                              control_source_block=max(s - 0, 0) // SUBSTEPS,
                              shifted_source_block=max(s - shift, 0) // SUBSTEPS)
                break
        out[str(shift)] = dict(shift_substeps=int(shift), shift_ms=float(shift * DT_MS),
                               probe_horizon_steps=PREFIX_STEPS, probe_horizon_ms=float(PREFIX_STEPS * DT_MS),
                               executed_steps_per_stream=executed,
                               first_differing_substep=first_any,
                               first_differing_time_ms=(None if first_any is None else float((first_any + 1) * DT_MS)),
                               first_differing_continuous_substep=first_state,
                               first_differing_continuous_detail=detail,
                               all_continuous_fields_identical_until_first_difference=True,
                               threshold_flip_any=bool(first_any is not None),
                               note='two networks (control stream and shifted stream) advance side by side from the '
                                    'same transferred initial state; every sub-step compares all six state arrays with '
                                    'exact float64 equality, so the reported first differing sub-step is exact '
                                    'within the probe horizon')
        del a, b
    dump(OUT / 'PREFIX_PROBE.json', out)
    return out


def compare_to_control(shift, zp, zc):
    sd = spike_difference(zp, zc)
    out = dict(label=label(shift), shift_substeps=int(shift), shift_ms=float(shift * DT_MS), dt_ms=DT_MS)
    windows = {}
    for w in WINDOWS_MS:
        n = w * SUBSTEPS
        entry = dict(window_ms=w, neural_steps=n)
        for key in ['pool_rates', 'raw_command', 'clipped_command']:
            entry[key] = metrics(zp[key][:w], zc[key][:w])
        entry['counts12000_1ms'] = metrics(zp['counts12000'][:w], zc['counts12000'][:w])
        entry['differing_spikes_cumulative'] = int(sd['per_step'][:n].sum())
        entry['extra_spikes_cumulative'] = int(sd['extra'][:n].sum())
        entry['missing_spikes_cumulative'] = int(sd['missing'][:n].sum())
        entry['differing_steps_cumulative'] = int((sd['per_step'][:n] > 0).sum())
        idx = WINDOWS_MS.index(w)
        prev = 0 if idx == 0 else WINDOWS_MS[idx - 1] * SUBSTEPS
        entry['differing_spikes_within_window'] = int(sd['per_step'][prev:n].sum())
        windows[str(w)] = entry
    out['windows'] = windows
    cp = zp['counts12000'].sum(0).astype(np.int64)
    cc = zc['counts12000'].sum(0).astype(np.int64)
    d = cp - cc
    out['final_spike_count'] = dict(
        shifted_total=int(cp.sum()), control_total=int(cc.sum()),
        difference_total=int(cp.sum() - cc.sum()),
        relative_difference=float(cp.sum() - cc.sum()) / float(cc.sum()),
        max_abs_per_cell_difference=int(np.abs(d).max()),
        argmax_cell=int(np.abs(d).argmax()),
        cells_with_difference=int((d != 0).sum()),
        cells_total=int(d.size),
        per_cell_pearson=float(np.corrcoef(cp, cc)[0, 1]),
        per_cell_spearman=float(_spearman(cp, cc)))
    first_diff = int(np.flatnonzero(sd['per_step'])[0]) if sd['per_step'].any() else None
    out['discrete_divergence'] = dict(
        first_differing_spike_step=first_diff,
        first_differing_spike_time_ms=(None if first_diff is None else float((first_diff + 1) * DT_MS)),
        differing_spikes_at_that_step=(None if first_diff is None else int(sd['per_step'][first_diff])),
        extra_at_that_step=(None if first_diff is None else int(sd['extra'][first_diff])),
        missing_at_that_step=(None if first_diff is None else int(sd['missing'][first_diff])),
        any_threshold_flip=bool(first_diff is not None))
    out['aggregate_200ms'] = dict(
        window_mean_pool_rates=window_mean_aggregate(zp['pool_rates'][:N_MS], zc['pool_rates'][:N_MS]),
        window_mean_raw_command=window_mean_aggregate(zp['raw_command'][:N_MS], zc['raw_command'][:N_MS]),
        window_mean_clipped_command=window_mean_aggregate(zp['clipped_command'][:N_MS], zc['clipped_command'][:N_MS]))
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def growth(comparisons):
    """Growth shape of D_shift versus the shift, and where it crosses the engineering threshold."""
    shifts = sorted(int(k) for k in comparisons)
    pts = [(s, comparisons[str(s)]['windows'][str(N_MS)]['pool_rates']['normalized_RMS']) for s in shifts]
    measured = [(s, v) for s, v in pts if s > 0]
    per_substep = {str(s): v / s for s, v in measured}
    vals = np.asarray([v for _, v in measured], float)
    prop = dict(per_substep_normalized=per_substep,
                min=float(vals.min()), max=float(vals.max()),
                max_over_min=float(vals.max() / max(vals.min(), 1e-300)),
                roughly_proportional=bool(vals.max() / max(vals.min(), 1e-300) <= 3.0),
                note='D_shift/substeps for each measured shift; roughly proportional if the ratio stays '
                     'within a factor 3 over the 8x shift span')
    x = np.log(np.asarray([s for s, _ in measured], float))
    y = np.log(np.maximum(vals, 1e-300))
    slope, intercept = np.polyfit(x, y, 1)
    loglog = dict(slope=float(slope), intercept=float(intercept),
                  fit_note='log(D) = intercept + slope*log(N); slope 1 means strictly proportional')
    cross = None
    for (s0, v0), (s1, v1) in zip(measured[:-1], measured[1:]):
        if (v0 - ENGINEERING_THRESHOLD) * (v1 - ENGINEERING_THRESHOLD) <= 0 and v0 != v1:
            t = (ENGINEERING_THRESHOLD - v0) / (v1 - v0)
            cross = dict(shift_substeps=float(s0 + t * (s1 - s0)), shift_ms=float((s0 + t * (s1 - s0)) * DT_MS),
                         within_measured_range=True,
                         bracket=[s0, s1], method='linear interpolation between measured points')
            break
    extrapolated = None
    if cross is None and np.any(vals >= ENGINEERING_THRESHOLD):
        smin = min(s for s, v in measured if v >= ENGINEERING_THRESHOLD)
        cross = dict(shift_substeps=float(smin), shift_ms=float(smin * DT_MS), within_measured_range=True,
                     bracket=[smin, smin], method='first measured shift at or above the threshold')
    if cross is None:
        n_ext = float(np.exp((np.log(ENGINEERING_THRESHOLD) - intercept) / slope))
        extrapolated = dict(shift_substeps=n_ext, shift_ms=n_ext * DT_MS,
                            method='log-log extrapolation of the measured points',
                            within_measured_range=False,
                            label='EXTRAPOLATED beyond the measured shift range')
    return dict(points=[dict(shift_substeps=s, shift_ms=s * DT_MS,
                             pointwise_pool_nRMS=(0.0 if s == 0 else comparisons[str(s)]['windows'][str(N_MS)]
                                                 ['pool_rates']['normalized_RMS']),
                             aggregate_window_mean_nRMS=(0.0 if s == 0 else comparisons[str(s)]['aggregate_200ms']
                                                         ['window_mean_pool_rates']['normalized_RMS']))
                        for s in shifts],
                proportionality=prop, loglog_fit=loglog,
                engineering_threshold=ENGINEERING_THRESHOLD,
                crossing_measured=cross, crossing_extrapolated=extrapolated,
                largest_measured_shift_substeps=int(shifts[-1]),
                largest_measured_shift_ms=float(shifts[-1] * DT_MS))


def verdict(comparisons, growth_rec):
    d_large = comparisons[str(SHIFTS[-1])]['windows'][str(N_MS)]['pool_rates']['normalized_RMS']
    phase_thr = PHASE_FRACTION * D_DT
    int_thr = INTEGRATOR_FRACTION * D_DT
    dt_ok = D_DT >= SEALED_DT_FLOOR
    if d_large >= phase_thr:
        reading = 'PHASE_EXPLAINS_DT'
        meaning = ('A pure 0.125 ms time shift of the injected-current stream reproduces at least half of the '
                   'adjacent-dt pointwise difference at 200 ms, with the integrator, dt, state, commands and readout '
                   'held fixed. The pointwise dt difference is therefore dominated by where discrete spikes and '
                   'held input blocks land on the sub-millisecond grid, not by accumulated integration error.')
    elif d_large <= int_thr and dt_ok:
        reading = 'INTEGRATOR_EXPLAINS_DT'
        meaning = ('A 0.125 ms time shift of the injected-current stream stays at or below a tenth of the adjacent-dt '
                   'pointwise difference at 200 ms while the sealed dt difference remains >= 0.25, so input grid '
                   'phase is not the dominant cause and the residual is consistent with integration/truncation.')
    else:
        reading = 'MIXED'
        meaning = ('The 0.125 ms input shift lands between the two pre-registered regimes, so input grid phase '
                   'explains part of the pointwise dt difference but the pre-registered reading does not attribute '
                   'it to phase alone.')
    agg_large = comparisons[str(SHIFTS[-1])]['aggregate_200ms']['window_mean_pool_rates']['normalized_RMS']
    sealed_agg = sealed_aggregate_anchors()['fine_0p03125_vs_0p015625']['window_mean']['normalized_RMS']
    return dict(reading=reading, meaning=meaning,
                rule=dict(phase_threshold=phase_thr, phase_rule='D_shift(0.125ms) >= 0.5*D_dt',
                          integrator_threshold=int_thr,
                          integrator_rule='D_shift(0.125ms) <= 0.1*D_dt AND sealed dt difference >= 0.25',
                          sealed_dt_difference=D_DT, sealed_dt_floor=SEALED_DT_FLOOR, sealed_dt_floor_satisfied=bool(dt_ok)),
                largest_shift=dict(substeps=int(SHIFTS[-1]), milliseconds=float(SHIFTS[-1] * DT_MS),
                                   pointwise_pool_nRMS_200ms=d_large),
                contrast=dict(pointwise_dt_nRMS_200ms=D_DT,
                              pointwise_shift_nRMS_200ms=d_large,
                              shift_over_dt=float(d_large / D_DT),
                              aggregate_dt_nRMS_200ms=sealed_agg,
                              aggregate_shift_nRMS_200ms=agg_large,
                              aggregate_shift_over_dt=float(agg_large / max(sealed_agg, 1e-300)),
                              note='pointwise and aggregate nRMS of the largest shift against the same sealed '
                                   'dt-pair values at the same 200 ms window'),
                engineering_threshold=ENGINEERING_THRESHOLD,
                crossing_measured=growth_rec['crossing_measured'],
                crossing_extrapolated=growth_rec['crossing_extrapolated'],
                implied_acceptance='if PHASE_EXPLAINS_DT, pointwise trajectory agreement is the wrong acceptance '
                                  'target for a coarser neural dt: the accept/reject metric must be applied to '
                                  'window-aggregated statistics (per-cell spike counts and window-mean pool rates), '
                                  'while a pointwise nRMS gate keeps rejecting configurations that differ only by '
                                  'sub-millisecond input grid phase',
                selected_timestep=None, model_selection=None, brain_model_invalid=False,
                biological_validity='NOT_ESTABLISHED')


# -------------------------------------------------------------------------------------- tests

def tests():
    rec = {}
    # T1 the sealed transferred initial state equals an independent rebuild (warm cache -> transfer -> RH .375).
    ini = load_sealed_initial()
    rebuilt, rebuilt_rates = rebuilt_initial()
    cmp_ini = net_exact(ini['net'], rebuilt)
    rates_eq = bool(np.array_equal(np.asarray(ini['rates'], float), rebuilt_rates))
    assert all(cmp_ini.values()) and rates_eq, (cmp_ini, rates_eq)
    rec['initial_state_equals_rebuild'] = dict(passed=True, fields=cmp_ini, rates_equal=rates_eq,
                                               source=str(_SEALED_INITIAL),
                                               note='the transferred initial state is taken from the sealed '
                                                    'numerics_followup pickle and independently reproduced from the '
                                                    'sealed 0.5 ms warm cache with RH .375 applied exactly once')

    # T2 the shift rule: N=0 is the sealed application; N>0 offsets the hold boundaries by exactly N sub-steps.
    b0 = block_index(0)
    assert np.array_equal(b0, (np.arange(N_STEP) // SUBSTEPS).astype(np.int16))
    rec['shift_zero_equals_sealed_application'] = dict(
        passed=True, blocks_per_ms=SUBSTEPS, first_three=list(b0[:3]), block_at_step_64=int(b0[64]),
        note='for N=0 applied_current(s) = sensory_current[s//64], bitwise the sealed application')
    rows = {}
    for n in SHIFTS:
        b = block_index(n)
        changes = np.flatnonzero(np.diff(b.astype(int)) != 0) + 1     # first sub-step of each new source block
        expect = np.arange(n + SUBSTEPS, N_STEP, SUBSTEPS)            # boundaries delayed by exactly n sub-steps
        rows[str(n)] = dict(changes_match_offset=bool(np.array_equal(changes, expect)),
                            first_change_substep=int(changes[0]),
                            first_change_ms=float(changes[0] * DT_MS),
                            n_boundaries=int(changes.size),
                            sample=list(zip(changes[:3].tolist(), [int(b[c]) for c in changes[:3]])))
        assert rows[str(n)]['changes_match_offset'], (n, changes[:5], expect[:5])
    rec['shift_rule_offsets_boundaries'] = dict(passed=True, per_shift=rows,
                                                note='every 1 ms hold boundary of the injected stream is delayed by '
                                                     'exactly N sub-steps: the boundaries land at N+64, N+128, ... '
                                                     'instead of 64, 128, ... (the first source block covers '
                                                     's = 0..N+63), and the first sample is held for s < N')
    t = block_index(8)
    assert np.array_equal(t, (np.maximum(np.arange(N_STEP) - 8, 0) // SUBSTEPS).astype(np.int16))
    assert int(t[64]) == 0 and int(block_index(0)[64]) == 1
    rec['shift_8_is_0125ms'] = dict(passed=True, shift_substeps=8, shift_ms=float(8 * DT_MS),
                                    first_differing_source_substep=64,
                                    control_block_at_64=int(block_index(0)[64]),
                                    shifted_block_at_64=int(t[64]))

    # T3 block indices fully determine the applied stream: reconstructed stream matches a delayed copy.
    tiny = np.arange(N_MS * 5, dtype=float).reshape(N_MS, 5) + 1.0        # 200 ms x 5 distinct samples
    held = np.repeat(tiny, SUBSTEPS, axis=0)
    for n in SHIFTS:
        b = block_index(n)
        want = held if n == 0 else np.concatenate([np.repeat(tiny[0:1], n, axis=0), held])[:N_STEP]
        got = tiny[b]
        assert got.shape == (N_STEP, 5) and np.array_equal(got, want), n
        assert np.array_equal(b, (np.maximum(np.arange(N_STEP) - n, 0) // SUBSTEPS).astype(np.int16)), n
    rec['applied_stream_reconstruction'] = dict(
        passed=True, shifts=SHIFTS, probe_shape=[N_STEP, 5],
        note='tiny[block_index(N)] reproduces the delayed stream built independently by np.repeat (1 ms hold) with '
             'the first sample repeated for the leading N sub-steps, for every declared shift; block_index also '
             'equals max(s-N,0)//64 exactly')

    # T4 metrics normalization and window slicing.
    x = np.arange(24, dtype=float).reshape(6, 4)
    y = x + 0.5
    m1 = metrics(x, y)
    assert abs(m1['absolute_RMS'] - 0.5) < 1e-12
    assert abs(m1['normalized_RMS'] - 0.5 / np.sqrt(np.mean(y * y))) < 1e-12
    assert metrics(x, x)['normalized_RMS'] == 0.0
    assert metrics(x, np.zeros_like(x))['normalized_RMS'] == float(np.sqrt(np.mean(x * x))) / 1e-9
    assert [w * SUBSTEPS for w in WINDOWS_MS] == [64, 320, 640, 1280, 3200, 6400, 12800]
    pa = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    wm_same = window_mean_aggregate(pa, pa)
    assert wm_same['normalized_RMS'] == 0.0 and wm_same['pools_within_10pct'] == 2
    assert wm_same['pearson_across_pools'] > 0.999
    wm_five = window_mean_aggregate(pa * 1.05, pa)
    assert wm_five['pools_within_10pct'] == 2 and abs(wm_five['normalized_RMS'] - 0.05) < 1e-12
    wm_bad = window_mean_aggregate(pa * 1.5, pa)
    assert wm_bad['pools_within_10pct'] == 0 and abs(wm_bad['normalized_RMS'] - 0.5) < 1e-12
    rec['metrics_and_windows'] = dict(passed=True, first_window=m1, window_steps=[w * SUBSTEPS for w in WINDOWS_MS],
                                      window_mean_convention=wm_same['definition'], denominator_floor_applied=True)

    # T5 sealed baselines reproduce the values frozen in the decision rule.
    base = sealed_dt_baseline()
    agg = sealed_aggregate_anchors()
    assert abs(base['fine_pair_200ms']['pool_rates'] - D_DT) < 5e-7
    assert abs(base['coarse_pair_200ms']['pool_rates'] - D_DT_COARSE) < 5e-7
    assert D_DT >= SEALED_DT_FLOOR
    fine_agg = agg['fine_0p03125_vs_0p015625']['window_mean']
    coarse_agg = agg['coarse_0p0625_vs_0p03125']['window_mean']
    assert 0.0 < fine_agg['normalized_RMS'] < D_DT
    rec['sealed_baselines'] = dict(passed=True, fine_pair_200ms=base['fine_pair_200ms'],
                                   coarse_pair_200ms=base['coarse_pair_200ms'],
                                   same_dt_method_pair_200ms=base['same_dt_method_pair_200ms'],
                                   sealed_aggregate_window_mean={'fine': fine_agg, 'coarse': coarse_agg},
                                   note='the sealed aggregate window-mean nRMS is far below the sealed pointwise '
                                        'nRMS at the same window, which is the contrast this stage tests')

    # T6 packbits cell mapping used by every spike difference statistic.
    pat = (np.arange(12000) % 5 == 0)
    full = np.unpackbits(np.packbits(pat), count=12000)
    assert np.array_equal(full, pat.astype(np.uint8))
    rec['packbits_cell_mapping'] = dict(passed=True)

    # T7 decision rule branches are reachable and mutually exclusive as written.
    def branch(d, dt=D_DT):
        if d >= PHASE_FRACTION * dt:
            return 'PHASE_EXPLAINS_DT'
        if d <= INTEGRATOR_FRACTION * dt and dt >= SEALED_DT_FLOOR:
            return 'INTEGRATOR_EXPLAINS_DT'
        return 'MIXED'
    probes = {str(v): branch(v) for v in [0.0, 0.01, 0.0287, 0.0288, 0.1, 0.1438, 0.1439, 0.2877, 1.0]}
    assert probes['0.1439'] == 'PHASE_EXPLAINS_DT' and probes['0.0287'] == 'INTEGRATOR_EXPLAINS_DT'
    assert probes['0.1'] == 'MIXED' and probes['0.0'] == 'INTEGRATOR_EXPLAINS_DT'
    assert branch(0.0, 0.2) == 'MIXED'
    rec['decision_rule_branches'] = dict(passed=True, probes=probes,
                                         thresholds=dict(phase=PHASE_FRACTION * D_DT,
                                                         integrator=INTEGRATOR_FRACTION * D_DT,
                                                         sealed_floor=SEALED_DT_FLOOR),
                                         note='with the sealed dt difference below 0.25 the integrator branch is '
                                              'unreachable, so MIXED is returned instead; recorded as designed')

    # T8 no physics body and no stochastic source is imported by this stage.
    import ast
    tree = ast.parse(Path(__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split('.')[0])
    forbidden = {'mujoco', 'mujoco_viewer', 'torch', 'random'}
    assert not (mods & forbidden), sorted(mods & forbidden)
    rec['no_physics_no_stochastic_imports'] = dict(passed=True, imports=sorted(mods),
                                                   checked_forbidden=sorted(forbidden))

    # T9 the transfer helper still matches the sealed semantics (guards accidental edits).
    arrival = []
    old = ConductanceNetwork(2, dt_ms=.5)
    old.steps = 20
    old.t_ms = 10.
    old._ring_i = 3
    old._ring[:] = False
    old._ring[(old._ring_i + 1) % 5, 0] = True
    old._ring[(old._ring_i + 4) % 5, 1] = True
    old.We = sparse.csr_matrix(np.diag([.001, .002]))
    old.Wi = sparse.csr_matrix(np.diag([.003, .004]))
    new = transfer(old, DT_MS, 'backward_euler')
    got = []
    for _ in range(round(3 / DT_MS)):
        start = new.t_ms
        ge = new.ge.copy()
        gi = new.gi.copy()
        new.step(-100.)
        e = new.ge / np.exp(-DT_MS / 3) - ge
        i = new.gi / np.exp(-DT_MS / 8) - gi
        if np.max(abs(e)) + np.max(abs(i)) > 1e-12:
            got.append([start, e.tolist(), i.tolist()])
    assert np.allclose([a[0] for a in got], [10.5, 12.])
    arrival.append(got)
    rec['transfer_absolute_event_times'] = dict(passed=True, cases=arrival)

    rec['passed'] = all(v.get('passed', True) for v in rec.values() if isinstance(v, dict))
    dump(OUT / 'MECHANISM_TESTS.json', rec)
    print('TESTS', rec['passed'], flush=True)
    return rec


# --------------------------------------------------------------------------------------- main

def prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'PREREGISTRATION.json').exists()
    base = sealed_dt_baseline()
    agg = sealed_aggregate_anchors()
    cfg = dict(
        scope='S5/P2 neural-only grid-phase experiment (s5-model-correction 20261106T063313Z); manual research '
              'diagnostic, not a product',
        question='Is the non-converging pointwise dt difference dominated by which 1 ms bin / sub-millisecond grid '
                 'phase each discrete spike and each held input block lands in, rather than by accumulated '
                 'integration error?',
        fixed_dt_ms=DT_MS, method=METHOD, substeps_per_command=SUBSTEPS, command_dt_ms=0.001,
        neural_steps=N_STEP, neural_elapsed_ms=N_MS, cells=12000, windows_ms=WINDOWS_MS,
        fixed_everything_else='same transferred initial state from the sealed numerics_followup '
                              'backward_euler_0p015625ms INITIAL_STATE.pkl, same RH .375 applied exactly once, same '
                              'causal 1 ms commands, same dt-aware 5 ms pool filter, unchanged readout '
                              'matrix/limits, one single dt and one single integrator for every run',
        the_only_change='the whole 200 x 12000 injected-current stream is shifted in time by N sub-steps',
        shift_implementation=('the stream is defined on the 1 ms command grid and held inside each 1 ms block, so '
                              'the sealed stage applies sensory_current[s//64] at neural sub-step s. The shift delays '
                              'the stream by N sub-step indices and holds the first sample for the leading N '
                              'sub-steps: k = max(s-N,0); block(s) = k//64; applied_current(s) = '
                              'sensory_current[block(s)]. N=0 is bitwise the sealed application. For 0<N<64 the first '
                              'source block covers s = 0..N+63 and every later 1 ms hold boundary is delayed by '
                              'exactly N sub-steps (boundaries at N+64, N+128, ... instead of 64, 128, ...), i.e. a '
                              'grid-phase offset of the injected stream; the applied current stays constant inside '
                              'each delayed 1 ms block and the command grid is unchanged. Identical rule in every run; '
                              'the exact per-sub-step applied block index is stored with every run as '
                              'applied_block_index.'),
        runs=[dict(label=label(s), shift_substeps=s, shift_ms=s * DT_MS) for s in SHIFTS],
        control_requirement='shift 0 must reproduce the sealed numerics_followup backward_euler_0p015625ms 200 ms '
                            'result bitwise (arrays, initial state and final state); otherwise stop and report',
        physics_runs=0, physical_steps=0, new_warm_runs=0,
        measurements=dict(
            windows='pool rates, raw command and clipped command nRMS at 1/5/10/20/50/100/200 ms versus the '
                    'control, same windows and the same normalized-RMS formula as the sealed stages',
            spikes='number of differing spikes (cumulative), final total spike-count difference, per-cell '
                   'spike-count correlation (Pearson and Spearman over 12000 cells)',
            aggregate='window-mean pool rate nRMS at 200 ms (per-pool time mean over the window, then nRMS across '
                      'the 66 pools); the same metric is computed read-only on the sealed dt pairs as the contrast',
            threshold='whether any threshold flip occurs and the first sub-step at which the continuous state '
                      'differs (exact full-state prefix probe, 96 sub-steps, both streams side by side)'),
        normalization='RMS(shifted-control)/max(RMS(control),1e-9); absolute and reference RMS retained',
        decision_rule=dict(
            D_dt=D_DT,
            D_dt_source='sealed numerics_followup supplement backward_euler 200 ms pool_rates nRMS, fine pair '
                        '0.03125 vs 0.015625 ms (asserted against ' + str(FOL / 'SUMMARY.json') + ')',
            phase='PHASE_EXPLAINS_DT if D_shift(0.125 ms) >= 0.5*D_dt',
            integrator='INTEGRATOR_EXPLAINS_DT if D_shift(0.125 ms) <= 0.1*D_dt while the sealed dt difference '
                       'stays >= 0.25',
            otherwise='MIXED',
            thresholds=dict(phase=PHASE_FRACTION * D_DT, integrator=INTEGRATOR_FRACTION * D_DT,
                            sealed_floor=SEALED_DT_FLOOR),
            D_shift_definition='200 ms pointwise pool_rates nRMS of the largest shift (8 sub-steps = 0.125 ms) '
                               'versus the control'),
        growth_report='whether D_shift is roughly proportional to the shift (D/N within a factor 3 over the 8x '
                      'span), the log-log slope, and the shift at which D_shift crosses 0.15 (the engineering '
                      'threshold), marked as interpolated if inside the measured range and as extrapolated if not',
        sealed_reference_anchors=dict(
            pointwise_200ms=base['fine_pair_200ms'], coarse_pointwise_200ms=base['coarse_pair_200ms'],
            same_dt_method_pair_200ms=base['same_dt_method_pair_200ms'],
            aggregate_window_mean=dict(fine=agg['fine_0p03125_vs_0p015625']['window_mean'],
                                      coarse=agg['coarse_0p0625_vs_0p03125']['window_mean'])),
        established_facts_respected=[
            'adjacent-dt pointwise pool-rate nRMS over 200 ms stays ~0.288-0.303 regardless of refinement',
            'aggregate statistics converge: totals 159756/161424/162288/162527, per-cell spike-count correlation '
            '0.999, window-mean pool-rate correlation 0.997',
            '1e-9 nA, 1e-6 mV and a 1e-6 ms event shift produced zero threshold flips in 12800 steps '
            '(numerics_chaos, BIAS_DOMINANT)'],
        forbidden_conclusions='no claim that the brain model is invalid; no claim of biological validity; no '
                              'timestep selection; no model selection; the conclusion is limited to whether '
                              'sub-millisecond input timing or the integrator explains the dt difference and what '
                              'acceptance metric follows',
        resources='5 neural runs of 12800 steps plus 8 prefix-probe runs of 96 steps; 0 physics, 0 warmups',
        units='C nF / g uS = ms; voltage mV, current nA; illustrative electrical parameters, not physiological '
              'calibration',
        importance='核心必需')
    dump(OUT / 'PREREGISTRATION.json', cfg)
    paths = [NUM / 'TRANSLATOR_IDENTITIES.json', NUM / 'SUMMARY.json',
             NUM / 'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz', NUM / 'CONFIGS/dt_0p5ms/WARM_CACHE.pkl',
             FOL / 'SUMMARY.json', _SEALED_INITIAL,
             FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz',
             FOL / 'RUNS/backward_euler_0p015625ms/FINAL_STATE.pkl',
             FOL / 'RUNS/backward_euler_0p015625ms/RESULT.json']
    paths += [FOL / 'RUNS' / n / 'NEURAL.npz' for n in SEALED_FOL_RUNS]
    paths += [SRC.parent / 'numerics/run.py', SRC.parent / 'numerics_followup/run.py',
              SRC.parent / 'numerics_chaos/run.py', R / 'engine/neural_cond.py']
    paths += sorted(SRC.glob('*.py')) + [OUT / 'PREREGISTRATION.json']
    dump(OUT / 'INPUT_HASHES_BEFORE.json', {str(p): sha(p) for p in paths})
    tests()
    dump(OUT / 'SOURCE_FREEZE.json', {str(p): sha(p) for p in sorted(SRC.glob('*.py'))})
    print('PREREGISTERED TESTED FROZEN', flush=True)


def checkhash():
    before = js(OUT / 'INPUT_HASHES_BEFORE.json')
    after = {p: sha(p) for p in before}
    assert after == before, [k for k in before if before[k] != after.get(k)]
    return after


def run():
    checkhash()
    assert js(OUT / 'MECHANISM_TESTS.json')['passed']
    results = []
    comparisons = {}
    res, control = run_case(0)
    results.append(res)
    control_check(control)
    dump(OUT / 'PROGRESS.json', dict(results=results, control_bitwise=True))
    probe = prefix_probe(SHIFTS)
    for shift in SHIFTS:
        if shift == 0:
            continue
        res, z = run_case(shift)
        results.append(res)
        comparisons[str(shift)] = compare_to_control(shift, z, control)
        del z
        dump(OUT / 'PROGRESS.json', dict(results=results, comparisons=comparisons, prefix_probe=probe))
    g = growth(comparisons)
    v = verdict(comparisons, g)
    summary = dict(
        stage='S5/P2 numerics_shift', out=str(OUT), dt_ms=DT_MS, method=METHOD,
        shifts_ms=[s * DT_MS for s in SHIFTS], cells=12000, neural_steps_per_run=N_STEP,
        results=results, comparisons=comparisons, prefix_probe=probe, growth=g, verdict=v,
        sealed_dt_baseline=sealed_dt_baseline(), sealed_aggregate_anchors=sealed_aggregate_anchors(),
        control_bitwise=js(OUT / 'CONTROL_BITWISE_CHECK.json'),
        neural_steps_total=sum(r['neural_steps'] for r in results)
        + sum(2 * v2['executed_steps_per_stream'] for v2 in probe.values()),
        full_neural_runs=len(results), prefix_probe_streams=2 * len(probe),
        prefix_probe_horizon_steps_per_stream=PREFIX_STEPS,
        physical_runs=0, physical_steps=0, new_warm_runs=0,
        selected_timestep=None, model_selection=None, brain_model_invalid=False,
        biological_validity='NOT_ESTABLISHED',
        stage_status='grid_phase_experiment_complete')
    dump(OUT / 'SUMMARY.json', summary)
    dump(OUT / 'INPUT_HASHES_AFTER.json', checkhash())
    print('DONE', v['reading'],
          'D_shift(0.125ms)=', v['largest_shift']['pointwise_pool_nRMS_200ms'],
          'agg=', v['contrast']['aggregate_shift_nRMS_200ms'],
          round(sum(r['wall_s'] for r in results), 2), flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    try:
        {'prereg': prereg, 'tests': tests, 'run': run}[action]()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / ('FAILURE_' + action + '_' + str(time.time_ns()) + '.json'), dict(traceback=traceback.format_exc()))
        raise
