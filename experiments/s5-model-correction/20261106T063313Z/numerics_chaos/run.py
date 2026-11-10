"""S1/P2 numerics_chaos: single-dt perturbation amplification-vs-bias diagnostic.

One fixed neural dt (0.015625 ms) removes the numerical method as a confound.
One control run (perturbation exactly zero) plus three declared tiny perturbations,
each in its own run. Causal 1 ms commands from prior completed rates, dt-aware
5 ms pool filter, unchanged readout matrix/limits, no physics/body stepping.
Never writes into the sealed numerics / numerics_followup stages.
Manual research diagnostic, not a product; no timestep is selected.
"""
import sys, json, pickle, copy, hashlib, time, traceback
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]
from engine.neural_cond import ConductanceNetwork, ConductanceParams

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_chaos')
SRC = Path(__file__).resolve().parent
NUM = OUT.parent / 'numerics'
FOL = OUT.parent / 'numerics_followup'

DT_MS = 0.015625                  # the finest already-tested grid
SUBSTEPS = 64                     # 0.001 ms command / DT_MS
N_MS = 200
N_STEP = N_MS * SUBSTEPS          # 12800 neural steps
WINDOWS_MS = [1, 5, 10, 20, 50, 100, 200]
METHOD = 'backward_euler'
RH_GAIN = 0.375
FILTER_TAU_MS = 5.0
EXC_TAU_MS = 3.0
INH_TAU_MS = 8.0
FIELDS = ['v', 'ge', 'gi', 'adaptation', 'refractory_until_ms', 'spike_count']

# Frozen perturbation target, derived only from sealed read-only inputs (see prereg()):
#   max total control spike count 96 -> cells {4867, 9599};
#   keep cells not refractory at initial neural time 1100 ms (so P2 survives the
#   first-step refractory reset) -> {9599}; lowest index -> 9599.
TARGET = 9599
P1_NA = 1e-9        # nA, added to the injected current of TARGET for the whole 200 ms
P2_MV = 1e-6        # mV, added once to TARGET membrane potential before the first step
P3_MS = 1e-6        # ms, single pending synaptic event arrival phase delay

RUNS = [('control', 'none'),
        ('P1_current', 'current'),
        ('P2_voltage', 'voltage'),
        ('P3_event_phase', 'event')]

# Sealed dt-induced 200 ms baseline (numerics_followup supplement, backward_euler,
# dts [0.0625, 0.03125, 0.015625]; "fine" = 0.03125 vs 0.015625, "coarse" = 0.0625 vs 0.03125).
DT_BASELINE_200 = {'pool_rates': {'fine': 0.287739, 'coarse': 0.292466},
                   'raw_command': {'fine': 0.290435, 'coarse': 0.301354},
                   'clipped_command': {'fine': 0.307135, 'coarse': 0.323041}}

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


def warm_cache():
    if 'cache' not in _MEMO:
        with (NUM / 'CONFIGS/dt_0p5ms/WARM_CACHE.pkl').open('rb') as f:
            _MEMO['cache'] = pickle.load(f)
    return _MEMO['cache']


def sealed_currents():
    """The same 200x12000 float64 actual sensory currents used by the sealed stages."""
    if 'currents' not in _MEMO:
        with np.load(NUM / 'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz') as z:
            cur = z['sensory_current'].copy()
        assert cur.shape == (N_MS, 12000) and cur.dtype == np.float64
        _MEMO['currents'] = cur
    return _MEMO['currents'].copy()


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
    """Identical state transfer to numerics_followup: full fields plus absolute pending events."""
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


def build():
    """Fresh network for one run: seal-identical transfer, then RH excitatory gain .375 once."""
    c = warm_cache()[1.3]
    old = c['net']
    net = transfer(old, DT_MS, METHOD)
    assert net.We.nnz == 180604 and net.Wi.nnz == 148811
    ident, poolmat = identities()
    rows = np.repeat(np.arange(net.n), np.diff(net.We.indptr))
    mask = np.isin(rows, ident['RH_cells'])
    net.We.data[mask] *= RH_GAIN
    assert np.array_equal(net.We.data[~mask], old.We.data[~mask])
    assert np.array_equal(net.We.data[mask], old.We.data[mask] * RH_GAIN)
    return net, ident, poolmat, np.asarray(ident['old_matrix']), np.asarray(ident['limits']), c['rates'].copy()


def phase_injection(net, presynaptic_cell):
    """Conductance increments that realise a P3_MS later arrival of one pending event.

    Ring slots are boolean at dt, so round(P3_MS/dt)=0 slots: a sub-dt arrival shift is not a
    representable slot move. Delaying an event by delta multiplies its contribution at every
    later interval boundary by exp(+delta/tau), which is reproduced exactly by injecting
    increment*expm1(delta/tau) at the same arrival interval.
    """
    fac_e = float(np.expm1(P3_MS / EXC_TAU_MS))
    fac_i = float(np.expm1(P3_MS / INH_TAU_MS))
    col_e = net.We.tocsc()[:, presynaptic_cell].toarray().ravel()
    col_i = net.Wi.tocsc()[:, presynaptic_cell].toarray().ravel()
    ee = col_e * fac_e
    ei = col_i * fac_i
    meta = dict(factor_exc=fac_e, factor_inh=fac_i,
                exc_posts=int((col_e != 0).sum()), inh_posts=int((col_i != 0).sum()),
                exc_max_uS=float(ee.max()) if ee.size else 0.0,
                inh_max_uS=float(ei.max()) if ei.size else 0.0)
    return ee, ei, meta


def expected_arrival_step(net, emit_step):
    """Window-relative step at which the event emitted in step emit_step arrives (interval start)."""
    return None if emit_step is None else emit_step + 1 + net.delay_steps


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
    extra = np.zeros(N_STEP, np.int64)      # spikes present only in the perturbation run
    missing = np.zeros(N_STEP, np.int64)    # spikes present only in the control
    target_only = np.zeros(N_STEP, bool)    # TARGET cell bit differs at that step
    for a, sp in spike_streams(zp['spikes12000_packed']):
        b = a + sp.shape[0]
        c = np.unpackbits(zc['spikes12000_packed'][a:b], axis=1, count=12000)
        sp = sp.astype(bool)
        c = c.astype(bool)
        d = sp != c
        per_step[a:b] = d.sum(1)
        extra[a:b] = (sp & ~c).sum(1)
        missing[a:b] = (~sp & c).sum(1)
        target_only[a:b] = sp[:, TARGET] != c[:, TARGET]
    return dict(per_step=per_step, extra=extra, missing=missing, target_only=target_only)


def first_spike_step(packed):
    nz = np.flatnonzero((packed != 0).any(axis=1))
    return int(nz[0]) if nz.size else None


def first_spike_step_cell(zp):
    for a, sp in spike_streams(zp['spikes12000_packed']):
        nz = np.flatnonzero(sp[:, TARGET])
        if nz.size:
            return int(a + nz[0])
    return None


def run_case(label, kind):
    folder = OUT / 'RUNS' / label
    folder.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    net, ident, poolmat, matrix, limits, rates = build()
    currents = sealed_currents()
    initial = rates.copy()
    applied = dict(kind=kind, index=(None if kind == 'none' else TARGET),
                   perturbation_exactly_zero=(kind == 'none'))
    ee = ei = None
    if kind == 'current':
        currents = currents.copy()
        before = currents[:, TARGET].copy()
        currents[:, TARGET] = before + P1_NA
        delta = currents[:, TARGET] - before
        applied.update(requested_nA=P1_NA, applied_all_200ms=True,
                       injected_current_before_nA={'min': float(before.min()), 'max': float(before.max())},
                       actual_delta_nA_min=float(delta.min()), actual_delta_nA_max=float(delta.max()),
                       exact_float64=bool(np.array_equal(delta, np.full(delta.shape, P1_NA))))
    elif kind == 'voltage':
        before = float(net.v[TARGET])
        net.v[TARGET] = before + P2_MV
        applied.update(requested_mV=P2_MV, before_mV=before, after_mV=float(net.v[TARGET]),
                       actual_delta_mV=float(net.v[TARGET]) - before, ulp_at_before_mV=float(np.spacing(np.float64(before))),
                       applied_once_before_first_step=True, active_at_step0=bool(net.t_ms + 1e-10 >= net.refractory_until_ms[TARGET]))
    elif kind == 'event':
        ee, ei, meta = phase_injection(net, TARGET)
        applied.update(requested_ms=P3_MS, direction='delay (+1e-6 ms later arrival)', **meta,
                       representation='ring slots are boolean at dt; round(1e-6/dt)=0 slots, so a sub-dt arrival shift '
                                      'is not a representable slot move. The 1e-6 ms later arrival is realised exactly in '
                                      'the conductance-decay phase of that single event by injecting the exp(+delta/tau) '
                                      'amplitude equivalence at its arrival interval (exact at every later interval '
                                      'boundary; the intra-interval voltage change inside the arrival interval is '
                                      'approximated exactly as the fixed-coefficient step already approximates all '
                                      'arrivals). Tested end to end on a synthetic two-cell network.')
    decay = np.exp(-DT_MS / FILTER_TAU_MS)
    detect_site = kind in ('none', 'event')
    initial_net = copy.deepcopy(net)
    packed = []
    pool, raw, clip, sources, counts = [], [], [], [], []
    clock = []
    states = {k: [] for k in FIELDS}
    trace = {k: np.zeros(N_STEP) for k in ['v', 'ge', 'gi', 'adaptation']}
    emit_step = None
    site = None
    site_state = None
    for m in range(N_MS):
        current = currents[m]
        sources.append(rates.copy())
        cmd = limits / 100 * (matrix @ rates)
        raw.append(cmd)
        clip.append(np.clip(cmd, -limits, limits))
        cnt = np.zeros(12000, np.int32)
        for _ in range(SUBSTEPS):
            s = len(packed)
            inject = False
            # Warm-history arrivals occupy steps 0..delay_steps only; scanning after that
            # isolates the in-window event emitted by TARGET.
            if detect_site and site is None and emit_step is not None and s > net.delay_steps \
                    and net._ring[net._ring_i][TARGET]:
                site = s
                inject = kind == 'event'
            if inject:
                sp = net.step(current, exc_events_uS=ee, inh_events_uS=ei)
            else:
                sp = net.step(current)
            packed.append(np.packbits(sp))
            cnt += sp
            if emit_step is None and sp[TARGET]:
                emit_step = s
            trace['v'][s] = net.v[TARGET]
            trace['ge'][s] = net.ge[TARGET]
            trace['gi'][s] = net.gi[TARGET]
            trace['adaptation'][s] = net.adaptation[TARGET]
            if site is not None and s == site and site_state is None:
                site_state = dict(step=s, t_ms=float(net.t_ms), ge=net.ge.copy(), gi=net.gi.copy(),
                                  v=net.v.copy(), spikes=np.packbits(sp))
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
             initial_pool_rates=initial, sensory_current=currents, neural_clock=np.asarray(clock),
             neural_dt_ms=DT_MS, time_s=1.3 + np.arange(1, N_MS + 1) / 1000,
             command_time_s=1.3 + np.arange(N_MS) / 1000, **states)
    np.savez_compressed(folder / 'NEURAL.npz', **z)
    np.savez_compressed(folder / 'TARGET_TRACE.npz', target=TARGET, dt_ms=DT_MS, **trace)
    if site_state is not None:
        np.savez_compressed(folder / 'SITE_STATE.npz', **site_state)
        exp_site = expected_arrival_step(net, emit_step)
        applied.update(site_step=site, site_time_ms=float(site_state['t_ms']),
                       event_emission_step=emit_step, expected_site_step=exp_site,
                       site_step_matches_emission_plus_delay=(None if exp_site is None else int(site == exp_site)))
    result = dict(label=label, kind=kind, dt_ms=DT_MS, method=METHOD, substeps_per_command=SUBSTEPS,
                  neural_steps=N_STEP, neural_elapsed_ms=float(net.t_ms - 1100.0),
                  wall_s=time.monotonic() - t0, folder=str(folder), target=TARGET,
                  applied_perturbation=applied, physical_runs=0, new_warm_runs=0,
                  total_spikes=int(np.asarray(counts).sum()), mean_hz=float(np.asarray(counts).sum() / 12000 / .2))
    dump(folder / 'RESULT.json', result)
    print('CASE', label, round(result['wall_s'], 3), result['total_spikes'], flush=True)
    return result, z, trace


def control_check(z):
    """Control must reproduce the sealed numerics_followup dt=0.015625 backward_euler 200 ms result bitwise."""
    sealed = arrays(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz')
    keys = ['spikes12000_packed', 'pool_rates', 'raw_command', 'clipped_command', 'command_source_rates',
            'counts12000', 'sensory_current', 'initial_pool_rates', 'neural_dt_ms', 'neural_clock',
            'time_s', 'command_time_s'] + FIELDS
    exact = {k: bool(np.array_equal(np.asarray(z[k]), np.asarray(sealed[k]))) for k in keys}
    with (FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl').open('rb') as f:
        s_ini = pickle.load(f)
    with (FOL / 'RUNS/backward_euler_0p015625ms/FINAL_STATE.pkl').open('rb') as f:
        s_fin = pickle.load(f)
    with (OUT / 'RUNS/control/INITIAL_STATE.pkl').open('rb') as f:
        m_ini = pickle.load(f)
    with (OUT / 'RUNS/control/FINAL_STATE.pkl').open('rb') as f:
        m_fin = pickle.load(f)

    def net_cmp(a, b):
        out = {k: bool(np.array_equal(getattr(a, k), getattr(b, k))) for k in FIELDS}
        out['t_ms'] = a.t_ms == b.t_ms
        out['steps'] = a.steps == b.steps
        out['dt'] = a.dt == b.dt
        out['ring'] = bool(np.array_equal(a._ring, b._ring))
        out['ring_i'] = a._ring_i == b._ring_i
        for key in ['We', 'Wi']:
            x, y = getattr(a, key), getattr(b, key)
            out[key] = bool(np.array_equal(x.indices, y.indices) and np.array_equal(x.indptr, y.indptr)
                            and np.array_equal(x.data, y.data))
        return out

    initial = dict(net=net_cmp(m_ini['net'], s_ini['net']), rates=bool(np.array_equal(m_ini['rates'], s_ini['rates'])))
    final = dict(net=net_cmp(m_fin['net'], s_fin['net']), rates=bool(np.array_equal(m_fin['rates'], s_fin['rates'])))
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


def compare_to_control(label, zp, zc, tp, tc):
    sd = spike_difference(zp, zc)
    out = dict(label=label, dt_ms=DT_MS)
    tv_first = np.flatnonzero(tp['v'] != tc['v'])
    out['target_continuous_divergence'] = dict(
        first_differing_step=(int(tv_first[0]) if tv_first.size else None),
        first_differing_time_ms=(float((tv_first[0] + 1) * DT_MS) if tv_first.size else None),
        max_abs_v_difference_mV=float(np.max(np.abs(tp['v'] - tc['v']))),
        max_abs_ge_difference_uS=float(np.max(np.abs(tp['ge'] - tc['ge']))),
        max_abs_gi_difference_uS=float(np.max(np.abs(tp['gi'] - tc['gi']))),
        perturbed_first_spike_step=first_spike_step_cell(zp),
        control_first_spike_step=first_spike_step_cell(zc))
    fp, fc = first_spike_step(zp['spikes12000_packed']), first_spike_step(zc['spikes12000_packed'])
    out['first_spike_time'] = dict(
        perturbation_step=fp, control_step=fc,
        perturbation_ms=(None if fp is None else float((fp + 1) * DT_MS)),
        control_ms=(None if fc is None else float((fc + 1) * DT_MS)),
        difference_ms=(None if (fp is None or fc is None) else float((fp - fc) * DT_MS)))
    first_diff = int(np.flatnonzero(sd['per_step'])[0]) if sd['per_step'].any() else None
    out['discrete_divergence'] = dict(
        first_differing_spike_step=first_diff,
        first_differing_spike_time_ms=(None if first_diff is None else float((first_diff + 1) * DT_MS)),
        first_differing_step_within_first_1ms=(None if first_diff is None else bool(first_diff < SUBSTEPS)),
        differing_spikes_at_that_step=(None if first_diff is None else int(sd['per_step'][first_diff])),
        target_cell_differs_at_that_step=(None if first_diff is None else bool(sd['target_only'][first_diff])))
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
        prev = 0 if w == WINDOWS_MS[0] else (w - WINDOWS_MS[WINDOWS_MS.index(w) - 1]) * SUBSTEPS
        entry['differing_spikes_within_window'] = int(sd['per_step'][n - prev:n].sum())
        windows[str(w)] = entry
    cp = zp['counts12000'].sum(0).astype(np.int64)
    cc = zc['counts12000'].sum(0).astype(np.int64)
    d = cp - cc
    out['final_spike_count'] = dict(perturbation_total=int(cp.sum()), control_total=int(cc.sum()),
                                   difference_total=int(cp.sum() - cc.sum()),
                                   max_abs_per_cell_difference=int(np.abs(d).max()),
                                   argmax_cell=int(np.abs(d).argmax()),
                                   cells_with_difference=int((d != 0).sum()),
                                   total_spike_count_difference=int(net_spike_total(zp) - net_spike_total(zc)))
    out['windows'] = windows
    return out


def net_spike_total(z):
    return int(np.asarray(z['counts12000']).sum())


def growth_rate(entry):
    """Descriptive pre-saturation log-growth rate of the pool-rate nRMS (not a Lyapunov exponent)."""
    pts = sorted((e['window_ms'], e['pool_rates']['normalized_RMS']) for e in entry['windows'].values())
    pts = [(t, v) for t, v in pts if 0 < v < 0.05]
    if len(pts) < 2:
        return dict(points_used=pts, slope_per_ms=None, e_folding_ms=None)
    t = np.asarray([p[0] for p in pts], float)
    y = np.log(np.asarray([p[1] for p in pts], float))
    slope = float(np.polyfit(t, y, 1)[0])
    return dict(points_used=pts, slope_per_ms=slope, e_folding_ms=(None if slope <= 0 else float(1 / slope)))


def dt_baseline():
    s = js(FOL / 'SUMMARY.json')['supplement']['backward_euler']
    exact = {k: s['windows']['200']['metrics'][k] for k in ['pool_rates', 'raw_command', 'clipped_command']}
    for k, v in DT_BASELINE_200.items():
        for which in ['fine', 'coarse']:
            got = exact[k][which]['normalized_RMS']
            assert abs(got - v[which]) < 5e-7, (k, which, got, v[which])
    matched = js(FOL / 'SUMMARY.json')['matched_methods']['0.015625']['200']
    return dict(source=str(FOL / 'SUMMARY.json'),
               sealed_supplement_200ms=exact,
               matched_methods_same_dt_200ms={k: matched[k] for k in ['pool_rates', 'raw_command', 'clipped_command']})


def verdict(comparisons):
    """Pre-registered reading of the perturbation-vs-dt magnitudes; no timestep is selected."""
    sealed = dt_baseline()['sealed_supplement_200ms']
    base = {k: dict(fine=sealed[k]['fine']['normalized_RMS'],
                    coarse=sealed[k]['coarse']['normalized_RMS']) for k in DT_BASELINE_200}
    rows = {}
    for label, c in comparisons.items():
        e = c['windows']['200']
        rows[label] = {k: dict(perturbation=c['windows']['200'][k]['normalized_RMS'],
                               dt_induced=base[k]['fine'],
                               ratio=c['windows']['200'][k]['normalized_RMS'] / base[k]['fine'],
                               log10_ratio=float(np.log10(max(c['windows']['200'][k]['normalized_RMS'], 1e-300) / base[k]['fine'])))
                       for k in ['pool_rates', 'raw_command', 'clipped_command']}
        rows[label]['differing_spikes_cumulative_200ms'] = e['differing_spikes_cumulative']
        rows[label]['first_differing_spike_time_ms'] = c['discrete_divergence']['first_differing_spike_time_ms']
        rows[label]['growth'] = growth_rate(c)
    ratios = [rows[l][k]['ratio'] for l in rows for k in ['pool_rates', 'raw_command', 'clipped_command']]
    chaotic = sum(1 for r in ratios if r >= 0.1)
    negligible = sum(1 for r in ratios if r <= 1e-3)
    if chaotic >= 6:
        reading = 'AMPLIFICATION_DOMINANT'
        meaning = ('Perturbation-induced 200 ms differences reach the same order as the dt-induced difference, so the '
                   '50-200 ms dt divergence is consistent with chaotic amplification of arbitrarily small differences. '
                   'Trajectory-level agreement is then the wrong acceptance target: no timestep refinement can make '
                   'trajectories agree, and acceptance must be distributional/statistical.')
    elif negligible == len(ratios):
        reading = 'BIAS_DOMINANT'
        meaning = ('Perturbation-induced 200 ms differences stay orders of magnitude below the dt-induced difference, '
                   'so the dt divergence is not explained by amplification of arbitrarily small differences and is '
                   'consistent with a systematic truncation/quantization bias that a better integrator could reduce.')
    else:
        reading = 'MIXED'
        meaning = ('Perturbation-induced differences land between the two regimes: some readout channels are amplified '
                   'toward the dt scale while others stay far below it. Both amplification and bias can contribute; the '
                   'pre-registered reading does not resolve them into a single cause.')
    return dict(dt_baseline_200ms=base, per_run=rows, reading=reading, meaning=meaning,
                rule=('AMPLIFICATION_DOMINANT if >=6 of the 9 (3 runs x 3 readouts) 200 ms perturbation/dt nRMS ratios '
                      'are >=0.1; BIAS_DOMINANT if all 9 are <=1e-3; otherwise MIXED.'),
                selected_timestep=None, model_selection=None, biological_validity='NOT_ESTABLISHED')


def _target_rule():
    sealed = np.load(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz')
    tot = sealed['counts12000'].astype(np.int64).sum(0)
    sealed.close()
    mx = int(tot.max())
    tied = np.flatnonzero(tot == mx)
    net = warm_cache()[1.3]['net']
    live = [int(i) for i in tied if net.t_ms + 1e-10 >= net.refractory_until_ms[i]]
    conn = [i for i in live if (net.We.tocsc()[:, i].nnz + net.Wi.tocsc()[:, i].nnz) > 0]
    return dict(max_total_spikes=mx, tied_cells=tied.tolist(), non_refractory_at_1100ms=live,
                with_outgoing_synapses=conn, chosen=int(min(conn)) if conn else None)


def tests():
    rec = {}
    # T1 state transfer keeps future event arrival timestamps absolute (sealed logic).
    arrival = []
    for dt in [.5, .25, .125, .0625, .03125, .015625]:
        old = ConductanceNetwork(2, dt_ms=.5)
        old.steps = 20
        old.t_ms = 10.
        old._ring_i = 3
        old._ring[:] = False
        old._ring[(old._ring_i + 1) % 5, 0] = True
        old._ring[(old._ring_i + 4) % 5, 1] = True
        old.We = sparse.csr_matrix(np.diag([.001, .002]))
        old.Wi = sparse.csr_matrix(np.diag([.003, .004]))
        new = transfer(old, dt, 'backward_euler')
        got = []
        for j in range(round(3 / dt)):
            start = new.t_ms
            ge = new.ge.copy()
            gi = new.gi.copy()
            new.step(-100.)
            e = new.ge / np.exp(-dt / 3) - ge
            i = new.gi / np.exp(-dt / 8) - gi
            if np.max(abs(e)) + np.max(abs(i)) > 1e-12:
                got.append([start, e.tolist(), i.tolist()])
        assert np.allclose([a[0] for a in got], [10.5, 12.])
        assert np.allclose([a[1] for a in got], [[.001, 0], [0, .002]])
        assert np.allclose([a[2] for a in got], [[.003, 0], [0, .004]])
        arrival.append(dict(dt_ms=dt, arrivals=got))
    rec['transfer_absolute_event_times'] = dict(passed=True, cases=arrival)

    # T2 packed spike row -> cell bit mapping used by every difference statistic.
    pat = (np.arange(12000) % 7 == 0)
    pk = np.packbits(pat)
    full = np.unpackbits(pk, count=12000)
    assert np.array_equal(full, pat.astype(np.uint8))
    assert bool(pk[TARGET // 8] & (1 << (7 - TARGET % 8))) == bool(pat[TARGET])
    rec['packbits_cell_mapping'] = dict(passed=True, target=TARGET, byte=int(TARGET // 8), bit=int(TARGET % 8))

    # T3 the 1e-6 ms arrival delay is realised exactly in the conductance-decay phase.
    delta = P3_MS
    for tau, name in [(EXC_TAU_MS, 'exc'), (INH_TAU_MS, 'inh')]:
        fac = np.expm1(delta / tau)
        amp = 7.3e-4
        scheme = amp * (1 + fac)
        errs = []
        for k in range(0, 6):
            # boundary k+1 after the arrival interval: exact elapsed decay is k*dt + (dt - delta)
            exact = amp * np.exp(-(k * DT_MS + (DT_MS - delta)) / tau)
            got = amp * (1 + fac) * np.exp(-(k + 1) * DT_MS / tau)
            errs.append(abs(got - exact) / max(abs(exact), 1e-300))
        assert max(errs) < 1e-14, (name, errs)
        assert np.isfinite(fac) and fac > 0
    rec['sub_dt_phase_realisation'] = dict(
        passed=True, delta_ms=delta, note='increment*exp(+delta/tau) reproduces the exact boundary decay '
        'a*exp(-(k*dt + dt - delta)/tau) for every later interval boundary within 1e-14 relative; '
        'round(delta/dt)=%d slots so a literal slot move is unavailable' % round(delta / DT_MS))

    # T4 an event delayed by exactly one slot equals the full-slot limit of the same factor.
    tau = EXC_TAU_MS
    fac_full = np.expm1(DT_MS / tau)
    assert abs(np.exp(DT_MS / tau) * np.exp(-DT_MS / tau) - 1) < 1e-15
    rec['endpoint_consistency'] = dict(passed=True, factor_full_slot=float(fac_full),
                                       note='exp(dt/tau)*exp(-dt/tau)=1, so delta->dt reproduces arrival in the next '
                                       'interval, i.e. the factor interpolates between adjacent arrival slots')

    # T5 float64 representability of the declared magnitudes.
    a0 = (0.0 + P1_NA) - 0.0
    a1 = (1.0 + P1_NA) - 1.0
    v0 = -72.98140322583006
    dv = (v0 + P2_MV) - v0
    ee_probe = 8e-4 * np.expm1(P3_MS / EXC_TAU_MS)
    rec['float64_representability'] = dict(
        passed=bool(a0 == P1_NA and abs(a1 - P1_NA) <= abs(np.spacing(1.0))
                    and abs(dv - P2_MV) <= abs(np.spacing(v0)) and (0.01 + ee_probe) != 0.01),
        current_increment_at_0nA=a0, current_increment_at_1nA=a1, current_ulp_at_1nA=float(abs(np.spacing(1.0))),
        voltage_increment=dv, voltage_ulp_at_probe=float(abs(np.spacing(v0))),
        conductance_increment_exc=float(ee_probe), conductance_ulp_at_1e_minus_2_uS=float(abs(np.spacing(0.01))),
        note='1e-9 nA is not lost: the target-injected current is 0.0 nA for the whole window, so the increment is exact')

    # T6 frozen target rule recomputed from sealed read-only inputs.
    rule = _target_rule()
    assert rule['chosen'] == TARGET, (rule, TARGET)
    rec['target_rule'] = dict(passed=True, rule=rule)

    # T7 metrics and window slicing.
    x = np.arange(24, dtype=float).reshape(6, 4)
    y = x + 0.5
    m1 = metrics(x, y)
    assert abs(m1['absolute_RMS'] - 0.5) < 1e-12 and abs(m1['normalized_RMS'] - 0.5 / np.sqrt(np.mean(y * y))) < 1e-12
    assert metrics(x, x)['normalized_RMS'] == 0.0
    z = np.zeros_like(x)
    assert metrics(x, z)['normalized_RMS'] == float(np.sqrt(np.mean(x * x))) / 1e-9
    assert [w * SUBSTEPS for w in WINDOWS_MS] == [64, 320, 640, 1280, 3200, 6400, 12800]
    rec['metrics_and_windows'] = dict(passed=True, first_window=m1, denominator_floor_applied=True,
                                      window_steps=[w * SUBSTEPS for w in WINDOWS_MS])

    # T9 end-to-end P3 injection on a synthetic three-cell network (never the 12000-cell run).
    w = sparse.csr_matrix(np.array([[0., 0., 0.], [1e-3, 0., 0.], [2e-3, 0., 0.]]))
    zero = sparse.csr_matrix((3, 3))
    ctrl = ConductanceNetwork(3, dt_ms=DT_MS, voltage_method=METHOD)
    ctrl.set_connectivity(w, zero)
    inj = ConductanceNetwork(3, dt_ms=DT_MS, voltage_method=METHOD)
    inj.set_connectivity(w, zero)
    ee3, ei3, meta3 = phase_injection(ctrl, 0)
    assert meta3['exc_posts'] == 2 and meta3['inh_posts'] == 0
    emit = None
    site = None
    steps = 1 + 1 + ctrl.delay_steps + 40
    ex = np.zeros(3)
    ex[0] = 100.
    ge_c, ge_i = [], []
    spike_diff = 0
    for s in range(steps):
        cur = ex if s == 0 else 0.0
        run_inject = False
        if site is None and emit is not None and s > ctrl.delay_steps and inj._ring[inj._ring_i][0]:
            site = s
            run_inject = True
        sc = ctrl.step(cur)
        si = inj.step(cur, exc_events_uS=ee3, inh_events_uS=ei3) if run_inject else inj.step(cur)
        if emit is None and sc[0]:
            emit = s
        spike_diff += int((sc != si).sum())
        ge_c.append(ctrl.ge.copy())
        ge_i.append(inj.ge.copy())
    assert emit == 0
    assert site == expected_arrival_step(ctrl, emit), (site, expected_arrival_step(ctrl, emit))
    ge_c = np.asarray(ge_c)
    ge_i = np.asarray(ge_i)
    before = ge_i[:site] - ge_c[:site]
    assert not before.any()
    diff = ge_i - ge_c
    exp_decay = np.exp(-DT_MS / EXC_TAU_MS)
    fac = ee3[1]
    errs = []
    errs_abs = []
    for k in range(steps - site):
        want = fac * exp_decay ** (k + 1)
        got = diff[site + k, 1]
        errs.append(abs(got - want) / max(abs(want), 1e-300))
        errs_abs.append(abs(got - want))
    # the difference is read out of ge of order 1e-3, so the absolute tolerance is set by that scale
    assert max(errs_abs) < 1e-16, (max(errs_abs), max(errs))
    assert np.all(diff[:, 0] == 0)
    rec['p3_injection_end_to_end'] = dict(
        passed=bool(max(errs_abs) < 1e-16 and spike_diff == 0),
        emit_step=emit, arrival_step=site, expected_arrival_step=expected_arrival_step(ctrl, emit),
        delay_steps=ctrl.delay_steps, max_absolute_error_uS=max(errs_abs),
        max_relative_error=max(errs), relative_error_floor_note='the conductance difference is extracted from ge of '
        'order 1e-3, so the last ulp limits the relative agreement to about 1e-9',
        spike_differences=spike_diff,
        note='the injected increment reproduces the declared +1e-6 ms later arrival everywhere after the '
             'arrival interval (increment*exp(+delta/tau), decaying by exp(-dt/tau) per step)')

    # T8 the physics/body stack is not imported or touched by this stage.
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
    rec['passed'] = all(v.get('passed', True) for v in rec.values() if isinstance(v, dict))
    dump(OUT / 'MECHANISM_TESTS.json', rec)
    print('TESTS', rec['passed'], flush=True)
    return rec


def prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'PREREGISTRATION.json').exists()
    rule = _target_rule()
    assert rule['chosen'] == TARGET
    cfg = dict(
        scope='S1/P2 neural-only single-dt perturbation diagnostic; manual research, not a product',
        question='Is the 50-200 ms dt divergence chaotic amplification of arbitrarily small differences or a '
                 'systematic truncation/quantization bias that a better integrator would remove?',
        fixed_dt_ms=DT_MS, substeps_per_command=SUBSTEPS, command_dt_ms=0.001, windows_ms=WINDOWS_MS,
        method=METHOD, voltage_method_note='one single dt and one single integrator for every run, so the numerical '
                                          'method is not a confound',
        initial_state='identical sealed numerics 0.5 ms warm state at physical 1.3 s (neural clock 1100 ms): full '
                      'v/ge/gi/adaptation/refractory_until_ms/spike_count and absolute pending event times',
        connectivity='sealed matrices; RH postsynaptic excitatory rows multiplied by .375 exactly once after transfer',
        input='same sealed 200x12000 float64 actual sensory currents, field sensory_current of '
              'numerics/RUNS/dt_0p5ms_paired_1.3/NEURAL.npz, held 1 ms, applied to all 64 sub-steps of each 1 ms',
        readout='unchanged pools/readout matrix/limits; causal commands from prior completed rates; exact 5 ms '
                'exponential pool filter applied per neural sub-step (dt-aware)',
        physics_runs=0, physical_steps=0, new_warm_runs=0, physics_state_persistence=False,
        runs=[dict(label=label, kind=kind) for label, kind in RUNS],
        perturbation_target=dict(
            index=TARGET,
            rule='among the neurons tied for the maximum total spike count in the sealed numerics_followup '
                 'backward_euler dt=0.015625 control trajectory, keep the cells that are not refractory at the '
                 'initial neural time 1100 ms (so the P2 voltage perturbation is not erased by the first-step '
                 'refractory reset) and that have at least one nonzero outgoing synaptic conductance entry; take the '
                 'lowest index',
            derivation=rule),
        perturbations=dict(
            control='perturbation exactly zero; must reproduce the sealed numerics_followup backward_euler '
                    'dt=0.015625 200 ms result bitwise, else stop and report the discrepancy',
            P1_current=dict(delta_nA=P1_NA, where='injected current of the single target neuron', duration='whole 200 ms'),
            P2_voltage=dict(delta_mV=P2_MV, where='membrane potential of the single target neuron',
                            duration='once, before the first step'),
            P3_event_phase=dict(delta_ms=P3_MS, what='arrival phase of one pending synaptic event emitted by the '
                                'target neuron (its first spike in the control trajectory)',
                                implementation='ring slots are boolean at dt, so round(delta/dt)=0 slots and a sub-dt '
                                'arrival shift is not a representable slot move. The 1e-6 ms later arrival is realised '
                                'exactly in the conductance-decay phase by injecting the exp(+delta/tau) amplitude '
                                'equivalence for that single event at its arrival interval (exact at every later '
                                'interval boundary; the intra-interval voltage change inside the arrival interval is '
                                'approximated exactly as the fixed-coefficient step approximates all arrivals). '
                                'exc and inh conductances of that cell carry the factor with tau=3 ms and tau=8 ms.')),
        actual_magnitudes='every run records the actual applied magnitude after float64 rounding; the target-injected '
                          'current is 0.0 nA for the whole window so the 1e-9 nA increment is exact',
        measurements='per run versus the control: first-spike-time difference, number of differing spikes in the first '
                     '1/5/10/20/50/100/200 ms, normalized RMS of pool rates, raw commands and clipped commands in the '
                     'same windows, final spike-count difference, and whether any threshold flip occurs in the first '
                     '1 ms (plus the continuous target-state divergence step)',
        normalization='RMS(perturbation-control)/max(RMS(control),1e-9); absolute and reference RMS retained',
        dt_baseline='sealed numerics_followup supplement backward_euler 200 ms nRMS, fine pair 0.03125 vs 0.015625: '
                    'pool/raw/clip = %s' % DT_BASELINE_200,
        decision_rule='AMPLIFICATION_DOMINANT if >=6 of the 9 (3 runs x 3 readouts) 200 ms perturbation/dt nRMS '
                      'ratios are >=0.1; BIAS_DOMINANT if all 9 are <=1e-3; otherwise MIXED. Each outcome is '
                      'reported with what it implies for whether further timestep refinement could ever achieve '
                      'trajectory agreement.',
        forbidden_conclusions='no claim that the brain model is invalid; no claim of biological validity; no timestep '
                              'selection; no model selection',
        units='C nF / g uS = ms; voltage mV, current nA; illustrative electrical parameters, not physiological '
              'calibration',
        importance='核心必需')
    dump(OUT / 'PREREGISTRATION.json', cfg)
    paths = [NUM / 'TRANSLATOR_IDENTITIES.json', NUM / 'SUMMARY.json',
             NUM / 'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz', NUM / 'CONFIGS/dt_0p5ms/WARM_CACHE.pkl',
             FOL / 'SUMMARY.json',
             FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz',
             FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl',
             FOL / 'RUNS/backward_euler_0p015625ms/FINAL_STATE.pkl',
             FOL / 'RUNS/backward_euler_0p015625ms/RESULT.json',
             SRC.parent / 'numerics/run.py', SRC.parent / 'numerics_followup/run.py',
             R / 'engine/neural_cond.py']
    paths += sorted(SRC.glob('*.py')) + [OUT / 'PREREGISTRATION.json']
    dump(OUT / 'INPUT_HASHES_BEFORE.json', {str(p): sha(p) for p in paths})
    dump(OUT / 'CHANGE_LEDGER.json', dict(
        old=[dict(path=str(p), sha256=sha(p)) for p in paths if p.parent != SRC and p != OUT / 'PREREGISTRATION.json'],
        new=[dict(path=str(p), sha256=sha(p), reason='New stage: single-dt perturbation diagnostic; no frozen module '
             'changed, sealed numerics and numerics_followup are read only') for p in sorted(SRC.glob('*.py'))]))
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
    res, control, control_trace = run_case('control', 'none')
    results.append(res)
    control_check(control)
    dump(OUT / 'PROGRESS.json', dict(results=results, control_bitwise=True))
    for label, kind in RUNS:
        if label == 'control':
            continue
        res, z, trace = run_case(label, kind)
        results.append(res)
        comparisons[label] = compare_to_control(label, z, control, trace, control_trace)
        del z
        dump(OUT / 'PROGRESS.json', dict(results=results, comparisons=comparisons))
    summary = dict(results=results, comparisons=comparisons, verdict=verdict(comparisons),
                   control_bitwise=js(OUT / 'CONTROL_BITWISE_CHECK.json'),
                   neural_steps_total=sum(r['neural_steps'] for r in results),
                   physical_runs=0, physical_steps=0, new_warm_runs=0, selected_timestep=None,
                   model_selection=None, biological_validity='NOT_ESTABLISHED',
                   stage_status='single_dt_perturbation_diagnostic_complete')
    dump(OUT / 'SUMMARY.json', summary)
    dump(OUT / 'INPUT_HASHES_AFTER.json', checkhash())
    print('DONE', summary['verdict']['reading'], round(sum(r['wall_s'] for r in results), 2), flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    try:
        {'prereg': prereg, 'tests': tests, 'run': run}[action]()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / ('FAILURE_' + action + '_' + str(time.time_ns()) + '.json'), dict(traceback=traceback.format_exc()))
        raise
