"""S5 model-correction / fine_phase: does a FINE neural timestep give a REPRODUCIBLE physical verdict?

Established facts, not re-litigated here:
  * at dt = 0.5 ms the physical outcome is fragile to sub-millisecond input phase: shifting the whole
    injected-current stream by 1 or 2 sub-steps (0.5 / 1.0 ms) moved the 1.3 s J from 1.590370 to
    3.251342 / 3.438951 and foot support from 0.732081 to 0.341220 / 0.290678;
  * at fine dt, different timesteps give similar J (3.222 / 3.333 / 3.236 at 0.25 / 0.125 / 0.0625 ms),
    so the 0.5 ms reading looks like the outlier;
  * at fixed fine dt, ANY nonzero input shift saturates the pointwise 1 ms metric while aggregate
    metrics (per-cell spike counts, window-mean pool rates) stay stable.

Question answered here: at ONE fixed fine timestep, does the PHYSICAL verdict (J, foot support, min
COM, D1-D8) survive a sub-millisecond input-phase shift, i.e. can this timestep serve as a
reproducible evaluation reference for later stages?

Fixed configuration (nothing else changes anywhere):
  neural dt 0.0625 ms, backward_euler, 16 neural sub-steps per 1 ms command, physics 12.5 us,
  unchanged body / matrices / encoder / readout / 5 ms dt-aware rate filter / 1 ms causal commands,
  RH excitatory gain exactly 0.375 applied once at release, five release phases
  [1.225, 1.275, 1.3, 1.325, 1.375] s, 0.2 s each.

Conditions
  F0  unshifted injected-current stream. Must reproduce the sealed dt = 0.0625 ms results bitwise
      (WARM.npz vs the sealed CONFIGS/dt_0p0625ms, warm cache vs the sealed WARM_CACHE.pkl, all five
      BODY/NATIVE/NEURAL releases vs RUNS/dt_0p0625ms_paired_*, and 1.3 s J = 3.2364115081859093).
      If it does not, the run stops and reports.
  F1  the 1200 x 12000 current stream delayed by 1 sub-step (0.0625 ms)
  F2  the 1200 x 12000 current stream delayed by 2 sub-steps (0.125 ms)
  F3  the 1200 x 12000 current stream delayed by 4 sub-steps (0.25 ms)

Shift rule (the sealed numerics_shift rule, re-expressed on the 0.0625 ms grid). The injected-current
stream is defined on the 1 ms command grid and held constant inside a 1 ms block. The sealed
application at global neural sub-step s (0 .. 19199 for the 1200-command warm replay) is
currents[s // 16]. The shift delays the stream by N sub-step indices and holds the first sample for
the leading N sub-steps:

    k        = max(s - N, 0)         # delayed source sub-step, first sample held
    block_s  = k // 16               # 1 ms source block whose value is applied at s
    applied_current(s) = currents[block_s]

For N = 0 this is bitwise the sealed application currents[s // 16]. Because max(s-N,0) clamps at 0,
the shifted index sequence for N >= 1 is exactly the unshifted one with N zeros prepended, i.e. a pure
N-sub-step delay with the first sample held: block 0 is held for N + 16 sub-steps and every remaining
hold boundary is offset by exactly N sub-steps. The final source block keeps 16 - N of its 16
sub-steps (the exact consequence of delaying a finite stream); the total applied length stays 19200
sub-steps and no extra sample is invented. The 1 ms command grid stays on the original boundaries and
remains causal. Nothing else changes.

Where the shift is applied, and where it cannot be. Each condition runs its OWN neural warm replay
from rest over its own shifted 1200 x 12000 stream (19200 sub-steps, 0.2 .. 1.4 s of sealed actual
float64 currents) and captures its neural state at the same absolute phase times. The release itself
is a closed loop: inside a release the injected current is produced by the sensory encoder from the
actual body at each 1 ms command boundary and therefore is not a fixed shiftable array. That rule is
kept identical for F0/F1/F2/F3 (one encoder sample per command held over all 16 sub-steps), so the
ONLY difference between conditions at a given phase is the neural warm state carried into the release
- its spike event pattern, pool rates and spike history - and hence the first motor command. The
physical starting states (integration snapshot, assist reference, encoder object) are the same sealed
PHYSICAL_CACHE.pkl entries in every condition: the body trajectory start is unchanged, only the neural
state differs. No condition ever reuses another condition's neural cache.

No new teacher recording, no optimization, no model or timestep selection. Manual research
diagnostic, not a product. Absolute forces/SI, walking, autonomous support and biological validity
remain NOT_ESTABLISHED, and no claim is made that the brain model is invalid.
"""
import sys
import json
import copy
import pickle
import time
import resource
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt
from translator import SpikeHistory
from engine.neural_cond import ConductanceNetwork, ConductanceParams

O = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/fine_phase')
SRC = Path(__file__).resolve().parent
S = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
NUM = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics')

rt.O = O
rt.b.O = O
js = rt.js
sha = rt.sha
dump = rt.dump

DT = 0.0000625              # FIXED fine neural timestep (s)
NSUB = 16                   # neural sub-steps per 1 ms command
CDT = 0.001                 # command / current-stream interval (s)
PDT = 0.0000125             # physics timestep (s)
WARM_CMDS = 1200            # 0.2 .. 1.4 s
REL_CMDS = 200              # 0.2 s release
WARM_SUBSTEPS = WARM_CMDS * NSUB      # 19200
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
CONDITIONS = [('F0', 0), ('F1', 1), ('F2', 2), ('F3', 4)]
SHIFT_OF = dict(CONDITIONS)
RH_GAIN = 0.375
FILTER_TAU = 0.005
SEALED_J_1P3 = 3.2364115081859093
SEALED_NAME = 'dt_0p0625ms'
D_GATE_ORDER = ['D1_no_nonfoot_ground_force', 'D2_no_free_fall', 'D3_foot_support_not_below_assist',
                'D4_foot_support_never_collapses', 'D4b_nonfoot_is_not_carrying_the_body',
                'D5_com_height_not_below_assist', 'D6_force_balance_consistent', 'D7_tripod_redundancy',
                'D8_still_above_absolute_floor']

READING_RULE = dict(
    frozen_before_running=True,
    spread_definition='At 1.3 s, spread = max(J) - min(J) over F0..F3 (four conditions); '
                      'footspread = max(foot_support_mean) - min(foot_support_mean) over F0..F3.',
    FINE_STABLE=dict(condition='spread <= 0.2 AND footspread <= 0.1',
                     meaning='the physical verdict at this fine timestep is reproducible across '
                             'sub-millisecond input phase at the pre-registered engineering tolerances '
                             '(0.2 is the J tolerance of the sealed physical gate, 0.1 the foot-support '
                             'tolerance), so this timestep can serve as an evaluation reference'),
    FINE_FRAGILE=dict(condition='spread > 1.0',
                      meaning='the physical verdict at this fine timestep is not reproducible across '
                              'sub-millisecond input phase: a pure input-phase shift moves J as much as '
                              'the whole timestep change did, so this timestep cannot serve as an '
                              'evaluation reference'),
    INTERMEDIATE=dict(condition='otherwise (not FINE_STABLE and not FINE_FRAGILE)',
                      meaning='neither band; the measured spread is reported as it is and no reference '
                              'claim is made'),
    precedence='FINE_STABLE and FINE_FRAGILE are mutually exclusive (0.2 < 1.0); the stable test is '
               'checked first because spread <= 0.2 already excludes spread > 1.0.',
    also_reported=['the same spreads at all five phases',
                   'whether the four phases other than 1.3 s show SMALLER spreads than 1.3 s',
                   'whether ANY condition passes all D1-D8'],
    limitation='This rule reads only whether the physical verdict is reproducible. It does not select a '
               'timestep and does not select a model; it does not by itself endorse the verdict value.')


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def exact(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if sparse.issparse(a):
        return (sparse.issparse(b) and a.shape == b.shape
                and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ['data', 'indices', 'indptr']))
    if isinstance(a, dict):
        return set(a) == set(b) and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if hasattr(a, '__len__') and hasattr(a, '__getitem__') and not isinstance(a, (str, bytes)):
        return exact(list(a), list(b))
    return bool(a == b)


def cleanup():
    """The legacy constructor writes O/COVERAGE.json; it lands here and is deleted, logged."""
    p = O / 'COVERAGE.json'
    if p.exists():
        dest = O / 'CLEANUP_LEDGER.json'
        v = js(dest) if dest.exists() else {'deleted': []}
        v['deleted'].append(dict(
            path=str(p), sha256=sha(p), bytes=p.stat().st_size, importance='可再生临时',
            reason='Regenerated legacy constructor write target (the legacy Trial dumps O/COVERAGE.json). '
                   'Pool identities are retained in TRANSLATOR_IDENTITIES.json; every historical coverage '
                   'file is untouched.'))
        dump(dest, v)
        p.unlink()


def checkhash():
    v = js(O / 'INPUT_HASHES_BEFORE.json')
    a = {p: sha(p) for p in v}
    assert a == v, 'frozen input changed'
    return a


def block_index(shift, substeps=WARM_SUBSTEPS):
    """Delayed 1 ms source block applied at each neural sub-step (see module docstring).

    The delay truncates the tail of the finite stream: with N >= 1 sub-steps of delay the final
    source block keeps 16 - N of its 16 sub-steps, and the total applied length is still ``substeps``.
    """
    s = np.arange(substeps)
    k = np.maximum(s - shift, 0)
    blk = (k // NSUB).astype(np.int64)
    assert blk[0] == 0 and blk.max() == (substeps - 1 - shift) // NSUB
    return blk


def label(cond, ph):
    return '%s_paired_%s' % (cond, ph)


def condition_dir(cond):
    return O / 'CONFIGS' / cond


def input_paths():
    paths = list(dict.fromkeys(rt.files() + [
        S / 'TRUE_CURRENTS.npz', S / 'DATA/PHYSICAL_CACHE.pkl', S / 'DATA/true_teacher/TEACHER.npz',
        S / 'SUMMARY.json', S / 'WARM_DATA_HASHES.json',
        NUM / 'SUMMARY.json', NUM / 'CLOCK_TESTS.json', NUM / 'TRANSLATOR_IDENTITIES.json',
        NUM / 'PREREGISTRATION.json',
        NUM / 'CONFIGS' / SEALED_NAME / 'WARM.npz',
        NUM / 'CONFIGS' / SEALED_NAME / 'WARM_CACHE.pkl',
        NUM / 'CONFIGS' / SEALED_NAME / 'PARAMETERS.json',
        NUM / 'CONFIGS' / SEALED_NAME / 'WARM_IDENTITY.json',
    ]))
    for ph in PHASES:
        for f in ['BODY.npz', 'NATIVE.npz', 'NEURAL.npz', 'RESULT.json']:
            paths.append(NUM / 'RUNS' / (SEALED_NAME + '_paired_' + str(ph)) / f)
    paths += sorted(SRC.glob('*.py'))
    assert all(p.exists() for p in paths), [str(p) for p in paths if not p.exists()]
    return paths


# --------------------------------------------------------------------------- static / synthetic

def static_checks():
    """Cheap checks only: no release, no warm replay, no teacher recording."""
    from scipy.signal import lfilter
    rec = {}
    # T1 the shift rule on the 0.0625 ms grid, for N = 0, 1, 2, 4.
    s = np.arange(WARM_SUBSTEPS)
    sealed_app = (s // NSUB).astype(np.int64)
    holds = lambda b: np.bincount(b, minlength=WARM_CMDS)
    rec['shift_rule'] = {}
    for cond, N in CONDITIONS:
        b = block_index(N)
        b0 = block_index(0)
        h = holds(b)
        rec['shift_rule'][cond] = dict(
            substeps=N, milliseconds=N * DT * 1000, blocks_per_ms=NSUB,
            first_blocks=b[:20].tolist(),
            first_block_hold_substeps=int(h[0]), last_block_hold_substeps=int(h[-1]),
            blocks_held_for_full_16_substeps=int(np.sum(h == NSUB)),
            blocks_used=int((h > 0).sum()),
            total_applied_substeps=int(h.sum()),
            equals_sealed_application=bool(np.array_equal(b, sealed_app)),
            is_pure_delay_of_unshifted=bool(N == 0 or (np.array_equal(b[N:], b0[:-N]) and not b[:N].any())))
    assert rec['shift_rule']['F0']['equals_sealed_application']
    for cond in ('F1', 'F2', 'F3'):
        assert rec['shift_rule'][cond]['is_pure_delay_of_unshifted'], cond
    assert all(v['total_applied_substeps'] == WARM_SUBSTEPS for v in rec['shift_rule'].values())
    assert [rec['shift_rule'][c]['first_block_hold_substeps'] for c, _ in CONDITIONS] == \
           [NSUB, 1 + NSUB, 2 + NSUB, 4 + NSUB]
    assert [rec['shift_rule'][c]['last_block_hold_substeps'] for c, _ in CONDITIONS] == \
           [16, 15, 14, 12]
    rec['shift_rule']['note'] = (
        'k=max(s-N,0); block=k//16. N=0 is bitwise the sealed application currents[s//16]. For N>=1 the '
        'shifted index sequence is the unshifted one delayed by exactly N sub-steps with the first sample '
        'held, so N=1/2/4 are a 0.0625/0.125/0.25 ms grid-phase offset of the same stream. Delaying a '
        'finite stream truncates its tail (the last source block loses N of its 16 sub-steps), never adds '
        'samples.')
    # T2 the dt-aware 5 ms pool filter at dt = 0.0625 ms (16 sub-steps) is the sealed linear recursion.
    rng = np.random.RandomState(20261106)
    sp = (rng.rand(160, 7) < 0.002).astype(np.int8)
    a = np.exp(-DT / FILTER_TAU)
    seq = np.zeros(7)
    ends = []
    for k in range(sp.shape[0]):
        seq = a * seq + (1 - a) * sp[k] / DT
        if (k + 1) % NSUB == 0:
            ends.append(seq.copy())
    ref = lfilter([(1 - a) / DT], [1, -a], sp, axis=0)[NSUB - 1::NSUB]
    ferr = float(np.max(np.abs(np.asarray(ends) - ref)))
    assert ferr < 1e-12, ferr
    rec['pool_filter'] = dict(dt_s=DT, substeps=NSUB, decay=float(a), filter_tau_s=FILTER_TAU,
                              max_abs_error_vs_lfilter=ferr, passed=ferr < 1e-12)
    # T3 synthetic clocks at dt = 0.0625 ms: 2 ms conduction delay = 32 steps, refractory, measured taus.
    ms = DT * 1000
    n = ConductanceNetwork(2, dt_ms=ms)
    n.set_connectivity(sparse.csr_matrix(([.001], ([1], [0])), shape=(2, 2)), sparse.csr_matrix((2, 2)))
    assert n.step([100., 0.])[0]
    emission = n.t_ms
    arrival = None
    for _ in range(round(3 / ms)):
        start = n.t_ms
        n.step(0.)
        if n.ge[1] > 0 and arrival is None:
            arrival = start
    ring = ConductanceNetwork(1, dt_ms=ms)
    times = []
    for _ in range(round(10 / ms)):
        if ring.step(100.)[0]:
            times.append(ring.t_ms)
    dec = ConductanceNetwork(1, dt_ms=ms)
    dec.ge[:] = .01
    dec.gi[:] = .02
    dec.adaptation[:] = .03
    dec.step(0.)
    taus = [-ms / np.log(dec.ge[0] / .01), -ms / np.log(dec.gi[0] / .02), -ms / np.log(dec.adaptation[0] / .03)]
    assert abs(arrival - emission - 2) < 1e-12
    assert np.allclose(taus, [3., 8., 100.], rtol=0, atol=1e-9)
    assert abs((times[-1] - times[0]) / (len(times) - 1) - (2 + ms)) < 1e-12
    assert n.delay_steps == round(2 / ms) == 32, n.delay_steps
    assert ring._ring.shape[0] == 33, ring._ring.shape
    rec['clocks'] = dict(dt_ms=ms, delay_ms=float(arrival - emission), delay_steps=int(n.delay_steps),
                         refractory_ms=2., earliest_spike_interval_ms=2 + ms,
                         measured_taus_ms=[float(v) for v in taus], ring_rows=int(ring._ring.shape[0]),
                         passed=True)
    # T4 sealed references are readable and carry the frozen anchors.
    sealed = js(NUM / 'SUMMARY.json')
    sealed_pairs = [r for r in sealed['results'] if 'paired' in r['label']]
    assert len(sealed_pairs) == 20
    a013 = next(r for r in sealed_pairs if r['dt_s'] == DT and r['release_time_s'] == 1.3)
    assert a013['J'] == SEALED_J_1P3, a013['J']
    anchors = {str(r['dt_s']): r['J'] for r in sealed_pairs if r['release_time_s'] == 1.3}
    # the task quotes rounded anchors; compare at 1e-9 absolute and keep the full precision values.
    quoted = {'0.0005': 1.5903704331878943, '0.00025': 3.222061473, '0.000125': 3.332859429,
              '6.25e-05': SEALED_J_1P3}
    assert set(anchors) == set(quoted), anchors
    assert all(abs(anchors[k] - quoted[k]) <= 1e-9 for k in quoted), anchors
    assert anchors['0.0005'] == 1.5903704331878943 and anchors['6.25e-05'] == SEALED_J_1P3, anchors
    warm_ident = js(NUM / 'CONFIGS' / SEALED_NAME / 'WARM_IDENTITY.json')
    assert warm_ident['neural_steps'] == WARM_SUBSTEPS, warm_ident['neural_steps']
    rec['sealed_reference'] = dict(
        stage1_paired_results=len(sealed_pairs),
        dt_list=sorted({r['dt_s'] for r in sealed_pairs}),
        J_at_1p3_by_dt=anchors,
        J_at_1p3_dt_0p0625_exactly_frozen=bool(a013['J'] == SEALED_J_1P3),
        sealed_warm_neural_steps=int(warm_ident['neural_steps']),
        all_stage1_D_gates_failed=bool(not any(r['validation']['passed'] for r in sealed['results'])))
    with (S / 'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:
        physical = pickle.load(f)
    assert len(physical) == 17 and all(ph in physical for ph in PHASES)
    rec['sealed_cache'] = dict(
        phases=len(physical), used_phases=PHASES, cache_fields=sorted(physical[1.3].keys()),
        note='each cache entry carries the physical integration snapshot, assist reference and encoder '
             'object (from PHYSICAL_CACHE) plus the neural net/rates/history')
    # T5 currents, teacher spikes and the RH mask.
    cur = arrays(S / 'TRUE_CURRENTS.npz')
    currents = cur['currents']
    assert currents.shape == (WARM_CMDS, 12000) and currents.dtype == np.float64
    assert np.max(abs(cur['time'] - (.2 + np.arange(WARM_CMDS) * CDT))) < 1e-11
    teacher = arrays(S / 'DATA/true_teacher/TEACHER.npz')
    # the sealed teacher records the 0.5 ms grid (2 sub-steps per command); the fine warm replay
    # produces 16 sub-steps per command, so no direct row-by-row teacher comparison exists at this dt.
    assert teacher['spikes204'].shape == (WARM_CMDS * 2, 204), teacher['spikes204'].shape
    ident = js(NUM / 'TRANSLATOR_IDENTITIES.json')
    rh = np.asarray(ident['RH_cells'])
    rows = js(rt.b.B / 'RUN/MAPPING.json')['rows']
    rh_local = np.array(sorted({int(c) for row in rows if row['leg'] == 'RH' for c in row['local_indices']}))
    assert np.array_equal(np.sort(rh), rh_local)
    mats = [sparse.load_npz(rt.b.AB / 'A' / ('ACTUAL_' + k + '.npz')) for k in ['We', 'Wi']]
    assert mats[0].nnz == 180604 and mats[1].nnz == 148811
    cellrows = np.repeat(np.arange(12000), np.diff(mats[0].indptr))
    rec['inputs'] = dict(RH_cells=int(rh.size), We_nnz=int(mats[0].nnz), Wi_nnz=int(mats[1].nnz),
                         exc_entries_scaled_at_release=int(np.isin(cellrows, rh).sum()), gain=RH_GAIN,
                         currents_shape=list(currents.shape), currents_dtype=str(currents.dtype),
                         teacher_spikes204_shape=list(teacher['spikes204'].shape),
                         note='RH .375 is applied once at release only, identical in every condition')
    del physical, currents
    dump(O / 'STATIC_CHECKS.json', rec)
    print('STATIC CHECKS OK', flush=True)
    return rec


def prereg():
    O.mkdir(parents=True, exist_ok=True)
    assert not (O / 'PREREGISTRATION.json').exists(), 'already pre-registered'
    cfg = dict(
        stage='S5 model-correction fine_phase: reproducibility of the physical verdict at one FIXED '
              'fine neural timestep; no optimization, no timestep selection, no model selection',
        question='At fixed dt = 0.0625 ms, is the physical verdict (J, foot support, min COM, D1-D8) a '
                 'reproducible property of the model, or an accident of the particular discrete event '
                 'pattern? I.e. can this fine timestep serve as the evaluation reference for later stages?',
        fixed=dict(neural_dt_s=DT, voltage_method='backward_euler', neural_substeps_per_command=NSUB,
                   command_dt_s=CDT, physics_dt_s=PDT, warm_window_s=[0.2, 1.4], release_duration_s=0.2,
                   release_phases_s=PHASES, rate_filter_tau_s=FILTER_TAU, RH_exc_gain_once=RH_GAIN,
                   params=vars(ConductanceParams()), zero_actor='identically zero',
                   unchanged='body, matrices, sensory encoder, readout matrix and limits, command '
                             'clipping, causal 1 ms commands, 5 ms dt-aware pool filter, physics clocks'),
        conditions={cond: dict(stream_shift_substeps=N, stream_shift_ms=N * DT * 1000,
                               warm_replay='own replay from rest over the whole shifted 1200x12000 '
                                           'stream, 1200 commands = 19200 sub-steps',
                               releases='five own releases from that own warm state',
                               neural_cache='never shared with another condition',
                               physical_start='same sealed PHYSICAL_CACHE.pkl entry (integration, assist, '
                                              'encoder); the body start trajectory is unchanged')
                    for cond, N in CONDITIONS},
        shift_rule=dict(
            rule='k=max(s-N,0); block=k//16; applied_current(s)=currents[block] on the 0.0625 ms grid '
                 '(16 sub-steps per 1 ms command); for N=0 this is bitwise currents[s//16]',
            first_sample_held=True,
            tail_truncation='delaying a finite 19200-sub-step stream drops N sub-steps from the tail: the '
                            'final source block keeps 16-N of its 16 sub-steps; the applied length stays '
                            '19200 sub-steps and no sample is invented',
            sub_millisecond_note='N=1 -> 0.0625 ms, N=2 -> 0.125 ms, N=4 -> 0.25 ms grid-phase offsets of '
                                 'the same stream, all strictly sub-millisecond',
            release_injection='inside a release the injected current is produced by the sensory encoder '
                              'from the actual body at each 1 ms command boundary (closed loop) and is not '
                              'a fixed shiftable array; that rule is identical in F0/F1/F2/F3, so the only '
                              'between-condition difference at a given phase is the neural warm state and '
                              'hence the first motor command'),
        measurements=dict(
            per_condition_and_phase=['J', 'J components', 'foot_support_mean', 'foot_support_p10',
                                     'com_z_min', 'first positive non-foot contact time', 'D1..D8 booleans',
                                     'release total spikes', 'release window-mean pool rates',
                                     'per-cell spike-count Pearson r vs F0'],
            warm=['total spikes', 'window-mean pool rates', 'pool-rate trajectory nRMS vs F0',
                  'per-cell spike-count Pearson r vs F0'],
            comparison='1.3 s spread across F0/F1/F2/F3 for J, foot support, min COM and first non-foot '
                       'time; per-phase spread for all five phases; D gates passed per condition'),
        reading_rule=READING_RULE,
        budget=dict(warm_replays=4, releases=20, physical_seconds=20 * 0.2, new_teacher_recordings=0,
                    neural_steps=4 * WARM_SUBSTEPS + 20 * REL_CMDS * NSUB),
        controls=dict(F0='must reproduce the sealed stage-1 dt_0p0625ms results bitwise (WARM.npz vs the '
                         'sealed CONFIGS/dt_0p0625ms, WARM_CACHE.pkl vs the sealed cache, all five '
                         'BODY/NATIVE/NEURAL releases, 1.3 s J = 3.2364115081859093); otherwise stop'),
        scope_limits='Manual research diagnostic, not a product. No walking, autonomous support or '
                     'biological validity claim. Absolute force values and SI units NOT_ESTABLISHED. No '
                     'timestep and no model is selected. The brain model is NOT claimed invalid. The only '
                     'conclusion drawn is whether a fine timestep yields a reproducible physical verdict '
                     'and thus whether it can serve as the evaluation reference for later stages.',
        importance='核心必需')
    dump(O / 'PREREGISTRATION.json', cfg)
    paths = input_paths()
    dump(O / 'INPUT_HASHES_BEFORE.json', {str(p): sha(p) for p in paths})
    dump(O / 'CHANGE_LEDGER.json', dict(
        old=[dict(path=str(p), sha256=sha(p)) for p in paths if p.parent not in (SRC, O)],
        new=[dict(path=str(p), sha256=sha(p),
                  reason='Declared fine-timestep reproducibility runtime: fixed-0.0625 ms body/neural '
                         'replay with a documented sub-step shift of the injected-current stream. '
                         'Read-only reuse of the sealed runtime and engine; nothing frozen is modified.')
             for p in sorted(SRC.glob('*.py'))],
        frozen_inputs_untouched=True))
    static_checks()
    print('PREREGISTERED', flush=True)


# ----------------------------------------------------------------------------------- runtime

class FinePhaseRuntime(rt.Runtime):
    def __init__(self):
        rt.O = O
        rt.b.O = O
        super().__init__()
        cleanup()
        self.set_dt(DT)
        rows = js(rt.b.B / 'RUN/MAPPING.json')['rows']
        self.legcells = [np.array(sorted({int(c) for row in rows if row['leg'] == leg
                                          for c in row['local_indices']}))
                         for leg in rt.b.LEGS]
        self.condition = None
        self.shift = 0
        dump(O / 'ANATOMICAL_IDENTITIES.json', dict(
            legs=rt.b.LEGS, cells=self.legcells,
            definition='unique mapped motor cells per anatomical leg; overlaps retained explicitly'))

    def set_dt(self, dt):
        self.neural_dt = dt
        self.n = round(CDT / dt)
        assert abs(self.n * dt - CDT) < 1e-15 and self.n == NSUB
        self.history = SpikeHistory(self.cells, dt=dt)

    def integrate(self, net, block, rates):
        """One 1 ms command: n sub-steps, each with its own injected-current row."""
        assert net.dt == self.neural_dt * 1000
        block = np.asarray(block, float)
        assert block.shape == (self.n, 12000)
        decay = np.exp(-self.neural_dt / FILTER_TAU)
        counts = np.zeros(12000, np.int32)
        self.last_spikes = []
        self.last_full = []
        for j in range(self.n):
            sp = net.step(current_nA=block[j])
            counts += sp
            self.last_full.append(np.packbits(sp))
            self.last_spikes.append(sp[self.cells].copy())
            self.history.consume(sp)
            rates = decay * rates + (1 - decay) * np.asarray(self.poolmat @ sp, float) / self.neural_dt
        return rates, counts

    def observe_neural(self, d, net, enc, rates):
        from run_neural_muscle_closure import observe_quantities
        current = np.zeros(12000) if self.zero_sensory else np.asarray(
            enc.encode(observe_quantities(self.m, d, self.q0, self.joints, self.geom))[0])
        # The release keeps the sealed grid: one encoder sample per 1 ms command boundary, held over
        # all 16 sub-steps, in every condition.
        rates, counts = self.integrate(net, np.repeat(current[None, :], self.n, axis=0), rates)
        return rates, counts, current

    def episode(self, label, ph, cut=False, sensory_cut=False):
        import train_dynamic_support as dst
        import dynamic_support_metrics as dsm
        folder = O / 'RUNS' / label
        folder.mkdir(parents=True, exist_ok=False)
        dump(folder / 'PREREGISTRATION.json', dict(
            label=label, condition=self.condition, shift_substeps=self.shift,
            shift_ms=self.shift * DT * 1000,
            parent=str(O / 'PREREGISTRATION.json'), parent_sha256=sha(O / 'PREREGISTRATION.json'),
            release_time_s=ph, duration_s=.2, neural_dt_s=self.neural_dt,
            neural_steps_per_command=self.n, command_dt_s=CDT, physics_dt_s=PDT,
            history_dt_s=self.history.dt, actor='identically zero; no translator',
            RH_exc_gain=RH_GAIN, cut=cut, sensory_cut=sensory_cut,
            warm_cache_of_this_condition_only=True))
        c = self.cache[ph]
        net = copy.deepcopy(c['net'])
        enc = copy.deepcopy(c['enc'])
        rates = c['rates'].copy()
        self.history.load_state_dict(c['history'])
        self.zero_sensory = sensory_cut
        rowids = np.repeat(np.arange(net.n), np.diff(net.We.indptr))
        net.We.data[np.isin(rowids, self.rh)] *= RH_GAIN
        d = self.mj.MjData(self.m)
        self.mj.mj_setState(self.m, d, c['integration'], self.mj.mjtState.mjSTATE_INTEGRATION)
        self.mj.mj_forward(self.m, d)
        q = d.qpos.copy()
        v = d.qvel.copy()
        d.ctrl[:] = 0 if cut else self.old_command(rates)
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        self.mj.mj_forward(self.m, d)
        assert np.array_equal(q, d.qpos) and np.array_equal(v, d.qvel)
        body = [self.bodyrow(d)]
        native = [self.contact(d)]
        logs = []
        start = time.monotonic()
        try:
            for k in range(REL_CMDS):
                features = self.history.features()
                old = self.old_command(rates)
                raw = self.limits / 100 * (self.matrix @ rates)
                ctrl = np.zeros(42) if cut else old.copy()
                source = rates.copy()
                rates, counts, cur = self.observe_neural(d, net, enc, rates)
                d.ctrl[:] = ctrl
                for j in range(80):
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    self.mj.mj_step(self.m, d)
                    self.mj.mj_forward(self.m, d)
                    native.append(self.contact(d))
                    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
                    if (j + 1) % 40 == 0:
                        body.append(self.bodyrow(d))
                logs.append(dict(time=float(d.time), features=features, source_rates=source,
                                 rates=rates.copy(), raw_command=raw, old_torque=old, applied=ctrl,
                                 spikes204=np.asarray(self.last_spikes),
                                 spikes12000_packed=np.asarray(self.last_full),
                                 sensory_current=cur.copy(),
                                 sensory_current_norm=float(np.linalg.norm(cur)),
                                 net_t_ms=net.t_ms, net_steps=net.steps))
            arr = np.asarray(native)
            scored = arr.copy()
            scored[:, 0] -= ph - 1.3
            old_duration = dst.RELEASE
            dst.RELEASE = .2
            try:
                J, parts = dst.objective(scored, c['assist'])
            finally:
                dst.RELEASE = old_duration
            gate = dsm.evaluate_native(c['assist'], arr)
            gate['checks']['D1_no_nonfoot_ground_force'] = bool(np.max(arr[:, 7]) == 0)
            gate['passed'] = all(gate['checks'].values())
            pos = np.flatnonzero(arr[:, 7] > 1e-9)
            packed = np.asarray([r['spikes12000_packed'] for r in logs])
            cell_counts = np.zeros(12000, np.int64)
            for row in packed:
                cell_counts += np.unpackbits(row, axis=1, count=12000).sum(0).astype(np.int64)
            pool_rates = np.asarray([r['rates'] for r in logs], float)
            assert sorted(gate['checks']) == sorted(D_GATE_ORDER), sorted(gate['checks'])
            result = dict(label=label, condition=self.condition, shift_substeps=self.shift,
                          shift_ms=self.shift * DT * 1000, dt_s=self.neural_dt, J=J, parts=parts,
                          validation=gate, gate_checks=gate['checks'], gate_booleans=[
                              dict(gate=k, value=bool(gate['checks'][k])) for k in D_GATE_ORDER],
                          release_time_s=ph, duration_s=.2, folder=str(folder),
                          cut=cut, sensory_cut=sensory_cut,
                          first_positive_nonfoot_s=float(arr[pos[0], 0] - ph) if len(pos) else None,
                          D_gates_passed=int(sum(gate['checks'].values())), D_gates_total=len(gate['checks']),
                          wall_s=time.monotonic() - start,
                          peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          warnings=int(d.warning.number.sum()), release_aux_max=0.,
                          release_external_max=0.,
                          aggregate_neural=dict(
                              release_total_spikes=int(cell_counts.sum()),
                              release_mean_allcell_hz=float(cell_counts.sum() / 12000 / .2),
                              release_motor_mean_hz=float(cell_counts[self.cells].mean() / .2),
                              release_window_mean_pool_rates=pool_rates.mean(0),
                              release_window_mean_cell_hz=cell_counts / .2,
                              release_neural_steps=int(REL_CMDS * NSUB),
                              definition='window = the whole 0.2 s release; pool rates are the saved '
                                         'command-end rates averaged over the 200 commands; cell rates are '
                                         'spike counts over 3200 sub-steps divided by 0.2 s'),
                          walking_certified=False)
            np.savez_compressed(folder / 'NEURAL_AGG.npz',
                                window_mean_pool_rates=pool_rates.mean(0),
                                window_mean_cell_hz=cell_counts / .2, pool_rates=pool_rates)
            dump(folder / 'RESULT.json', result)
        except BaseException:
            import traceback
            dump(folder / 'FAILURE.json', dict(traceback=traceback.format_exc()))
            raise
        finally:
            np.savez_compressed(folder / 'NATIVE.npz', rows=np.asarray(native))
            np.savez_compressed(folder / 'BODY.npz', **{k: np.array([r[k] for r in body]) for k in body[0]})
            if logs:
                np.savez_compressed(folder / 'NEURAL.npz',
                                    **{k: np.array([r[k] for r in logs]) for k in logs[0]})
            self.zero_sensory = False
        print('RELEASE', label, round(J, 6), 'D', result['D_gates_passed'], gate['passed'], flush=True)
        return result


def warm(t, cond, physical, currents):
    """Own neural warm replay from rest over this condition's shifted stream."""
    start = time.monotonic()
    shift = SHIFT_OF[cond]
    t.condition = cond
    t.shift = shift
    t.set_dt(DT)
    folder = condition_dir(cond)
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / 'PARAMETERS.json', dict(
        condition=cond, shift_substeps=shift, shift_ms=shift * DT * 1000,
        neural_dt_s=DT, steps_per_command=t.n, command_dt_s=CDT, physics_dt_s=PDT,
        params=vars(ConductanceParams()), history_dt_s=DT, rate_filter_tau_s=FILTER_TAU,
        connectivity='original', RH_release_only=RH_GAIN,
        shift_rule='k=max(s-N,0); block=k//16; applied_current(s)=currents[block]',
        current_source=str(S / 'TRUE_CURRENTS.npz'), current_sha256=sha(S / 'TRUE_CURRENTS.npz'),
        physical_source=str(S / 'DATA/PHYSICAL_CACHE.pkl')))
    net = ConductanceNetwork(12000, dt_ms=DT * 1000)
    net.set_connectivity(*[sparse.load_npz(rt.b.AB / 'A' / ('ACTUAL_' + k + '.npz')) for k in ['We', 'Wi']])
    bidx = block_index(shift)
    rates = np.zeros(66)
    t.history.reset()
    cache = {}
    packed = []
    rr = []
    legs = []
    cell_filtered = np.zeros(12000)
    cell_rates = []
    checks = []
    decay = np.exp(-DT / FILTER_TAU)
    # capture EVERY sealed physical phase (17), exactly as the sealed warm did, so the whole cache can
    # be compared bitwise; only the five release phases are aggregated.
    captures = {round((ph - .2) / CDT): ph for ph in physical}
    cap_counts = {}
    for k in range(WARM_CMDS):
        block = currents[bidx[k * NSUB:(k + 1) * NSUB]]
        rates, counts = t.integrate(net, block, rates)
        packed.extend(t.last_full)
        rr.append(rates.copy())
        for s in t.last_full:
            cell_filtered = decay * cell_filtered + (1 - decay) * np.unpackbits(s, count=12000) / DT
        cell_rates.append(cell_filtered.copy())
        legs.append([cell_filtered[c].mean() for c in t.legcells])
        if k + 1 in captures:
            ph = captures[k + 1]
            c = dict(physical[ph])
            c.update(net=copy.deepcopy(net), rates=rates.copy(), history=t.history.state_dict())
            cache[ph] = c
            if ph in PHASES:
                cap_counts[ph] = np.unpackbits(np.asarray(t.last_full), axis=1,
                                               count=12000).sum(0).astype(np.int64)
                # NOTE: the sealed PHYSICAL_CACHE net/rates/history were recorded at dt = 0.5 ms, so
                # they CANNOT be an equality reference here (different timestep). The F0 bitwise control
                # is checked in run() against sealed WARM.npz and WARM_CACHE.pkl instead.
                checks.append(dict(phase=ph, capture_command=k + 1,
                                   reference='sealed CONFIGS/dt_0p0625ms checked in run(), not against '
                                             'PHYSICAL_CACHE (that cache is a 0.5 ms state)'))
    packed = np.asarray(packed)
    spikes = np.unpackbits(packed, axis=1, count=12000)[:, t.cells]
    rr = np.asarray(rr)
    source_rates = np.vstack([np.zeros(66), rr[:-1]])
    raw = source_rates @ t.matrix.T * t.limits / 100
    np.savez_compressed(folder / 'WARM.npz',
                        spikes12000_packed=packed, spikes204=spikes, cells=t.cells, pool_rates=rr,
                        command_source_rates=source_rates, leg_rates=np.asarray(legs),
                        cell_filtered_rates=np.asarray(cell_rates), raw_command=raw,
                        clipped_command=np.clip(raw, -t.limits, t.limits),
                        time=.2 + np.arange(1, WARM_CMDS + 1) * CDT, neural_dt_s=DT,
                        shift_substeps=np.int64(shift), shift_ms=np.float64(shift * DT * 1000),
                        applied_block_index=bidx)
    with (folder / 'WARM_CACHE.pkl').open('wb') as f:
        pickle.dump(cache, f, protocol=5)
    cell_counts = np.unpackbits(packed, axis=1, count=12000).sum(0).astype(np.int64)
    np.savez_compressed(folder / 'WARM_AGG.npz',
                        window_mean_pool_rates=rr.mean(0), window_mean_cell_hz=cell_counts / 1.2,
                        pool_rates=rr,
                        **{('cap_rates_' + str(ph)): cache[ph]['rates'] for ph in PHASES},
                        **{('cap_counts_' + str(ph)): cap_counts[ph] for ph in PHASES})
    del cell_rates, spikes
    summary = dict(
        condition=cond, shift_substeps=shift, shift_ms=shift * DT * 1000, dt_s=DT,
        neural_steps=int(packed.shape[0]), neural_commands=WARM_CMDS,
        total_spikes=int(cell_counts.sum()), mean_allcell_hz=float(cell_counts.sum() / 12000 / 1.2),
        motor_mean_hz=float(cell_counts[t.cells].mean() / 1.2),
        window_mean_pool_rates=[float(v) for v in rr.mean(0)],
        leg_mean_hz=[float(v) for v in np.asarray(legs).mean(0)],
        capture_phase_pool_rates={str(ph): [float(v) for v in cache[ph]['rates']] for ph in PHASES},
        capture_phase_total_spikes={str(ph): int(cap_counts[ph].sum()) for ph in PHASES},
        baseline_exact_checks=checks,
        cache_sha256=sha(folder / 'WARM_CACHE.pkl'),
        warm_npz_sha256=sha(folder / 'WARM.npz'),
        wall_s=time.monotonic() - start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    dump(folder / 'WARM_SUMMARY.json', summary)
    dump(folder / 'WARM_IDENTITY.json', dict(
        condition=cond, shift_substeps=shift, shift_ms=shift * DT * 1000,
        current_source=str(S / 'TRUE_CURRENTS.npz'), current_sha256=sha(S / 'TRUE_CURRENTS.npz'),
        physical_source=str(S / 'DATA/PHYSICAL_CACHE.pkl'), physical_sha256=sha(S / 'DATA/PHYSICAL_CACHE.pkl'),
        neural_steps=int(packed.shape[0]), wall_s=summary['wall_s'],
        baseline_exact_checks=checks, cache_sha256=summary['cache_sha256'],
        own_cache_only='this cache is used only by this condition and is never shared'))
    t.cache = cache
    print('WARM', cond, shift, int(packed.shape[0]), len(cache), round(summary['wall_s'], 1), flush=True)
    return summary


def nrm(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b * b)), 1e-9))


def compare(warm_summaries, results):
    """Cross-condition comparison at each phase plus the pre-registered 1.3 s reading."""
    conds = [c for c, _ in CONDITIONS]
    agg = {cond: arrays(condition_dir(cond) / 'WARM_AGG.npz') for cond in conds}
    warm_compare = {}
    for cond, shift in CONDITIONS:
        if cond == 'F0':
            warm_compare[cond] = dict(reference=True, shift_substeps=0, shift_ms=0.0)
            continue
        a, b = agg[cond], agg['F0']
        per_phase = {}
        for ph in PHASES:
            x = a['cap_counts_' + str(ph)].astype(np.int64)
            y = b['cap_counts_' + str(ph)].astype(np.int64)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
            per_phase[str(ph)] = dict(
                pool_rates_nRMS_at_capture=nrm(a['cap_rates_' + str(ph)], b['cap_rates_' + str(ph)]),
                pool_rates_max_abs_diff=float(np.max(np.abs(a['cap_rates_' + str(ph)] - b['cap_rates_' + str(ph)]))),
                cell_spike_counts_pearson_r=r,
                cell_spike_counts_total_abs_diff=int(np.abs(x - y).sum()),
                cell_spike_counts_F0=int(y.sum()), cell_spike_counts_this=int(x.sum()))
        warm_compare[cond] = dict(
            shift_substeps=shift, shift_ms=shift * DT * 1000,
            pool_rates_trajectory_nRMS=nrm(a['pool_rates'], b['pool_rates']),
            pool_rates_trajectory_max_abs_diff=float(np.max(np.abs(a['pool_rates'] - b['pool_rates']))),
            window_mean_pool_rates_nRMS=nrm(a['window_mean_pool_rates'], b['window_mean_pool_rates']),
            window_mean_pool_rates_pearson_r=float(np.corrcoef(a['window_mean_pool_rates'],
                                                               b['window_mean_pool_rates'])[0, 1]),
            cell_spike_counts_pearson_r_whole_warm=(
                float(np.corrcoef(a['window_mean_cell_hz'], b['window_mean_cell_hz'])[0, 1])
                if a['window_mean_cell_hz'].std() > 0 and b['window_mean_cell_hz'].std() > 0 else None),
            total_spikes=int(warm_summaries[cond]['total_spikes']),
            total_spikes_F0=int(warm_summaries['F0']['total_spikes']),
            mean_allcell_hz=warm_summaries[cond]['mean_allcell_hz'],
            mean_allcell_hz_F0=warm_summaries['F0']['mean_allcell_hz'],
            mean_allcell_hz_ratio=float(warm_summaries[cond]['mean_allcell_hz']
                                        / warm_summaries['F0']['mean_allcell_hz']),
            per_phase=per_phase)
    release_compare = {}
    for ph in PHASES:
        rows = sorted([r for r in results if r['release_time_s'] == ph], key=lambda r: conds.index(r['condition']))
        first = lambda r: .2 if r['first_positive_nonfoot_s'] is None else r['first_positive_nonfoot_s']
        J = {r['condition']: r['J'] for r in rows}
        foot = {r['condition']: r['parts']['foot_support_mean'] for r in rows}
        com = {r['condition']: r['parts']['com_z_min'] for r in rows}
        nf = {r['condition']: first(r) for r in rows}
        D = {r['condition']: dict(checks=r['validation']['checks'], passed=r['validation']['passed'],
                                  n_passed=r['D_gates_passed']) for r in rows}
        rel = {}
        for cond in conds:
            if cond == 'F0':
                rel[cond] = dict(reference=True)
                continue
            a = next(r for r in rows if r['condition'] == cond)['aggregate_neural']
            b = next(r for r in rows if r['condition'] == 'F0')['aggregate_neural']
            x = a['release_window_mean_cell_hz']
            y = b['release_window_mean_cell_hz']
            rel[cond] = dict(
                release_window_mean_pool_rates_nRMS=nrm(a['release_window_mean_pool_rates'],
                                                        b['release_window_mean_pool_rates']),
                release_window_mean_cell_hz_pearson_r=(float(np.corrcoef(x, y)[0, 1])
                                                       if x.std() > 0 and y.std() > 0 else None),
                release_mean_allcell_hz_ratio=float(x.mean() / y.mean()) if y.mean() else None,
                release_total_spikes=int(a['release_total_spikes']),
                release_total_spikes_F0=int(b['release_total_spikes']))
        release_compare[str(ph)] = dict(
            J=J, J_spread=float(max(J.values()) - min(J.values())),
            foot_support_mean=foot, foot_support_spread=float(max(foot.values()) - min(foot.values())),
            com_z_min=com, com_z_min_spread=float(max(com.values()) - min(com.values())),
            first_positive_nonfoot_s=nf,
            first_positive_nonfoot_spread=float(max(nf.values()) - min(nf.values())),
            D=D, neural_vs_F0=rel,
            any_D_gate_pass=any(v['passed'] for v in D.values()),
            D_checks_identical_across_conditions=bool(all(
                D[c]['checks'] == D['F0']['checks'] for c in conds)))
    main = release_compare['1.3']
    jspread = main['J_spread']
    fspread = main['foot_support_spread']
    if jspread <= 0.2 and fspread <= 0.1:
        verdict = 'FINE_STABLE'
    elif jspread > 1.0:
        verdict = 'FINE_FRAGILE'
    else:
        verdict = 'INTERMEDIATE'
    other = [float(p) for p in PHASES if p != 1.3]
    other_spreads = {str(ph): release_compare[str(ph)]['J_spread'] for ph in other}
    other_foot = {str(ph): release_compare[str(ph)]['foot_support_spread'] for ph in other}
    sealed = js(NUM / 'SUMMARY.json')
    dt_ref = {}
    for r in sealed['results']:
        if 'paired' in r['label']:
            dt_ref.setdefault(str(r['release_time_s']), {})[str(r['dt_s'])] = dict(
                J=r['J'], foot_support_mean=r['parts']['foot_support_mean'],
                com_z_min=r['parts']['com_z_min'],
                first_positive_nonfoot_s=r['first_positive_nonfoot_s'],
                D_gates_passed=int(sum(r['validation']['checks'].values())), D_passed=r['validation']['passed'])
    return dict(
        conditions=[dict(condition=c, shift_substeps=s, shift_ms=s * DT * 1000) for c, s in CONDITIONS],
        warm_neural=warm_compare, release=release_compare,
        reading_rule=READING_RULE, verdict=verdict,
        verdict_numbers=dict(
            J_spread_at_1p3=jspread, foot_support_spread_at_1p3=fspread,
            com_z_min_spread_at_1p3=main['com_z_min_spread'],
            first_positive_nonfoot_spread_at_1p3=main['first_positive_nonfoot_spread'],
            J_by_condition_at_1p3=main['J'],
            foot_support_by_condition_at_1p3=main['foot_support_mean'],
            D_gates_passed_by_condition_at_1p3={c: main['D'][c]['n_passed'] for c in main['D']},
            any_condition_passes_all_D_gates_at_1p3=main['any_D_gate_pass'],
            any_condition_passes_all_D_gates_at_any_phase=any(
                release_compare[str(ph)]['any_D_gate_pass'] for ph in PHASES),
            J_spread_all_phases=other_spreads, J_spread_at_1p3_only=dict(**{'1.3': jspread}),
            foot_support_spread_all_phases=other_foot,
            phases_other_than_1p3_with_smaller_J_spread=[str(ph) for ph in other
                                                         if release_compare[str(ph)]['J_spread'] < jspread],
            all_other_phases_smaller_J_spread=bool(all(
                release_compare[str(ph)]['J_spread'] < jspread for ph in other)),
            all_other_phases_smaller_foot_spread=bool(all(
                release_compare[str(ph)]['foot_support_spread'] < fspread for ph in other)),
            max_other_phase_J_spread=float(max(release_compare[str(ph)]['J_spread'] for ph in other)),
            max_other_phase_foot_spread=float(max(release_compare[str(ph)]['foot_support_spread'] for ph in other))),
        reference_sealed_dt_vs_J=dt_ref,
        reference_note='Sealed stage-1 dt sweep, read-only, used only as the yardstick for how much the '
                       'physical verdict moved when the whole timestep changed.')


def run():
    checkhash()
    assert all(sha(p) == h for p, h in js(S / 'WARM_DATA_HASHES.json').items())
    t = FinePhaseRuntime()
    with (S / 'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:
        physical = pickle.load(f)
    cur = arrays(S / 'TRUE_CURRENTS.npz')
    currents = cur['currents']
    assert currents.shape == (WARM_CMDS, 12000) and currents.dtype == np.float64
    assert len(physical) == 17
    conds = [c for c, _ in CONDITIONS]
    results = []
    warm_summaries = {}
    neural_steps = 0
    for cond, shift in CONDITIONS:
        ws = warm(t, cond, physical, currents)
        warm_summaries[cond] = ws
        neural_steps += WARM_SUBSTEPS
        for ph in [1.3] + [v for v in PHASES if v != 1.3]:
            r = t.episode(label(cond, ph), ph)
            results.append(r)
            neural_steps += REL_CMDS * NSUB
            if cond == 'F0':
                old = next(x for x in js(NUM / 'SUMMARY.json')['results']
                           if x['label'] == SEALED_NAME + '_paired_' + str(ph))
                same = {}
                for f in ['BODY.npz', 'NATIVE.npz', 'NEURAL.npz']:
                    a = arrays(Path(old['folder']) / f)
                    b = arrays(Path(r['folder']) / f)
                    assert set(a) == set(b), (f, sorted(set(a) ^ set(b)))
                    same[f] = {k: bool(np.array_equal(a[k], b[k])) for k in a}
                    assert all(same[f].values()), (ph, f, [k for k, v in same[f].items() if not v])
                assert r['J'] == old['J'], (r['J'], old['J'])
                assert r['validation']['checks'] == old['validation']['checks']
                assert r['first_positive_nonfoot_s'] == old['first_positive_nonfoot_s']
                if ph == 1.3:
                    assert r['J'] == SEALED_J_1P3, r['J']
                dump(Path(r['folder']) / 'BASELINE_EXACT.json', dict(
                    sealed_folder=old['folder'], all_arrays_exact=same, J_exact=r['J'] == old['J'],
                    D_checks_exact=r['validation']['checks'] == old['validation']['checks'],
                    note='F0 control reproduces the sealed stage-1 dt_0p0625ms release bitwise'))
        if cond == 'F0':
            z = arrays(condition_dir('F0') / 'WARM.npz')
            sz = arrays(NUM / 'CONFIGS' / SEALED_NAME / 'WARM.npz')
            same = {k: bool(np.array_equal(z[k], sz[k])) for k in sz}
            assert all(same.values()), [k for k, v in same.items() if not v]
            dump(condition_dir('F0') / 'BASELINE_EXACT_WARM.json', dict(
                sealed_warm=str(NUM / 'CONFIGS' / SEALED_NAME / 'WARM.npz'),
                arrays_exact_vs_sealed=same, extra_keys_in_this_run=[k for k in z if k not in sz],
                note='every array the sealed stage wrote is bitwise reproduced; the extra keys only '
                     'record the shift rule that was applied (N=0 here)'))
            with (condition_dir('F0') / 'WARM_CACHE.pkl').open('rb') as f:
                mine = pickle.load(f)
            with (NUM / 'CONFIGS' / SEALED_NAME / 'WARM_CACHE.pkl').open('rb') as f:
                theirs = pickle.load(f)
            assert sorted(mine) == sorted(theirs), (sorted(mine), sorted(theirs))
            cache_same = {}
            for ph in sorted(mine):
                cache_same[str(ph)] = dict(
                    net_vars_exact=exact(vars(mine[ph]['net']), vars(theirs[ph]['net'])),
                    rates_exact=exact(mine[ph]['rates'], theirs[ph]['rates']),
                    history_exact=exact(mine[ph]['history'], theirs[ph]['history']),
                    integration_exact=exact(mine[ph]['integration'], theirs[ph]['integration']),
                    assist_exact=exact(mine[ph]['assist'], theirs[ph]['assist']))
                assert all(cache_same[str(ph)].values()), (ph, cache_same[str(ph)])
            del mine, theirs
            dump(condition_dir('F0') / 'BASELINE_EXACT_CACHE.json', dict(
                sealed_cache=str(NUM / 'CONFIGS' / SEALED_NAME / 'WARM_CACHE.pkl'),
                per_phase=cache_same, all_exact=True,
                note='the F0 warm neural cache (net, rates, history) and the shared physical fields '
                     '(integration, assist) are bitwise equal to the sealed cache'))
        else:
            a = arrays(condition_dir(cond) / 'WARM_AGG.npz')
            b = arrays(condition_dir('F0') / 'WARM_AGG.npz')
            mine = arrays(condition_dir(cond) / 'WARM.npz')
            ref = arrays(condition_dir('F0') / 'WARM.npz')
            effect = dict(
                condition=cond,
                pool_rates_trajectory_array_equal_to_F0=bool(np.array_equal(mine['pool_rates'], ref['pool_rates'])),
                spikes12000_packed_array_equal_to_F0=bool(np.array_equal(mine['spikes12000_packed'],
                                                                        ref['spikes12000_packed'])),
                window_mean_cell_hz_array_equal_to_F0=bool(np.array_equal(a['window_mean_cell_hz'],
                                                                         b['window_mean_cell_hz'])),
                total_spikes_equal_F0=bool(ws['total_spikes'] == warm_summaries['F0']['total_spikes']),
                cap_rates_array_equal_per_phase={str(ph): bool(np.array_equal(a['cap_rates_' + str(ph)],
                                                                             b['cap_rates_' + str(ph)]))
                                                 for ph in PHASES},
                applied_block_index_differs_from_F0=bool(
                    not np.array_equal(mine['applied_block_index'], ref['applied_block_index'])),
                note='the shift is only meaningful if it actually changed the warm neural event pattern; '
                     'these are the checks that it did')
            dump(condition_dir(cond) / 'SHIFT_EFFECT.json', effect)
            assert not effect['pool_rates_trajectory_array_equal_to_F0'], cond
            assert not effect['spikes12000_packed_array_equal_to_F0'], cond
            assert effect['applied_block_index_differs_from_F0'], cond
            del mine, ref
        dump(O / 'PROGRESS.json', dict(done=[r['label'] for r in results], neural_steps=neural_steps))
    comparison = compare(warm_summaries, results)
    dump(O / 'CROSS_CONDITION.json', comparison)
    summary = dict(
        experiment='S5 model-correction fine_phase: reproducibility of the physical verdict at the fixed '
                   'fine neural timestep 0.0625 ms',
        source_dir=str(SRC), output_dir=str(O),
        fixed=js(O / 'PREREGISTRATION.json')['fixed'],
        conditions=[dict(condition=c, shift_substeps=s, shift_ms=s * DT * 1000) for c, s in CONDITIONS],
        results=results, warm=warm_summaries, comparison=comparison,
        verdict=comparison['verdict'], verdict_numbers=comparison['verdict_numbers'],
        warm_replays=len(CONDITIONS), physical_releases=len(results),
        physical_seconds=round(.2 * len(results), 6), new_teacher_recordings=0,
        neural_steps=neural_steps, selected_model=None, selected_timestep=None,
        walking_certified=False, autonomous_support_certified=False,
        absolute_forces='NOT_ESTABLISHED', biological_validity='NOT_ESTABLISHED',
        brain_model_invalid_claim='NOT_MADE',
        scope='Manual research reproducibility diagnostic, not a product; no timestep and no model is '
              'selected by this run.')
    dump(O / 'SUMMARY.json', summary)
    dump(O / 'INPUT_HASHES_AFTER.json', checkhash())
    print('DONE', len(results), comparison['verdict'], flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    try:
        {'prereg': prereg, 'run': run, 'checks': static_checks}[action]()
    except BaseException:
        import traceback
        O.mkdir(parents=True, exist_ok=True)
        dump(O / ('FAILURE_' + action + '_' + str(time.time_ns()) + '.json'),
             dict(traceback=traceback.format_exc()))
        raise
