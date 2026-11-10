"""S5 sense-drive GAIN SWEEP: does lowering the sensory encoder gain bring the spiking
network into a sparse operating regime and restore task-relevant motor information and
physical support?

BOUNDARIES (frozen before running, see prereg.py)
  * Changing `ChannelConfig.current_nA_per_unit` is an ENGINEERING parameter change. It is
    NOT biological calibration and no measured fly current is claimed anywhere here.
  * The neurons generate the command. No external CPG, PD law, preset trajectory, state
    machine or lookup table produces any motion. The readout is an offline ridge fit of the
    interface from filtered neural rates to the recorded teacher torque, exactly as the
    previous stage (pop_readout / readout_fit) fitted it.
  * Only gain 0.3 is a RECORDED encoder stream (TRUE_CURRENTS.npz). Every other gain uses the
    SAME teacher trajectory with the encoder output rescaled linearly; this is declared as a
    linear rescaling of the recorded encoder output, not a re-recording.
"""
import json
import hashlib
import pickle
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ workspace
R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'      # runtime + translator
SRC = Path(__file__).resolve().parent                             # frozen source dir
RT1 = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z')
O = RT1 / 'gain_sweep'                                            # this task output dir

# read-only inputs
SEALED_NUMERICS = RT1 / 'numerics'                                # sealed clocks + warm caches
SEALED_CONFIGS = SEALED_NUMERICS / 'CONFIGS'
POPREADOUT = RT1 / 'pop_readout'                                  # previous stage artifacts
STRENGTH = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
CURRENTS = STRENGTH / 'TRUE_CURRENTS.npz'                         # recorded at gain 0.3
PHYSICAL = STRENGTH / 'DATA/PHYSICAL_CACHE.pkl'
TEACHER = STRENGTH / 'DATA/true_teacher'
META = R / 'data/banc888/banc_888_meta.feather'
FLYWIRE = R / 'data/flywire/banc_connectome.npz'
HANDOVER = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
IDENTITIES_NPZ = HANDOVER / 'IDENTITIES.npz'                      # 12000 root ids, tier order
TRANSLATOR_IDENTITIES = RT1 / 'sensory/CANDIDATES/TRANSLATOR_IDENTITIES.json'

# ------------------------------------------------------------------ experiment
GAINS = [0.3, 0.1, 0.03, 0.01]        # nA per unit; 0.3 is ChannelConfig's current default
REF_GAIN = 0.3                        # the gain TRUE_CURRENTS.npz was recorded at
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
DT_MAIN = 0.0005                      # neural timestep for every gain
DT_FINE = 0.0000625                   # second timestep, top-two gains only (rule in prereg)
DTS = [DT_MAIN, DT_FINE]
CDT = 0.001                           # 1 ms causal command clock
DURATION = 0.2                        # release duration (s)
PHYSICS_DT = 0.0000125                # 12.5 us physics
RH_GAIN = 0.375                       # applied exactly once at release
FILTER_TAU = 0.005                    # 5 ms rate filter
FIT_LO, FIT_HI = 0, 600               # 0.2-0.8 s of the teacher window (1 ms intervals)
HELD_LO, HELD_HI = 600, 1200          # 0.8-1.4 s, held out
GAMMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
CV_FOLDS = 5
TEACHER_STRIDE = 2                    # teacher BODY.npz is stored at 0.5 ms
TEACHER_START = 400
# reading rule thresholds, frozen in PREREGISTRATION.json before any run
RESTORED_MOTOR_R2 = 0.7               # motor-cell held-out R^2 needed for INFORMATION_RESTORED
BASELINE_MOTOR_R2 = 0.43              # parent-measured value at the current gain (check only)
SUPPORT_PHASES_NEEDED = 4             # >= 4 of 5 phases passing all D gates
DENSE_READOUT_MEAN_HZ = 40.0          # dense-regime relief: readout-204 mean rate ceiling
DENSE_HYPER_SHARE = 0.02              # share of cells above 100 Hz allowed in a sparse regime
DENSE_SILENT_SHARE = 0.40             # share of cells with zero spikes allowed
DENSE_MIN_MEAN_HZ = 5.0               # the population must still fire
MAX_RELEASES = 35
MAX_WARMS = 6                         # 4 gains at 0.5 ms + top-two at 0.0625 ms


def gain_tag(g):
    return 'gain_' + ('%g' % g).replace('.', 'p')


def dt_tag(dt):
    return 'dt_' + ('%g' % (dt * 1000)).replace('.', 'p') + 'ms'


def phase_index(ph):
    i = round((ph - 0.2) / CDT) - 1
    assert abs(0.2 + (i + 1) * CDT - ph) < 1e-12, (ph, i)
    return i


# ------------------------------------------------------------------ io helpers
def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def js(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                     default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                                     else x.item()) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    """Structural + bitwise equality, as used by the sealed numerics stage."""
    from scipy import sparse
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if sparse.issparse(a):
        return (sparse.issparse(b) and a.shape == b.shape
                and all(np.array_equal(getattr(a, k), getattr(b, k))
                        for k in ['data', 'indices', 'indptr']))
    if isinstance(a, dict):
        return set(a) == set(b) and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return bool(a == b)


# ------------------------------------------------------------------ inputs
def teacher_torque():
    """Applied teacher joint torque at the 1200 command boundaries (0.2-1.4 s)."""
    body = arrays(TEACHER / 'BODY.npz')
    time = body['time']
    assert time.shape == (2801,) and abs(time[0]) < 1e-9 and abs(time[-1] - 1.4) < 1e-9
    ctrl = body['ctrl']
    assert ctrl.shape == (2801, 42)
    idx = TEACHER_START + TEACHER_STRIDE * np.arange(1200)
    assert abs(time[idx[0]] - 0.2) < 1e-9 and abs(time[idx[-1]] - 1.399) < 1e-9
    return ctrl[idx].copy()


def recorded_currents():
    z = arrays(CURRENTS)
    cur = z['currents']
    assert cur.shape == (1200, 12000) and cur.dtype == np.float64
    return cur


def currents_for_gain(gain):
    """Current stream for one gain.

    `TRUE_CURRENTS.npz` is the encoder output recorded at gain 0.3. `ChannelSensoryEncoder`
    multiplies every channel quantity by `cfg.current_nA_per_unit` exactly once, so its output
    is a linear function of the gain (verified as a static check). This function therefore
    rescales that recorded stream; it does not re-record it (no teacher run is made here).
    At gain 0.3 the factor is 0.3/0.3 == 1.0 and the product is bitwise the recorded array.
    """
    return recorded_currents() * (gain / REF_GAIN)


def motor_cells_204():
    cells = np.asarray(js(TRANSLATOR_IDENTITIES)['cells'], int)
    assert cells.shape == (204,) and len(set(cells.tolist())) == 204
    return cells


def anatomy():
    """Per-tier-cell annotation read from the dataset, joined by banc_888_id."""
    import pyarrow.feather as pf
    ids = np.load(IDENTITIES_NPZ)['root_ids']
    assert ids.shape == (12000,)
    t = pf.read_table(META, memory_map=True)

    def col(name):
        v = t.column(name).to_numpy(zero_copy_only=False)
        return np.array([('NA' if x is None else str(x)) for x in v], dtype=object)

    b888 = col('banc_888_id').astype(np.int64)
    assert len(b888) == t.num_rows
    row_of = {int(v): k for k, v in enumerate(b888)}
    rows = np.array([row_of.get(int(i), -1) for i in ids], int)
    unmatched = int((rows < 0).sum())
    safe = np.where(rows < 0, 0, rows)
    # every column is materialised ONCE; indexing it per tier cell row is a pure gather
    sup = col('super_class')
    cls = col('cell_class')
    fun = col('cell_function')
    pick = lambda a: np.where(rows < 0, 'NA', a[safe])
    out = dict(unmatched=unmatched, matched=int(12000 - unmatched), meta_rows=int(t.num_rows),
               key='banc_888_id',
               super_class=pick(sup), cell_class=pick(cls), cell_function=pick(fun),
               ids=ids, meta_rows_of_tier=safe)
    return out


def pool_matrix():
    """The sealed 66-pool projection (mean rate per pool), rebuilt without MuJoCo.

    Identical construction to the legacy trial module: pool keys are the distinct non-empty
    `local_indices` tuples of POOL_TO_JOINT.json in file order.
    """
    from scipy import sparse
    from_plant = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/'
                      '20261105T060730Z/POOL_TO_JOINT.json')
    rows = js(from_plant)['muscle_to_joint']
    pools = []
    lookup = {}
    for r in rows:
        key = tuple(r['local_indices'])
        if key and key not in lookup:
            lookup[key] = len(pools)
            pools.append(np.array(key, int))
    assert len(pools) == 66 and len(set(np.concatenate(pools))) == 204
    rows_of = np.repeat(np.arange(66), [len(p) for p in pools])
    mat = sparse.csr_matrix((np.concatenate([np.full(len(p), 1 / len(p)) for p in pools]),
                             (rows_of, np.concatenate(pools))), shape=(66, 12000))
    return mat, pools


#: canonical leg order of the sealed runtime (rt.b.LEGS); the MAPPING file groups rows as
#: LF, RF, LM, RM, LH, RH, so the per-leg vectors must NOT follow file order.
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def leg_cells_and_rh():
    rows = js(HANDOVER / 'RUN/MAPPING.json')['rows']
    assert set(r['leg'] for r in rows) == set(LEGS)
    legs = LEGS
    legcells = [np.array(sorted({int(c) for row in rows if row['leg'] == leg
                                 for c in row['local_indices']}), int) for leg in legs]
    rh = np.array(sorted({int(i) for row in rows if row['leg'] == 'RH'
                          for i in row['local_indices']}), int)
    return legs, legcells, rh


# ------------------------------------------------------------------ ridge (interface fit)
# Copied from the previous stage's readout fitting code
# (/experiments/s5-model-correction/20261106T063313Z/pop_readout/static.py, functions
# fold_spec / cv_from_spec / fit_ridge / alpha_scale / ridge_solve), so the information
# metric is the SAME estimator the parent used for 0.43 (204 cells) and 0.86 (12000 cells).
def alpha_scale(Xfit):
    X = np.asarray(Xfit, float)
    return float((X * X).sum() / len(X))


def r2_score(Y, P):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    ss = float(((Y - P) ** 2).sum())
    tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss / tot


def ridge_solve(X, Y, alpha):
    """Ridge with no constant term, dual form (d >> n). Returns W (42 x d)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    A = X @ X.T
    A[np.diag_indices_from(A)] += float(alpha)
    Z = np.linalg.solve(A, Y)
    return (X.T @ Z).T.copy()


def fold_spec(Xfit, folds=CV_FOLDS):
    n = len(Xfit)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    spec = []
    for f in range(folds):
        va = slice(bounds[f], bounds[f + 1])
        tr = np.r_[np.arange(0, bounds[f]), np.arange(bounds[f + 1], n)]
        Xtr = Xfit[tr]
        s, U = np.linalg.eigh(Xtr @ Xtr.T)
        spec.append(dict(va=va, Kva=Xfit[va] @ Xtr.T, U=U, s=s, tr=tr))
    return spec


def cv_from_spec(spec, Yfit, alphas):
    for sp in spec:
        sp['UtY'] = sp['U'].T @ Yfit[sp['tr']]
    curve = []
    for alpha in alphas:
        pred = np.zeros_like(Yfit)
        for sp in spec:
            pred[sp['va']] = sp['Kva'] @ (sp['U'] @ (sp['UtY'] / (sp['s'] + float(alpha))[:, None]))
        curve.append(dict(alpha=float(alpha), cv_r2=float(r2_score(Yfit, pred))))
    return curve


def fit_ridge(Xfit, Yfit, tag, log=print):
    """Scale-free gamma grid, 5 blocked CV folds inside the fit window, final fit on all 600."""
    s = alpha_scale(Xfit)
    alphas = [g * s for g in GAMMAS]
    spec = fold_spec(Xfit)
    curve = cv_from_spec(spec, Yfit, alphas)
    del spec
    best = max(curve, key=lambda g: g['cv_r2'])
    for c in curve:
        c['gamma'] = c['alpha'] / s
    W = ridge_solve(Xfit, Yfit, best['alpha'])
    log('  %-34s s=%.6g gamma=%.3g alpha=%.6g cv_r2=%.4f' %
        (tag, s, best['alpha'] / s, best['alpha'], best['cv_r2']))
    return dict(W=W, alpha=float(best['alpha']), gamma=float(best['alpha'] / s), scale=float(s),
                cv_curve=curve, cv_best=float(best['cv_r2']))


def fit_information(source, teacher, cells=None, tag='', log=print):
    """Fit filtered rates -> teacher torque; report R^2 in the fit and held-out windows.

    Protocol identical to the previous stage's P2/P3: per-cell fit-window mean removed from
    the features, the per-joint fit-window mean torque added back as one fixed offset.
    """
    X = np.asarray(source, float)
    if cells is not None:
        X = X[:, cells]
    Yfit, Yheld = teacher[FIT_LO:FIT_HI], teacher[HELD_LO:HELD_HI]
    Xfit, Xheld = X[FIT_LO:FIT_HI], X[HELD_LO:HELD_HI]
    m = Xfit.mean(0)
    Ybar = Yfit.mean(0)
    f = fit_ridge(Xfit - m, Yfit - Ybar, tag, log)
    W = f['W']
    b = Ybar - (W @ m)
    pred = X @ W.T + b
    return dict(W=W, b=b, mean=m, alpha=f['alpha'], gamma=f['gamma'], scale=f['scale'],
                cv_best=f['cv_best'], cv_curve=f['cv_curve'],
                r2_fit=float(r2_score(Yfit, pred[FIT_LO:FIT_HI])),
                r2_held=float(r2_score(Yheld, pred[HELD_LO:HELD_HI])),
                n_features=int(X.shape[1]), constant_term=True,
                offset_rms=float(np.sqrt((b ** 2).mean())),
                feature_mean_fit_hz=float(m.mean()),
                feature_rms_fit_hz=float(np.sqrt((Xfit ** 2).mean())),
                cells=None if cells is None else np.asarray(cells, int))


# ------------------------------------------------------------------ rate diagnostics
def cell_filter_from_spikes(packed, dt, n_interval_steps=None):
    """Replay the sealed 5 ms per-cell filter from packed 12000-bit words, one row per 1 ms."""
    decay = float(np.exp(-dt / FILTER_TAU))
    n = round(CDT / dt) if n_interval_steps is None else n_interval_steps
    assert len(packed) % n == 0, (len(packed), n)
    f = np.zeros(12000)
    out = np.empty((len(packed) // n, 12000))
    for i, p in enumerate(packed):
        f = decay * f + (1 - decay) * np.unpackbits(p, count=12000) / dt
        if (i + 1) % n == 0:
            out[(i + 1) // n - 1] = f
    return out


def rate_diagnostics(cell, packed, ann, readout_cells):
    """Per-super_class rate regime plus the sparseness descriptors the reading rule uses.

    mean rate of a cell  = mean over the 1200 one-ms interval ends of the 5 ms filtered rate
                           (identical definition to the parent's published 163.8/94.5/120.5)
    temporal CV of a cell = std/mean over the window; the reported motor CV is the MEDIAN over
    the 204 readout cells (this reproduces the parent's 0.72).
    silent                = zero spikes in the 1.2 s window.
    """
    spikes = np.unpackbits(packed, axis=1, count=12000)
    per_cell_hz = cell.mean(0)
    total_spikes = spikes.sum(0)
    sc = ann['super_class']
    per_class = {}
    for cls in sorted(set(sc)):
        m = sc == cls
        v = per_cell_hz[m]
        per_class[cls] = dict(n_cells=int(m.sum()), mean_hz=float(v.mean()),
                              median_hz=float(np.median(v)),
                              p90_hz=float(np.percentile(v, 90)),
                              max_hz=float(v.max()),
                              silent_fraction=float((total_spikes[m] == 0).mean()),
                              mean_spikes=int(total_spikes[m].sum()))
    r204 = per_cell_hz[readout_cells]
    cv = cell[:, readout_cells].std(0) / np.maximum(cell[:, readout_cells].mean(0), 1e-12)
    sil = total_spikes == 0
    desc = dict(
        window_s=1.2, intervals=cell.shape[0],
        all12000=dict(mean_hz=float(per_cell_hz.mean()),
                      median_hz=float(np.median(per_cell_hz)),
                      silent_fraction=float(sil.mean()),
                      hyper_share_above_100hz=float((per_cell_hz > 100).mean()),
                      total_spikes=int(total_spikes.sum())),
        readout_204=dict(mean_hz=float(r204.mean()), median_hz=float(np.median(r204)),
                         max_hz=float(r204.max()),
                         temporal_cv_median=float(np.median(cv)),
                         temporal_cv_mean=float(cv.mean()),
                         silent_fraction=float((total_spikes[readout_cells] == 0).mean()),
                         hyper_share_above_100hz=float((r204 > 100).mean())),
        per_super_class=per_class,
        leg_rates_hz=None)
    return desc


def dense_relief(diag):
    """The pre-registered mechanical statement 'the dense regime is actually relieved'."""
    cond = {
        'A_readout204_mean_at_most_%.1f_Hz' % DENSE_READOUT_MEAN_HZ:
            bool(diag['readout_204']['mean_hz'] <= DENSE_READOUT_MEAN_HZ),
        'B_hyper_share_above_100Hz_at_most_%.2f' % DENSE_HYPER_SHARE:
            bool(diag['all12000']['hyper_share_above_100hz'] <= DENSE_HYPER_SHARE),
        'C_silent_share_at_most_%.2f' % DENSE_SILENT_SHARE:
            bool(diag['all12000']['silent_fraction'] <= DENSE_SILENT_SHARE),
        'D_population_mean_at_least_%.1f_Hz' % DENSE_MIN_MEAN_HZ:
            bool(diag['all12000']['mean_hz'] >= DENSE_MIN_MEAN_HZ),
    }
    return dict(conditions=cond, any_failed=[k for k, v in cond.items() if not v],
                DENSE_RELIEVED=bool(all(cond.values())))
