"""Shared constants and read-only loaders for the WHOLE-POPULATION readout test (S5).

Question: can a readout that uses the WHOLE neural population (all 12000 filtered
cell rates) replace the 204-cell / 66-pool readout, so that the SAME sealed neural
warm states physically support the body?

Scope: this is OFFLINE ENGINEERING FITTING of the readout interface, not biological
calibration.  The neurons still generate the command; no non-neural generator
produces any motion.  Nothing here writes into a sealed directory.
"""
import json
import hashlib
import pickle
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- workspace
R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'   # runtime + translator
SRC = Path(__file__).resolve().parent                           # frozen source dir
RT1 = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z')
O = RT1 / 'pop_readout'                                        # this task output dir

# sealed read-only inputs
CONFIGS = RT1 / 'numerics/CONFIGS'                 # WARM.npz / WARM_CACHE.pkl per neural dt
SEALED_RUNS = RT1 / 'numerics/RUNS'                # sealed R0 (current readout) releases
NUMERICS_SUMMARY = RT1 / 'numerics/SUMMARY.json'
IDENTITIES = RT1 / 'sensory/CANDIDATES/TRANSLATOR_IDENTITIES.json'
PRIOR = RT1 / 'readout_fit'                        # previous stage: structure + release loop
TEACHER = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z/DATA/true_teacher')

# cell annotation (the task's required join key is banc_888_id, NOT root_id)
META = R / 'data/banc888/banc_888_meta.feather'
B888_TO_ROOT = R / 'data/banc888/b888_to_root.npy'
HANDOVER = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
IDENTITIES_NPZ = HANDOVER / 'IDENTITIES.npz'       # 12000 root ids in tier (local-index) order
MAPPING = HANDOVER / 'RUN/MAPPING.json'

# ---------------------------------------------------------------- experiment
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
DTS = [(0.0005, 'dt_0p5ms'), (0.0000625, 'dt_0p0625ms')]
TAGS = ['R0', 'P1', 'P2', 'P3']       # R0 = sealed control; P1/P2 whole population; P3 204 cells
CDT = 0.001                 # 1 ms causal command clock
DURATION = 0.2              # release duration (s)
RH_GAIN = 0.375             # applied exactly once at release
FIT_LO, FIT_HI = 0, 600     # 0.2-0.8 s of the teacher window, in 1 ms intervals
HELD_LO, HELD_HI = 600, 1200  # 0.8-1.4 s, held out
TREND_MS = 100              # task's trend-removal length, used only for the interface diagnostic
# alpha is parameterised SCALE-FREE: alpha = gamma * s, with s = mean over fit-window
# samples of the squared feature norm (i.e. trace(X X^T)/n, the mean eigenvalue scale of
# the ridge kernel).  Declared before the fit so the grid covers the whole range from
# 'no regularisation' (gamma << 1) to 'prediction shrunk to the constant' (gamma >> 1).
GAMMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
CV_FOLDS = 5                # contiguous blocked folds inside the FIT window only

# Teacher torque sampling: the teacher BODY.npz is stored at 0.5 ms (2801 samples,
# 0..1.4 s).  Command interval k covers [0.2+k*1ms, 0.2+(k+1)*1ms); its source
# rates are causal (state at the interval START).  The matched teacher sample is
# therefore 0.5 ms index 400+2k.  This reproduces the parent's published teacher
# RMS 3.022 (full window) and 3.003 (held-out window) exactly.
TEACHER_STRIDE = 2
TEACHER_START = 400
# index of the warm state at release phase ph inside WARM.npz cell_filtered_rates
def phase_index(ph):
    i = round((ph - 0.2) / CDT) - 1
    assert abs(0.2 + (i + 1) * CDT - ph) < 1e-12, (ph, i)
    return i


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def js(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False,
                                     default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def decoder_identity():
    ident = js(IDENTITIES)
    matrix = np.array(ident['old_matrix'], float)
    limits = np.asarray(ident['limits'], float)
    assert matrix.shape == (42, 66) and limits.shape == (42,)
    return matrix, limits


def motor_cells_204():
    """The 204 cells the current readout actually consumes (union of the 66 pools)."""
    cells = np.asarray(js(IDENTITIES)['cells'], int)
    assert cells.shape == (204,) and len(set(cells.tolist())) == 204
    return cells


def warm(tag):
    """Sealed warm cache trace + the interval-start 5 ms-filtered per-cell rate features."""
    z = arrays(CONFIGS / tag / 'WARM.npz')
    cell = z['cell_filtered_rates']
    assert cell.shape == (1200, 12000)
    # The sealed causal convention: the command of interval k uses the state at the
    # START of interval k, i.e. the end-of-interval state of k-1 (zero before 0.2 s),
    # exactly as command_source_rates = vstack([zeros, rr[:-1]]).
    source = np.vstack([np.zeros((1, 12000)), cell[:-1]])
    return dict(z=z, cell=cell, source=source, pool_source=z['command_source_rates'])


def alpha_scale(Xfit):
    """alpha = gamma * alpha_scale(Xfit)."""
    X = np.asarray(Xfit, float)
    return float((X * X).sum() / len(X))


def r2_score(Y, P):
    ss = float(((Y - P) ** 2).sum())
    tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss / tot


def ridge_solve(X, Y, alpha):
    """Ridge with NO constant term, dual form (d >> n).  Returns W (42 x d), b = 0.

    min_beta ||Y - X beta||^2 + alpha ||beta||^2
    beta = X^T (X X^T + alpha I)^-1 Y      (n x n solve, n = number of intervals)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    A = X @ X.T
    A[np.diag_indices_from(A)] += float(alpha)
    Z = np.linalg.solve(A, Y)          # (n, 42)
    return (X.T @ Z).T.copy(), np.zeros(Y.shape[1])


def ridge_predict(W, b, X):
    return np.asarray(X, float) @ W.T + b


def scale_free_stats(raw, clipped, limits, target, lo, hi):
    """Interface description of one readout command against the teacher torque."""
    def safe_corr(a, b):
        a = np.asarray(a, float) - np.mean(a)
        b = np.asarray(b, float) - np.mean(b)
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return 0.0 if d == 0.0 else float((a * b).sum() / d)

    sat = np.abs(np.abs(clipped) - limits[None, :]) < 1e-12
    per_joint_r = [safe_corr(clipped[:, j], target[:, j]) for j in range(clipped.shape[1])]
    return dict(
        overall_correlation_with_teacher=safe_corr(clipped.ravel(), target.ravel()),
        per_joint_correlation_median=float(np.median(per_joint_r)),
        per_joint_correlation_min=float(np.min(per_joint_r)),
        per_joint_correlation_max=float(np.max(per_joint_r)),
        per_joint_above_0p5=int(np.sum(np.asarray(per_joint_r) > 0.5)),
        per_joint_correlations=[float(v) for v in per_joint_r],
        sign_agreement=float(np.mean(np.sign(clipped) == np.sign(target))),
        command_rms=float(np.sqrt((clipped ** 2).mean())),
        teacher_rms=float(np.sqrt((target ** 2).mean())),
        command_rms_over_teacher_rms=float(np.sqrt((clipped ** 2).mean())
                                           / np.sqrt((target ** 2).mean())),
        unclipped_command_rms=float(np.sqrt((raw ** 2).mean())),
        unclipped_absmax_over_limit_median=float(np.median(
            [np.abs(raw[:, j]).max() / limits[j] for j in range(len(limits))])),
        saturation_fraction_overall=float(sat.mean()),
        saturation_fraction_fit_window=float(sat[lo:hi].mean()),
        saturation_fraction_held_window=float(sat[hi:].mean()),
        joints_with_any_saturation=int(np.sum(sat.any(0))),
        r2_fit_window=float(r2_score(target[lo:hi], raw[lo:hi])),
        r2_held_window=float(r2_score(target[hi:], raw[hi:])),
        r2_held_window_after_clipping=float(r2_score(target[hi:], clipped[hi:])),
        r2_held_window_if_prediction_were_only_the_fit_window_mean=float(
            r2_score(target[hi:], np.repeat(target[lo:hi].mean(0)[None, :], len(target[hi:]), 0))),
        r2_fit_window_if_prediction_were_only_the_fit_window_mean=float(
            r2_score(target[lo:hi], np.repeat(target[lo:hi].mean(0)[None, :], hi - lo, 0))),
    )
