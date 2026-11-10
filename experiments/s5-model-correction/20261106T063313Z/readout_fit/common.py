"""Shared constants and read-only loaders for the S5 readout-correction experiment.

Scope: replace ONLY the neural-rate -> joint-torque readout interface with an
offline least-squares linear readout, and measure whether the SAME sealed neural
warm states can physically support the body. The neurons still generate the
command; the readout is an interface that is fitted to the teacher torque offline.

Nothing here writes into a sealed directory.
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
O = RT1 / 'readout_fit'                                        # this task output dir

# sealed read-only inputs
CONFIGS = RT1 / 'numerics/CONFIGS'          # WARM.npz / WARM_CACHE.pkl per neural dt
SEALED_RUNS = RT1 / 'numerics/RUNS'         # sealed R0 (current readout) releases
NUMERICS_SUMMARY = RT1 / 'numerics/SUMMARY.json'
IDENTITIES = RT1 / 'sensory/CANDIDATES/TRANSLATOR_IDENTITIES.json'
TEACHER = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z/DATA/true_teacher')

# ---------------------------------------------------------------- experiment
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
DTS = [(0.0005, 'dt_0p5ms'), (0.0000625, 'dt_0p0625ms')]
TAGS = ['R0', 'R1', 'R2']
CDT = 0.001                 # 1 ms causal command clock
DURATION = 0.2              # release duration (s)
RH_GAIN = 0.375             # applied exactly once at release
FIT_LO, FIT_HI = 0, 600     # 0.2-0.8 s of the teacher window, in 1 ms intervals
HELD_LO, HELD_HI = 600, 1200  # 0.8-1.4 s, held out

# Teacher torque sampling: the teacher BODY.npz is stored at 0.5 ms (2801 samples,
# 0..1.4 s). Command interval k covers [0.2+k*1ms, 0.2+(k+1)*1ms); its source
# rates are causal (end of the previous interval). The matched teacher sample is
# therefore the one AT the interval start: 0.5 ms index 400+2k.
# This convention reproduces the parent's published teacher RMS 3.022 (full
# window) and 3.003 (held-out window) exactly.
TEACHER_STRIDE = 2
TEACHER_START = 400


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
    # the stored grid accumulates float error (~1.6e-12), so the guard is at 1e-9
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


def warm_rates(tag):
    """Pool rates (1200x66) of the sealed warm cache of one neural timestep."""
    z = arrays(CONFIGS / tag / 'WARM.npz')
    source = z['command_source_rates']
    assert source.shape == (1200, 66)
    return z, source


def sealed_label(tag, phase):
    return f'{tag}_paired_{phase}'


def r2_score(Y, P):
    ss = float(((Y - P) ** 2).sum())
    tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss / tot


def fit_readout(source, target, lo, hi):
    """Ordinary least squares with a per-joint bias. Returns W (42x66), b (42,)."""
    X = source[lo:hi]
    Y = target[lo:hi]
    G = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(G, Y, rcond=None)
    return coef[:66].T.copy(), coef[66].copy()


def readout_command(W, b, source, limits, scale=1.0):
    """Declared readout: clip(scale * (W @ rates + b), +/- limits)."""
    raw = scale * (source @ W.T + b)
    return raw, np.clip(raw, -limits, limits)
