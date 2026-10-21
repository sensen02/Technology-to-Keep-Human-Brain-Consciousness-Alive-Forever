"""Targeted extension: required muscle-force multiplier with a much larger exploration cap.

The pose/cone sensitivity run uses a declared cap of 8.0 on the strength multiplier and reports
"cap reached" for the new seated pose and for the v6 seated candidate. This script re-uses the
matrices that run already wrote (no new model solves, no mj_step) and bisects the multiplier up to
CAP_MAX for every cone resolution, so the comparison carries real numbers where they exist.

Read-only on the sensitivity directory except for one added file; relaxation level only.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
KS = (4, 8, 16, 32, 64)
CAP_MAX = 1e4
TAN_HALF_DEN = 10 ** 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ellipse_rays(k):
    pts = set()
    for i in range(k):
        theta = 2 * math.pi * i / k
        if abs(math.sin(theta)) < 1e-12:
            pts.add((1.0 if math.cos(theta) > 0 else -1.0, 0.0))
        elif abs(math.cos(theta)) < 1e-12:
            pts.add((0.0, 1.0 if math.sin(theta) > 0 else -1.0))
        else:
            t = round(math.tan(theta / 2) * TAN_HALF_DEN) / TAN_HALF_DEN
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return sorted(pts)


def cone_block(meta, dirs):
    cols = []
    for c in meta['active_contacts']:
        frame = np.asarray(c['frame'])
        mu = np.asarray(c['mu'])
        jac_t = np.asarray(c['J']).T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + p * t1 + q * t2))
    return np.asarray(cols).T


def feasible(muscle_cols, cone_cols, target, lo, hi, s):
    A = np.hstack([muscle_cols * s, cone_cols])
    scale = np.maximum(1.0, np.maximum(np.abs(target), np.max(np.abs(A), axis=1)))
    res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=target / scale,
                  bounds=list(zip(lo, hi)) + [(0, None)] * (A.shape[1] - len(lo)), method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res.status == 0


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        DESK.glob('S2-pose-cone-sensitivity-*'))[-1]
    report = dict(status='RUNNING', source_dir=str(src), cap_max=CAP_MAX,
                  manual_research_not_product=True, standing=False,
                  scope='required muscle-force multiplier, larger exploration cap; relaxation level only')
    try:
        results = {}
        for label in ('new_seated_tarsus_only', 'v6_seated', 'canonical_declared'):
            mdir = src / ('matrices_' + label)
            if not mdir.exists():
                continue
            meta = json.loads((mdir / 'metadata.json').read_text())
            with np.load(mdir / 'native_diamond.npz') as z:
                A = z['A'].copy()
                target = z['target'].copy()
                lo = z['lo'].copy()
                hi = z['hi'].copy()
            nu = len(lo)
            muscle_cols = A[:, :nu]
            entry = {}
            for k in KS:
                cone = cone_block(meta, ellipse_rays(k))
                f1 = feasible(muscle_cols, cone, target, lo, hi, 1.0)
                low, high = (0.5, 1.0) if f1 else (1.0, 2.0)
                while high <= CAP_MAX and not feasible(muscle_cols, cone, target, lo, hi, high):
                    high *= 1.25
                if high > CAP_MAX:
                    entry[str(k)] = dict(feasible_at_s_1=bool(f1), cap_reached=True,
                                         infeasible_up_to_s=high / 1.25, cap_max=CAP_MAX)
                    continue
                for _ in range(30):
                    mid = 0.5 * (low + high)
                    if feasible(muscle_cols, cone, target, lo, hi, mid):
                        high = mid
                    else:
                        low = mid
                entry[str(k)] = dict(feasible_at_s_1=bool(f1), cap_reached=False,
                                     infeasible_below=low, feasible_above=high,
                                     threshold_midpoint=0.5 * (low + high))
            results[label] = entry
        report['results'] = results
        report['reading'] = (
            'Multiplier s scales the muscle columns (equivalently the activation box). "cap_reached" '
            'means no feasibility was found up to cap_max and NO threshold is claimed. This is a '
            'static relaxation sensitivity, not a strength requirement of a real fly and not evidence '
            'of standing.')
        report['status'] = 'PASS_RAISED_CAP_THRESHOLDS_REPORTED'
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1200:])
    (src / 'threshold-cap-raised.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(src / 'threshold-cap-raised.json')
    print(report['status'])
    if report['status'].startswith('PASS'):
        for label, entry in report['results'].items():
            print('%-24s' % label, {k: (v.get('threshold_midpoint'), v.get('cap_reached')) for k, v in entry.items()})
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
