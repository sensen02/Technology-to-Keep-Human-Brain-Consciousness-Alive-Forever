"""Algebraic muscle-force scale requirement; no model, forward, optimization or integration."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import lsq_linear
S = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-phase-capacity/20261101T193630Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-force-scale-requirement/20261101T201309Z')
FLOOR = 0.0001
SCALES = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 64.0]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    z = np.load(S / 'STATIC_ARRAYS.npz')
    A = z['A'][:, 6:, :]
    passive = z['passive'][:, 6:]
    target = z['target'][:, 6:]
    times = z['times']
    groups = z['group_matrix']
    G = groups if groups.shape[1] == 66 else None
    rows = []
    for k in range(A.shape[0]):
        base = passive[k] + A[k] @ np.full(90, FLOOR)
        den = float(np.linalg.norm(target[k]))
        entry = dict(time_s=float(times[k]), target_norm=den, scales={})
        for scale in SCALES:
            Bm = A[k] @ G * scale if G is not None else A[k] * scale
            y = target[k] - base
            fit = lsq_linear(Bm, y, bounds=(0., 1. - FLOOR), method='bvls', tol=1e-12, max_iter=2000)
            resid = Bm @ fit.x - y
            entry['scales'][str(scale)] = dict(
                relative_residual=float(np.linalg.norm(resid) / den),
                saturated=int((fit.x > 1. - FLOOR - 1e-9).sum()),
                success=bool(fit.success),
            )
        rows.append(entry)
    summary = dict(
        scope='Bounded linear algebra on saved 3470 unit-activation torque matrices; pooled 66 groups',
        activation_box=[FLOOR, 1.0],
        meaning='Scale multiplies every muscle torque uniformly. It is a hypothetical plant strength change, not a calibrated physiological value and not a claim that stronger muscles suffice.',
        phases=rows,
        interpretation=[
            'A decreasing residual with scale shows the frozen torque targets are amplitude-limited at this posture.',
            'A residual that stays high even at large scale shows a direction/architecture limit instead.',
            'Neither outcome licenses editing the muscle model without an independent source for a new force scale.',
        ],
    )
    (OUT / 'SCALE_SWEEP.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for r in rows:
        print(round(r['time_s'], 3), [round(r['scales'][str(s)]['relative_residual'], 4) for s in SCALES])


if __name__ == '__main__':
    main()
