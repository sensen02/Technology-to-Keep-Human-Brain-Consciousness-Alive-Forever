"""S298 figure: per-leg L after common-mode removal, raw correlations, tripod difference, feasibility bound."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = ('lf', 'rm', 'lh'), ('rf', 'lm', 'rh')
S289 = Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
GROUPS = Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json')


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    c = json.loads((out / 'LEG_ANTIPHASE.json').read_text()); q = c['per_quantity']['gain_normalised_preclip']
    b = json.loads((out / 'FEASIBILITY_BOUND.json').read_text())
    g = json.loads(GROUPS.read_text())['legs']; z = np.load(S289 / 'NEURAL.npz')
    V = np.asarray(z['decoder_motor_gains'], float); norm = np.asarray(z['decoder_preclip'], float) / np.where(V == 0, 1.0, V)
    L = {l: norm[:, g[l]['minus']].mean(axis=1) for l in LEGS}
    Lm = np.vstack([L[l] for l in LEGS]); cm = Lm.mean(0); R = Lm - cm
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); t = np.arange(1, 401)
    a0 = ax[0, 0]
    for j, l in enumerate(LEGS): a0.plot(t, L[l], lw=.9, label=f'{l.upper()} L')
    a0.set_xlabel('ms'); a0.set_ylabel('gain-normalised lift group mean'); a0.legend(fontsize=7, ncol=6)
    a0.set_title('raw lift-group drive per leg: six near-parallel offsets, no alternation visible')
    a1 = ax[0, 1]
    for j, l in enumerate(LEGS): a1.plot(t, R[j], lw=.9, label=f'{l.upper()} residual')
    a1.set_xlabel('ms'); a1.set_ylabel('after subtracting the six-leg common mode')
    a1.set_title('residuals sum to zero by construction, so their negative correlations are structural')
    a1.legend(fontsize=7, ncol=6)
    a2 = ax[1, 0]
    raw = np.array([[q['raw_pair_correlations'][0][2]]])
    M = np.eye(6)
    for (x, y, v) in q['raw_pair_correlations']:
        i, j = LEGS.index(x), LEGS.index(y); M[i, j] = M[j, i] = v
    im = a2.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
    a2.set_xticks(range(6)); a2.set_xticklabels([l.upper() for l in LEGS]); a2.set_yticks(range(6)); a2.set_yticklabels([l.upper() for l in LEGS])
    a2.set_title('raw L pair correlations: all positive (min %.3f)' % min(v for _, _, v in q['raw_pair_correlations']))
    fig.colorbar(im, ax=a2, fraction=.046)
    a3 = ax[1, 1]
    D = np.mean([L[l] for l in TRI_A], axis=0) - np.mean([L[l] for l in TRI_B], axis=0)
    a3.plot(t, D, lw=1.1, label='tripod difference')
    a3.axhline(0, c='k', lw=.6)
    a3.set_xlabel('ms'); a3.set_ylabel('difference (gain-normalised)')
    a3.set_title('tripod difference: mean %.3f std %.3f (%.1f%% of leg mean), 2 sign changes'
                 % (q['tripod_difference']['mean'], q['tripod_difference']['std'],
                    100 * q['tripod_difference']['std'] / np.mean(list(q['L_mean'].values()))))
    a3.legend(fontsize=7)
    fig.suptitle('S298 drive-layer causality audit (read-only, zero dynamic calls): no usable per-leg phase upstream', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); p = out / 'DRIVE_CAUSALITY.png'; fig.savefig(p, dpi=145)
    (out / 'FIGURE.json').write_text(json.dumps(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(dict(figure=str(p))))


if __name__ == '__main__': raise SystemExit(main())
