"""S296 figure: per-leg lift/press drive projection, thresholds, ceilings, tripod difference."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = ('lf', 'rm', 'lh'), ('rf', 'lm', 'rh')


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    res = json.loads((out / 'DRIVE_PROJECTION.json').read_text())
    pl = res['arms']['s292_original_mask']['per_leg']; t = np.arange(1, 401)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 11))
    ax0 = fig.add_subplot(3, 1, 1)
    for leg in LEGS:
        L = np.asarray(pl[leg]['L']['series']); P = np.asarray(pl[leg]['P']['series'])
        ax0.plot(t, L, lw=1.1, label=f'L {leg.upper()} (lift)')
        ax0.plot(t, P, lw=.8, ls='--', alpha=.55)
    for thr in (.3, .5, .8, 1.0):
        ax0.axhline(thr, lw=.7, ls=':', c='k')
        ax0.text(402, thr, str(thr), fontsize=7, va='center')
    ax0.set_xlim(0, 400); ax0.set_ylim(0, 1.05); ax0.set_xlabel('ms')
    ax0.set_ylabel('group-mean command (1.0 = S295 lift-arm level)')
    ax0.set_title('S292 baseline: lift projection L (solid) sits at a near-constant offset, press projection P (dashed); thresholds 0.3/0.5/0.8/1.0')
    ax0.legend(fontsize=7, ncol=6)
    ax1 = fig.add_subplot(3, 1, 2)
    ceil = [pl[l]['L']['ceiling'] for l in LEGS]; mx = [pl[l]['L']['max'] for l in LEGS]; mean = [pl[l]['L']['mean'] for l in LEGS]
    x = np.arange(6); w = .27
    ax1.bar(x - w, mean, w, label='L mean')
    ax1.bar(x, mx, w, label='L max')
    ax1.bar(x + w, ceil, w, label='structural ceiling (masked members removed)')
    ax1.axhline(.8, c='r', ls='--', lw=.9); ax1.axhline(1.0, c='k', ls=':', lw=.9)
    ax1.set_xticks(x); ax1.set_xticklabels([l.upper() for l in LEGS]); ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('fraction'); ax1.legend(fontsize=7)
    ax1.set_title('cellings below 0.8 for lm/lh/rm/rh: four legs cannot reach the 0.8 threshold even at full activation of their active members')
    ax2 = fig.add_subplot(3, 1, 3)
    A = np.mean([np.asarray(pl[l]['L']['series']) for l in TRI_A], axis=0)
    B = np.mean([np.asarray(pl[l]['L']['series']) for l in TRI_B], axis=0)
    ax2.plot(t, A - B, lw=1.2, label='tripod difference (LF+RM+LH) - (RF+LM+RH)')
    ax2.axhline(0, c='k', lw=.6); ax2.axhspan(-.05, .05, color='gray', alpha=.15)
    ax2.set_xlim(0, 400); ax2.set_xlabel('ms'); ax2.set_ylabel('difference')
    ax2.set_title('tripod difference is a near-constant negative offset with small ripple; no large-amplitude anti-phase alternation')
    ax2.legend(fontsize=7)
    fig.suptitle('S296 drive time-structure audit (read-only, zero dynamic calls): drive is essentially constant, not rhythmic', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); p = out / 'DRIVE_STRUCTURE.png'; fig.savefig(p, dpi=145)
    (out / 'FIGURE.json').write_text(json.dumps(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(dict(figure=str(p))))


if __name__ == '__main__': raise SystemExit(main())
