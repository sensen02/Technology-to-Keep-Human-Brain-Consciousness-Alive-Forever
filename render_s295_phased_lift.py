"""S295 figure: differential lift per arm, support band, and a representative load-transfer trace."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
W = 24.446354755878


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    ph = json.loads((out / 'PHASED.json').read_text()); der = json.loads((out / 'DIFFERENTIAL.json').read_text())['vs_passive_same_index']
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ids = [k for k in der if k != 'passive_zero']
    ids.sort(key=lambda k: -der[k]['differential_lift_max'])
    a = ax[0, 0]
    colors = ['tab:red' if der[k]['differential_lift_max'] >= 1 else 'tab:gray' for k in ids]
    a.bar(range(len(ids)), [der[k]['differential_lift_max'] for k in ids], color=colors)
    a.axhline(1.0, c='k', ls='--', lw=.9); a.axhline(.366, c='tab:blue', ls=':', lw=.8); a.axhline(.488, c='tab:green', ls=':', lw=.8)
    a.set_xticks(range(len(ids))); a.set_xticklabels(ids, rotation=60, ha='right', fontsize=7)
    a.set_ylabel('differential lift in hold phase (native)')
    a.set_title('differential lift vs passive (1.0 threshold; blue/green = S294 static best target/any)')
    b = ax[0, 1]
    band = [der[k]['support_band_fraction'] for k in ids]
    b.bar(range(len(ids)), band, color='tab:purple')
    b.axhline(.8, c='k', ls='--', lw=.9); b.set_ylim(0, 1.05)
    b.set_xticks(range(len(ids))); b.set_xticklabels(ids, rotation=60, ha='right', fontsize=7)
    b.set_ylabel('fraction of hold samples with support ratio in [0.9, 1.2]')
    b.set_title('support retention during the hold phase (0.8 bar)')
    c = ax[1, 0]
    passive = np.load(out / 'runs' / 'passive_zero.npz'); t = np.arange(1200) * .1
    for rid, style in (('lf_minus', '-'), ('lf_plus', '--')):
        z = np.load(out / 'runs' / (rid + '.npz'))
        c.plot(t, z['foot_z'][:, 0] - passive['foot_z'][:, 0], style, lw=1.2, label=rid + ' LF foot minus passive')
    c.axvspan(20, 40, color='gray', alpha=.15); c.axvspan(40, 100, color='orange', alpha=.12); c.axvspan(100, 120, color='gray', alpha=.15)
    c.set_xlabel('ms'); c.set_ylabel('relative LF foot height (native)')
    c.set_title('phases: rest 0-20, ramp 20-40, hold 40-100, release 100-120'); c.legend(fontsize=7)
    d = ax[1, 1]
    z = np.load(out / 'runs' / 'lf_minus.npz')
    d.plot(t, z['fz'][:, 0], label='LF (target) foot Fz')
    d.plot(t, z['fz'].sum(axis=1) - z['fz'][:, 0], label='other five legs Fz')
    d.axhline(W, c='k', ls=':', lw=.8, label='body weight')
    d.axvspan(40, 100, color='orange', alpha=.12)
    d.set_xlabel('ms'); d.set_ylabel('vertical contact force (native)')
    d.set_title('lf_minus load transfer: target unloads, other legs carry the body')
    d.legend(fontsize=7)
    fig.suptitle('S295 phased open-loop lift probe: diagnostic only, not neural control, not gait, no support or walking claim', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); p = out / 'PHASED_LIFT.png'; fig.savefig(p, dpi=150)
    (out / 'FIGURE.json').write_text(json.dumps(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(dict(figure=str(p))))


if __name__ == '__main__': raise SystemExit(main())
