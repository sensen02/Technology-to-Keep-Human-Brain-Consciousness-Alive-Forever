"""S297 figure: delivered modulation per channel, per-leg projection, support ratio and contact timing."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

S289 = Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
S295 = Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
W = 24.446354755878


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    s289 = np.load(S289 / 'NEURAL.npz'); pre = np.asarray(s289['decoder_preclip'], float); V = np.asarray(s289['decoder_motor_gains'], float)
    nz = np.flatnonzero(V != 0); m = (np.arange(1, 401) >= 20) & (np.arange(1, 401) < 400)
    groups = json.loads((S295 / 'DIRECTION_GROUPS.json').read_text())['legs']
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    a = {arm: np.load(out / arm / 'NEURAL.npz') for arm in ('original_mask', 'decoder_ac_gain')}
    t = {arm: np.load(out / arm / 'TRAJECTORY.npz') for arm in a}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    p0 = ax[0, 0]
    dbase = np.array([a['original_mask']['decoder_ac_applied'][m, i].std() / pre[m, i].std() for i in nz])
    dcand = np.array([a['decoder_ac_gain']['decoder_ac_applied'][m, i].std() / pre[m, i].std() for i in nz])
    p0.scatter(dbase, dcand, s=18)
    p0.plot([1e-4, 1], [1e-4, 1], 'k:', lw=.8)
    p0.set_xscale('log'); p0.set_yscale('log'); p0.set_xlabel('delivered modulation, baseline (std cmd / std preclip)')
    p0.set_ylabel('delivered modulation, AC-gain arm')
    p0.set_title('delivered modulation per channel (median %.4f -> %.4f, x%.2f)' % (np.median(dbase), np.median(dcand), np.median(dcand) / np.median(dbase)))
    p1 = ax[0, 1]
    x = np.arange(6); w = .2
    for k, (arm, alpha) in enumerate((('original_mask', .9), ('decoder_ac_gain', .55))):
        L = [np.asarray(a[arm]['decoder_ac_applied'])[:, groups[l]['minus']].mean(axis=1).std() for l in LEGS]
        P = [np.asarray(a[arm]['decoder_ac_applied'])[:, groups[l]['plus']].mean(axis=1).std() for l in LEGS]
        p1.bar(x + (k - .5) * 2 * w, L, w, alpha=alpha, label=f'{arm} L std')
        p1.bar(x + (k + .5) * 2 * w, P, w, alpha=alpha, label=f'{arm} P std')
    p1.set_xticks(x); p1.set_xticklabels([l.upper() for l in LEGS]); p1.legend(fontsize=6); p1.set_title('per-leg lift/press projection std before vs after')
    p2 = ax[1, 0]
    for arm in a:
        tt = t[arm]['time'] * 1000; r = t[arm]['foot_force_world'][:, :, 2].sum(1) / W
        p2.plot(tt, r, lw=.9, label=arm)
    p2.axhspan(.9, 1.1, color='green', alpha=.12); p2.axhline(1.0, c='k', lw=.6)
    p2.axvline(253.4, c='tab:blue', ls=':', lw=.9); p2.axvline(36.7, c='tab:red', ls=':', lw=.9)
    p2.set_xlabel('ms'); p2.set_ylabel('sum(Fz)/(m g)'); p2.set_ylim(0, 2.0)
    p2.set_title('support ratio: AC gain leaves the 0.9-1.1 band and the body touches down at 36.7 ms instead of 253.4 ms')
    p2.legend(fontsize=7)
    p3 = ax[1, 1]
    for arm in a:
        tt = t[arm]['time'] * 1000; p3.plot(tt, t[arm]['com'][:, 2] - t[arm]['com'][0, 2], lw=.9, label=arm)
    p3.set_xlabel('ms'); p3.set_ylabel('COM dz (native)'); p3.legend(fontsize=7)
    p3.set_title('COM vertical drop is similar, so the damage is in early bearing, not in the final depth')
    fig.suptitle('S297 decoder dynamic-range recalibration (open-loop engineering diagnostic, no support or walking claim)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); p = out / 'DECODER_RANGE.png'; fig.savefig(p, dpi=145)
    (out / 'FIGURE.json').write_text(json.dumps(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(dict(figure=str(p))))


if __name__ == '__main__': raise SystemExit(main())
