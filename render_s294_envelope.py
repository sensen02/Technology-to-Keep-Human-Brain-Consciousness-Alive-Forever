"""S294 figure: baseline support ratio, passive control, and the declared probe envelope."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
WEIGHT = 24.446354755878


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    env = json.loads((out / 'ENVELOPE.json').read_text()); der = json.loads((out / 'DERIVED_LIFT.json').read_text())['vs_zero_control']
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(S292 / 'TRAJECTORY.npz'); t = z['time'] * 1000; ratio = z['foot_force_world'][:, :, 2].sum(axis=1) / WEIGHT
    z9 = np.load(S289 / 'TRAJECTORY.npz'); r9 = z9['foot_force_world'][:, :, 2].sum(axis=1) / WEIGHT
    zero = np.load(out / 'runs' / 'zero_control.npz'); tz = np.arange(500) * .1
    zr = zero['total'] / WEIGHT
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    a.plot(t, ratio, lw=.9, label='S292 current-sample baseline (identical to S289 decoder_original)')
    a.plot(np.arange(len(r9)) / 10, r9, lw=.6, ls='--', alpha=.7, label='S289 decoder_original')
    a.axhline(1.0, c='k', lw=.8); a.axvline(253.4, c='r', ls=':', lw=.8)
    a.annotate('first non-foot contact 253.4 ms', xy=(253.4, .2), xytext=(160, .05), fontsize=8,
               arrowprops=dict(arrowstyle='->', lw=.7))
    a.set_xlabel('ms'); a.set_ylabel('sum(Fz) / (m g)'); a.set_title('baseline support ratio: about 1.0 and rising COM until the body touches down')
    a.legend(fontsize=7); a.set_ylim(-.05, 2.2)
    b = ax[0, 1]
    b.plot(tz, zr, lw=.9, label='zero-command probe')
    b.plot(tz, zero['com'][:, 2] / zero['com'][0, 2], lw=.9, label='COM z (normalised)')
    b.axhline(1.0, c='k', lw=.8); b.set_xlabel('ms'); b.set_title('passive control: no command, body falls 0.72 in 50 ms with impact spikes')
    b.legend(fontsize=7)
    c = ax[1, 0]
    kinds = {'press': 'o', 'lift': 's', 'all15': '^', 'all90': '*', 'zero': 'x'}
    for k, marker in kinds.items():
        xs = [env['runs'][i]['support_ratio']['mean'] for i in der if env['runs'][i]['kind'] == k]
        ys = [der[i]['peak'] for i in der if env['runs'][i]['kind'] == k]
        if xs: c.scatter(xs, ys, marker=marker, s=32, label=k)
    c.axhline(1.0, c='r', ls='--', lw=.8); c.axhline(0.0, c='k', lw=.6)
    c.set_xlabel('mean support ratio over 50 ms'); c.set_ylabel('commanded-leg lift above passive control (native)')
    c.set_title('no declared constant command lifts the target leg to 1.0; best differential %.3f' % max(v['peak'] for v in der.values()))
    c.legend(fontsize=7)
    d = ax[1, 1]
    ids = sorted(env['runs'], key=lambda i: env['runs'][i]['com']['dz'])
    d.bar(range(len(ids)), [env['runs'][i]['com']['dz'] for i in ids], width=.8)
    d.set_xlabel('run (sorted)'); d.set_ylabel('COM dz over 50 ms (native)')
    d.set_title('every declared run sinks the body (baseline in its first 50 ms rises by +0.056)')
    fig.suptitle('S294 open-loop capability envelope: diagnostic only, not neural control, not support or walking evidence', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); p = out / 'ENVELOPE.png'; fig.savefig(p, dpi=150)
    (out / 'FIGURE.json').write_text(json.dumps(dict(path=str(p), sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(dict(figure=str(p), best_diff=max(v['peak'] for v in der.values()))))


if __name__ == '__main__': raise SystemExit(main())
