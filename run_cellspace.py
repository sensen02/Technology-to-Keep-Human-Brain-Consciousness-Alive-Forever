"""Run the intracellular spatial layer and quantify the lumped-model error.

Single-cell / subcellular scope only. No experimental dataset in this project
constrains intracellular calcium gradients, so this layer is a numerical study
of whether spatial resolution changes the observables, not a validated model.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def main():
    import cellspace

    metrics = {'scope': ('single-cell spatial reaction-diffusion; no experimental data in this '
                         'project constrains intracellular gradients')}

    print('running cellspace.self_test() ...')
    try:
        metrics['self_test'] = cellspace.self_test()
    except Exception as exc:
        metrics['self_test'] = f'{type(exc).__name__}: {exc}'

    print('spatial vs lumped at three diffusion coefficients ...')
    comparison = {}
    for d_ca in (10.0, 100.0, 1e5):
        try:
            r = cellspace.compare_with_lumped(mode='radial1d', n_shells=64, t_end=5.0,
                                              tear=0.5, D_ca=d_ca)
            if isinstance(r, dict):
                # Nested run dicts carry large arrays; keep scalars only.
                r = {k: v for k, v in r.items()
                     if isinstance(v, (int, float, str, bool, type(None)))
                     or (isinstance(v, (list, tuple)) and len(v) <= 8)}
            comparison[f'D_ca_{d_ca:g}'] = r
        except Exception as exc:
            comparison[f'D_ca_{d_ca:g}'] = f'{type(exc).__name__}: {exc}'
    metrics['spatial_vs_lumped'] = comparison

    print('realistic stimulus run ...')
    spatial = cellspace.run(mode='radial1d', n_shells=64, t_end=5.0, tear=0.5)
    lumped = cellspace.run_lumped(t_end=5.0, tear=0.5)
    t = np.asarray(spatial['times'])
    mean_c = np.asarray(spatial['mean_c'])
    lump_c = np.asarray(lumped['mean_c'] if 'mean_c' in lumped else lumped['c']).ravel()
    rel = np.abs(mean_c - lump_c) / np.maximum(np.abs(lumped['c']).max(), 1e-12)
    metrics['mean_vs_lumped'] = {'max_abs_diff_uM': float(np.max(np.abs(mean_c - lump_c))),
                                 'max_rel_diff_vs_peak': float(np.max(rel))}
    fields = np.asarray(spatial['c'])
    grad = (fields.max(axis=1) - fields.min(axis=1)) / np.maximum(np.abs(fields.mean(axis=1)), 1e-12)
    metrics['gradient'] = {'max_relative_spread': float(np.nanmax(grad)),
                           'time_of_max_s': float(t[int(np.nanargmax(grad))])}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    ax = axs[0]
    ax.plot(t, mean_c, label='spatial, volume-weighted mean')
    ax.plot(np.asarray(lumped['times']), lump_c, '--', label='well-mixed ODE')
    ax.set(xlabel='Time (s)', ylabel='Cytosolic Ca (uM)',
           title='Spatial vs lumped, same stimulus and parameters')
    ax.legend(fontsize=8)
    ax = axs[1]
    for frac in (0.1, 0.5, 1.0):
        i = int(np.argmin(np.abs(t - t[-1] * frac)))
        ax.plot(np.asarray(spatial['r_um']), fields[i], label=f't={t[i]:.2f} s')
    ax.set(xlabel='Radius (um)', ylabel='Cytosolic Ca (uM)',
           title='Radial profiles: gradient size is the quantity that matters')
    ax.legend(fontsize=8)
    ax = axs[2]
    ax.plot(t, grad)
    ax.set(xlabel='Time (s)', ylabel='(max-min)/mean over shells',
           title=f"Max relative spread {np.nanmax(grad):.3g}")
    fig.suptitle('Intracellular spatial layer: does resolving the cytosol change the answer?\n'
                 'Single-cell scope; no experimental constraint on intracellular gradients', fontsize=11)
    fig.savefig(OUT / 'cellspace_validation.png', dpi=160)
    plt.close(fig)

    # The convergence order and the large-D lumped limit were previously only
    # reachable by calling cellspace.test_2_/test_4_ by hand, which is why the
    # published numbers (L-infinity order 1.92/1.96; D -> infinity residual 0.50%)
    # had NO stored artefact and could only be reproduced interactively.  They are
    # now part of the stored result.
    print('running the full seven-test numerical suite ...')
    try:
        suite = cellspace.full_test_suite()
        metrics['full_suite'] = suite
        order = suite.get('test_2_analytic_gaussian_diffusion', {}).get('orders') \
            if isinstance(suite.get('test_2_analytic_gaussian_diffusion'), dict) else None
        lumped = suite.get('test_4_lumped_large_D_limit')
        print('  convergence order entries :', json.dumps(order, default=str)[:200])
        print('  large-D lumped limit      :', json.dumps(lumped, default=str)[:300])
    except Exception as exc:
        metrics['full_suite'] = '%s: %s' % (type(exc).__name__, exc)
        print('  suite failed:', exc)

    (OUT / 'metrics_cellspace.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'self_test'}, indent=2, default=str)[:2500])


if __name__ == '__main__':
    main()
