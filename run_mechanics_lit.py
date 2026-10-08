"""Compare the computed mechanics layer against published quantitative anchors.

Published anchors used (sources in MEASURED_ANCHORS.md):
  - shape index lower bound / solid-fluid threshold: q = P/sqrt(A) ~ 3.72 (hexagon)
    and 3.81 (disordered threshold), measured in vivo for embryonic epithelium.
  - junctional T1 rate in the pupal notum, 12-13.5 hAPF: 8.5e-4 per minute per
    junction (fluctuation-driven, reversible).
  - tissue relaxation time in the pupal thorax: tau ~ 10 s (range 5-20 s).
  - wound closure: notum closes within ~3 h for a ~40 um wound.

The mechanical parameters of this model are ILLUSTRATIVE, so these comparisons
are consistency checks and constraints, not validation.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

ANCHORS = {
    'shape_index_hexagon': 3.72,
    'shape_index_disordered_threshold': 3.81,
    't1_rate_per_min_per_junction': 8.5e-4,
    't1_rate_note': 'pupal notum 12-13.5 hAPF, fluctuation-driven and reversible',
    'relaxation_time_s': 10.0,
    'relaxation_time_range_s': [5.0, 20.0],
    'notum_closure_time_s': 3 * 3600.0,
    'notum_wound_diameter_um': 40.0,
}


def exp_fit(t, y):
    def model(t, a, b, tau):
        return a + b * np.exp(-t / tau)
    span = float(np.ptp(y))
    p0 = [y[-1], y[0] - y[-1], max(1.0, np.ptp(t) / 5)]
    bounds = ([-np.inf, -10 * max(span, 1.0), 1e-3], [np.inf, 10 * max(span, 1.0), 1e5])
    pars, cov = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
    pred = model(t, *pars)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'tau_s': float(pars[2]), 'offset': float(pars[0]), 'amplitude': float(pars[1]),
            'r_squared': float(1 - ss_res / ss_tot) if ss_tot > 0 else None}


def main():
    import mechanics

    metrics = {'anchors': ANCHORS,
               'scope': ('consistency comparison against published values from a different '
                         'stage and, where noted, a different tissue; mechanical parameters '
                         'here are illustrative')}

    print('relaxed control run (no injury) for shape statistics ...')
    relaxed = mechanics.run(n_side=24, mode='cut', cut_halfwidth=0.0, t_end=20.0,
                            sample_dt=5.0, lambda_cut=1.0, log_level=0)
    A = relaxed['areas'][-1]
    P = relaxed['perimeters'][-1]
    alive = relaxed['cell_alive'][-1]
    q = P[alive] / np.sqrt(A[alive])
    metrics['shape_index'] = {
        'mean': float(q.mean()), 'median': float(np.median(q)), 'sd': float(q.std()),
        'min': float(q.min()), 'max': float(q.max()),
        'published_hexagon': ANCHORS['shape_index_hexagon'],
        'published_disordered_threshold': ANCHORS['shape_index_disordered_threshold'],
        'difference_from_hexagon': float(q.mean() - ANCHORS['shape_index_hexagon']),
        'difference_from_threshold': float(q.mean() - ANCHORS['shape_index_disordered_threshold']),
        'note': ('A regular hexagonal lattice must sit near 3.72; this is a geometry check. '
                 'It does not test tissue fluidity.'),
    }
    metrics['area_stats'] = {'mean_um2': float(A[alive].mean()), 'sd_um2': float(A[alive].std())}

    mech = dict(np.load(OUT / 'mechanics_cut_n24.npz', allow_pickle=False))
    t = np.asarray(mech['times'], dtype=float)
    E = np.asarray(mech['energy'], dtype=float)
    cut = np.asarray(mech['cut_length'], dtype=float)
    metrics['relaxation_energy'] = exp_fit(t, E)
    metrics['relaxation_cut_length'] = exp_fit(t, cut)
    tau = metrics['relaxation_energy']['tau_s']
    metrics['relaxation_vs_published'] = {
        'model_tau_s': tau,
        'published_tau_s': ANCHORS['relaxation_time_s'],
        'published_range_s': ANCHORS['relaxation_time_range_s'],
        'ratio_model_over_published': tau / ANCHORS['relaxation_time_s'],
        'inside_published_range': bool(ANCHORS['relaxation_time_range_s'][0] <= tau
                                       <= ANCHORS['relaxation_time_range_s'][1]),
        'note': ('Because the mechanical parameters are illustrative, a mismatch is a '
                 'calibration statement: matching tau would fix the ratio of the '
                 'viscosity-like coefficient to the elastic coefficients.'),
    }

    events = json.loads(str(mech['events_json']))
    n_junctions = 3 * int(alive.sum()) / 2.0
    duration_min = float(t[-1]) / 60.0
    t1_events = len(events.get('t1', []))
    metrics['t1_rate'] = {
        't1_events_in_run': t1_events,
        'run_duration_s': float(t[-1]),
        'approx_junctions': n_junctions,
        'model_rate_per_min_per_junction': (t1_events / n_junctions / duration_min
                                            if duration_min > 0 else None),
        'published_rate_per_min_per_junction': ANCHORS['t1_rate_per_min_per_junction'],
        'published_note': ANCHORS['t1_rate_note'],
        'interpretation': ('The model has no stochastic junctional line tension, so it '
                           'cannot produce the fluctuation-driven reversible T1s seen in '
                           'the notum. The missing mechanism is identified, not fitted.'),
    }

    closure = {'model_cut_length_initial_um': float(cut[0]),
               'model_cut_length_final_um': float(cut[-1]),
               'model_fraction_closed': float(1 - cut[-1] / cut[0]),
               'model_elapsed_s': float(t[-1]),
               'notum_closure_time_s': ANCHORS['notum_closure_time_s'],
               'notum_wound_diameter_um': ANCHORS['notum_wound_diameter_um'],
               'geometry_mismatch_note': ('The simulated cut is a long slit (initial free '
                                          'boundary {:.0f} um) whereas the published wound is '
                                          'about 40 um across, so closure times are not '
                                          'directly comparable.'
                                          ).format(float(cut[0]))}
    metrics['closure'] = closure

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    ax = axs[0]
    ax.hist(q, bins=25, color='steelblue')
    ax.axvline(ANCHORS['shape_index_hexagon'], color='k', ls='--', label='hexagon 3.72')
    ax.axvline(ANCHORS['shape_index_disordered_threshold'], color='r', ls=':',
               label='published threshold 3.81')
    ax.set(xlabel='Shape index P/sqrt(A)', ylabel='Cells',
           title=f'Relaxed lattice, mean {q.mean():.3f}')
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(t, E, '.', ms=3, label='energy')
    tt = np.linspace(t[0], t[-1], 200)
    r = metrics['relaxation_energy']
    ax.plot(tt, r['offset'] + r['amplitude'] * np.exp(-tt / r['tau_s']),
            label=f"fit tau={r['tau_s']:.2f} s (R2={r['r_squared']:.3f})")
    ax.set(xlabel='Time (s)', ylabel='Energy (illustrative units)',
           title=f"Published tau ~ {ANCHORS['relaxation_time_s']:.0f} s (notum, 18-26 hAPF)")
    ax.legend(fontsize=8)
    ax = axs[2]
    ax.plot(t, cut)
    ax.set(xlabel='Time (s)', ylabel='Free cut boundary (um)',
           title=f"Cut closes {100 * closure['model_fraction_closed']:.0f}% in {t[-1]:.0f} s")
    fig.suptitle('Computed mechanics vs published anchors: consistency checks, not validation\n'
                 'Mechanical parameters are illustrative, so a mismatch is a calibration statement',
                 fontsize=11)
    fig.savefig(OUT / 'mechanics_literature.png', dpi=160)
    plt.close(fig)

    (OUT / 'metrics_mechanics_lit.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
