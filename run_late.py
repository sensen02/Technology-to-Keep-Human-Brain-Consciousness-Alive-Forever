"""Late-phase (delayed ligand/receptor) comparison against the experimental trace.

The ligand parameters come from the same publication whose experiment is used for
comparison, so any agreement is retrospective, not independent validation.
No parameter was tuned to improve the fit.
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
    import model_late, measurement

    exp = model_late.experimental_control_radius()
    metrics = {'scope': 'retrospective comparison; ligand parameters fitted in the same context'}

    print('cross-checking early phase against model.run ...')
    metrics['early_crosscheck'] = model_late.cross_check_early()

    print('running late phase at n_side=45 (2025 cells) ...')
    res = model_late.run(n_side=45, t_end=300.0, sample_dt=0.5)
    m = measurement.measure(res['c'], res['baseline_c'], res['polygons'], res['gcamp'], res['ablated'])
    t, r = res['t'], m['radius']
    for target in (19.26, 25.68, 49.22, 126.26, 300.0):
        metrics[f'radius_{str(target).replace(".", "_")}'] = float(np.interp(target, t, r))
    metrics['censored_frames'] = int(m['censored'].sum())
    metrics['experimental'] = {str(x): float(exp['radius_um'][int(np.argmin(np.abs(exp['t_s'] - x)))])
                               for x in (19.26, 49.22, 126.26)}
    pattern = model_late.early_pattern(res, m)
    metrics['pattern'] = pattern
    # Same descriptive statistics for the experiment, for side-by-side reading.
    t_exp = exp['t_s']
    r_exp = np.where(exp['radius_um'] <= 0, np.nan, exp['radius_um'])
    metrics['experimental_pattern'] = model_late.early_pattern({'times': t_exp}, {'radius': r_exp})
    metrics['numerical_health'] = {
        'nonfinite_c': bool(~np.isfinite(res['c']).all()),
        'min_c': float(res['c'].min()), 'max_c': float(res['c'].max()),
        'min_ip3': float(res['ip3'].min()), 'max_rho': float(np.max(res['rho'])),
    }
    metrics['runtime_note'] = 'CPU only; see metadata for solver statistics'

    fig, axs = plt.subplots(2, 1, figsize=(11, 9), layout='constrained')
    ax = axs[0]
    keep = (exp['t_s'] >= 0) & (exp['t_s'] <= 300)
    ax.plot(exp['t_s'][keep], exp['radius_um'][keep], 'o', ms=3, color='black', label='Experimental single wound')
    ax.plot(t, r, label='Model with delayed ligand/receptor layer (2025 cells)')
    for x, lab in [(49.22, 'trough'), (126.26, 'late peak')]:
        ax.axvline(x, color='gray', lw=.6, ls=':')
    ax.set(xlabel='Time after injury (s)', ylabel='GCaMP half-height radius (um)',
           title='Late phase: expansion / recession / second expansion')
    ax.legend(fontsize=9)
    ax = axs[1]
    ax.plot(t, r - np.interp(t, exp['t_s'], exp['radius_um']), label='Model minus experiment')
    ax.axhline(0, color='k', lw=.7)
    ax.set(xlabel='Time (s)', ylabel='Residual (um)',
           title='Largest disagreement near t=49 s: the model does not reproduce the deep trough')
    ax.legend(fontsize=9)
    fig.suptitle('Delayed ligand layer: synthetic model vs one experimental trace\n'
                 'Parameters partly calibrated in the same experimental context; retrospective only',
                 fontsize=11)
    fig.savefig(OUT / 'late_validation.png', dpi=160)
    plt.close(fig)

    (OUT / 'metrics_late.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
