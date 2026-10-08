"""Run the cell-state / metabolism / death layer and plot its behaviour.

Synthetic, uncalibrated module: parameters are illustrative except ATP order of
magnitude. Nothing here shows that any real cell lives or dies.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def main():
    import cellstate

    metrics = {}
    control = cellstate.run(t_end=300.0, dt=0.1, sample_dt=5.0)
    anoxia = cellstate.run(t_end=300.0, dt=0.1, sample_dt=5.0, oxygen=0.0)
    no_repair = cellstate.run(t_end=300.0, dt=0.1, sample_dt=5.0, repair_enabled=False,
                              tear_scale=1.0)
    metrics['control'] = {
        'death_fraction_final': float(control['death_fraction'][-1]),
        'death_fraction_tissue_final': float(control['death_fraction_tissue'][-1]),
        'min_integrity': float(control['integrity'].min()),
        'max_V': float(control['V'].max()),
        'min_atp_fraction': float(control['atp'].min() / control['atp'][0].mean()),
    }
    metrics['anoxia_oxygen0'] = {
        'time_atp_depleted_median_s': float(np.median(control['times'][np.isfinite(np.where(control['atp'].mean(axis=1) > 1e-9, 0, np.nan))])) if False else None,
        'death_fraction_final': float(anoxia['death_fraction'][-1]),
        'min_atp_fraction': float(anoxia['atp'].min() / control['atp'][0].mean()),
    }
    metrics['no_repair'] = {
        'death_fraction_final': float(no_repair['death_fraction'][-1]),
        'min_integrity': float(no_repair['integrity'].min()),
    }
    metrics['state_codes'] = {str(k): v for k, v in cellstate.STATE_CODES.items()}
    metrics['param_sources_note'] = ('Parameters in cellstate.py are marked illustrative '
                                     'unless a citation is recorded in PARAM_SOURCES. '
                                     'The module is uncalibrated.')
    metrics['scope_warning'] = ('These states are MODEL VERDICTS, not measurements of real '
                                'cell survival. No verified cytosolic-calcium or critical-volume '
                                'death threshold exists for this tissue; Drosophila inner-membrane '
                                'channel behaviour differs from the mammalian PTP.')

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    t = control['times']
    ax = axs[0, 0]
    for res, label in [(control, 'nominal oxygen'), (anoxia, 'anoxia (oxygen=0)')]:
        ax.plot(res['times'], res['atp'].mean(axis=1), label=label)
    ax.set(xlabel='Time (s)', ylabel='Mean ATP (uM, module units)',
           title='ATP proxy: uncalibrated, illustrative parameters')
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for res, label in [(control, 'nominal, repair ON'), (no_repair, 'repair disabled')]:
        ax.plot(res['times'], res['integrity'].mean(axis=1), label=label)
    ax.set(xlabel='Time (s)', ylabel='Mean membrane integrity (0-1)',
           title='Repair layer: only qualitative behaviour is trustworthy')
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    for res, label in [(control, 'nominal'), (anoxia, 'anoxia')]:
        ax.plot(res['times'], res['V'].mean(axis=1), label=label)
    ax.set(xlabel='Time (s)', ylabel='Mean relative volume', title='Volume: swelling response')
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    for res, label in [(control, 'nominal'), (anoxia, 'anoxia'), (no_repair, 'repair off')]:
        ax.plot(res['times'], res['death_fraction'], label=label)
    ax.set(xlabel='Time (s)', ylabel='Fraction of cells flagged irreversible or lysed',
           title='Death triage with hysteresis: a MODEL VERDICT, not measured survival')
    ax.legend(fontsize=8)
    fig.suptitle('Cell-state layer: volume / ATP / membrane repair / death triage\n'
                 'Uncalibrated illustrative module; not evidence about any real cell', fontsize=11)
    fig.savefig(OUT / 'cellstate_validation.png', dpi=160)
    plt.close(fig)

    print('running cellstate.self_test() ...')
    metrics['self_test'] = cellstate.self_test(quick=False)
    (OUT / 'metrics_cellstate.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'self_test'}, indent=2, default=str))


if __name__ == '__main__':
    main()
