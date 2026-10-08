"""Small manually implemented IDEALIZED integration demo, not fly physiology.
Run with venv/bin/python run_isolation_scenario.py. Writes integrated_* only.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import numpy as np
from engine.isolation_scenario import IsolationConfig, run_isolation_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'outputs'/'brain_isolation')
    parser.add_argument('--chemical-gain-uS', type=float, default=0.)
    parser.add_argument('--support-gain-nA', type=float, default=0.)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config = IsolationConfig(chemical_target_gain_uS=args.chemical_gain_uS,
                             support_target_gain_nA=args.support_gain_nA)
    scenarios = ('intact', 'sham', 'neck_cut', 'eye_loss', 'neck_cut_supported')
    results = {s: run_isolation_scenario(replace(config, scenario=s)) for s in scenarios}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), constrained_layout=True)
    colors = dict(zip(scenarios, ('black', 'gray', 'tab:red', 'tab:blue', 'tab:green')))
    for s, r in results.items():
        t, p = r['times_ms'], r['probes']
        for ax, probe in zip(axes[:3], ('brain_mV', 'neck_distal_mV', 'body_mV')):
            ax.plot(t, p[probe], color=colors[s], label=s, alpha=.8, lw=1)
            ax.set_ylabel(probe+' [mV]')
        axes[4].plot(t, p['support_proxy'], color=colors[s], label=s)
        arrays = {'times_ms': t, 'input_times_ms': r['input_times_ms'], **p, **r['inputs']}
        np.savez_compressed(args.output_dir/f'integrated_{s}.npz', **arrays)
    r = results['intact']
    axes[3].plot(r['times_ms'], r['probes']['chemical_response'], label='target receptor response')
    axes[3].set_ylabel('chemical response [1]')
    axes[3].set_title('FAST illustrative kinetics; chemical dt=1 ms = .001 s; mapping gain default0')
    axes[4].set_ylabel('NONPHYSIOLOGICAL\nsupport availability [1]')
    axes[4].set_title('Support proxy is NOT oxygen/ATP; brain-current hypothesis gain default0')
    axes[5].plot(r['input_times_ms'], r['inputs']['artifact_unfiltered_mV'], label='raw stimulus artifact')
    axes[5].plot(r['input_times_ms'], r['inputs']['artifact_recorded_mV'], label='bandlimited artifact-only recorder')
    axes[5].set_ylabel('artifact [mV], NOT Vm')
    axes[5].set_xlabel('neural time [ms]; full plotted duration = 0.12 seconds')
    for ax in axes:
        ax.axvline(config.phase_ms, color='purple', ls='--', label='scheduled phase' if ax is axes[0] else None)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[5].legend(fontsize=8)
    fig.suptitle('IDEALIZED eye(point) → brain(point) → neck(squid-HH) → body(point)\n'
                 'Actual axial cut after common baseline; NOT reconstructed anatomy or fly physiology')
    plot = args.output_dir/'integrated_comparison.png'
    fig.savefig(plot, dpi=140)
    plt.close(fig)
    metadata = {s: {k: v for k, v in r.items() if k not in ('times_ms', 'input_times_ms', 'probes', 'inputs')} for s, r in results.items()}
    report = args.output_dir/'integrated_report.json'
    report.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({s: r['metrics'] for s, r in results.items()}, indent=2))
    print(plot.resolve())
    print(report.resolve())


if __name__ == '__main__':
    main()
