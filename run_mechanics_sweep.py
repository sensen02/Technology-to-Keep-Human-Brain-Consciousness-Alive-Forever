"""Does the published closure time constrain the purse-string tension?

Runs the computed mechanics at two purse-string tensions and reports the closure
rate. The point is to show which parameter combination the published notum
closure time (about 3 h for a ~40 um wound) actually constrains, and to state
how many independent measured anchors are needed to pin the illustrative
parameters used here.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def main():
    import mechanics

    metrics = {'scope': ('illustrative mechanical parameters; sweep shows which parameter '
                         'the published closure timescale constrains')}
    results = {}
    for lc in (0.25, 1.0, 4.0):
        cache = OUT / f'mechanics_cut_n24_lc{lc}.npz'
        if cache.exists():
            d = dict(np.load(cache, allow_pickle=False))
        else:
            d = mechanics.run(n_side=24, mode='cut', cut_halfwidth=8.0, t_end=30.0,
                              sample_dt=2.0, lambda_cut=lc, log_level=0)
            np.savez_compressed(cache, times=d['times'], cut_length=d['cut_length'],
                                hole_area=d['hole_area'], energy=d['energy'])
        t = np.asarray(d['times'], dtype=float)
        cut = np.asarray(d['cut_length'], dtype=float)
        rate = (cut[0] - cut[-1]) / t[-1]
        results[f'lambda_cut_{lc}'] = {
            'cut_initial_um': float(cut[0]), 'cut_final_um': float(cut[-1]),
            'elapsed_s': float(t[-1]),
            'boundary_retraction_rate_um_per_s': float(rate),
            'fraction_closed': float(1 - cut[-1] / cut[0]),
        }
    metrics['purse_string_sweep'] = results
    rates = [v['boundary_retraction_rate_um_per_s'] for v in results.values()]
    metrics['rate_range_um_per_s'] = [float(min(rates)), float(max(rates))]
    metrics['interpretation'] = (
        'Closure rate scales with purse-string tension but the model still closes a 155 um '
        'free boundary many times faster than the published notum closes a ~40 um wound '
        '(about 3 h). Calibrating the relaxation time fixes the viscosity-to-elasticity '
        'ratio; the closure rate is needed as a SECOND independent anchor to fix the '
        'purse-string tension. With only one anchor the closure prediction stays unfalsifiable.')
    metrics['anchors_needed'] = {
        'measured_anchor_1': 'relaxation time ~10 s (5-20 s), notum 18-26 hAPF',
        'measured_anchor_2': 'closure time ~3 h for a ~40 um wound, notum 12-18 hAPF',
        'constrains': ['eta / (k_area * A0)', 'lambda_cut / eta'],
        'note': ('Both anchors come from the same tissue but DIFFERENT developmental stages; '
                 'MEASURED_ANCHORS.md warns against mixing those windows.'),
    }
    (OUT / 'metrics_mechanics_sweep.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
