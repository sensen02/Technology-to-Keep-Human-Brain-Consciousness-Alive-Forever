"""S5/P2 numerics_shift post-run addendum: proportionality recheck and saturation diagnostic.

ANALYSIS ONLY. Does not re-run any neural replay, does not modify run.py (frozen by SOURCE_FREEZE.json)
and does not change any stored number; every value written here is recomputed from the saved NEURAL.npz
arrays and SUMMARY.json.

Two post-run facts documented here:
 1. Implementation mismatch of the pre-registered growth question. PREREGISTRATION.json asks whether
    D_shift is roughly proportional to the shift, defined there as D/N staying within a factor 3 over
    the 8x shift span. The frozen run.py computed the *spread of D itself* instead of the spread of D/N,
    so its stored flag 'roughly_proportional' answers a different question. Both answers are reported
    here; the pre-registered wording is evaluated explicitly and it is NOT proportional.
 2. Post-hoc (not pre-registered) saturation diagnostic: the pointwise and the aggregate metrics of
    every pair among the five runs, so the reader can see whether the shifted runs differ from the
    control more than they differ from each other.
"""
import json, time
from pathlib import Path
import numpy as np

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift')
SRC = Path(__file__).resolve().parent
SHIFTS = [0, 1, 2, 4, 8]
DT = 0.015625
NMS = 200
D_DT = 0.287739
SPAN_FACTOR = 3.0


def js(p):
    return json.loads(Path(p).read_text())


def lab(s):
    return 'RUNS/shift_%03d_substeps/NEURAL.npz' % s


def nrms(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b ** 2)))
    return num / max(den, 1e-9)


def aggregates(x, y):
    mx = np.asarray(x, float).mean(0)
    my = np.asarray(y, float).mean(0)
    rel = np.abs(mx - my) / np.maximum(np.abs(my), 1e-12)
    return dict(window_mean_nRMS=nrms(mx, my), pools_within_10pct=int((rel <= 0.10).sum()),
                pearson_across_pools=float(np.corrcoef(mx, my)[0, 1]))


def main():
    summ = js(OUT / 'SUMMARY.json')
    data = {}
    for s in SHIFTS:
        with np.load(OUT / lab(s)) as z:
            data[s] = {k: z[k].copy() for k in
                       ['pool_rates', 'raw_command', 'clipped_command', 'counts12000']}
    counts = {s: data[s]['counts12000'].astype(np.int64).sum(0) for s in SHIFTS}

    # 1) proportionality as pre-registered (D/N) versus what the frozen code measured (D spread)
    pts = [(p['shift_substeps'], p['pointwise_pool_nRMS']) for p in summ['growth']['points'] if p['shift_substeps'] > 0]
    dvals = np.asarray([v for _, v in pts], float)
    dn = {str(s): v / s for s, v in pts}
    dn_vals = np.asarray(list(dn.values()), float)
    dn_spread = float(dn_vals.max() / dn_vals.min())
    d_spread = float(dvals.max() / dvals.min())
    prop = dict(
        pre_registered_criterion='D/N (nRMS per sub-step of shift) stays within a factor %g over the 8x shift span'
                                 % SPAN_FACTOR,
        per_substep_normalized=dn,
        per_substep_spread=dn_spread,
        pointwise_values={str(s): v for s, v in pts},
        pointwise_spread=d_spread,
        proportional_by_pre_registered_criterion=bool(dn_spread <= SPAN_FACTOR),
        answer='NOT_PROPORTIONAL_SATURATED: D/N falls by a factor %.2f while the shift grows 8x, and D itself '
               'changes by only a factor %.3f, so D_shift saturates already at the smallest nonzero shift '
               '(1 sub-step = 0.015625 ms).' % (dn_spread, d_spread),
        frozen_flag_recorded=summ['growth']['proportionality']['roughly_proportional'],
        implementation_mismatch='the frozen run.py computed max(D)/min(D) instead of max(D/N)/min(D/N) for the '
                                'pre-registered question, so its stored flag answers "is D flat across the shift '
                                'span" (yes, factor %.3f) rather than "is D proportional to the shift" (no, factor '
                                '%.2f in D/N). No stored numeric result changes; the correct answer to the '
                                'pre-registered wording is the one above.' % (d_spread, dn_spread))

    smallest = min(pts, key=lambda t: t[0])
    largest = max(pts, key=lambda t: t[0])
    crossing = dict(
        engineering_threshold=0.15,
        smallest_measured_shift_substeps=int(smallest[0]),
        smallest_measured_shift_ms=smallest[0] * DT,
        smallest_measured_D=smallest[1],
        largest_measured_shift_substeps=int(largest[0]),
        largest_measured_shift_ms=largest[0] * DT,
        largest_measured_D=largest[1],
        crossing_shift='the 0.15 engineering threshold is already exceeded at the smallest measured nonzero '
                       'shift (1 sub-step = 0.015625 ms); the pointwise metric cannot be pushed below 0.15 by '
                       'any nonzero sub-millisecond input shift inside this range',
        only_way_to_threshold='shift exactly 0 (the control), where D_shift is 0 by construction',
        ratio_smallest_over_dt=smallest[1] / D_DT,
        ratio_largest_over_dt=largest[1] / D_DT)

    # 2) post-hoc saturation diagnostic: all pairs among the five runs
    pairs = [(a, b) for i, a in enumerate(SHIFTS) for b in SHIFTS[i + 1:]]
    matrix = []
    for a, b in pairs:
        agg = aggregates(data[b]['pool_rates'], data[a]['pool_rates'])
        matrix.append(dict(
            a_substeps=a, b_substeps=b, a_ms=a * DT, b_ms=b * DT,
            pointwise_pool_nRMS=nrms(data[b]['pool_rates'], data[a]['pool_rates']),
            pointwise_raw_command_nRMS=nrms(data[b]['raw_command'], data[a]['raw_command']),
            pointwise_clipped_command_nRMS=nrms(data[b]['clipped_command'], data[a]['clipped_command']),
            window_mean_pool_nRMS=agg['window_mean_nRMS'],
            pools_within_10pct=agg['pools_within_10pct'],
            per_cell_spike_count_pearson=float(np.corrcoef(counts[b], counts[a])[0, 1]),
            total_spike_difference=int(counts[b].sum() - counts[a].sum())))
    shifted_vs_control = [m['pointwise_pool_nRMS'] for m in matrix if m['a_substeps'] == 0]
    shifted_vs_shifted = [m['pointwise_pool_nRMS'] for m in matrix if m['a_substeps'] != 0]
    sat = dict(
        definition='pointwise pool-rate nRMS over 200 ms between every unordered pair of runs; a is the reference '
                   'denominator, so each row is RMS(b-a)/RMS(a)',
        pairs=matrix,
        shift_vs_control={'min': min(shifted_vs_control), 'max': max(shifted_vs_control),
                          'ratios': shifted_vs_control},
        shift_vs_shift={'min': min(shifted_vs_shifted), 'max': max(shifted_vs_shifted),
                        'ratios': shifted_vs_shifted},
        overlap='the shifted-versus-shifted nRMS range overlaps the shifted-versus-control range, so two runs '
                'that differ only by 0.015625 ms of input grid phase are as far apart as each of them is from '
                'the control; the 200 ms pointwise pool metric is measuring a discretization-level floor, not a '
                'linear response to the size of the shift',
        post_hoc=True,
        post_hoc_note='this pairwise diagnostic was NOT part of the pre-registered decision rule; it is reported as '
                      'a post-run observation computed from the already frozen arrays')

    per_cell = {str(s): float(np.corrcoef(counts[s], counts[0])[0, 1]) for s in SHIFTS if s != 0}
    totals = {str(s): int(counts[s].sum()) for s in SHIFTS}

    out = dict(
        stage='S5/P2 numerics_shift post-run addendum', analysis_only=True, neural_replays_run=0,
        written_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_script=str(SRC / 'postrun_addendum.py'),
        frozen_source_untouched=True, frozen_reading=summ['verdict']['reading'],
        d_dt=D_DT,
        proportionality=prop, threshold_crossing=crossing, saturation_diagnostic=sat,
        per_cell_spike_count_pearson_vs_control=per_cell, total_spikes=totals,
        aggregate_window_mean_nRMS_vs_control={str(p['shift_substeps']): p['aggregate_window_mean_nRMS']
                                               for p in summ['growth']['points']},
        sealed_aggregate_dt_nRMS=summ['sealed_aggregate_anchors']['fine_0p03125_vs_0p015625']['window_mean']
        ['normalized_RMS'],
        conclusions_allowed='only whether sub-millisecond input timing or the integrator explains the dt difference, '
                            'and what acceptance metric follows',
        biological_validity='NOT_ESTABLISHED', selected_timestep=None, model_selection=None,
        importance='核心必需')
    p = OUT / 'POSTRUN_ADDENDUM.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + '\n')
    print('ADDENDUM', p)
    print(' D/N spread %.3f proportional=%s' % (dn_spread, prop['proportional_by_pre_registered_criterion']))
    print(' shift-vs-control pointwise [%.4f..%.4f]  shift-vs-shift [%.4f..%.4f]'
          % (min(shifted_vs_control), max(shifted_vs_control), min(shifted_vs_shifted), max(shifted_vs_shifted)))
    print(' totals', totals, 'per-cell pearson', {k: round(v, 6) for k, v in per_cell.items()})


if __name__ == '__main__':
    main()
