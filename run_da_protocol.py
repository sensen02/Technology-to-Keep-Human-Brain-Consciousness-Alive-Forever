"""Demo: measured 6x ACh protocol -> illustrative dopamine release/clearance model.

Run: /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python run_da_protocol.py

Writes (all under /run/media/sensen/Data2/cell_wound_prototype/outputs/brain_isolation/):
  da_protocol_comparison.png, da_protocol_report.json, da_protocol_traces.npz

WHAT IS REAL HERE: only the protocol timing (6 boluses of 0.2 pmol ACh, 10 min
apart, 1 min record per stimulus, 5 s pre-bolus baseline, 15 s representative
trace, ~20 min post-dissection rest) and the QUALITATIVE target pattern (young
release constant; aged release declining; t1/2 not significantly different) are
from the source study (Dumitrescu/Copeland/Venton, PMC9897283). The ACh pulse is
the STIMULUS; the modelled transmitter is dopamine. Every pool size, volume,
time constant, efficacy and noise level is ILLUSTRATIVE and swept. There is NO
quantitative fit to the paper and none is claimed: its absolute
concentration-vs-time values exist only in figures and could not be extracted.
No statement is made about consciousness, viability or medical use.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from engine.da_protocol import (
    DEFAULT_DECLINE_THRESHOLD, DEFAULT_T_HALF_TOLERANCE, DAKinetics,
    ProtocolSpec, SOURCE, build_registry, pattern_verdict, release_sequence,
    simulate, sweep_clearance, sweep_pool_refill_efficacy, sweep_shape_coupling,
)

OUT = Path(__file__).resolve().parent / 'outputs' / 'brain_isolation'
PNG = OUT / 'da_protocol_comparison.png'
JSON = OUT / 'da_protocol_report.json'
NPZ = OUT / 'da_protocol_traces.npz'

HONESTY = [
    "Only the protocol timing and the qualitative young/aged pattern come from "
    "the source study (" + SOURCE + ").",
    "The ACh bolus is a STIMULUS dose; the modelled transmitter is dopamine. "
    "The 0.2 pmol ACh dose is never treated as a dopamine amount.",
    "The tissue volume (1e8 um^3), pool size, refilling time constant, release "
    "efficacy, release time course, clearance time constant and noise level are "
    "ILLUSTRATIVE choices, swept in this report, NOT measured parameters.",
    "The 'aging knob' is a HYPOTHESIS (a change in releasable pool size and/or "
    "refilling rate); the source study measured no pool or clearance parameter.",
    "No quantitative fit to the source study is claimed or possible now: its "
    "absolute concentration-vs-time values are only in figures and the PDF "
    "could not be retrieved, so no curve digitisation or fitting was done.",
    "No claim is made about consciousness, viability, welfare or medical use.",
    "'t1/2 not significantly different' in the source is a non-rejection, not "
    "proof of equality; a small clearance change could escape detection.",
    "The source also reports regional inconsistency (mushroom body vs central "
    "complex), which this single-region model does not represent.",
]

SCENARIOS = {
    'young_control': dict(
        kinetics=DAKinetics(label='young control (illustrative)'),
        role='reference: large fast-refilling pool -> release approximately constant'),
    'aged_hypothesis_pool': dict(
        kinetics=DAKinetics(pool_capacity_pmol=0.03, refill_tau_s=1800.0,
                            label='aged HYPOTHESIS: small pool + slow refilling'),
        role='HYPOTHESIS: pool/refilling change -> declining release, t1/2 unchanged'),
    'aged_pool_size_only': dict(
        kinetics=DAKinetics(pool_capacity_pmol=0.03, refill_tau_s=30.0,
                            label='pool size only, fast refilling'),
        role='control: small pool alone does NOT decline when refilling is fast'),
    'aged_hypothesis_clearance_fast': dict(
        kinetics=DAKinetics(clearance_tau_s=1.25,
                            label='aged HYPOTHESIS: 4x faster clearance'),
        role='control: clearance change -> release still flat AND t1/2 moves'),
    'aged_hypothesis_clearance_slow': dict(
        kinetics=DAKinetics(clearance_tau_s=20.0,
                            label='aged HYPOTHESIS: 4x slower clearance'),
        role='control: same conclusion in the other direction'),
}
PLOT_ORDER = ('young_control', 'aged_hypothesis_pool', 'aged_pool_size_only',
              'aged_hypothesis_clearance_fast')


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def build_figure(protocol, results, sweep, clearance, coupling):
    colors = dict(zip(PLOT_ORDER, ('tab:blue', 'tab:red', 'tab:green', 'tab:purple')))
    fig, ax = plt.subplots(3, 3, figsize=(19.0, 13.5), layout='constrained')
    fig.suptitle(
        "MEASURED protocol (PMC9897283): 6 x 0.2 pmol ACh stimulus, 10 min apart, 1 min record, "
        "5 s pre-bolus baseline, ~20 min rest — the ACh bolus is the STIMULUS; the modelled transmitter is dopamine\n"
        "Illustrative dopamine release/clearance model: pool size, refilling, release efficacy, release kinetics, "
        "clearance and the 1e8 um^3 volume are ALL ILLUSTRATIVE and swept\n"
        "Only the protocol timing and the qualitative young/aged pattern are measured; no quantitative fit to the "
        "paper is claimed (its absolute concentration values are figure-only and could not be extracted)",
        fontsize=10.5)

    # (0,0) full session -----------------------------------------------------
    a = ax[0, 0]
    floor = 1e-3
    for name in PLOT_ORDER:
        r = results[name]
        a.plot(r.timeline_time_s / 60.0, np.maximum(r.timeline_nM, floor),
               label=f"{name}", color=colors[name], lw=1.1)
        a.plot(np.asarray(r.bolus_times_s) / 60.0, r.peak_nM, 'o', ms=3.5,
               color=colors[name])
    for t in protocol.bolus_times_s:
        a.axvline(t / 60.0, color='0.6', ls=':', lw=0.8)
    a.set_yscale('log')
    a.set_ylim(floor, 3e4)
    a.set_xlim(-21, 52)
    a.set_title("Full session: 20 min rest, then 6 boluses 10 min apart", fontsize=10)
    a.set_xlabel("time from first record start (min)", fontsize=9)
    a.set_ylabel("extracellular DA (nM, log; clipped at 1e-3 nM)", fontsize=9)
    a.annotate("vertical dotted lines = the 6 measured bolus times "
               "(0.08, 10.08, 20.08, 30.08, 40.08, 50.08 min);\n"
               "circles = peak of each record; the flat 1e-3 nM line is a plot clip",
               xy=(0.02, 0.80), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
    a.legend(fontsize=6.5, loc='upper right', ncol=1, framealpha=0.92)
    a.tick_params(labelsize=8)

    # (0,1)/(0,2) representative 15 s traces --------------------------------
    for col, idx, label in ((1, 0, "stimulus 1"), (2, 5, "stimulus 6")):
        a = ax[0, col]
        for name in PLOT_ORDER:
            r = results[name]
            t = r.record_time_s[idx] - protocol.bolus_times_s[idx]
            a.plot(t, r.representative_nM[idx], label=name, color=colors[name], lw=1.4)
        a.axvline(0.0, color='0.5', ls='--', lw=0.9)
        a.set_xlim(-5, 10)
        a.set_title(f"Representative 15 s trace, {label} (bolus at t = 0)", fontsize=10)
        a.set_xlabel("time from bolus (s)", fontsize=9)
        a.set_ylabel("extracellular DA (nM)", fontsize=9)
        a.tick_params(labelsize=8)
        if idx == 5:
            noisy = simulate(protocol, DAKinetics(noise_std_nM=2.0,
                                                  label='young + illustrative 2 nM noise'),
                             sample_dt_s=0.02, seed=12345)
            a.plot(noisy.record_time_s[idx] - protocol.bolus_times_s[idx],
                   noisy.representative_nM[idx], color='0.55', lw=0.7,
                   label='same young trace + illustrative 2 nM noise (seed 12345)')
        top = max(float(np.nanmax(results[n].representative_nM[idx]))
                  for n in PLOT_ORDER)
        if idx == 5:
            top = max(top, float(np.nanmax(noisy.representative_nM[idx])))
        a.set_ylim(0, 1.55 * max(top, 1.0))
        a.legend(fontsize=6.5, loc='upper center', ncol=2, framealpha=0.95)
    ax[0, 2].annotate("identical trace shapes (amplitude only)\n"
                      "-> t1/2 must stay the same",
                      xy=(0.02, 0.62), xycoords='axes fraction', fontsize=7.5,
                      bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))

    # (1,0) released amount --------------------------------------------------
    a = ax[1, 0]
    idxs = np.arange(1, 7)
    for name in PLOT_ORDER:
        a.plot(idxs, results[name].released_pmol, 'o-', label=name,
               color=colors[name], ms=4)
    a.set_title("Released dopamine per stimulus (pool-limited)", fontsize=10)
    a.set_xlabel("stimulus index n (measured: 6, 10 min apart)", fontsize=9)
    a.set_ylabel("released DA (pmol)", fontsize=9)
    a.set_xticks(idxs)
    a.set_ylim(0.0, 1.65 * float(np.max([results[n].released_pmol[0] for n in PLOT_ORDER])))
    a.legend(fontsize=6.5, loc='upper center', ncol=2, framealpha=0.95)
    a.tick_params(labelsize=8)
    a.annotate("stimulus 1 is identical for every pool scenario\n"
               "(every pool starts full): declines appear from n = 2",
               xy=(0.03, 0.05), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))

    # (1,1) normalised release ----------------------------------------------
    a = ax[1, 1]
    for name in PLOT_ORDER:
        r = results[name]
        a.plot(idxs, r.released_pmol / r.released_pmol[0], 'o-', label=name,
               color=colors[name], ms=4)
    a.axhline(DEFAULT_DECLINE_THRESHOLD, color='k', ls='--', lw=1.0)
    a.set_title("Release normalised to stimulus 1 (acceptance target)", fontsize=10)
    a.set_xlabel("stimulus index n", fontsize=9)
    a.set_ylabel("A_n / A_1 (1 = constant release)", fontsize=9)
    a.set_xticks(idxs)
    a.set_ylim(0.4, 1.35)
    a.legend(fontsize=6.5, loc='upper center', ncol=2, framealpha=0.95)
    a.tick_params(labelsize=8)
    a.annotate(f"dashed line: decline criterion A6/A1 <= {DEFAULT_DECLINE_THRESHOLD:g}\n"
               "young = flat (as measured); aged-pool HYPOTHESIS = declining",
               xy=(0.03, 0.03), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))

    # (1,2) t1/2 -------------------------------------------------------------
    a = ax[1, 2]
    for name in PLOT_ORDER:
        a.plot(idxs, results[name].measured_t_half_s, 'o-', label=name,
               color=colors[name], ms=4)
    a.set_title("Half-decay time of each trace (measured from the trace)", fontsize=10)
    a.set_xlabel("stimulus index n", fontsize=9)
    a.set_ylabel("t1/2 (s)", fontsize=9)
    a.set_xticks(idxs)
    a.set_ylim(0, 11.5)
    a.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=0.95)
    a.tick_params(labelsize=8)
    a.annotate("pool change: t1/2 invariant (<1e-13 relative)\n"
               "clearance change: t1/2 moves by ~65% ->\n"
               "only the release route reproduces the measured pattern",
               xy=(0.30, 0.44), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    # (2,0) peaks ------------------------------------------------------------
    a = ax[2, 0]
    for name in PLOT_ORDER:
        a.plot(idxs, results[name].measured_peak_nM, 'o-', label=name,
               color=colors[name], ms=4)
    a.set_title("Peak concentration per stimulus (what FSCV would report)", fontsize=10)
    a.set_xlabel("stimulus index n", fontsize=9)
    a.set_ylabel("peak extracellular DA (nM)", fontsize=9)
    a.set_xticks(idxs)
    a.set_ylim(0, 132)
    a.legend(fontsize=6.5, loc='lower left', ncol=2, framealpha=0.95)
    a.tick_params(labelsize=8)
    a.annotate("1 nM = 1e-3 uM; peaks inherit the ILLUSTRATIVE 1e8 um^3 volume,\n"
               "so absolute values are NOT calibrated to the paper (uM-level in figures only)",
               xy=(0.03, 0.78), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    # (2,1) region heat map --------------------------------------------------
    a = ax[2, 1]
    ax_ = sweep['axes']
    pools = np.array(ax_['pool_sizes_pmol'])
    refills = np.array(ax_['refill_taus_s'])
    ie = ax_['efficacies'].index(0.05)
    ratio = np.array(sweep['decline_ratio_grid'][ie])          # [refill, pool]
    grid = np.array(sweep['acceptance_grid'][ie])
    X, Y = np.meshgrid(np.log10(pools), np.log10(refills))
    mesh = a.pcolormesh(X, Y, ratio, cmap='viridis_r', vmin=0.0, vmax=1.0,
                        shading='nearest')
    a.contour(X, Y, ratio, levels=[DEFAULT_DECLINE_THRESHOLD], colors='white',
              linewidths=1.2)
    a.contourf(X, Y, grid.astype(float), levels=[0.5, 1.5], colors='none',
               hatches=['//'], edgecolor='white')
    a.plot(np.log10(1.0), np.log10(30.0), marker='*', ms=15, color='tab:blue',
           mec='white', mew=0.8)
    a.plot(np.log10(0.03), np.log10(1800.0), marker='*', ms=15, color='tab:red',
           mec='white', mew=0.8)
    a.annotate('young (1 pmol, 30 s):\nno decline', (np.log10(1.0), np.log10(30.0)),
               textcoords='offset points', xytext=(-8, 12), fontsize=7.5, color='white',
               bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1.5))
    a.annotate('aged HYPOTHESIS\n(0.03 pmol, 1800 s)', (np.log10(0.03), np.log10(1800.0)),
               textcoords='offset points', xytext=(6, 6), fontsize=7.5, color='white',
               bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1.5))
    a.set_xticks(np.log10(pools))
    a.set_xticklabels([f"{p:g}" for p in pools], fontsize=7.5, rotation=45)
    a.set_yticks(np.log10(refills))
    a.set_yticklabels([f"{t:g}" for t in refills], fontsize=7.5)
    a.set_xlabel("releasable pool capacity (pmol, ILLUSTRATIVE, log)", fontsize=9)
    a.set_ylabel("pool refilling time constant (s, ILLUSTRATIVE, log)", fontsize=9)
    a.set_title(f"Acceptance region at efficacy = 0.05:\n"
                f"A6/A1 <= {DEFAULT_DECLINE_THRESHOLD:g} AND t1/2 spread <= "
                f"{100 * DEFAULT_T_HALF_TOLERANCE:g}%  (hatched = accepted)",
                fontsize=10)
    cb = fig.colorbar(mesh, ax=a, pad=0.02)
    cb.set_label("release ratio A6/A1  (=1: no decline)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # (2,2) where invariance breaks -----------------------------------------
    a = ax[2, 2]
    cps = [row['release_shape_coupling'] for row in coupling['rows']]
    spreads = [100.0 * row['t_half_relative_spread'] for row in coupling['rows']]
    a.plot(cps, spreads, 'o-', color='tab:red',
           label='aged-pool HYPOTHESIS, release waveform\nshortens with pool depletion')
    a.axhline(100.0 * DEFAULT_T_HALF_TOLERANCE, color='k', ls='--', lw=1.0,
              label=f"t1/2 invariance tolerance ({100 * DEFAULT_T_HALF_TOLERANCE:g}%)")
    a.set_title("Where the t1/2 invariance breaks", fontsize=10)
    a.set_xlabel("release_shape_coupling (0 = amplitude-only pool effect)", fontsize=9)
    a.set_ylabel("within-run t1/2 spread (% of mean)", fontsize=9)
    a.set_xlim(-0.1, 2.1)
    a.set_ylim(0, max(12, 1.3 * max(spreads)))
    a.legend(fontsize=7, loc='upper left')
    a.tick_params(labelsize=8)
    a.annotate("coupling 0: spread ~1e-14 % (structural)\n"
               "coupling 1: 6.3% > tolerance -> fails\n"
               "so the claim survives only if depletion changes\n"
               "release AMPLITUDE, not release TIME COURSE",
               xy=(0.02, 0.30), xycoords='axes fraction', fontsize=7.5,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    for axis in ax.ravel():
        axis.title.set_multialignment('center')
    fig.savefig(PNG, dpi=140)
    plt.close(fig)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = ProtocolSpec()
    checks = []

    def check(name, ok, detail):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})
        assert ok, f"FAILED check {name}: {detail}"

    # ---- 1+2. run the measured protocol, verify its timing -----------------
    results = {}
    for name, spec in SCENARIOS.items():
        results[name] = simulate(protocol, spec['kinetics'], sample_dt_s=0.02,
                                 coarse_dt_s=0.5, seed=12345)

    timing = protocol.timing_report()
    r0 = results['young_control']
    check('6 boluses at the measured times',
          list(r0.bolus_times_s) == [5.0, 605.0, 1205.0, 1805.0, 2405.0, 3005.0],
          f"bolus times {list(r0.bolus_times_s)} s")
    check('10 min between boluses',
          all(s == 600.0 for s in timing['bolus_spacing_s']),
          f"spacings {timing['bolus_spacing_s']} s")
    check('bolus 5 s into each 1 min record',
          bool(np.allclose(r0.record_time_s[:, 0] - r0.bolus_times_s, -5.0)),
          'record start = bolus - 5 s for all 6 stimuli')
    check('sampling covers each record exactly',
          int(r0.record_time_s.shape[1]) == protocol.n_record_samples(0.02) == 3001
          and bool(np.allclose(r0.record_time_s[:, -1] - r0.bolus_times_s, 55.0)),
          '3001 samples per 60 s record at 0.02 s, last sample at bolus + 55 s')
    check('representative trace is 15 s',
          float(protocol.representative_trace_s) == 15.0,
          '15 s = 5 s baseline + 10 s of the evoked event')
    check('~20 min post-dissection rest is represented',
          float(protocol.post_dissection_rest_s) == 1200.0
          and float(r0.timeline_time_s[0]) == -1200.0,
          'timeline starts 1200 s before the first record')

    # ---- amounts, positivity, ledgers --------------------------------------
    for name, r in results.items():
        scale = max(1e-18, abs(r.ledger['released_total_pmol']))
        check(f'pool ledger closes ({name})',
              abs(r.ledger['pool_ledger_residual_pmol']) < 1e-13 * scale,
              f"residual {r.ledger['pool_ledger_residual_pmol']:.3e} pmol")
        check(f'extracellular ledger closes ({name})',
              abs(r.ledger['extracellular_ledger_residual_pmol']) < 1e-11 * scale,
              f"released {r.ledger['released_total_pmol']:.6g} pmol = "
              f"cleared {r.ledger['cleared_total_pmol']:.6g} + "
              f"remaining {r.ledger['extracellular_at_session_end_pmol']:.3e}")
        check(f'amounts nonnegative and finite ({name})',
              bool(np.all(r.released_pmol >= 0) and np.all(r.record_nM >= 0)
                   and np.all(np.isfinite(r.record_nM))
                   and np.all(r.released_pmol <= r.pool_before_pmol + 1e-15)),
              'no negative or nonfinite amount/concentration; release <= pool')

    # ---- pattern verdicts --------------------------------------------------
    verdicts = {name: pattern_verdict(r) for name, r in results.items()}
    young, aged = results['young_control'], results['aged_hypothesis_pool']
    fast, pc = results['aged_pool_size_only'], results['aged_hypothesis_clearance_fast']
    check('young control reproduces the measured plateau',
          not verdicts['young_control']['release_declines_enough']
          and verdicts['young_control']['t_half_approximately_constant'],
          f"release ratio A6/A1 = {young.release_ratio_last_first:.6f}")
    check('aged pool hypothesis reproduces decline + constant t1/2',
          verdicts['aged_hypothesis_pool']['reproduces_measured_pattern'],
          f"A6/A1 = {aged.release_ratio_last_first:.4f} "
          f"({100 * aged.relative_release_decline:.1f}% decline), t1/2 spread "
          f"{aged.t_half_relative_spread:.2e}")
    check('pool size alone (fast refilling) does NOT decline',
          not verdicts['aged_pool_size_only']['release_declines_enough'],
          f"A6/A1 = {fast.release_ratio_last_first:.6f} at "
          f"refill_tau/ISI = {fast.kinetics.refill_tau_over_isi(protocol):.3f}")
    check('clearance change cannot produce the release decline',
          abs(pc.release_ratio_last_first - 1.0) < 1e-8,
          f"clearance 4x faster: A6/A1 = {pc.release_ratio_last_first:.10f}")
    check('pool-only change leaves t1/2 invariant (exact)',
          abs(aged.measured_t_half_s[0] - young.measured_t_half_s[0])
          / young.measured_t_half_s[0] < 1e-12,
          f"young {young.measured_t_half_s[0]:.6f} s vs aged-pool "
          f"{aged.measured_t_half_s[0]:.6f} s")
    check('clearance-only change moves t1/2',
          abs(pc.measured_t_half_s[0] - young.measured_t_half_s[0])
          / young.measured_t_half_s[0] > 0.5,
          f"young {young.measured_t_half_s[0]:.4f} s vs faster clearance "
          f"{pc.measured_t_half_s[0]:.4f} s "
          f"({100 * (pc.measured_t_half_s[0] / young.measured_t_half_s[0] - 1):.1f}%)")

    # ---- 3. sweeps ---------------------------------------------------------
    sweep = sweep_pool_refill_efficacy(protocol, DAKinetics())
    clearance = sweep_clearance(protocol, DAKinetics())
    coupling = sweep_shape_coupling(protocol, aged.kinetics)
    check('a nonempty parameter region reproduces the pattern',
          sweep['region_found'] and 0 < sweep['n_accepted'] < sweep['n_combinations'],
          f"{sweep['n_accepted']} of {sweep['n_combinations']} combinations accepted "
          f"({100 * sweep['fraction_accepted']:.1f}%)")
    check('no clearance value reproduces the decline with a fixed pool',
          all(abs(row['release_ratio_last_first'] - 1.0) < 1e-8
              for row in clearance['rows']),
          'A6/A1 = 1 for every clearance time constant in the sweep')
    check('t1/2 invariance breaks as release shape couples to pool content',
          coupling['rows'][0]['t_half_relative_spread'] < 1e-12
          and [row for row in coupling['rows']
               if row['release_shape_coupling'] == 1.0][0]['t_half_relative_spread']
          > DEFAULT_T_HALF_TOLERANCE,
          'coupling 0: ~1e-14 (structural); coupling 1: '
          f"{100 * [row for row in coupling['rows'] if row['release_shape_coupling'] == 1.0][0]['t_half_relative_spread']:.2f}% "
          f"> {100 * DEFAULT_T_HALF_TOLERANCE:g}% tolerance")

    # boundaries in words
    ax = sweep['axes']
    effs = ax['efficacies']
    pools = ax['pool_sizes_pmol']
    refills = ax['refill_taus_s']
    grid = np.array(sweep['acceptance_grid'])
    dec = np.array(sweep['decline_ratio_grid'])
    region_text = []
    for ie, eff in enumerate(effs):
        rows = []
        for it, tau in enumerate(refills):
            ok = [pools[ip] for ip in range(len(pools)) if grid[ie, it, ip]]
            rows.append((tau, max(ok) if ok else None,
                         float(dec[ie, it, max(range(len(pools)),
                                               key=lambda ip: dec[ie, it, ip])])))
        largest = max((t for t, m, _ in rows if m is not None), default=None)
        region_text.append({
            'stimulus_efficacy_pmol_DA_per_pmol_ACh': eff,
            'max_pool_size_with_slowest_declining_refill_pmol': (
                max((m for t, m, _ in rows if m is not None), default=None)),
            'largest_refill_tau_still_accepting_s': largest,
            'refill_taus_with_no_declining_pool_size_s': [
                t for t, m, _ in rows if m is None],
            'best_case_release_ratio_in_each_refill_row': {
                f"{t:g}": r for t, _, r in rows},
        })
    check('region boundaries are interpretable (large pools and fast refilling fail)',
          all(entry['max_pool_size_with_slowest_declining_refill_pmol'] is not None
              for entry in region_text)
          and all(grid[ie, 0, :].sum() == 0 for ie in range(len(effs))),
          'no pool size declines when refilling is much faster than the 600 s gap')

    # ---- report ------------------------------------------------------------
    registry = build_registry(protocol, DAKinetics())
    release_fraction_young = float(young.released_pmol[0] / DAKinetics().pool_capacity_pmol)
    report = {
        'title': ('Illustrative dopamine release/clearance model driven by the '
                  'measured 6x ACh protocol of PMC9897283'),
        'measured_inputs': timing,
        'measured_qualitative_target': {
            'young_control': 'release approximately constant across 6 stimulations',
            'aged': 'release decreases across stimulations',
            't_half': 'not significantly different between groups -> the change is '
                      'in release amount, not clearance speed',
            'source': SOURCE,
        },
        'illustrative_inputs': {
            'default_kinetics': jsonable(
                {k: v for k, v in vars(DAKinetics()).items()}),
            'swept_axes': {
                'pool_capacity_pmol': ax['pool_sizes_pmol'],
                'refill_tau_s': ax['refill_taus_s'],
                'stimulus_efficacy_pmol_DA_per_pmol_ACh': ax['efficacies'],
                'clearance_tau_s': [row['clearance_tau_s'] for row in clearance['rows']],
                'release_shape_coupling': [row['release_shape_coupling']
                                           for row in coupling['rows']],
                'voxel_volume_um3': [DAKinetics().voxel_volume_um3],
            },
        },
        'units_and_conversion': {
            'amount': 'pmol (1 pmol = 1e-12 mol)',
            'concentration': 'nM and uM',
            'time': 's',
            'volume': 'um^3 (1 um^3 = 1e-15 L)',
            'pmol_to_nM': DAKinetics().conversion_report(protocol),
            'worked_example': ('one release of '
                               f'{float(young.released_pmol[0]):.4g} pmol uniformly in '
                               f'{DAKinetics().voxel_volume_um3:.3g} um^3 gives '
                               f'{float(young.q_nM[0]):.4g} nM = '
                               f'{float(young.q_nM[0]) * 1e-3:.4g} uM, and the trace peak '
                               f'is {float(young.peak_nM[0]):.4g} nM = '
                               f'{float(young.peak_nM[0]) * 1e-3:.4g} uM'),
            'assumed_volume_is_illustrative': True,
            'ach_dose_is_a_stimulus_not_dopamine': True,
        },
        'per_stimulus_release_fraction_of_pool_young':
            release_fraction_young,
        'protocol_reproduction': {
            '6_boluses_10_min_apart': True,
            'record_per_stimulus_s': protocol.recording_duration_s,
            'representative_trace_s': protocol.representative_trace_s,
            'pool_must_deplete': ('a finite pool always loses the released amount; '
                                  'whether it VISIBLY declines depends on the per-stimulus '
                                  'release fraction and on refill_tau/ISI'),
            'refill_tau_over_ISI_young':
                young.kinetics.refill_tau_over_isi(protocol),
            'refill_recovery_per_interval_young':
                young.kinetics.refill_recovery_per_interval(protocol),
            'clearance_time_constant_s': young.kinetics.clearance_tau_s,
            'clearance_half_life_s': young.kinetics.clearance_half_life_s,
            'measured_t_half_from_trace_s': [float(x) for x in young.measured_t_half_s],
        },
        'scenarios': {name: dict({'role': spec['role']},
                                 **jsonable(results[name].summary()))
                      for name, spec in SCENARIOS.items()},
        'verdicts': {name: jsonable(v) for name, v in verdicts.items()},
        'parameter_region': {
            'criteria': sweep['criteria'],
            'n_combinations': sweep['n_combinations'],
            'n_accepted': sweep['n_accepted'],
            'fraction_accepted': sweep['fraction_accepted'],
            'region_found': sweep['region_found'],
            'boundaries': jsonable(sweep['boundaries_largest_pool_still_declining']),
            'summary_by_efficacy': region_text,
            'interpretation': sweep['interpretation'],
            'decline_ratio_grid': sweep['decline_ratio_grid'],
            't_half_relative_spread_grid': sweep['t_half_relative_spread_grid'],
            'acceptance_grid_axes': {
                'order': ['stimulus_efficacy', 'refill_tau_s', 'pool_capacity_pmol'],
                'values': {'stimulus_efficacy': ax['efficacies'],
                           'refill_tau_s': ax['refill_taus_s'],
                           'pool_capacity_pmol': ax['pool_sizes_pmol']},
            },
        },
        'clearance_sweep': jsonable(clearance),
        'shape_coupling_sweep': jsonable(coupling),
        'key_structural_claim': {
            'statement': ('HYPOTHESIS (not a measured mechanism): an aged change in the '
                          'releasable pool (size and/or refilling rate) reproduces '
                          '"release declines across stimulations while t1/2 stays the '
                          'same", whereas any change in clearance moves t1/2 and leaves '
                          'the release pattern flat'),
            'evidence_in_this_run': {
                'pool_only_t1_half_relative_change':
                    abs(aged.measured_t_half_s[0] - young.measured_t_half_s[0])
                    / young.measured_t_half_s[0],
                'pool_release_ratio_A6_A1': aged.release_ratio_last_first,
                'clearance_only_t1_half_relative_change':
                    abs(pc.measured_t_half_s[0] - young.measured_t_half_s[0])
                    / young.measured_t_half_s[0],
                'clearance_release_ratio_A6_A1': pc.release_ratio_last_first,
            },
            'caveat': ('because the pool and the release efficacy both scale the trace '
                       'amplitude, this qualitative pattern CANNOT by itself identify '
                       'which one changed, nor distinguish a pool change from any other '
                       'amplitude-only change'),
        },
        'what_a_real_fit_would_need': [
            'the absolute concentration-vs-time values of the source study, which are '
            'only in its figures: PDF retrieval was blocked (1,817-byte interception '
            'page) and no supplementary numeric table exists, so no curve digitisation '
            'or fitting was possible now',
            'an independently measured clearance rate constant and releasable pool '
            'capacity for this preparation (the source reports none)',
            'the detected volume / electrode sensitivity, to convert pmol to the nM '
            'reported by FSCV',
            'receptor subtype and release-probability measurements to justify the '
            'stimulus efficacy parameter',
            'per-region data (the source shows regionally inconsistent aged effects: '
            'central complex vs mushroom body)',
        ],
        'checklist_passed': all(c['passed'] for c in checks),
        'checks': checks,
        'honesty': HONESTY,
        'provenance_registry': {
            'summary': registry.summary(),
            'measured_names': registry.measured_names(),
            'unmeasured_names': registry.unmeasured_names(),
            'table': registry.snapshot(),
        },
        'wall_clock_seconds': None,
    }

    # ---- figure + artefacts ------------------------------------------------
    build_figure(protocol, results, sweep, clearance, coupling)
    traces = {}
    for name, r in results.items():
        traces[f'{name}__timeline_time_s'] = r.timeline_time_s
        traces[f'{name}__timeline_nM'] = r.timeline_nM
        traces[f'{name}__record_time_s'] = r.record_time_s
        traces[f'{name}__record_nM'] = r.record_nM
        traces[f'{name}__representative_nM'] = r.representative_nM
        traces[f'{name}__released_pmol'] = r.released_pmol
        traces[f'{name}__pool_before_pmol'] = r.pool_before_pmol
        traces[f'{name}__peak_nM'] = r.measured_peak_nM
        traces[f'{name}__t_half_s'] = r.measured_t_half_s
    traces['sweep_pool_sizes_pmol'] = np.array(sweep['axes']['pool_sizes_pmol'])
    traces['sweep_refill_taus_s'] = np.array(sweep['axes']['refill_taus_s'])
    traces['sweep_efficacies'] = np.array(sweep['axes']['efficacies'])
    traces['sweep_decline_ratio'] = np.array(sweep['decline_ratio_grid'])
    traces['sweep_acceptance'] = np.array(sweep['acceptance_grid'])
    traces['sweep_t_half_spread'] = np.array(sweep['t_half_relative_spread_grid'])
    np.savez_compressed(NPZ, **traces)

    report['wall_clock_seconds'] = time.perf_counter() - start
    JSON.write_text(json.dumps(jsonable(report), indent=2))

    print(json.dumps(jsonable({
        'png': str(PNG), 'json': str(JSON), 'npz': str(NPZ),
        'protocol_timing': timing,
        'checks': {'n_checks': len(checks),
                   'all_passed': all(c['passed'] for c in checks)},
        'scenarios': {name: {'released_pmol': [round(float(x), 6) for x in r.released_pmol],
                             'release_ratio_A6_A1': round(r.release_ratio_last_first, 6),
                             't_half_s_measured': [round(float(x), 6) for x in r.measured_t_half_s],
                             't_half_relative_spread': r.t_half_relative_spread,
                             'peak_nM_stim1': round(float(r.measured_peak_nM[0]), 4)}
                      for name, r in results.items()},
        'region': {'n_combinations': sweep['n_combinations'],
                   'n_accepted': sweep['n_accepted'],
                   'fraction_accepted': sweep['fraction_accepted'],
                   'summary_by_efficacy': region_text},
        'clearance_sweep': {
            'release_ratio_all_one':
                all(abs(row['release_ratio_last_first'] - 1.0) < 1e-8
                    for row in clearance['rows']),
            't_half_s_vs_clearance_tau_s':
                [[row['clearance_tau_s'], round(row['t_half_s'][0], 4)]
                 for row in clearance['rows']]},
        'wall_clock_seconds': report['wall_clock_seconds'],
    }), indent=2))


if __name__ == '__main__':
    main()
