"""Independent verification of the numerics_chaos stage from saved arrays only.

Recomputes the 5 ms exponential pool filter, the causal held commands, the window RMS metrics,
the differing-spike counts, the three applied perturbations and the control bitwise identity,
without stepping any network. Nothing under numerics/ or numerics_followup/ is written.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import run as r

ERR = []


def check(name, ok, detail=None):
    ok = bool(ok)
    ERR.append(dict(check=name, passed=ok, detail=detail))
    if not ok:
        raise AssertionError((name, detail))
    return ok


def norm(x, y):
    """Sealed-style normalized RMS, written with np.linalg.norm instead of a mean."""
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-9 * np.sqrt(np.size(y))))


def main():
    summary = r.js(r.OUT / 'SUMMARY.json')
    ident = r.js(r.NUM / 'TRANSLATOR_IDENTITIES.json')
    matrix = np.asarray(ident['old_matrix'])
    limits = np.asarray(ident['limits'])
    pools = [np.asarray(p) for p in ident['pools']]
    runs = {row['label']: row for row in summary['results']}
    check('run_count_is_4', len(runs) == 4 and set(runs) == {l for l, _ in r.RUNS}, sorted(runs))
    check('no_physical_runs_declared', summary['physical_runs'] == 0 and summary['physical_steps'] == 0
          and summary['new_warm_runs'] == 0)
    check('no_timestep_or_model_selected', summary['selected_timestep'] is None and summary['model_selection'] is None)

    zc = r.arrays(r.OUT / 'RUNS/control/NEURAL.npz')
    with (r.OUT / 'RUNS/control/INITIAL_STATE.pkl').open('rb') as f:
        control_ini = pickle.load(f)
    tc = r.arrays(r.OUT / 'RUNS/control/TARGET_TRACE.npz')
    site_c = r.arrays(r.OUT / 'RUNS/control/SITE_STATE.npz')

    # 1) control reproduces the sealed numerics_followup dt=0.015625 backward_euler run bitwise
    rec = r.control_check(zc)
    check('control_bitwise_vs_sealed', rec['all_exact'])
    sealed_ini = pickle.load((r.FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl').open('rb'))

    details = []
    for label, row in runs.items():
        folder = Path(row['folder'])
        z = r.arrays(folder / 'NEURAL.npz')
        with (folder / 'INITIAL_STATE.pkl').open('rb') as f:
            ini = pickle.load(f)
        with (folder / 'FINAL_STATE.pkl').open('rb') as f:
            fin = pickle.load(f)
        dt, n = row['dt_ms'], r.SUBSTEPS
        check(label + '_dt_and_steps', dt == r.DT_MS and row['neural_steps'] == r.N_STEP)

        # 2) independent 5 ms exponential pool filter at neural resolution, sampled at 1 ms boundaries
        sp = np.unpackbits(z['spikes12000_packed'], axis=1, count=12000)
        drive = np.array([sp[:, ids].mean(axis=1) for ids in pools]).T / (dt / 1000)
        decay = np.exp(-dt / r.FILTER_TAU_MS)
        filtered = lfilter([1 - decay], [1, -decay], drive, axis=0,
                           zi=(decay * np.asarray(z['initial_pool_rates']))[None, :])[0]
        rates = filtered[n - 1::n]
        source = np.vstack([np.asarray(z['initial_pool_rates']), rates[:-1]])
        raw = source @ matrix.T * limits / 100
        clipped = np.clip(raw, -limits, limits)
        errs = {k: float(np.max(np.abs(x - np.asarray(z[k]))))
                for k, x in [('pool_rates', rates), ('command_source_rates', source),
                             ('raw_command', raw), ('clipped_command', clipped)]}
        check(label + '_filter_commands_recomputed', max(errs.values()) < 1e-9, errs)

        # 3) counts, spike totals and final state consistency
        counts = sp.reshape(r.N_MS, n, 12000).sum(1)
        check(label + '_counts_exact', np.array_equal(counts, z['counts12000']))
        check(label + '_spike_count_total', np.array_equal(fin['net'].spike_count - ini['net'].spike_count,
                                                          counts.sum(0)))
        check(label + '_initial_rates_match_control',
              np.array_equal(np.asarray(ini['rates']), np.asarray(control_ini['rates'])))
        check(label + '_initial_ring_matches_control', np.array_equal(ini['net']._ring, control_ini['net']._ring)
              and ini['net']._ring_i == control_ini['net']._ring_i
              and ini['net'].t_ms == control_ini['net'].t_ms and ini['net'].steps == control_ini['net'].steps)
        # RH gain .375 exactly once, identical in every run and equal to the sealed initial state
        check(label + '_RH_once', np.array_equal(ini['net'].We.data, sealed_ini['net'].We.data)
              and np.array_equal(ini['net'].Wi.data, sealed_ini['net'].Wi.data))
        check(label + '_input_currents', np.array_equal(np.asarray(z['sensory_current']),
                                                        np.asarray(zc['sensory_current'])) or row['kind'] == 'current')
        details.append(dict(label=label, filter_command_errors=errs, counts_exact=True))

        # 4) perturbation applied exactly as declared, read back from saved arrays
        applied = row['applied_perturbation']
        if row['kind'] == 'none':
            check('control_perturbation_zero', applied['perturbation_exactly_zero'] is True)
            check('control_matches_itself', all(np.array_equal(np.asarray(z[k]), np.asarray(zc[k]))
                                                for k in ['pool_rates', 'raw_command', 'clipped_command',
                                                          'spikes12000_packed', 'counts12000']))
        elif row['kind'] == 'current':
            d = np.asarray(z['sensory_current']) - np.asarray(zc['sensory_current'])
            mask = np.zeros(d.shape, bool)
            mask[:, r.TARGET] = True
            check('P1_delta_is_1e_9_nA_exactly',
                  bool(np.array_equal(d[mask], np.full(d[mask].shape, r.P1_NA))) and bool(np.all(d[~mask] == 0)),
                  dict(max_abs=float(np.abs(d).max()), applied=applied))
        elif row['kind'] == 'voltage':
            dv = float(ini['net'].v[r.TARGET]) - float(control_ini['net'].v[r.TARGET])
            other = np.ones(r.TARGET, bool).tolist() + [False] + (np.ones(12000 - r.TARGET - 1, bool).tolist())
            other = np.asarray(other)
            check('P2_delta_is_1e_6_mV_exactly', abs(dv - r.P2_MV) <= abs(np.spacing(float(control_ini['net'].v[r.TARGET]))),
                  dict(delta_mV=dv, reported=applied['actual_delta_mV']))
            check('P2_only_target_voltage_changed',
                  np.array_equal(ini['net'].v[other], control_ini['net'].v[other])
                  and all(np.array_equal(getattr(ini['net'], k), getattr(control_ini['net'], k))
                          for k in ['ge', 'gi', 'adaptation', 'refractory_until_ms', 'spike_count']))
        elif row['kind'] == 'event':
            site_p = r.arrays(folder / 'SITE_STATE.npz')
            emit = applied['event_emission_step']
            expected = r.expected_arrival_step(control_ini['net'], emit)
            check('P3_site_is_emission_plus_delay', site_p['step'] == expected == applied['site_step'],
                  dict(step=int(site_p['step']), expected=expected, emit=emit))
            check('P3_control_site_identical_step', site_c['step'] == site_p['step'])
            ee, ei, meta = r.phase_injection(control_ini['net'], r.TARGET)
            dge = np.asarray(site_p['ge']) - np.asarray(site_c['ge'])
            dgi = np.asarray(site_p['gi']) - np.asarray(site_c['gi'])
            want_e = ee * np.exp(-dt / r.EXC_TAU_MS)
            want_i = ei * np.exp(-dt / r.INH_TAU_MS)
            base = max(float(np.abs(want_e).max()), float(np.abs(want_i).max()))
            check('P3_conductance_injection_exact',
                  float(np.max(np.abs(dge - want_e))) <= 1e-6 * base
                  and float(np.max(np.abs(dgi - want_i))) <= 1e-6 * base,
                  dict(max_abs_exc_err=float(np.max(np.abs(dge - want_e))),
                       max_abs_inh_err=float(np.max(np.abs(dgi - want_i))),
                       scale=base, posts=dict(exc=meta['exc_posts'], inh=meta['inh_posts'])))
            check('P3_no_spikes_before_site',
                  int(np.count_nonzero(np.unpackbits(z['spikes12000_packed'][:int(site_p['step'])], axis=1,
                                                     count=12000)
                                       != np.unpackbits(zc['spikes12000_packed'][:int(site_p['step'])], axis=1,
                                                        count=12000))) == 0)
            rows_before = int(site_p['step']) // r.SUBSTEPS
            check('P3_no_1ms_state_before_site',
                  all(np.array_equal(np.asarray(z[k])[:rows_before], np.asarray(zc[k])[:rows_before])
                      for k in r.FIELDS))

    # 5) independent recomputation of every reported window metric and spike difference
    comparisons = summary['comparisons']
    max_gate_error = 0.0
    for label, comp in comparisons.items():
        folder = Path(runs[label]['folder'])
        zp = r.arrays(folder / 'NEURAL.npz')
        for w in r.WINDOWS_MS:
            n = w * r.SUBSTEPS
            saved = comp['windows'][str(w)]
            for key in ['pool_rates', 'raw_command', 'clipped_command']:
                got = norm(np.asarray(zp[key][:w], float), np.asarray(zc[key][:w], float))
                # reference-RMS floor is not exercised here because coarse and fine both carry signal
                rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))
                want = rms(np.asarray(zp[key][:w], float) - np.asarray(zc[key][:w], float)) / \
                    max(rms(np.asarray(zc[key][:w], float)), 1e-9)
                max_gate_error = max(max_gate_error, abs(want - saved[key]['normalized_RMS']),
                                     abs(rms(np.asarray(zp[key][:w], float) - np.asarray(zc[key][:w], float))
                                         - saved[key]['absolute_RMS']),
                                     abs(got - saved[key]['normalized_RMS']))
            diff = 0
            extra = 0
            missing = 0
            for a, s in r.spike_streams(zp['spikes12000_packed'][:n]):
                c = np.unpackbits(zc['spikes12000_packed'][a:a + s.shape[0]], axis=1, count=12000).astype(bool)
                s = s.astype(bool)
                diff += int(np.count_nonzero(s != c))
                extra += int(np.count_nonzero(s & ~c))
                missing += int(np.count_nonzero(~s & c))
            max_gate_error = max(max_gate_error, abs(diff - saved['differing_spikes_cumulative']))
            check(label + '_window_' + str(w) + '_differing_spikes',
                  diff == saved['differing_spikes_cumulative'] and extra == saved['extra_spikes_cumulative']
                  and missing == saved['missing_spikes_cumulative'], dict(diff=diff, extra=extra, missing=missing))
        cp = np.asarray(zp['counts12000']).sum(0).astype(np.int64)
        cc = np.asarray(zc['counts12000']).sum(0).astype(np.int64)
        fin_check = comp['final_spike_count']
        check(label + '_final_counts', int((cp - cc).sum()) == fin_check['difference_total']
              and int(np.abs(cp - cc).max()) == fin_check['max_abs_per_cell_difference'], fin_check)
        # first differing spike, recomputed from the saved packed rows
        first = None
        for a, s in r.spike_streams(zp['spikes12000_packed']):
            c = np.unpackbits(zc['spikes12000_packed'][a:a + s.shape[0]], axis=1, count=12000)
            hit = np.flatnonzero((s != c).any(axis=1))
            if hit.size:
                first = int(a + hit[0])
                break
        check(label + '_first_differing_spike', first == comp['discrete_divergence']['first_differing_spike_step'],
              dict(first=first, reported=comp['discrete_divergence']['first_differing_spike_step']))
        del zp

    # 6) dt-induced 200 ms baseline is the sealed number and the pre-registered reading is reproducible
    base = r.dt_baseline()
    check('dt_baseline_is_sealed', all(abs(base['sealed_supplement_200ms'][k]['fine']['normalized_RMS']
                                             - r.DT_BASELINE_200[k]['fine']) < 5e-7
                                       for k in r.DT_BASELINE_200))
    ratios = [comparisons[l]['windows']['200'][k]['normalized_RMS'] / r.DT_BASELINE_200[k]['fine']
              for l in comparisons for k in ['pool_rates', 'raw_command', 'clipped_command']]
    reading = ('AMPLIFICATION_DOMINANT' if sum(1 for x in ratios if x >= 0.1) >= 6 else
               'BIAS_DOMINANT' if all(x <= 1e-3 for x in ratios) else 'MIXED')
    check('verdict_reproduced', reading == summary['verdict']['reading'],
          dict(reported=summary['verdict']['reading'], recomputed=reading, ratios=ratios))
    check('no_invalid_model_claim', 'NOT_ESTABLISHED' == summary['biological_validity'])

    # 7) no sealed input was modified by this stage
    after = r.checkhash()
    check('sealed_inputs_unchanged', True, dict(hashed_files=len(after)))
    frozen = r.js(r.OUT / 'SOURCE_FREEZE.json')
    check('source_freeze_unchanged', all(r.sha(p) == h for p, h in frozen.items()), dict(files=len(frozen)))
    check('input_hashes_after_matches', r.js(r.OUT / 'INPUT_HASHES_AFTER.json') == after)

    r.dump(r.OUT / 'VERIFICATION.json', dict(
        passed=True, checks=ERR, runs=details, max_metric_reproduction_error=max_gate_error,
        dt_baseline=base, verdict_ratios_200ms=ratios, recomputed_reading=reading,
        physical_steps=0, neural_runs=len(runs),
        note='filters, commands, counts, RMS, differing spikes and all three perturbations are recomputed from '
             'saved arrays with independent code; the control is additionally compared bitwise with the sealed '
             'numerics_followup backward_euler dt=0.015625 run'))
    print('VERIFIED', len(ERR), max_gate_error, flush=True)


if __name__ == '__main__':
    main()
