"""Multirate composed-scenario demo: run the scenario set and write outputs.

Writes (all under outputs/brain_isolation/):
  * multirate_comparison.png  -- legibility-checked comparison figure
  * multirate_report.json     -- every metric, count, clock and honesty caveat
  * multirate_traces.npz      -- per-step traces for all scenarios

Run with: /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python -B
          /run/media/sensen/Data2/cell_wound_prototype/run_multirate_scenario.py
"""
from dataclasses import replace
from pathlib import Path
import json
import resource
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.multirate_scenario import (MultirateConfig, SubnetworkConfig, SCENARIOS,
                                       select_banc_subnetwork, run_multirate_scenario,
                                       NECK_ROTATION, NECK_TRANSLATION_UM)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'brain_isolation'
SUB_CFG = SubnetworkConfig(max_neurons=20000)
BASE = MultirateConfig(duration_ms=40.0)


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _pack(npz_payload, prefix, data):
    """Pack a scalar/array dict, turning lists-of-dicts into dicts-of-lists."""
    scalars, arrays, dicts = {}, {}, {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            dicts[key] = value
        elif isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value)
            if arr.dtype == object:
                dicts[key] = list(value)
            else:
                arrays[key] = arr
        else:
            scalars[key] = value
    for key, value in scalars.items():
        npz_payload[prefix + key] = np.asarray(value)
    for key, value in arrays.items():
        npz_payload[prefix + key] = np.asarray(value)
    for key, value in dicts.items():
        if not value:
            continue
        keys = sorted({k for item in value for k in item})
        for k in keys:
            column = [item.get(k) for item in value]
            if all(isinstance(v, (int, float, np.integer, np.floating, bool)) or v is None
                   for v in column):
                npz_payload[prefix + key + '.' + k] = np.array(
                    [np.nan if v is None else v for v in column])
            else:
                npz_payload[prefix + key + '.' + k] = np.array(
                    ['' if v is None else str(v) for v in column])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    print('[1/4] selecting the bounded deterministic BANC subnetwork ...', flush=True)
    sub = select_banc_subnetwork(SUB_CFG, ROOT)
    counts, excluded = sub['counts'], sub['excluded']
    print('      neurons=%d  exc_edges=%d  inh_edges=%d  visual_edges=%d'
          % (sub['n'], counts['exc_matrix_nnz'], counts['inh_matrix_nnz'],
             sub['visual']['pure_histamine_visual_rows_inside_subnetwork']), flush=True)

    print('[2/4] running %d scenarios on the shared subnetwork ...' % len(SCENARIOS), flush=True)
    results, timings = {}, {}
    for name in SCENARIOS:
        cfg = replace(BASE, scenario=name)
        before = _peak_rss_mb()
        t0 = time.perf_counter()
        results[name] = run_multirate_scenario(cfg, sub)
        timings[name] = {'wall_seconds': time.perf_counter() - t0,
                         'peak_rss_mb_before': before, 'peak_rss_mb_after': _peak_rss_mb()}
        m = results[name]['metrics']
        print('      %-16s wall %6.2fs  fine baseline/post %d/%d  coarse interface spikes %d'
              % (name, timings[name]['wall_seconds'], m['fine_tier_baseline_spikes'],
                 m['fine_tier_post_spikes'], m['coarse_interface_post_spikes']), flush=True)

    print('[3/4] writing multirate_traces.npz and multirate_report.json ...', flush=True)
    payload = {}
    for name, result in results.items():
        _pack(payload, name + '.', result['probes'])
        _pack(payload, name + '.', result['inputs'])
        payload[name + '.times_ms'] = result['times_ms']
        payload[name + '.input_times_ms'] = result['input_times_ms']
        for key, value in result['slow_history'].items():
            payload[name + '.slow_history.' + key] = np.asarray(value)
        payload[name + '.histamine.targets'] = result['histamine']['targets']
    np.savez_compressed(OUT / 'multirate_traces.npz', **payload)

    elapsed = time.perf_counter() - t_start
    report = {
        'status': 'completed',
        'what_this_is': 'one bounded multirate scenario joining BANC coarse tier, graded histamine '
                        'vision, injury chemistry and an idealized HH neck cable with an electrode',
        'what_this_is_not': 'NOT a whole-brain, whole-nervous-system, viability, survival or '
                            'consciousness model; NOT calibrated fly physiology',
        'outputs': {'figure': str(OUT / 'multirate_comparison.png'),
                    'report': str(OUT / 'multirate_report.json'),
                    'traces': str(OUT / 'multirate_traces.npz')},
        'scenarios': list(SCENARIOS),
        'scenario_equivalence': ('every scenario uses the same subnetwork, the same initial state '
                                 '(rest -65 mV, zero synaptic state, zero adaptation) and the same '
                                 'seed; only the scheduled intervention differs'),
        'subnetwork': {
            'selection_rule': sub['selection_rule'],
            'counts': counts, 'excluded_edge_accounting': excluded,
            'photo_seed_info': sub['photo_seed_info'],
            'visual_projection': {k: _jsonable(v) for k, v in sub['visual'].items()
                                  if not isinstance(v, np.ndarray) or v.ndim == 0},
            'digest': sub['digest'],
            'body_part_counts': sub['body_part_counts'],
            'dataset': sub['dataset'],
        },
        'regions': results['intact']['regions'],
        'runs': {name: {'config': results[name]['config'],
                        'metrics': _jsonable(results[name]['metrics']),
                        'timings': timings[name],
                        'provenance': _jsonable(results[name]['provenance'])}
                 for name in SCENARIOS},
        'paired_comparisons': {
            'neck_cut_vs_intact': {
                'fine_post_spikes': [results['intact']['metrics']['fine_tier_post_spikes'],
                                     results['neck_cut']['metrics']['fine_tier_post_spikes']],
                'coarse_interface_post_spikes': [
                    results['intact']['metrics']['coarse_interface_post_spikes'],
                    results['neck_cut']['metrics']['coarse_interface_post_spikes']],
                'reading': 'the cut removes body-side output while the brain-side coarse tier is '
                           'unchanged, which is structurally guaranteed by the ideal unidirectional '
                           'cable and is NOT evidence about real isolation resilience'},
            'eye_loss_vs_intact': {
                'histamine_delivered_uS': [results['intact']['metrics']['histamine_increment_total_uS'],
                                           results['eye_loss']['metrics']['histamine_increment_total_uS']],
                'post_phase_histamine_increment_uS': [
                    float(results['intact']['inputs']['histamine_increment_uS'][
                        results['intact']['input_times_ms'] >= BASE.phase_ms].sum()),
                    float(results['eye_loss']['inputs']['histamine_increment_uS'][
                        results['eye_loss']['input_times_ms'] >= BASE.phase_ms].sum())],
                'reading': 'zeroing the ideal eye interface removes the graded histamine drive to '
                           'its targets; it does not model eye damage or phototransduction loss'},
            'sham_vs_intact': {
                'identical_probes': all(
                    np.array_equal(results['intact']['probes'][k], results['sham']['probes'][k])
                    for k in results['intact']['probes']),
                'reading': 'sham reaches the scheduled phase and changes nothing'},
        },
        'measured_performance': {
            'wall_seconds_total_demo': elapsed,
            'wall_seconds_per_scenario': {k: v['wall_seconds'] for k, v in timings.items()},
            'peak_rss_mb_end_of_demo': _peak_rss_mb(),
            'peak_rss_mb_per_scenario': {k: v['peak_rss_mb_after'] for k, v in timings.items()},
            'machine_note': 'single process, single thread per sparse matmul; peak RSS is the '
                            'process high-water mark, so later scenarios reuse the same peak'},
        'honesty_limitations': LIMITATIONS,
    }
    (OUT / 'multirate_report.json').write_text(json.dumps(report, indent=2, default=str))

    print('[4/4] rendering the comparison figure ...', flush=True)
    render(results, sub, OUT / 'multirate_comparison.png')
    print('      total wall %.1fs  peak RSS %.0f MB' % (elapsed, _peak_rss_mb()), flush=True)
    print('      figure %s' % (OUT / 'multirate_comparison.png'), flush=True)
    print('      report %s' % (OUT / 'multirate_report.json'), flush=True)
    print('      traces %s' % (OUT / 'multirate_traces.npz'), flush=True)


LIMITATIONS = [
    'IDEAL_EYE_IF / IDEAL_NECK_RELAY_IF / IDEAL_BODY_IF are INVENTED placeholder cells. They are '
    'NOT anatomy and are absent from the BANC dataset.',
    'The composed eye->brain->neck->body path MIXES a real BANC subgraph with invented interface '
    'cells and an invented cable; only intra-subnetwork connectivity is data-derived.',
    'The BANC subnetwork is a deterministic SELECTION of at most ~20000 neurons out of 153962, not '
    'the whole nervous system and not the whole brain.',
    'All physiology parameters are ILLUSTRATIVE and UNCALIBRATED: point ConductanceParams, '
    'synapse->uS scale and per-edge synapse cap, graded-release scale, cable geometry, HH '
    'temperature, electrode conductivity/geometry, and every placeholder current gain.',
    'Both chemistry->neural gains (injury_k_target_gain_uS, injury_ligand_target_gain_uS) default '
    'to 0.0 and are labelled HYPOTHESIS; with the defaults the injury chemistry does NOT reach the '
    'neural tier.',
    'The reduced potassium chemistry is an externally clamped electroneutral salt exchange with an '
    'explicit reservoir ledger. It is NOT electrodiffusion: no pumps, no ATP, no charge or energy '
    'ledger, no survival meaning. The Nernst Ek is a diagnostic and can go extreme.',
    'The recorder signal is electrode artifact with an explicit zero neural contribution. It is '
    'never Vm and never a neural recording, and no source is reconstructed.',
    'No consciousness, viability, survival, recovery or "rescue" claim is made or implied.',
    'Transmitter labels do not measure receptor sign: ACh->exc and GABA/glutamate->inh are explicit '
    'assumptions, and mixed/unknown/histamine/modulator rows are excluded and counted, so the '
    'coarse graph is partial by construction.',
    'Graded visual drive uses only exact pure-histamine photoreceptor labels; mixed ACh/histamine '
    'photoreceptor rows are excluded, so the visual pathway here is explicitly incomplete.',
    'Only a small number of pure-histamine photoreceptors survive the connected-selection step: '
    "the degree-ranked core and a sparse photoreceptor's modelled partners usually lie in different "
    'components, so most requested seeds are dropped by the largest-connected-component step. The '
    'graded projection therefore reaches tens of targets, not thousands (see counts).',
    'The number of BANC neurons that fire, and their rates, follow from an illustrative constant '
    'placeholder current; they are a numerical regime choice, NOT a measured firing-rate prediction.',
    'Coarse-tier firing is driven by a constant illustrative placeholder current, so its rates are '
    'a numerical regime choice, not a prediction; the histogram of rates is not calibrated.',
    'The neck cable is an ideal straight uniform cylinder; a "neck cut" is a sealed axial '
    'disconnect of one section, not tissue damage.',
    'The electrode field is applied as an externally held extracellular voltage command, not a '
    'solved tissue field; the cable placement is an invented rigid transform.',
]


def render(results, sub, path):
    base = BASE
    names = list(SCENARIOS)
    colors = {'intact': '#1f77b4', 'sham': '#7f7f7f', 'neck_cut': '#d62728',
              'eye_loss': '#ff7f0e', 'injury_at_phase': '#2ca02c'}
    fig = plt.figure(figsize=(17.5, 11.5), layout='constrained')
    grid = fig.add_gridspec(3, 3)

    # 1. fine-tier distal (body-side) membrane voltage, with region shading
    ax = fig.add_subplot(grid[0, 0])
    for name in ('intact', 'sham', 'neck_cut'):
        r = results[name]
        ax.plot(r['times_ms'], r['probes']['neck_distal_mV'], lw=1.2,
                color=colors[name], label=name)
    ax.axvspan(0, base.phase_ms, color='gray', alpha=.12)
    ax.axvline(base.phase_ms, color='k', ls=':', lw=1)
    ax.set(xlabel='time (ms)', ylabel='distal cable mV (Vm)',
           title='(d) fine neck tier: body-side output voltage\n'
                 'shaded = baseline, cut at %.0f ms' % base.phase_ms)
    ax.legend(fontsize=7, loc='lower left')
    ax.text(.02, .99, 'ideal HH cylinder - NOT anatomy', transform=ax.transAxes,
            fontsize=7, va='bottom', color='dimgray')

    # 2. graded histamine: event increment and target conductance
    ax = fig.add_subplot(grid[0, 1])
    for name in ('intact', 'eye_loss'):
        r = results[name]
        ax.plot(r['input_times_ms'], r['inputs']['histamine_increment_uS'], lw=1.1,
                color=colors[name], label=name)
    ax.axvline(base.phase_ms, color='k', ls=':', lw=1)
    ax.set(xlabel='time (ms)', ylabel='g_hist event increment (uS)',
           title='(b) dedicated histamine channel (NOT the inhibitory channel)')
    ax.set_ylim(0, 0.52)
    ax.legend(fontsize=7, loc='upper left')
    axt = ax.twinx()
    axt.plot(results['intact']['times_ms'],
             results['intact']['probes']['histamine_target_g_hist_uS'], lw=1.0, ls='--',
             color='#1f77b4', alpha=.7, label='intact target sum (right)')
    axt.plot(results['eye_loss']['times_ms'],
             results['eye_loss']['probes']['histamine_target_g_hist_uS'], lw=1.0, ls='--',
             color='#ff7f0e', alpha=.7, label='eye_loss target sum (right)')
    axt.set_ylabel('target sum g_hist (uS)', fontsize=8)
    axt.legend(fontsize=6, loc='lower right', framealpha=.95)
    axt.tick_params(labelsize=7)

    # 3. chemical response and injury event
    ax = fig.add_subplot(grid[0, 2])
    for name in names:
        style = dict(lw=1.1, color=colors[name], label=name)
        if name != 'intact':
            style['alpha'] = .75
        ax.plot(results[name]['times_ms'], results[name]['probes']['chemical_response'], **style)
    ax.axvline(base.phase_ms, color='k', ls=':', lw=1)
    ax.set(xlabel='time (ms)', ylabel='ligand response (dimensionless)',
           title='(c) slow ligand chemistry clock (%.1f ms)'
                 % base.chemistry_dt_ms)
    ax.legend(fontsize=6, ncol=2)
    ax.text(.05, .84, 'injury_at_phase rewrites the barrier at %.0f ms' % base.phase_ms,
            transform=ax.transAxes, fontsize=7, color='dimgray', va='top')

    # 4. accumulated potassium / chemistry clocks (injury branch)
    ax = fig.add_subplot(grid[1, 0])
    r = results['injury_at_phase']
    ax.plot(r['times_ms'], r['probes']['outside_potassium_mM'], color='#2ca02c', lw=1.2,
            label='extracellular K (mM)')
    ax.axvline(base.phase_ms, color='k', ls=':', lw=1)
    ax.set(xlabel='time (ms)', ylabel='extracellular K (mM)',
           title='(c) reduced K chemistry (%.1f ms clock)' % base.potassium_dt_ms)
    ax2 = ax.twinx()
    ax2.plot(r['times_ms'], r['probes']['ek_mV'], color='#8c564b', lw=1.0, ls='--',
             label='Nernst Ek (mV, diagnostic)')
    ax2.set_ylabel('Ek (mV, diagnostic)', fontsize=8)
    ax2.tick_params(labelsize=7)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='center left')
    ax.text(.02, .97, 'electroneutral reduced exchange, NOT electrodiffusion',
            transform=ax.transAxes, fontsize=7, va='top', color='dimgray')

    # 5. artifact-only recorder
    ax = fig.add_subplot(grid[1, 1])
    r = results['intact']
    ax.plot(r['input_times_ms'], r['inputs']['artifact_recorded_mV'], color='#9467bd', lw=1.2)
    ax.set(xlabel='time (ms)', ylabel='recorded artifact (mV, filtered + noise)',
           title='(e) ARTIFACT-ONLY recorder output\nneural contribution is exactly 0 mV')
    win = ((r['input_times_ms'] >= base.stimulus_start_ms - 2)
           & (r['input_times_ms'] <= base.stimulus_start_ms + base.stimulus_duration_ms + 3))
    axins = ax.inset_axes([0.32, 0.52, 0.64, 0.36])
    axins.plot(r['input_times_ms'][win], r['inputs']['artifact_recorded_mV'][win],
               color='#9467bd', lw=1.2)
    axins.set_title('stimulus window', fontsize=7)
    axins.tick_params(labelsize=6)
    ax.text(.02, .10, 'never Vm, never a neural recording',
            transform=ax.transAxes, fontsize=7, color='dimgray')

    # 6. region label chart: who is real data and who is invented (grid slot 1,2)
    ax = fig.add_subplot(grid[1, 2])
    ax.axis('off')
    ax.set_title('(a) region provenance', fontsize=11)
    rows = [
        ('region', 'cells', 'source'),
        ('IDEAL_EYE_IF', '1 point', 'INVENTED interface cell'),
        ('IDEAL_NECK_RELAY_IF', '1 point', 'INVENTED interface cell'),
        ('IDEAL_BODY_IF', '1 point', 'INVENTED interface cell'),
        ('BANC_SELECTED', '%d points' % sub['n'], 'REAL BANC subnetwork (selection)'),
        ('neck cable', '%d segs' % (base.neck_sections * base.neck_nseg_per_section),
         'INVENTED ideal HH cylinder'),
        ('recorder', '1 point', 'INVENTED artifact-only probe'),
    ]
    for i, (a, b, c) in enumerate(rows):
        weight = 'bold' if i == 0 else 'normal'
        colour = 'black' if i == 0 or 'REAL' in c else 'darkred'
        ax.text(0.0, 0.93 - i * 0.11, a, fontsize=8, weight=weight, color=colour)
        ax.text(0.40, 0.93 - i * 0.11, b, fontsize=8, weight=weight, color=colour)
        ax.text(0.62, 0.93 - i * 0.11, c, fontsize=8, weight=weight, color=colour)
    ax.text(0.0, 0.16, 'BANC subnetwork: %d of %d neurons, %d of %d annotated rows\n'
                       'pure-histamine photoreceptors kept: %d, visual rows: %d'
            % (sub['n'], sub['counts']['neurons_total_in_dataset'],
               sub['excluded']['retained_rows'], sub['excluded']['total_annotated_rows_in_dataset'],
               sub['counts']['pure_histamine_photoreceptors_in_final'],
               sub['visual']['pure_histamine_visual_rows_inside_subnetwork']),
            fontsize=8, color='black')
    ax.text(0.0, 0.03, 'red = invented, not anatomy', fontsize=8, color='darkred')

    # 7. body-side output spike counts
    ax = fig.add_subplot(grid[2, 0])
    width = .36
    x = np.arange(len(names))
    base_counts = [results[n]['metrics']['fine_tier_baseline_spikes'] for n in names]
    post_counts = [results[n]['metrics']['fine_tier_post_spikes'] for n in names]
    ax.bar(x - width / 2, base_counts, width, label='baseline (<= %.0f ms)' % base.phase_ms,
           color='#aec7e8', edgecolor='k', lw=.4)
    ax.bar(x + width / 2, post_counts, width, label='post phase', color='#1f77b4', edgecolor='k', lw=.4)
    for xi, (b, p) in enumerate(zip(base_counts, post_counts)):
        ax.text(xi - width / 2, b + .05, str(b), ha='center', fontsize=8)
        ax.text(xi + width / 2, p + .05, str(p), ha='center', fontsize=8)
    ax.set_xticks(x, names, rotation=16, fontsize=8)
    ax.set(ylabel='body-side distal spikes', ylim=(0, max(base_counts + post_counts) + 1),
           title='(d) body-side output spikes per scenario')
    ax.legend(fontsize=7)

    # 8. visual target drive, intact vs eye_loss
    ax = fig.add_subplot(grid[2, 1])
    width = .36
    x = np.arange(2)
    post_mask = {n: results[n]['times_ms'] >= base.phase_ms for n in ('intact', 'eye_loss')}
    means = [float(np.mean(results[n]['probes']['histamine_target_mean_mV'][post_mask[n]]))
             for n in ('intact', 'eye_loss')]
    ghists = [float(np.mean(results[n]['probes']['histamine_target_g_hist_uS'][post_mask[n]]))
              for n in ('intact', 'eye_loss')]
    ax.bar(x - width / 2, means, width, color='#ffbb78', edgecolor='k', lw=.4)
    ax.set_xticks(x, ['intact', 'eye_loss'])
    ax.set(ylabel='mean target Vm after phase (mV)',
           title='(b) graded histamine drive reaches its targets')
    for xi, value in enumerate(means):
        ax.text(xi - width / 2, value - 2.5, '%.1f' % value, ha='center', fontsize=8, color='k')
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, ghists, width, color='#2ca02c', edgecolor='k', lw=.4)
    ax2.set_ylabel('mean target sum g_hist (uS)', fontsize=8)
    for xi, value in enumerate(ghists):
        ax2.text(xi + width / 2, value + .8, '%.2f' % value, ha='center', fontsize=8, color='k')
    ax.set_ylim(min(means) - 8, max(means) + 3)
    ax2.set_ylim(0, max(ghists) * 1.3)

    # 9. honesty notes
    ax = fig.add_subplot(grid[2, 2])
    ax.axis('off')
    ax.set_title('Honesty: what this run does NOT establish', fontsize=11)
    notes = [
        'IDEAL_*_IF cells are INVENTED, not anatomy.',
        'BANC tier is a SELECTION (<= ~20000 of 153962 neurons).',
        'eye->brain->neck->body mixes real BANC with invented cells.',
        'All parameters illustrative / uncalibrated.',
        'Chemistry->neural gains default 0 (HYPOTHESIS).',
        'Reduced K chemistry is NOT electrodiffusion.',
        'Recorder output is artifact only - never Vm.',
        'No consciousness / viability / survival / rescue claim.',
        'Unidirectional ideal cable makes "brain survives the cut"',
        '   structurally trivial, not a resilience finding.',
        'Mixed ACh/histamine visual rows are excluded and counted.',
    ]
    for i, line in enumerate(notes):
        ax.text(0.0, 0.94 - i * 0.085, line, fontsize=8)

    fig.suptitle('Multirate scenario: real BANC subnetwork + graded histamine vision + reduced '
                 'injury chemistry + idealized HH neck + artifact-only recorder\n'
                 'BOUNDED COMPOSED PROTOTYPE - illustrative, uncalibrated, NOT a survival or '
                 'consciousness model', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
