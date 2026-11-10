"""Step 5: build the tables, the figure, SUMMARY.json, the inventory, the manifest and the
Chinese report from the stored artifacts.  Every number comes from GAIN_METRICS_OFFLINE.json
(simulation) or VERIFICATION.json (independent recomputation); this file computes no physics.
"""
import sys
import csv
import json
import datetime
from pathlib import Path

import numpy as np

import common as C

js, sha, dump = C.js, C.sha, C.dump

BLOCKS = []          # (gain, dt, readout, [(phase, result), ...])
OFF = None
VER = None


def load():
    global OFF, VER
    OFF = js(C.O / 'GAIN_METRICS_OFFLINE.json')
    VER = js(C.O / 'VERIFICATION.json')
    runs = sorted(p for p in (C.O / 'RUNS').glob('*') if p.is_dir())
    groups = {}
    for folder in runs:
        r = js(folder / 'RESULT.json')
        key = (float(r['gain_nA_per_unit']), float(r['dt_s']), r['readout'])
        groups.setdefault(key, []).append(r)
    for key in sorted(groups, key=lambda k: (-k[0], k[1], k[2])):
        rows = sorted(groups[key], key=lambda r: r['release_time_s'])
        BLOCKS.append((key[0], key[1], key[2], rows))


def table_physical():
    out = []
    for gain, dt, readout, rows in BLOCKS:
        for r in rows:
            p = r['parts']
            c = r['validation']['checks']
            out.append(dict(
                gain=gain, dt_ms=dt * 1000, readout=readout,
                phase=r['release_time_s'], J=r['J'],
                foot_support_mean=p['foot_support_mean'], foot_support_p10=p['foot_support_p10'],
                com_z_min=p['com_z_min'], nonfoot_share=p['nonfoot_share'],
                free_fall_fraction=p['free_fall_fraction'],
                bearing_legs_median=p['bearing_legs_median'],
                first_nonfoot_ms=(None if r['first_positive_nonfoot_s'] is None
                                  else r['first_positive_nonfoot_s'] * 1000),
                D_pass=int(sum(c.values())), D_total=len(c),
                D_passed_all=bool(r['validation']['passed']),
                saturation=r['applied_saturation_fraction'],
                command_rms=r['applied_command_rms'],
                readout_r2_held=r['readout_r2_held'],
                failed_checks=[k for k, v in c.items() if not v]))
    return out


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def rate_rows():
    rows = []
    for gain in C.GAINS:
        d = OFF['results']['%g' % gain]['diagnostics']
        row = dict(gain=gain, population_mean_hz=d['all12000']['mean_hz'],
                   population_median_hz=d['all12000']['median_hz'],
                   population_silent_fraction=d['all12000']['silent_fraction'],
                   hyper_share_above_100hz=d['all12000']['hyper_share_above_100hz'],
                   total_spikes=d['all12000']['total_spikes'],
                   readout204_mean_hz=d['readout_204']['mean_hz'],
                   readout204_median_hz=d['readout_204']['median_hz'],
                   readout204_max_hz=d['readout_204']['max_hz'],
                   motor_cell_temporal_cv_median=d['readout_204']['temporal_cv_median'],
                   readout204_silent_fraction=d['readout_204']['silent_fraction'])
        for cls, v in d['per_super_class'].items():
            row['class:%s:mean_hz' % cls] = v['mean_hz']
            row['class:%s:median_hz' % cls] = v['median_hz']
            row['class:%s:silent' % cls] = v['silent_fraction']
        rows.append(row)
    return rows


def info_rows():
    rows = []
    for gain in C.GAINS:
        i = OFF['results']['%g' % gain]['information']
        v = VER['checks']['_summary']['per_gain']['%g' % gain]['information']
        rows.append(dict(
            gain=gain,
            motor204_r2_fit=i['motor_204']['r2_fit'], motor204_r2_held=i['motor_204']['r2_held'],
            motor204_gamma=i['motor_204']['gamma'], motor204_alpha=i['motor_204']['alpha'],
            motor204_r2_held_recomputed=v['motor_204']['r2_held_recomputed'],
            all12000_r2_fit=i['all_12000']['r2_fit'],
            all12000_r2_held=i['all_12000']['r2_held'],
            all12000_gamma=i['all_12000']['gamma'], all12000_alpha=i['all_12000']['alpha'],
            all12000_r2_held_recomputed=v['all_12000']['r2_held_recomputed'],
            offset_rms_204=i['motor_204']['offset_rms'],
            feature_rms_fit_hz_204=i['motor_204']['feature_rms_fit_hz'],
            dense_relieved=OFF['results']['%g' % gain]['dense_relief']['DENSE_RELIEVED']))
    return rows


def figure(phys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    gains = C.GAINS
    col = {0.3: '#1f77b4', 0.1: '#2ca02c', 0.03: '#ff7f0e', 0.01: '#d62728'}

    # (a) rate regime vs gain
    ax = axes[0][0]
    pop = [OFF['results']['%g' % g]['diagnostics']['all12000']['mean_hz'] for g in gains]
    med = [OFF['results']['%g' % g]['diagnostics']['all12000']['median_hz'] for g in gains]
    r204 = [OFF['results']['%g' % g]['diagnostics']['readout_204']['mean_hz'] for g in gains]
    sen = [OFF['results']['%g' % g]['diagnostics']['per_super_class']['sensory']['mean_hz']
           for g in gains]
    mot = [OFF['results']['%g' % g]['diagnostics']['per_super_class']['motor']['mean_hz']
           for g in gains]
    for ys, lab in [(sen, 'sensory mean'), (r204, 'readout-204 mean'), (mot, 'motor mean'),
                    (pop, 'all-12000 mean'), (med, 'all-12000 median')]:
        ax.plot(gains, np.maximum(ys, 1e-2), 'o-', label=lab)
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('sensory encoder gain (nA per unit)'); ax.set_ylabel('mean rate (Hz)')
    ax.set_title('(a) rate regime vs gain, 0.5 ms, 0.2-1.4 s')
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # (b) information vs gain
    ax = axes[0][1]
    m = [OFF['results']['%g' % g]['information']['motor_204']['r2_held'] for g in gains]
    a = [OFF['results']['%g' % g]['information']['all_12000']['r2_held'] for g in gains]
    ax.plot(gains, m, 'o-', color='#9467bd', label='204 motor cells (held R2)')
    ax.plot(gains, a, 's-', color='#8c564b', label='all 12000 cells (held R2)')
    ax.axhline(C.RESTORED_MOTOR_R2, color='red', ls='--', lw=1,
               label='pre-registered 0.7 bar')
    ax.axhline(C.BASELINE_MOTOR_R2, color='gray', ls=':', lw=1, label='parent 0.43 (current gain)')
    for x, y in zip(gains, m):
        ax.annotate('%.3f' % y, (x, y), textcoords='offset points', xytext=(4, -10), fontsize=8)
    ax.set_xscale('log'); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel('sensory encoder gain (nA per unit)'); ax.set_ylabel('held-out R2 (0.8-1.4 s)')
    ax.set_title('(b) linearly decodable teacher torque vs gain')
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # (c) total spikes
    ax = axes[0][2]
    sp = [OFF['results']['%g' % g]['diagnostics']['all12000']['total_spikes'] for g in gains]
    ax.plot(gains, sp, 'o-', color='#17becf')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('sensory encoder gain (nA per unit)'); ax.set_ylabel('spikes in 1.2 s')
    ax.set_title('(c) total spikes (12000 cells, 0.2-1.4 s)')
    ax.grid(alpha=.3)

    # (d) J per phase per gain
    ax = axes[1][0]
    for (gain, dt, readout, rows) in BLOCKS:
        ph = [r['release_time_s'] for r in rows]
        jj = [r['J'] for r in rows]
        ax.plot(ph, jj, 'o--' if readout == 'all12000' else 'o-',
                color=col[gain], alpha=.9 if readout == 'motor204' else .5,
                label='gain %g %s %.4g ms' % (gain, readout, dt * 1000))
    ax.set_xlabel('release phase (s)'); ax.set_ylabel('J (lower is better)')
    ax.set_title('(d) physical objective per phase')
    ax.legend(fontsize=6); ax.grid(alpha=.3)

    # (e) foot support mean per phase
    ax = axes[1][1]
    for (gain, dt, readout, rows) in BLOCKS:
        ax.plot([r['release_time_s'] for r in rows],
                [r['parts']['foot_support_mean'] for r in rows], 'o-', color=col[gain],
                alpha=.9 if readout == 'motor204' else .5,
                label='gain %g %s %.4g ms' % (gain, readout, dt * 1000))
    ax.axhline(1.0, color='k', ls=':', lw=1, label='assisted reference 1.0 W')
    ax.axhline(0.9, color='red', ls='--', lw=1, label='D3 floor (assist-0.10)')
    ax.axhline(0.5, color='orange', ls='--', lw=1, label='D4 p10 floor 0.5')
    ax.set_xlabel('release phase (s)'); ax.set_ylabel('mean foot support (body weights)')
    ax.set_title('(e) foot support after release'); ax.legend(fontsize=6); ax.grid(alpha=.3)

    # (f) min COM per phase
    ax = axes[1][2]
    for (gain, dt, readout, rows) in BLOCKS:
        ax.plot([r['release_time_s'] for r in rows],
                [r['parts']['com_z_min'] for r in rows], 'o-', color=col[gain],
                alpha=.9 if readout == 'motor204' else .5,
                label='gain %g %s %.4g ms' % (gain, readout, dt * 1000))
    ax.axhline(0.8 * 1.2072528224791905, color='red', ls='--', lw=1, label='D8 floor 0.9658')
    ax.axhline(1.2374996556699485, color='k', ls=':', lw=1, label='assisted COM mean')
    ax.set_xlabel('release phase (s)'); ax.set_ylabel('min COM height (model length)')
    ax.set_title('(f) minimum COM after release'); ax.legend(fontsize=6); ax.grid(alpha=.3)

    fig.suptitle('S5 gain sweep: sensory encoder gain vs sparseness, motor information and support '
                 '(engineering parameter sweep; no walking claim)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = C.O / 'FIGURES' / 'GAIN_SWEEP.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.savefig(C.O / 'GAIN_SWEEP.png', dpi=150)
    plt.close(fig)
    return path


def block_rows():
    out = {}
    for gain, dt, readout, rows in BLOCKS:
        key = '%s|%s|%s' % (C.gain_tag(gain), C.dt_tag(dt), readout)
        per = [dict(phase=r['release_time_s'], J=r['J'],
                    foot_support_mean=r['parts']['foot_support_mean'],
                    foot_support_p10=r['parts']['foot_support_p10'],
                    com_z_min=r['parts']['com_z_min'],
                    nonfoot_share=r['parts']['nonfoot_share'],
                    free_fall_fraction=r['parts']['free_fall_fraction'],
                    bearing_legs_median=r['parts']['bearing_legs_median'],
                    first_nonfoot_ms=(None if r['first_positive_nonfoot_s'] is None
                                      else r['first_positive_nonfoot_s'] * 1000),
                    D_pass=int(sum(r['validation']['checks'].values())),
                    D_total=len(r['validation']['checks']),
                    D_passed_all=bool(r['validation']['passed']),
                    saturation=r['applied_saturation_fraction'],
                    failed=[k for k, v in r['validation']['checks'].items() if not v])
               for r in rows]
        J = np.array([p['J'] for p in per])
        out[key] = dict(gain=gain, dt_s=dt, readout=readout, per_phase=per,
                        J=dict(min=float(J.min()), median=float(np.median(J)),
                               max=float(J.max())),
                        D_pass_counts=[p['D_pass'] for p in per],
                        phases_passing_all=[p['phase'] for p in per if p['D_passed_all']],
                        phases_passing_all_count=int(sum(p['D_passed_all'] for p in per)),
                        best_D_pass_count=int(max(p['D_pass'] for p in per)),
                        median_support=float(np.median([p['foot_support_mean'] for p in per])),
                        min_com=float(min(p['com_z_min'] for p in per)),
                        saturation_max=float(max(p['saturation'] for p in per)))
    return out


def dose_response(blocks):
    """Gain dose-response summary, per timestep and readout.  No verdict, only the shape."""
    out = {}
    for dt in C.DTS:
        for readout in ('motor204', 'all12000'):
            keys = [k for k, v in blocks.items() if v['dt_s'] == dt and v['readout'] == readout]
            if not keys:
                continue
            rows = []
            for k in sorted(keys, key=lambda k: -blocks[k]['gain']):
                v = blocks[k]
                rows.append(dict(gain=v['gain'], J_median=v['J']['median'],
                                 J_min=v['J']['min'], J_max=v['J']['max'],
                                 D_pass_counts=v['D_pass_counts'],
                                 best_D_pass_count=v['best_D_pass_count'],
                                 phases_passing_all=v['phases_passing_all_count'],
                                 median_foot_support=v['median_support'],
                                 min_com=v['min_com'],
                                 saturation_max=v['saturation_max'],
                                 motor_r2_held=OFF['results']['%g' % v['gain']]
                                 ['information']['motor_204']['r2_held']))
            Js = [r['J_median'] for r in rows]
            out['%s|%s' % (C.dt_tag(dt), readout)] = dict(
                rows=rows,
                J_median_increases_as_gain_falls=bool(all(Js[i] < Js[i + 1]
                                                          for i in range(len(Js) - 1))),
                D_count_increases_as_gain_falls=bool(all(
                    rows[i]['best_D_pass_count'] <= rows[i + 1]['best_D_pass_count']
                    for i in range(len(rows) - 1))),
                note='per-phase numbers are in per_phase; nothing here averages phases into a '
                     'verdict')
    return out


def main():
    load()
    phys = table_physical()
    rrows = rate_rows()
    irows = info_rows()
    write_csv(C.O / 'RATE_TABLE.csv', rrows,
              ['gain', 'population_mean_hz', 'population_median_hz', 'population_silent_fraction',
               'hyper_share_above_100hz', 'total_spikes', 'readout204_mean_hz',
               'readout204_median_hz', 'readout204_max_hz',
               'motor_cell_temporal_cv_median', 'readout204_silent_fraction']
              + [k for k in rrows[0] if k.startswith('class:')])
    write_csv(C.O / 'INFORMATION_TABLE.csv', irows,
              ['gain', 'motor204_r2_fit', 'motor204_r2_held', 'motor204_gamma', 'motor204_alpha',
               'motor204_r2_held_recomputed', 'all12000_r2_fit', 'all12000_r2_held',
               'all12000_gamma', 'all12000_alpha', 'all12000_r2_held_recomputed',
               'offset_rms_204', 'feature_rms_fit_hz_204', 'dense_relieved'])
    write_csv(C.O / 'RATE_BY_CLASS.csv',
              [dict(gain=g, cls=cls, **v)
               for g in C.GAINS
               for cls, v in OFF['results']['%g' % g]['diagnostics']['per_super_class'].items()],
              ['gain', 'cls', 'n_cells', 'mean_hz', 'median_hz', 'p90_hz', 'max_hz',
               'silent_fraction', 'mean_spikes'])
    write_csv(C.O / 'PHYSICAL_TABLE.csv', phys,
              ['gain', 'dt_ms', 'readout', 'phase', 'J', 'foot_support_mean', 'foot_support_p10',
               'com_z_min', 'nonfoot_share', 'free_fall_fraction', 'bearing_legs_median',
               'first_nonfoot_ms', 'D_pass', 'D_total', 'D_passed_all', 'saturation',
               'command_rms', 'readout_r2_held', 'failed_checks'])
    blocks = block_rows()
    dose = dose_response(blocks)
    fig = figure(phys)

    summ = VER['checks']['_summary']
    verdict = summ['verdict']
    ctrl = js(C.O / 'STATIC_GAIN0P3_REFERENCE.json')
    sealed_r0 = {}
    sealed = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/'
                  'numerics/SUMMARY.json')
    if sealed.exists():
        for r in js(sealed)['results']:
            if r['label'].startswith('dt_0p5ms_paired'):
                sealed_r0[str(r['release_time_s'])] = dict(
                    J=r['J'], D_pass=int(sum(r['validation']['checks'].values())),
                    foot_support_mean=r['parts']['foot_support_mean'],
                    com_z_min=r['parts']['com_z_min'])

    summary = dict(
        experiment='S5 sense-drive gain sweep: sparseness, motor information and physical support',
        label='ENGINEERING parameter sweep (one number, the encoder nA-per-unit gain), NOT '
              'biological calibration; the neurons generate the command; the readout is an '
              'offline ridge fit of the interface',
        gains=C.GAINS, reference_gain=C.REF_GAIN, neural_dt_s=C.DT_MAIN,
        second_timestep_s=C.DT_FINE, release_phases=C.PHASES, duration_s=C.DURATION,
        releases=sum(len(v) for _, _, _, v in BLOCKS),
        warms=len([p for p in (C.O / 'CONFIGS').glob('*') if p.is_dir()]),
        teacher_recordings=0,
        reference_gain_bitwise_reproduction=OFF['reference_gain_bitwise'],
        rate_regime={g: OFF['results'][g]['diagnostics']['all12000'] for g in
                     OFF['results']},
        rate_regime_readout204={g: OFF['results'][g]['diagnostics']['readout_204']
                                for g in OFF['results']},
        dense_relief={g: OFF['results'][g]['dense_relief'] for g in OFF['results']},
        information={g: OFF['results'][g]['information'] for g in OFF['results']},
        information_recomputed=summ['motor_held_r2'],
        information_recomputed_all=summ['all_cell_held_r2'],
        physical_blocks=blocks, physical_dose_response=dose,
        sealed_current_decoder_reference=sealed_r0,
        reading_rule=js(C.O / 'PREREGISTRATION.json')['reading_rule'],
        verdict=dict(
            INFORMATION_RESTORED=summ['INFORMATION_RESTORED'],
            DRIVE_DEPENDENT_SUPPORT=summ['DRIVE_DEPENDENT_SUPPORT'],
            overall=verdict,
            motor_held_r2_best=dict(gain=max(C.GAINS, key=lambda g: summ['motor_held_r2']['%g' % g]),
                                    value=max(summ['motor_held_r2'].values())),
            phases_passing_all_D_any_block=bool(any(v['phases_passing_all_count'] > 0
                                                    for v in blocks.values())),
            support_by_block={k: dict(phases_passing_all=v['phases_passing_all_count'],
                                      D_pass_counts=v['D_pass_counts']) for k, v in blocks.items()}),
        figure=str(fig),
        verification=str(C.O / 'VERIFICATION.json'),
        verification_all_passed=VER['all_passed'],
        boundaries=dict(walking='NOT_CLAIMED', autonomous_support='NOT_CLAIMED',
                        biological_validity='NOT_ESTABLISHED',
                        absolute_forces='NOT_ESTABLISHED', brain_model_invalid='NOT_CLAIMED',
                        timestep_selected=None, model_selected=None, gain_selected=None,
                        phase_robustness='every conclusion is per phase; the 0.0625 ms arm is '
                                         'labelled a single-timestep observation'),
        importance='核心必需')
    dump(C.O / 'SUMMARY.json', summary)
    print('TABLES+FIGURE+SUMMARY written', verdict, flush=True)
    return summary


if __name__ == '__main__':
    main()
