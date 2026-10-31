"""S204: calibration arms verdict + figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')
S203 = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')

cen = json.load(open(RUN / 'READOUT_CENSUS.json'))
rate = json.load(open(RUN / 'RATE_DISTRIBUTION.json'))
bt = json.load(open(RUN / 'DECODER_TRANSFER_BITWISE.json'))
reach = json.load(open(RUN / 'REACHABILITY.json'))
dec = json.load(open(RUN / 'SATURATION_DECOMPOSITION.json'))
tr = json.load(open(RUN / 'DECODER_TRANSFER.json'))

H0 = 100.0
ol = bt['open_loop_gate']['per_arm']
cl = bt['closed_loop_arms']['per_arm']
gate_cal = any(ol[a]['nonempty']['off_rail_fraction'] > 0.5 for a in ('H1', 'H2'))
gate_pool = all(cl[a]['nonempty']['off_rail_fraction'] <= 0.25 for a in ('H1', 'H2')) and \
            all(ol[a]['nonempty']['off_rail_fraction'] <= 0.25 for a in ('H1', 'H2'))
verdict = ('SATURATION_IS_CALIBRATION' if gate_cal else
           ('POOL_TOO_SMALL_FOR_GRADATION' if gate_pool else 'UNDETERMINED'))

at_zero_by_arm = {a: ol[a]['nonempty']['at_zero'] for a in ol}
cal = dict(
    round='S204',
    question='is the pre-collapse railing a property of the body or of the decoder calibration?',
    verdict=verdict,
    verdict_fields=dict(
        frozen_gate_calibration_rule='off-rail fraction of non-empty actuators at H1 or H2 > 50% with bitwise identical input',
        frozen_gate_pool_too_small_rule='off-rail fraction <= 25% even at H2 (>= 10*H0)',
        open_loop_off_rail={a: ol[a]['nonempty']['off_rail_fraction'] for a in ol},
        closed_loop_off_rail={a: cl[a]['nonempty']['off_rail_fraction'] for a in cl},
        calibration_gate_fires=bool(gate_cal),
        pool_too_small_gate_fires=bool(gate_pool)),
    arms=dict(
        H0=dict(rate_for_full_control_hz=H0, role='current value',
                open_loop=ol['H0'], closed_loop=cl['H0']),
        H1=dict(rate_for_full_control_hz=rate['derivation']['H1'], role='2 * median measured pool rate (data-derived)',
                derivation='median of the 222 pool-window rates of the H0 arm = %.4f Hz, so H1 = 2*median = %.4f Hz'
                           % (rate['summary']['median'], rate['derivation']['H1']),
                open_loop=ol['H1'], closed_loop=cl['H1']),
        H2=dict(rate_for_full_control_hz=rate['derivation']['H2'],
                role='max(10*H0, 2*P90) -> the pre-registered >=10x probe',
                derivation=('P90 of the same distribution = %.4f Hz, 2*P90 = %.4f Hz, 10*H0 = %.4f Hz, so '
                            'H2 = max(...) = %.4f Hz (came from %s)'
                            % (rate['summary']['p90'], 2 * rate['summary']['p90'], 10 * H0,
                               rate['derivation']['H2'], rate['derivation']['H2_came_from'])),
                open_loop=ol['H2'], closed_loop=cl['H2'])),
    reading=dict(
        transfer='ctrl = clip(rates/rate_full,0,1) with rates a 5 ms exponential average of "pool mean spikes per 0.5 ms sample / 0.0005 s". '
                 'A steady per-cell rate f therefore maps to ctrl = f/rate_full, independent of pool size, and ONE spike injects '
                 '%.1f Hz of filtered rate, i.e. %.3f of full scale at 100 Hz. At H0 a single spike on a 1-cell pool already clips.'
                 % (tr['filtered_gain_per_single_spike'], tr['filtered_gain_per_single_spike'] / H0),
        measurement='the measured pool rate distribution has median %.1f Hz and P90 %.1f Hz against a full scale of %.0f Hz: the '
                    'typical pool sits EXACTLY at full scale, so at H0 roughly half of the pools are pinned at ctrl=1 by construction.'
                    % (rate['summary']['median'], rate['summary']['p90'], H0),
        calibration_result='raising the full scale to %.0f Hz takes the off-rail fraction of non-empty actuators from %.1f%% to '
                           '%.1f%% (ceiling entries fall from %d to %d), so the railing at H0 is dominated by the calibration and '
                           'NOT by the body.'
                           % (rate['derivation']['H2'], 100 * ol['H0']['nonempty']['off_rail_fraction'],
                              100 * ol['H2']['nonempty']['off_rail_fraction'], ol['H0']['nonempty']['at_one'],
                              ol['H2']['nonempty']['at_one']),
        residual_zero='the floor does not move: %d of the 222 non-empty actuator-window values have ZERO spikes in every arm, so '
                      'that part of the railing is an INPUT-side silence (no spikes reached the pool), not calibration.'
                      % at_zero_by_arm['H0'],
        not_an_improvement='no movement claim is made. Raising the full scale also lowers the command amplitude (median ctrl %.3f at '
                           'H0 -> %.3f at H2), which is why the verdict is about gradation, not about better walking.'
                           % (ol['H0']['nonempty']['quantiles']['0.5'], ol['H2']['nonempty']['quantiles']['0.5'])),
    structural_facts=dict(
        n_tier=cen['n_tier'], n_decoder_rows=cen['n_decoder_rows'],
        n_structurally_uncommandable=cen['n_empty_rows'],
        structurally_uncommandable_actuators=[e['actuator'] for e in cen['empty_actuators']],
        n_nonempty_rows=cen['n_nonempty_rows'], unique_pool_cells=cen['unique_pool_cells'],
        coverage_fraction=cen['coverage_fraction'],
        pool_size=dict(min=cen['pool_size_min'], mean=cen['pool_size_mean'], max=cen['pool_size_max'],
                       histogram=cen['pool_size_histogram']),
        reachability=dict(label=reach['label'], share_2hops=reach['reachable_nonpool_share'],
                          share_3hops=reach['reachable_nonpool_share_3hops'],
                          note=reach['margin_note'])),
    saturation_recomputation=dict(
        s203_original=dec['s203_numbers'],
        corrected=dec['per_arm'],
        note=('S203 counted 84.7%% of all 90 rows at a rail over the 4 command windows t = 5..20 ms (command 0 was already '
              'excluded there). Recomputing on commands 1-3 of the 0.02 s arms gives 84.1%% over 90 rows and %.1f%% over the 74 '
              'commandable rows, i.e. the structural-zero rows are %.1f%% of the rail figure rather than its explanation.'
              % (100 * dec['per_arm']['short3_A']['at_rail_excl_structural'],
                 100 * dec['per_arm']['short3_A']['structural_zero_share_of_the_rail_figure']))),
    suggested_calibration=dict(
        value_hz=float(rate['derivation']['H1']),
        rationale=('with H=%d Hz the median pool (%.0f Hz) maps to ctrl~0.5 and the P90 pool to ctrl=2.0 -> clipped; H between '
                   'H1 and 400-500 Hz keeps the top decile reachable while giving the median real resolution. This is a '
                   'SUGGESTION for a later round, not applied here beyond the three measured arms.'
                   % (rate['derivation']['H1'], rate['summary']['median']))),
    calls=dict(model_loads_for_arms=3, mj_step_arms=3 * 2000, forward_arms=3,
               offline_model_loads=2, offline_forwards=0, lp=0, neural_ticks=3 * 10 * 10),
)
(RUN / 'CALIBRATION_ARMS.json').write_text(json.dumps(cal, indent=2, default=str))

# ---------------------------------------------------------------- FIG1 pool size vs required rate
fig, ax = plt.subplots(figsize=(10, 5.5))
ks = np.arange(1, 8)
ax.axhline(100, color='tab:red', lw=2, label='full scale H0 = 100 Hz: per-cell rate for ctrl=1.0 (independent of k)')
ax.axhline(50, color='tab:red', lw=1, ls='--', label='per-cell rate for ctrl=0.5')
ax.axhline(2000, color='k', lw=1, ls=':', label='physical ceiling: one spike per 0.5 ms sample = 2000 Hz')
ax.plot(ks, np.full(ks.size, rate['summary']['median']), 'o', color='tab:blue', ms=8,
        label='measured MEDIAN pool rate = %.0f Hz' % rate['summary']['median'])
ax.plot(ks, np.full(ks.size, rate['summary']['p90']), 's', color='tab:green', ms=7,
        label='measured P90 pool rate = %.0f Hz' % rate['summary']['p90'])
ax.plot(ks, np.full(ks.size, tr['filtered_gain_per_single_spike']), '^', color='tab:purple', ms=7,
        label='filtered rate injected by ONE spike = %.0f Hz' % tr['filtered_gain_per_single_spike'])
ax.set_yscale('log')
ax.set_xlabel('pool size k (cells)')
ax.set_ylabel('per-cell firing rate (Hz, log)')
ax.set_title('FIG1  Required single-cell rate vs pool size, and where the measured pools sit\n'
             'flat in k: the full scale (100 Hz) equals the MEDIAN pool rate, and one spike is already 1.9x full scale',
             fontsize=9)
ax.legend(fontsize=6.5, loc='lower left', framealpha=.95)
ax.grid(alpha=.3, which='both')
fig.tight_layout()
fig.savefig(RUN / 'FIG1_POOLSIZE_VS_REQUIRED_RATE.png', dpi=150)

# ---------------------------------------------------------------- FIG2 saturation decomposition
fig, ax = plt.subplots(figsize=(11, 5.5))
arms = ['short3_A', 'short3_B']
labels = ['A no tactile', 'B raw force']
classes = ['structural_zero', 'pool_zero_spikes', 'ctrl_at_one', 'spiked_ctrl_below_one']
colors = ['0.6', 'tab:blue', 'tab:red', 'tab:green']
w = 0.35
for j, arm in enumerate(arms):
    bottom = 0.0
    for cls, col in zip(classes, colors):
        v = dec['per_arm'][arm]['counts'][cls]
        ax.bar(j, v, bottom=bottom, color=col, width=w,
               label=cls if j == 0 else None, edgecolor='k', lw=.4)
        if v:
            ax.text(j, bottom + v / 2, str(v), ha='center', va='center', fontsize=8)
        bottom += v
ax.set_xticks([0, 1])
ax.set_xticklabels(['A no tactile', 'B raw force'])
ax.set_ylabel('actuator-window values (3 windows x 90 rows = 270)')
ax.axhline(3 * 74, color='k', ls='--', lw=1)
ax.text(0.5, 3 * 74 + 3, 'ceiling for the 74 commandable rows = 222', ha='center', fontsize=8)
ax.set_title('FIG2  Pre-collapse saturation decomposed (commands 1-3)\n'
             'S203 counted 84.7% at rail over 90 rows incl. command 0; 48 of those values are the 16 uncommandable rows',
             fontsize=9)
ax.legend(fontsize=7, loc='lower right')
fig.tight_layout()
fig.savefig(RUN / 'FIG2_SATURATION_DECOMPOSITION.png', dpi=150)

# ---------------------------------------------------------------- FIG3 arm ctrl traces
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, arm in zip(axs, ('H0', 'H1', 'H2')):
    z = np.load(RUN / arm / 'TRACE.npz')
    img = z['ctrl'][:4].T
    im = ax.imshow(img, aspect='auto', vmin=0, vmax=1, cmap='viridis', origin='lower')
    zi = [c['actuator_index'] for c in cen['rows'] if c['pool_size'] == 0]
    ax.plot(np.full(len(zi), -0.35), zi, '|', color='red', ms=6, mew=1.6)
    ax.set_title('%s  rate_full=%.0f Hz' % (arm, cl[arm]['rate_full']), fontsize=9)
    ax.set_xlabel('command window (0 = cold start)')
    ax.set_xticks(range(4))
axs[0].set_ylabel('actuator index (0-89)')
fig.colorbar(im, ax=axs, fraction=0.02, label='ctrl')
fig.suptitle('FIG3  Per-actuator ctrl, first four command windows, three calibration arms (red ticks = the 16 structurally '
             'uncommandable rows)\nwindow 0 is all-zero in every arm; raising the full scale spreads the same spike ticket over '
             'the 0-1 range instead of pinning it at 1', fontsize=9)
fig.savefig(RUN / 'FIG3_ARM_CTRL_TRACES.png', dpi=150)

print(json.dumps(dict(verdict=verdict, off_rail_open=cal['verdict_fields']['open_loop_off_rail'],
                      off_rail_closed=cal['verdict_fields']['closed_loop_off_rail'],
                      gates=dict(cal=gate_cal, pool=gate_pool)), indent=1))
