"""S202 analysis: channel-scale audit + three figures. Engine-free, artifact-only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('/home/sensen/Desktop/engine修复验收/s202-tactile-channel/20261025T362000000000Z')
ARMS = {'A': 'A', 'A_REPEAT': 'A_REPEAT', 'B': 'B', 'C': 'C', 'D': 'D'}
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
C0 = 0.3      # current_nA_per_unit
REF_F = 1.0   # reference_force (N)

S = {k: json.load(open(BASE / v / 'SUMMARY.json')) for k, v in ARMS.items()}
T = {k: np.load(BASE / v / 'TRACE.npz') for k, v in ARMS.items()}

tab = {}
for k in ARMS:
    s, z = S[k], T[k]
    tab[k] = dict(
        per_leg_force_max={L: s['force_channel']['per_leg_max'][L] for L in LEGS},
        per_leg_current_nA={L: s['force_channel']['per_leg_max'][L] * C0 for L in LEGS},
        touch_units_mean={L: s['touch_channel']['per_leg_mean'][L] for L in LEGS},
        touch_current_nA={L: s['touch_channel']['per_leg_mean'][L] * C0 for L in LEGS},
        cur_max_nA=float(np.max(z['cur_max'])),
        ctrl_max=float(np.max(z['ctrl_max'])),
        act_max=float(np.max(z['act_max'])),
        sat_interval_frac=s['sat_interval_frac'],
        ctrl_sat_sample_frac=float(np.mean(z['ctrl_max'] >= 0.9999)),
        act_sat_sample_frac=float(np.mean(z['act_max'] > 0.999)),
    )

# FIG 1: four-arm rootz overlay
fig, ax = plt.subplots(figsize=(11, 5.5))
for k, c, ls in (('A', 'k', '-'), ('A_REPEAT', '0.6', '--'), ('B', 'tab:red', '-'),
                 ('C', 'tab:blue', '-'), ('D', 'tab:green', '--')):
    ax.plot(T[k]['t'], T[k]['rootz'], label=k, color=c, ls=ls, lw=1.1)
ax.set_xlabel('t (s)')
ax.set_ylabel('root z (m)')
ax.legend(fontsize=8, ncol=3)
ax.set_title('S202 four-arm root-z overlay (5 s): identical start/neural/decoder/physics, only the tactile channel differs\n'
             'A_REPEAT == A bitwise (noise floor 0); D (sign-flipped) == A in trajectory; B/C differ')
axi = ax.inset_axes([0.07, 0.60, 0.34, 0.32])
for k, c in (('A', 'k'), ('B', 'tab:red'), ('C', 'tab:blue'), ('D', 'tab:green')):
    m = T[k]['t'] <= 0.06
    axi.plot(T[k]['t'][m], T[k]['rootz'][m], color=c, lw=1.2)
axi.set_title('first 60 ms (collapse happens here, all arms)', fontsize=7)
axi.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(BASE / 'FIG1_FOUR_ARM_ROOTZ.png', dpi=150)

# FIG 2: per-leg tactile force time series
fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for i, L in enumerate(LEGS):
    a = axs[i // 3, i % 3]
    a.plot(T['B']['t'], T['B']['fch'][:, i], color='tab:red', lw=.9, label='B raw')
    a.plot(T['C']['t'], T['C']['fch'][:, i], color='tab:blue', lw=.9, label='C /WPL')
    a.plot(T['D']['t'], T['D']['fch'][:, i], color='tab:green', lw=.7, ls='--', label='D sign-flipped')
    a.axhline(C0 * REF_F, color='0.5', ls=':', lw=.8)
    a.set_title('%s  maxB=%.3f (=%.3f nA)  maxC=%.3f' % (
        L, tab['B']['per_leg_force_max'][L], tab['B']['per_leg_current_nA'][L],
        tab['C']['per_leg_force_max'][L]), fontsize=8)
    a.set_ylabel('tactile force unit')
    a.tick_params(labelsize=7)
    if i == 0:
        a.legend(fontsize=6)
axs[1, 1].set_xlabel('t (s)')
fig.suptitle('S202 per-leg tactile force fed to the channel encoder '
             '(declared scale 1 unit = 0.3 nA at reference_force 1.0 N; dotted line = 0.3 units)')
fig.tight_layout()
fig.savefig(BASE / 'FIG2_PER_LEG_TACTILE_FORCE.png', dpi=150)

# FIG 3: corrected contact timeline
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for k, c in (('A', 'k'), ('B', 'tab:red'), ('C', 'tab:blue'), ('D', 'tab:green')):
    axs[0].plot(T[k]['t'], T[k]['bodyfloor'], color=c, lw=.9, label=k)
    axs[1].plot(T[k]['t'], T[k]['nfeet'], color=c, lw=.9, label=k)
axs[0].set_ylabel('body (non-foot) floor contacts')
axs[1].set_ylabel('foot (tarsus) contacts')
for a in axs:
    a.legend(fontsize=7)
    a.grid(alpha=.3)
axs[1].set_xlabel('t (s)')
axs[0].set_title('S202 corrected contact census (floor contact WITHOUT tarsus = body contact):\n'
                 'body-floor contact from 0.020 s in EVERY arm; old S197-S200 "max_nonfoot = 0" must be read as withdrawn')
fig.tight_layout()
fig.savefig(BASE / 'FIG3_CORRECTED_CONTACT_TIMELINE.png', dpi=150)

verdict = dict(
    saturation_independent_of_input_scale=True,
    evidence=('arm A has force identically 0 AND touch 0, yet ctrl_max=%.6f, act_max=%.6f, '
              'sat_interval_frac=%.3f (ctrl saturated in %.1f%% of samples)'
              % (tab['A']['ctrl_max'], tab['A']['act_max'], tab['A']['sat_interval_frac'],
                 100 * tab['A']['ctrl_sat_sample_frac'])),
    force_magnitude_vs_scale=('per-leg raw force max (B) %s ; injected current at declared 0.3 nA/unit %s nA'
                              % (json.dumps(tab['B']['per_leg_force_max']),
                                 json.dumps({k: round(v, 4) for k, v in tab['B']['per_leg_current_nA'].items()}))),
    sentence_zh='饱和与输入标度无关',
    detail_zh=('A 臂触觉输入恒为 0，ctrl 仍到 1.0、act 仍到 1.0（饱和样本占比 %.1f%%），'
               '所以饱和不是触觉输入标度造成的；B 臂最大触觉电流仅 %.3f nA 也照样饱和。'
               '这与冻结的静态不可行性一致：42 行 LP 最小最大残差 3.3086，电机回路被顶到限幅。'
               % (100 * tab['A']['ctrl_sat_sample_frac'], tab['B']['cur_max_nA'])),
)
audit = dict(
    declared=dict(current_nA_per_unit=C0, reference_force_N=REF_F,
                  reading='1 unit of force at reference_force=1.0 N injects 0.3 nA; raw force in N is used as the unit'),
    per_arm=tab,
    verdict=verdict,
)
(BASE / 'CHANNEL_SCALE_AUDIT.json').write_text(json.dumps(audit, indent=2, default=float))

print(json.dumps(dict(
    A=dict(ctrl=tab['A']['ctrl_max'], act=tab['A']['act_max'], sat=tab['A']['sat_interval_frac'],
           ctrl_sat_frac=tab['A']['ctrl_sat_sample_frac']),
    cur_max={k: round(tab[k]['cur_max_nA'], 4) for k in ARMS},
    B_current_nA={k: round(v, 4) for k, v in tab['B']['per_leg_current_nA'].items()},
    C_current_nA={k: round(v, 4) for k, v in tab['C']['per_leg_current_nA'].items()},
), indent=1))
for f in ('FIG1_FOUR_ARM_ROOTZ.png', 'FIG2_PER_LEG_TACTILE_FORCE.png', 'FIG3_CORRECTED_CONTACT_TIMELINE.png'):
    print(f, (BASE / f).stat().st_size)
