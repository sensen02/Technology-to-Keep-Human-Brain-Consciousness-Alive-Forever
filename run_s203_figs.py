"""S203 figures: (1) ctrl divergence timeline, (2) log-scale max|dqpos| growth, (3) tactile force vs rail state."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

R = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')

def ld(n):
    return json.load(open(R / n / 'SUMMARY.json')), np.load(R / n / 'TRACE.npz')

SA, ZA = ld('A'); SD, ZD = ld('A_d1'); SE, ZE = ld('A_d2'); SB, ZB = ld('B'); SF, ZF = ld('B_d1')
s3A, z3A = ld('short3_A'); s3B, z3B = ld('short3_B'); s3C, z3C = ld('short3_C')
s3D, z3D = ld('short3_D'); s3T, z3T = ld('short3_T')
names = SA['actuator_names']
tA = np.arange(ZA['ctrl'].shape[0]) * 0.005

# ---- FIG 1: first-command probe + ctrl divergence -------------------------------------------
arms = ('A', 'T', 'B', 'C', 'D')
z = {'A': z3A, 'T': z3T, 'B': z3B, 'C': z3C, 'D': z3D}
s = {'A': s3A, 'T': s3T, 'B': s3B, 'C': s3C, 'D': s3D}
first_ctrl = [float(np.max(z[a]['ctrl'][0])) for a in arms]
first_spk = [int(z[a]['spikes'][0]) for a in arms]
first_nz = [int(np.sum(z[a]['ctrl'][0] > 1e-12)) for a in arms]
d_ctrl = np.abs(ZA['ctrl'] - ZB['ctrl'])
mx = d_ctrl.max(axis=1)
first = int(np.argmax(mx > 0))
tp = np.arange(z3A['ctrl'].shape[0]) * 0.005

fig, axs = plt.subplots(3, 1, figsize=(12, 12))
x = np.arange(len(arms))
axs[0].bar(x - 0.2, first_ctrl, width=0.4, color='tab:red', label='max ctrl in the FIRST command')
axs[0].bar(x + 0.2, [v / max(first_spk) if max(first_spk) else 0 for v in first_spk], width=0.4,
           color='tab:blue', label='spikes in the first command (normalised)')
for i, a in enumerate(arms):
    axs[0].text(i - 0.2, first_ctrl[i] + 0.02, '%.3f' % first_ctrl[i], ha='center', fontsize=8)
    axs[0].text(i + 0.2, (first_spk[i] / max(first_spk) if max(first_spk) else 0) + 0.02,
                '%d' % first_spk[i], ha='center', fontsize=8)
axs[0].set_xticks(x)
axs[0].set_xticklabels(['A no tactile', 'T touch only', 'B raw force', 'C force/weight', 'D sign-flipped'],
                       fontsize=8)
axs[0].set_ylabel('first command / normalised spikes')
axs[0].legend(fontsize=8, loc='upper left')
axs[0].set_title('FIG1a  FIRST COMMAND PROBE: identical release pose, identical physics (all six tarsi on the floor, 1.70-3.90 N);\n'
                 'only the tactile channel differs. A and T (touch-only) emit an ALL-ZERO command; only positive force moves the motors')
axs[0].set_ylim(0, 1.15)

for arm, zz, c, lbl in (('B', z3B, 'tab:red', 'A vs B'), ('C', z3C, 'tab:orange', 'A vs C'),
                        ('D', z3D, 'tab:green', 'A vs D'), ('T', z3T, 'tab:purple', 'A vs T (touch only)')):
    axs[1].plot(tp * 1000, np.abs(z3A['ctrl'] - zz['ctrl']).max(axis=1), 'o-', color=c, label=lbl, lw=1.4)
axs[1].set_xlabel('t (ms), one point per 5 ms command interval (instrumented 0.02 s arms)')
axs[1].set_ylabel('max |dctrl|')
axs[1].legend(fontsize=8)
axs[1].set_title('FIG1b  pre-collapse divergence (4 command intervals = 800 body steps): D and T are exactly 0 everywhere '
                 '(T is bitwise A), B and C differ from the very first command')

axs[2].plot(tA, np.maximum(mx, 1e-6), color='tab:red', lw=.9)
axs[2].set_yscale('log')
axs[2].set_ylim(1e-6, 1.6)
axs[2].axvline(0.020, color='k', ls=':', lw=1.2)
axs[2].text(0.05, 3e-6, 'body-floor contact 0.020 s', fontsize=8)
axs[2].set_xlabel('t (s), one point per 5 ms command interval (5 s arms)')
axs[2].set_ylabel('max |dctrl| A vs B (log)')
n_cmd_diff = int((mx > 0).sum())
n_val_diff = int((d_ctrl > 0).sum())
axs[2].set_title('FIG1c  A vs B over the whole 5 s: %d/%d commands differ (%.0f%%), %d/%d actuator-values differ (%.1f%%), '
                 'max|dctrl| = %.3f' % (n_cmd_diff, len(mx), 100 * n_cmd_diff / len(mx), n_val_diff, d_ctrl.size,
                                       100 * n_val_diff / d_ctrl.size, d_ctrl.max()))
fig.tight_layout()
fig.savefig(R / 'FIG1_CTRL_DIVERGENCE_TIMELINE.png', dpi=150)

# ---- FIG 2: log-scale max|dqpos| growth ------------------------------------------------------
TIMES = (0.005, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0)
def curve(Qb, Qp):
    out = []
    for t in TIMES:
        k = min(int(round(t / 0.005)) - 1, Qb.shape[0] - 1)
        out.append(max(float(np.max(np.abs(Qb[k] - Qp[k]))), 1e-18))
    return out
c1 = curve(ZA['qpos'], ZD['qpos'])
c2 = curve(ZA['qpos'], ZE['qpos'])
c3 = curve(ZB['qpos'], ZF['qpos'])
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(TIMES, c1, 'o-', color='tab:blue', label='A vs A_d1  (delta = 1e-9 rad on joint_LFCoxa_pitch)')
ax.semilogy(TIMES, c2, 's-', color='tab:red', label='A vs A_d2  (delta = 1e-6 rad)')
ax.semilogy(TIMES, c3, '^-', color='tab:green', label='B vs B_d1  (delta = 1e-9 rad, tactile on)')
ax.axhline(1e-9, color='0.6', ls=':', lw=.9)
ax.text(3.0, 1.3e-9, 'the perturbation itself (1e-9)', fontsize=7, color='0.4')
ax.axhline(0.05, color='k', ls='--', lw=1.0)
ax.text(3.0, 0.062, 'frozen chaos gate on end-state displacement: 0.05 m', fontsize=7)
ax.axvline(0.020, color='k', ls=':', lw=1.2)
ax.text(0.022, 1e-7, 'collapse 0.020 s', fontsize=7, rotation=90)
ax.set_xlabel('t (s)')
ax.set_ylabel('max |dqpos| vs the unperturbed arm (log, 49 entries)')
ax.legend(fontsize=8)
ax.set_title('FIG2  Sensitivity of the trajectory to an initial-value perturbation\n'
             'delta = 1e-9 stays at ~1e-9..1e-8 (no growth); delta = 1e-6 grows to ~7e-3 by 0.1 s and to 0.28 max|dqpos| at 5 s')
ax.grid(alpha=.3, which='both')
fig.tight_layout()
fig.savefig(R / 'FIG2_DELTAQPOS_GROWTH.png', dpi=150)

# ---- FIG 3: six-leg tactile force vs rail state ---------------------------------------------
leg_idx = {L: [i for i, n in enumerate(names) if n.startswith(L)] for L in LEGS}
ctrlB = ZB['ctrl']
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i, L in enumerate(LEGS):
    axs[0].plot(tA, ZB['force'][:, i], lw=.9, label=L)
axs[0].set_ylabel('tactile force per leg (N)')
axs[0].legend(fontsize=7, ncol=6)
axs[0].set_title('FIG3  Arm B: per-leg tactile force (top) and the fraction of that leg\'s 15 actuators sitting at a decode rail (bottom)\n'
                 'rail = ctrl clipped to 0 or 1; the rail fraction is ~0.87 pre-collapse in the NO-TACTILE arm too')
for L in LEGS:
    sub = ctrlB[:, leg_idx[L]]
    rail = np.mean((sub <= 1e-12) | (sub >= 1.0 - 1e-12), axis=1)
    axs[1].plot(tA, rail, lw=.9, label=L)
subA = ZA['ctrl']
railA = np.mean((subA <= 1e-12) | (subA >= 1.0 - 1e-12), axis=1)
axs[1].plot(tA, railA, color='k', lw=1.6, ls='--', label='A (all 90)')
axs[1].axvline(0.020, color='k', ls=':', lw=1.0)
axs[1].set_ylabel('fraction of actuators at a rail')
axs[1].set_xlabel('t (s)')
axs[1].legend(fontsize=7, ncol=7)
axs[1].set_ylim(0.4, 1.02)
fig.tight_layout()
fig.savefig(R / 'FIG3_TACTILE_VS_SATURATION.png', dpi=150)

print(json.dumps(dict(fig1=dict(first_ctrl=first_ctrl, first_spikes=first_spk, first_nonzero=first_nz,
                               first_divergence_s=float(tA[first]), cmd_diff=n_cmd_diff,
                               val_diff=n_val_diff, max_dctrl=float(d_ctrl.max())),
                      fig2=dict(A_d1=c1, A_d2=c2, B_d1=c3),
                      fig3=dict(railA_mean=float(railA.mean()))), indent=1))
for f in ('FIG1_CTRL_DIVERGENCE_TIMELINE.png', 'FIG2_DELTAQPOS_GROWTH.png', 'FIG3_TACTILE_VS_SATURATION.png'):
    print(f, (R / f).stat().st_size)
