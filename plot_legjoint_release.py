"""S178 figures from saved results only (no engine)."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--a', type=Path, required=True); ap.add_argument('--b', type=Path, required=True)
args = ap.parse_args(); A, B = args.a, args.b
la = json.loads((A/'STAGE_B_LEDGER.json').read_text()) if (A/'STAGE_B_LEDGER.json').exists() else None
lb = json.loads((B/'STAGE_B_LEDGER.json').read_text())
sb = json.loads((B/'STAGE_B_RESULT.json').read_text())
si = lb['history'][0]
trials = [e for e in lb['events'] if e['role'] == 'block_trial']
rej_root = [e for e in lb['events'] if e['role'] == 'REJECT_ROOT_NOT_BALANCED']

fig, axs = plt.subplots(2, 2, figsize=(16, 11))
ax = axs[0, 0]
ok = [t for t in trials if t['root_ok']]
bad = [t for t in trials if not t['root_ok']]
ax.semilogy([t['minimax'] for t in ok], [max(t['root_norm'], 1e-16) for t in ok], 'o', ms=6, label='root balanced')
ax.semilogy([t['minimax'] for t in bad], [max(t['root_norm'], 1e-16) for t in bad], 'x', color='tab:red', ms=5, label='root NOT balanced (rejected)')
ax.axhline(1e-8, color='r', ls='--', lw=.8); ax.axvline(1e-8, color='r', ls='--', lw=.8)
ax.scatter([si['minimax']], [max(si['root_norm'], 1e-16)], s=200, marker='*', color='k', label='S177 baseline')
ax.set_xlabel('42-row LP minimax (joint rows)'); ax.set_ylabel('root |residual6|inf')
ax.set_xscale('log'); ax.legend(fontsize=8); ax.set_title('SEPARATE objectives: root balance vs 42-row minimax\n(never merged into one score)')
ax = axs[0, 1]
names = ['root', 'z', 'roll', 'pitch', 'yaw', 'x/y']
ax.bar(range(2), [1e-12/1e-8, 3.3086/1e-8], color=['tab:green', 'tab:orange'])
ax.set_yscale('log'); ax.set_xticks(range(2), ['root |r6|inf / 1e-8', '42-row minimax / 1e-8'])
ax.set_title('distance to each frozen threshold at final point\n(root satisfied, joints not)')
ax = axs[1, 0]
rows = np.arange(6, 48)
final = json.loads((B/'STAGE_B_RESULT.json').read_text())['final']
viol = final['row_violation']
heat = np.array([viol])
im = ax.imshow(heat, aspect='auto', cmap='magma')
ax.set_yticks([0]); ax.set_yticklabels(['final'])
ax.set_xticks(range(0, 42, 5)); ax.set_xticklabels([str(rows[i]) for i in range(0, 42, 5)], fontsize=8)
ax.set_xlabel('joint row DOF (6..47)'); ax.set_title(f'per-row interval violation at final point; unreachable={final["unreachable"]}')
plt.colorbar(im, ax=ax)
ax = axs[1, 1]
ax.plot(range(len(trials)), [t['minimax'] for t in trials], '.-', label='trial 42-row minimax')
ax.axhline(si['minimax'], color='k', ls='--', lw=1, label='baseline 3.3086')
ax.set_xlabel('trial order'); ax.set_ylabel('42-row minimax'); ax.legend(fontsize=8)
ax.set_title(f'trials={len(trials)} root-rejected={len(rej_root)} accepted={lb["ledger"]["accepted_moves"]}; '
             f'contact-set changes={lb["ledger"]["contact_set_changes"]}')
fig.suptitle('S178 leg-joint release from the S177 root-balanced baseline (Stage B)')
fig.tight_layout(); fig.savefig(B/'ROOT_JOINT_SEPARATION.png', dpi=150)

# Stage A figure: minimum angle needed per unreachable DOF and per-joint reachability share
rowsA = list(csv.DictReader((A/'STAGE_A_SIGNFLIP.csv').open()))
fig2, axs2 = plt.subplots(1, 2, figsize=(17, 7))
dofs = sorted({int(r['dof']) for r in rowsA})
minneed = []; legs = []; bestjoint = []
for d in dofs:
    sub = [r for r in rowsA if int(r['dof']) == d]
    vals = [float(r['angle_needed_for_sign_flip']) for r in sub if r['angle_needed_for_sign_flip'] not in ('', 'None')]
    minneed.append(min(vals) if vals else np.nan); legs.append(sub[0]['leg'])
    bestjoint.append(min((r for r in sub if r['angle_needed_for_sign_flip'] not in ('', 'None')),
                         key=lambda r: float(r['angle_needed_for_sign_flip']))['joint_name'] if vals else '-')
y = np.arange(len(dofs))
axs2[0].barh(y, minneed, color=['tab:green' if (v is not None and v <= 0.6) else 'tab:orange' for v in minneed])
axs2[0].axvline(0.6, color='r', ls='--', label='frozen one-sided window 0.6 rad')
axs2[0].set_xscale('log'); axs2[0].set_yticks(y)
axs2[0].set_yticklabels([f'DOF {d} ({l})' for d, l in zip(dofs, legs)], fontsize=9)
for i, v in enumerate(minneed):
    axs2[0].text(v*1.15 if v else 0.01, i, f'{v:.3g} rad\n{bestjoint[i]}', va='center', fontsize=7)
axs2[0].set_xlabel('minimum joint angle needed to flip the dominant muscle channel (rad)')
axs2[0].set_title('Stage A: sign-flip demand vs frozen 0.6 rad window\n(green = achievable inside the window)')
axs2[0].legend(fontsize=8); axs2[0].set_xlim(1e-2, 5e2)
share = []
for d in dofs:
    sub = [r for r in rowsA if int(r['dof']) == d]
    share.append(100.0*sum(1 for r in sub if r['row_reachable_at_some_grid'] == 'True')/len(sub))
axs2[1].barh(y, share, color='tab:blue')
axs2[1].set_yticks(y); axs2[1].set_yticklabels([f'DOF {d}' for d in dofs], fontsize=9)
axs2[1].set_xlabel('% of that leg\'s 7 joints where the row interval becomes reachable')
axs2[1].set_xlim(0, 105)
axs2[1].set_title('empirical interval reachability inside the window\n(target moves with contact forces: NECESSARY condition only)')
fig2.tight_layout(); fig2.savefig(A/'STAGE_A_SIGNS.png', dpi=150)
print('saved ROOT_JOINT_SEPARATION.png and STAGE_A_SIGNS.png')
