"""S179 figures from saved results only (no engine)."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True); args = ap.parse_args(); p = args.run
c = json.loads((p/'CANDIDATES.json').read_text()); rm = json.loads((p/'ROW_DOF_MAP.json').read_text())
s = json.loads((p/'SUMMARY.json').read_text())
fig, axs = plt.subplots(2, 2, figsize=(16, 11))
col = {'ROOT_BALANCED': 'tab:blue', 'ROOT_DIVERGE': 'tab:red', 'ROOT_CONTACT_SET_CHANGE': 'tab:orange', 'ROOT_RESIDUAL_FLOOR': 'tab:purple'}
ax = axs[0, 0]
for k, v in col.items():
    sub = [x for x in c if x['root_failure_class'] == k]
    if sub: ax.scatter([x['minimax'] for x in sub], [max(x['root_norm'], 1e-13) for x in sub], label=f'{k} ({len(sub)})', s=55, c=v)
ax.axhline(1e-8, color='k', ls='--', lw=.8); ax.axvline(1e-8, color='k', ls='--', lw=.8)
ax.scatter([3.3086208677913731], [1.3464784842653899e-12], marker='*', s=260, c='k', label='S177 baseline')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('42-row LP minimax (joint rows)'); ax.set_ylabel('root |residual6|inf')
ax.legend(fontsize=7); ax.set_title('two independent axes: root balance vs joint minimax\n(class-coloured; never merged into one score)')
ax = axs[0, 1]
ax.bar(range(len(c)), [len(x['unreachable']) for x in c],
       color=[col[x['root_failure_class']] for x in c])
ax.axhline(6, color='k', ls='--', lw=.9, label='baseline 6 unreachable rows')
ax.set_xlabel('candidate'); ax.set_ylabel('# unreachable joint rows'); ax.legend(fontsize=8)
ax.set_title('unreachable-row count per candidate\n(root-failed candidates cannot be credited with a lower count)')
ax = axs[1, 0]
V = np.array([x['row_violation'] for x in c])
im = ax.imshow(V.T, aspect='auto', cmap='magma', origin='lower')
ax.set_yticks(range(0, 42, 5))
ax.set_yticklabels([f'{rm["rows"][i]["dof_name"]}/{rm["rows"][i]["dof"]}' for i in range(0, 42, 5)], fontsize=6)
ax.set_xlabel('candidate'); ax.set_title('per-row interval violation heat map (rows 0..41 -> DOF/name)')
plt.colorbar(im, ax=ax)
ax = axs[1, 1]
tr = [x['target_row_reachable'] for x in c]
ax.step(range(len(c)), [len(x['unreachable']) for x in c], where='mid', label='unreachable count')
ax.scatter([i for i, v in enumerate(tr) if v], [0]*sum(tr), marker='v', c='g', label='target row 36 reachable')
ax.scatter([i for i, v in enumerate(tr) if not v], [-0.5]*sum(1 for v in tr if not v), marker='x', c='r', s=12,
           label='target row 36 NOT reachable (all)')
ax.set_ylim(-1, 14); ax.set_xlabel('candidate'); ax.legend(fontsize=8)
ax.set_title('target row 36 (DOF 42, RMCoxa_pitch) never became reachable')
fig.suptitle(f'S179 targeted RM-leg search: {s["candidates"]} candidates, selection={s["selection"]}, forwards={s["calls"]["forward"]}')
fig.tight_layout(); fig.savefig(p/'S179_SELECTED.png', dpi=150)
print('saved S179_SELECTED.png')
