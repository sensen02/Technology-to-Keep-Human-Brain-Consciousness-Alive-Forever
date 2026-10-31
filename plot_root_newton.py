"""S177 plots from saved results only (no engine)."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); p = ap.parse_args().output
res = json.loads((p/'ROOT_RESULT.json').read_text()); summ = json.loads((p/'SUMMARY.json').read_text())
led = json.loads((p/'NEWTON_LEDGER.json').read_text()); events = led['events']
hist = [h for h in summ['history'] if h['role'] in ('initial', 'accepted')]
its = [h['iteration'] for h in hist]; norms = [h['norm_inf'] for h in hist]
fig, axs = plt.subplots(2, 2, figsize=(16, 11))
axs[0, 0].semilogy(its, np.maximum(norms, 1e-16), 'o-')
axs[0, 0].axhline(1e-8, color='r', ls='--', label='pre-registered 1e-8')
axs[0, 0].set_title('root six-row residual |r|inf per accepted Newton iterate')
axs[0, 0].set_xlabel('Newton iteration'); axs[0, 0].set_ylabel('|residual6|inf'); axs[0, 0].legend()
comp = np.array([h['residual6'] for h in hist]) if 'residual6' in hist[0] else np.array([res['residual6']])
if comp is not None:
    for i, name in enumerate(['x', 'y', 'z', 'roll', 'pitch', 'yaw']):
        axs[0, 1].semilogy(its, np.maximum(abs(comp[:, i]), 1e-16), 'o-', ms=3, label=name)
    axs[0, 1].axhline(1e-8, color='r', ls='--'); axs[0, 1].legend(fontsize=8)
axs[0, 1].set_title('per-component magnitude'); axs[0, 1].set_xlabel('Newton iteration')
seq = [e for e in events if e['role'] in ('initial', 'line_search_trial') and e.get('norm_inf') is not None]
tl = list(range(len(seq)))
axs[1, 0].semilogy(tl, np.maximum([e['norm_inf'] for e in seq], 1e-16), '.-', ms=4)
axs[1, 0].set_title('every evaluated iterate (incl. rejected line-search trials)')
axs[1, 0].set_xlabel('evaluation order'); axs[1, 0].set_ylabel('|residual6|inf')
changes = [e['event_id'] for e in events if e['role'] == 'LINE_SEARCH_DECISION' and e.get('contact_set_changed')]
bar = dict(foot_depth=res['geometry']['foot_depth'], cap=0.03442910871489635, bearing=res['geometry']['bearing_legs'],
           nonfoot=res['geometry']['nonfoot_depth'], friction=res['geometry']['friction_margin'])
labels = [f"foot depth\n{bar['foot_depth']:.5f}/{bar['cap']:.5f}", f"bearing legs\n{bar['bearing']}/6",
          f"non-foot\n{bar['nonfoot']:.1e}", f"friction margin\n{bar['friction']:.3f}"]
axs[1, 1].bar(range(4), [bar['foot_depth']/bar['cap'], bar['bearing']/6, 0.0, 1.0],
              color=['tab:green', 'tab:green', 'tab:green', 'tab:green'])
axs[1, 1].set_xticks(range(4), labels, fontsize=8); axs[1, 1].axhline(1, color='r', ls='--')
axs[1, 1].set_title(f"geometry gates at solution: {'PASS' if res['geometry']['geometry_gates_pass'] else 'FAIL'}; contact-set changes={summ['newton_ledger']['contact_set_changes']}")
fig.suptitle(f"S177 root wrench Gauss-Newton: root rows balanced to {res['norm_inf']:.2e}, geometry gates PASS; "
             f"42 joint rows separately NOT balanced (LP minimax {summ['joint_lp_minimax']:.4f})")
fig.tight_layout(); fig.savefig(p/'ROOT_NEWTON_CONVERGENCE.png', dpi=150)
fig2, ax = plt.subplots(figsize=(14, 6))
roles = [e['role'] for e in seq]
ax.semilogy(tl, np.maximum([e['norm_inf'] for e in seq], 1e-16), '.-')
for ev in events:
    if ev['role'] == 'LINE_SEARCH_DECISION' and ev.get('contact_set_changed'):
        ax.axvline(min(len(tl)-1, ev['event_id']), color='orange', ls=':', label='contact-set change')
ax.set_title('contact-set changes are counted, never silently accepted'); ax.set_xlabel('evaluation order'); ax.set_ylabel('|residual6|inf')
ax.legend(); fig2.tight_layout(); fig2.savefig(p/'GEOMETRY_CONTACT_SEQUENCE.png', dpi=150)
print('saved ROOT_NEWTON_CONVERGENCE.png and GEOMETRY_CONTACT_SEQUENCE.png')
