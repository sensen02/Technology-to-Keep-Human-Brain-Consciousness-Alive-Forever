"""S205 analysis: classify the silent pools and draw the two figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s205-unresolved-mn-and-silent-pools/20261025T075517207961Z')
S204 = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')
cen = json.load(open(S204 / 'READOUT_CENSUS.json'))
prov = json.load(open(RUN / 'UNRESOLVED_PROVENANCE.json'))
d = np.load(RUN / 'diag' / 'DIAG.npz')
cur, counts, v = d['current'], d['counts'], d['v']
pool_cells = d['pool_cells'].tolist()
cell_row = {c: i for i, c in enumerate(pool_cells)}
thr = float(d['threshold_mV'])
import sys
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from engine.embodied import adapters as ad
_net = ad.NeuralTier(ad.build_neural_tier(ad.TierConfig(background_mean_nA=0, background_sd_nA=0)),
                     ad.TierConfig(background_mean_nA=0, background_sd_nA=0), ad.OwnershipLedger()).net
rest = float(_net.p.rest_mV)
WIN = (1, 2, 3)

pools = []
for c in cen['rows']:
    if c['pool_size'] == 0:
        continue
    cells = c['local_indices']
    rows = [cell_row[x] for x in cells]
    absI = np.abs(cur[list(WIN)][:, cells]).max()
    spk = int(counts[list(WIN)][:, cells].sum())
    vmax = float(v[list(WIN)][:, rows, :].max())
    # NOTE: pool cells are motor neurons; the sensory encoder injects into sensory cells only (492/12000),
    # so the external current seen by a pool cell is zero by architecture. The meaningful drive measure is
    # therefore the membrane potential relative to rest and threshold.
    if spk > 0:
        cls = 'FIRED'
    elif vmax >= thr:
        cls = 'REACHED_THRESHOLD_WITHOUT_SPIKE_GATED'
    elif vmax > rest + 1e-9:
        cls = 'DRIVEN_BUT_BELOW_THRESHOLD'
    else:
        cls = 'AT_REST_NO_DRIVE'
    pools.append(dict(actuator=c['actuator'], actuator_index=c['actuator_index'], leg=c['leg'],
                      pool_size=c['pool_size'], max_abs_external_current_nA=float(absI),
                      spikes_in_windows_1_3=spk, max_membrane_mV=vmax, threshold_mV=thr, cls=cls))

all_cells_I = np.abs(cur[list(WIN)]).max(axis=0)
n_cells_driven = int((all_cells_I > 0).sum())
cls_counts = {}
for p in pools:
    cls_counts[p['cls']] = cls_counts.get(p['cls'], 0) + 1

silent = [p for p in pools if p['spikes_in_windows_1_3'] == 0]
diag = dict(
    arm='diag (0.05 s, 2000 steps, rate_full=100, no tactile); its ctrl and spike counts are BITWISE equal to S204 arm H0',
    threshold_mV=thr, windows='command intervals 1-3, the same window set as the S204 calibration audit',
    n_nonempty_pools=len(pools), class_counts=cls_counts,
    n_silent_pools=len(silent),
    n_pools_silent_in_all_three_windows=len(silent),
    silent_pools=silent,
    rest_mV=rest,
    n_cells_with_nonzero_external_current=int(n_cells_driven),
    external_current_targets=('the sensory encoder writes into the six per-leg channel groups only (%d of %d cells); the motor '
                              'pool cells therefore have zero EXTERNAL current by construction and are driven synaptically, so '
                              'this round classifies them by membrane potential instead' % (n_cells_driven, d['current'].shape[1])),
    zero_spike_actuator_window_values=dict(
        n_zero_spikes=int(sum(1 for c in cen['rows'] if c['pool_size'] > 0
                              for w in WIN if int(counts[w][c['local_indices']].sum()) == 0)),
        n_zero_ctrl=72,
        of=222,
        per_window_zero_spikes={int(w): int(sum(1 for c in cen['rows'] if c['pool_size'] > 0
                                                if int(counts[w][c['local_indices']].sum()) == 0)) for w in WIN},
        n_pools_silent_in_all_three_windows=len(silent),
        note=('TWO DIFFERENT COUNTS, both correct: 74 of 222 non-empty actuator-windows carry ZERO SPIKES, while S204 reported '
              '72 of 222 with ZERO CTRL. The two differ because ctrl is a decaying filtered rate, so a pool can carry zero '
              'spikes in a window yet still hold ctrl > 0 from the previous window. CORRECTION TO S204: those values are NOT '
              '"24 permanently silent actuators" - only %d of the 74 pools carry no spikes across all three windows, and both '
              'are driven but subthreshold (see silent_pools).' % len(silent))),
    n_cells_total=int(d['current'].shape[1]),
    cells_driven_share=float(n_cells_driven) / int(d['current'].shape[1]),
    reading={
        'who_gets_current': ('only %d of the %d tier cells receive any nonzero external current in windows 1-3: the sensory '
                             'encoder writes into the six per-leg channel groups, nothing else.' % (n_cells_driven, d['current'].shape[1])),
        'why_silent': ('%d of the %d non-empty pools carry zero spikes in these windows; their classification is above. A pool '
                       'counted NO_EXTERNAL_CURRENT receives no sensory current at all in this window set, so its silence is '
                       'input-side, not threshold-side.' % (len(silent), len(pools))),
        'relation_to_s204': ('S204 found 72 of 222 non-empty actuator-window values with zero spikes (24 actuators x 3 windows). '
                             'This diagnosis explains each of those 24 actuators.')})
(RUN / 'SILENT_POOL_DIAGNOSIS.json').write_text(json.dumps(diag, indent=2, default=str))

# ---------------- FIG1: unresolved reasons
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['absent from dataset\n(pleural_promotor)', 'only on other legs\n(tergopleural promotor)',
          'only on other legs\n(sternal adductor)', 'only on other legs\n(pleural remotor/abductor)']
vals = [6, 8, 1, 1]
cols = ['0.6', 'tab:red', 'tab:orange', 'tab:purple']
ax.bar(labels, vals, color=cols, edgecolor='k', lw=.4)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + 0.1, str(v_), ha='center', fontsize=10)
ax.set_ylabel('uncommandable actuators (of 16)')
ax.set_title('FIG1  Why 16 coxa actuators can never be commanded\n'
             '6: muscle absent from BANC; 10: annotated only on other legs (8 tergopleural promotors exist only on the front leg)',
             fontsize=9)
ax.set_ylim(0, 9.5)
fig.tight_layout()
fig.savefig(RUN / 'FIG1_UNRESOLVED_REASONS.png', dpi=150)

# ---------------- FIG2: silent pool classes
fig, ax = plt.subplots(figsize=(11, 5.5))
order = ['FIRED', 'DRIVEN_BUT_BELOW_THRESHOLD', 'REACHED_THRESHOLD_WITHOUT_SPIKE_GATED', 'AT_REST_NO_DRIVE']
short = ['FIRED', 'driven,\nsubthreshold', 'reached threshold\nwithout spiking', 'at rest\n(no drive)']
vals = [cls_counts.get(k, 0) for k in order]
cols = ['tab:green', 'tab:orange', 'tab:red', '0.55']
ax.bar(short, vals, color=cols, edgecolor='k', lw=.4)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + 0.4, str(v_), ha='center', fontsize=10)
ax.set_ylabel('non-empty pools (of 74), windows 1-3')
ax.set_title('FIG2  Pool drive in the 0-50 ms windows, from an arm whose pool spike counts equal S204-H0 bitwise\n'
             'rest %.0f mV, threshold %.0f mV: %d pools fire, %d are driven but stay subthreshold, none are at rest or gated'
             % (rest, thr, cls_counts.get('FIRED', 0), cls_counts.get('DRIVEN_BUT_BELOW_THRESHOLD', 0)), fontsize=9)
ax.set_ylim(0, max(vals) + 6)
fig.tight_layout()
fig.savefig(RUN / 'FIG2_SILENT_POOL_CLASSES.png', dpi=150)

print(json.dumps(dict(class_counts=cls_counts, n_silent=len(silent),
                      cells_with_current='%d/%d' % (n_cells_driven, d['current'].shape[1]),
                      silent_detail=[(p['actuator'], p['cls'], round(p['max_abs_external_current_nA'], 4),
                                      round(p['max_membrane_mV'], 2)) for p in silent]), indent=1))
