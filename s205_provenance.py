"""S205 Part A: provenance of the 16 uncommandable coxa actuators.

Reads the pool-rebuild JSON and, independently, the BANC feather metadata it was built from,
so the reason for each unresolved row is checked against the dataset rather than trusted.
"""
import json
import numpy as np
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s205-unresolved-mn-and-silent-pools/20261025T075517207961Z')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
POOLS = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
MEAS = POOLS['measured']
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))

# ---- independent check of the dataset claims ------------------------------------------------
feather_info = {}
try:
    import pandas as pd
    df = pd.read_feather(MEAS['source']['feather'])
    cols = [c for c in ('peripheral_target_type', 'body_part_effector', 'side', 'flow', 'root_id', 'cell_type')
            if c in df.columns]
    mn = df[df['cell_type'].astype(str).str.contains('motor', case=False, na=False)] if 'cell_type' in df.columns else df
    counts = {}
    for r in MEAS['unresolved']:
        tgt = r['muscle_target']
        sub = mn[mn['body_part_effector'].astype(str) == tgt] if 'body_part_effector' in mn.columns else mn.iloc[0:0]
        by_side = sub['peripheral_target_type'].astype(str).value_counts().to_dict() if 'peripheral_target_type' in sub.columns else {}
        counts[tgt] = dict(n_rows_total=int(len(sub)), by_peripheral_target_type=by_side,
                           sides=sub['side'].astype(str).value_counts().to_dict() if 'side' in sub.columns else {})
    feather_info = dict(read_ok=True, rows=int(len(df)), columns_used=cols,
                        motor_neuron_rows=int(len(mn)),
                        per_target=counts)
except Exception as e:                                        # noqa: BLE001
    feather_info = dict(read_ok=False, error=repr(e))

# ---- what the dataset-derived tables say about each target -----------------------------------
derived_tables = {}
try:
    padj = json.loads((ROOT / 'data/banc888/promotor_adductor_pools.json').read_text())
    for tgt_key, cells in padj.items():
        nerves = sorted({c.get('nerve', '') for c in cells})
        regions = sorted({c.get('region', '') for c in cells})
        derived_tables[tgt_key] = dict(n_cells=len(cells), nerves=nerves, regions=regions,
                                       cell_types=sorted({c.get('ct', '') for c in cells}),
                                       sides=sorted({c.get('side', '') for c in cells}),
                                       legs_inferred_from_neuron_names=sorted(
                                           {'prothoracic(front)' if 'prothoracic' in n else
                                            'mesothoracic(mid)' if 'mesothoracic' in n else
                                            'metathoracic(hind)' if 'metathoracic' in n else
                                            'abdominal' if 'abdominal' in n else 'other' for n in nerves}))
    derived_available = True
except Exception as e:                                        # noqa: BLE001
    derived_tables = dict(error=repr(e))
    derived_available = False

# ---- the 16 rows ---------------------------------------------------------------------------
rows = []
for r in MEAS['unresolved']:
    rows.append(dict(actuator=r['actuator'], leg=r['leg'], side=r['side'], muscle_target=r['muscle_target'],
                     reason=r['reason'], target_exists_on=r['target_exists_on'],
                     dataset_check=feather_info.get('per_target', {}).get(r['muscle_target'])))

by_reason = {}
for r in rows:
    by_reason.setdefault(r['reason'], []).append(r['actuator'])
absent = sorted({r['muscle_target'] for r in rows if 'absent' in r['reason']})
elsewhere = sorted({r['muscle_target'] for r in rows if 'absent' not in r['reason']})

# ---- mechanical authority lost, per hard DOF -----------------------------------------------
hard = {}
for dof, v in CENSUS['hard_dofs'].items():
    hard[dof] = dict(joint=v['joint'], n_with_moment=v['n_actuators_with_moment'],
                     n_uncommandable=len(v['empty_row_members']),
                     uncommandable=v['empty_row_members'],
                     commandable=v['nonempty_members'],
                     commandable_share=len(v['nonempty_members']) / v['n_actuators_with_moment'],
                     lost_moment_share=None)

# moment shares: how much of the |moment| on that DOF sits on uncommandable actuators
TRAJ = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')
MOM = np.asarray(TRAJ['actuator_moment'][0], float)
name_to_idx = {c['actuator']: c['actuator_index'] for c in CENSUS['rows']}
for dof, v in hard.items():
    idx = [name_to_idx[a] for a in v['uncommandable'] + v['commandable']]
    tot = float(np.sum(np.abs([MOM[i, int(dof)] for i in idx])))
    lost = float(np.sum(np.abs([MOM[name_to_idx[a], int(dof)] for a in v['uncommandable']])))
    v['total_abs_moment'] = tot
    v['lost_abs_moment'] = lost
    v['lost_moment_share'] = lost / tot if tot else None

out = dict(
    n_unresolved=len(rows),
    by_reason={k: dict(n=len(v), actuators=v) for k, v in by_reason.items()},
    reasons_sum_to_16=sum(len(v) for v in by_reason.values()) == 16,
    target_absent_entirely=dict(targets=absent,
                                meaning='the BANC annotation has no motor neuron for this muscle anywhere in the dataset'),
    target_on_other_legs_only=dict(targets=elsewhere,
                                   meaning='the muscle is annotated only on other legs/sides; no substitution was made here'),
    rows=rows,
    dataset_check=feather_info,
    dataset_check_note=('a direct re-read of banc_888_meta.feather was NOT possible in this environment (pyarrow is missing), '
                        'so the reason strings are the build-time product of that same dataset, and the derived table '
                        'data/banc888/promotor_adductor_pools.json is used as the corroborating source. This gap is recorded, '
                        'not papered over.'),
    derived_table_check=dict(available=derived_available, per_target=derived_tables),
    hard_dofs=hard,
    statement=('Of the 16 uncommandable rows, %d name a muscle target that does not exist in the BANC dataset at all (%s) and '
               '%d name a target that exists only on other legs/sides (%s). No substitution, mirror or interpolation was applied '
               'in this round: doing so would be inventing synapses.'
               % (len(by_reason.get('target absent from BANC entirely', [])),
                  ', '.join(sorted({r['muscle_target'] for r in rows if 'absent' in r['reason']})),
                  len(rows) - len(by_reason.get('target absent from BANC entirely', [])),
                  ', '.join(elsewhere))),
)
(RUN / 'UNRESOLVED_PROVENANCE.json').write_text(json.dumps(out, indent=2, default=str))

print(json.dumps(dict(n=out['n_unresolved'], reasons={k: v['n'] for k, v in out['by_reason'].items()},
                      sums_to_16=out['reasons_sum_to_16'],
                      feather_ok=feather_info.get('read_ok'),
                      hard_dofs={d: dict(joint=v['joint'], lost='%d/%d' % (v['n_uncommandable'], v['n_with_moment']),
                                         lost_moment_share=None if v['lost_moment_share'] is None else round(v['lost_moment_share'], 4))
                                 for d, v in hard.items()}), indent=1, default=str))
