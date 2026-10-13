"""Export the annotated motor pools for the two added tarsus muscles (needs pandas).

Run with venv/bin/python. Writes one record per new actuator, listing the exact BANC root IDs of
the motor neurons annotated for that muscle on that leg and side. Nothing is inferred.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
OUT = HERE / 'outputs/model_branches/tarsus_dof/tarsus_motor_pools.json'
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
EFFECTOR = {'LF': 'front_leg', 'RF': 'front_leg', 'LM': 'middle_leg', 'RM': 'middle_leg',
            'LH': 'hind_leg', 'RH': 'hind_leg'}
TARGETS = (('tarsus_depressor_muscle', 'depressor'), ('tarsus_levator_muscle', 'levator'))


def main():
    f = pd.read_feather(FEATHER, columns=['banc_888_id', 'flow', 'peripheral_target_type',
                                          'body_part_effector', 'side', 'cell_type'])
    f = f[f['flow'] == 'efferent']
    records = []
    for leg in LEGS:
        for target, suffix in TARGETS:
            side = 'left' if leg[0] == 'L' else 'right'
            sel = f[(f['peripheral_target_type'] == target) & (f['body_part_effector'] == EFFECTOR[leg])
                    & (f['side'] == side)]
            roots = sorted(int(x) for x in sel['banc_888_id'])
            records.append(dict(actuator=f'{leg}Tarsus_{suffix}', leg=leg, side=side, muscle_target=target,
                                n_pool=len(roots), pool_root_ids=roots,
                                pool_cell_types={str(k): int(v) for k, v in sel['cell_type'].value_counts().items()}))
    doc = dict(status='EXPORTED_TARSUS_MOTOR_POOLS',
               label='manual read-only export from the BANC annotation; no cell is borrowed or inferred',
               feather=str(FEATHER), feather_sha256=hashlib.sha256(FEATHER.read_bytes()).hexdigest(),
               records=records,
               totals={r['actuator']: r['n_pool'] for r in records})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps(doc['totals'], indent=2))


if __name__ == '__main__':
    main()
