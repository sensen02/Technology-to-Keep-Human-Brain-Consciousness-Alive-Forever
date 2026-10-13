"""Export BANC leg-efferent muscle coverage (needs pandas; run with venv/bin/python).

Read-only. Writes the per-leg/side cell counts for every annotated leg muscle target,
so the mujoco-side gap audit can read it without needing pandas.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
OUT = HERE / 'outputs/model_branches/model_annotation_gap/banc_leg_muscle_coverage.json'
LEGS = ['front_leg', 'middle_leg', 'hind_leg']


def main():
    f = pd.read_feather(FEATHER, columns=['flow', 'body_part_effector', 'peripheral_target_type', 'side'])
    e = f[(f['flow'] == 'efferent') & (f['body_part_effector'].isin(LEGS))]
    coverage = {}
    for target, group in e.groupby('peripheral_target_type'):
        counts = group.groupby(['body_part_effector', 'side']).size()
        coverage[str(target)] = {f'{k[0]}/{k[1]}': int(v) for k, v in counts.items()}
    doc = dict(status='EXPORTED_BANC_LEG_MUSCLE_COVERAGE',
               label='manual read-only export, not an interpretation',
               feather=str(FEATHER), feather_sha256=hashlib.sha256(FEATHER.read_bytes()).hexdigest(),
               efferent_leg_cells=int(len(e)), legs=LEGS,
               leg_muscle_targets=coverage,
               note='counts are annotated efferent cells per (leg, side) for each peripheral_target_type')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + '\n')
    print(OUT)
    print(json.dumps(coverage, indent=2))


if __name__ == '__main__':
    main()
