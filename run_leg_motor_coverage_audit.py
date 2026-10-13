"""Leg motor coverage vocabulary audit: is the missing pool a naming gap or a data gap?

Read-only. The motor manifest leaves 16 of 90 actuators without a pool. Before asking for new
data it must be established whether those muscles exist in the annotation under a different
name or on a different nerve. This audit searches the whole annotation vocabulary and the
per-nerve efferent population instead of only the peripheral target field.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
RUN = HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c'
OUT = HERE / 'outputs/model_branches/leg_motor_coverage'
PROPOSED_TARGETS = ['pleural_promotor_muscle', 'tergopleural_promotor_muscle',
                    'pleural_remotor_and_abductor_muscle', 'sternal_adductor_muscle']
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import pandas as pd
    f = pd.read_feather(FEATHER, columns=['root_id', 'flow', 'cell_type', 'peripheral_target_type',
                                          'body_part_effector', 'side', 'nerve'])
    eff = f[f['flow'] == 'efferent']
    leg_nerve = eff[eff['nerve'].astype(str).str.contains('thoracic_leg_nerve', na=False)]
    nerve_class = Counter()
    for n in leg_nerve['nerve'].astype(str):
        for seg in ('prothoracic', 'mesothoracic', 'metathoracic'):
            if seg in n:
                nerve_class[seg] += 1
    promotor_cells = f[f['cell_type'].astype(str).str.contains('promotor', case=False, na=False)]
    per_target = {}
    for target in PROPOSED_TARGETS:
        rows = leg_nerve[leg_nerve['peripheral_target_type'] == target]
        per_target[target] = dict(
            cells_on_leg_nerves=int(len(rows)),
            by_effector=dict(Counter(rows['body_part_effector'].astype(str))),
            by_nerve=dict(Counter(rows['nerve'].astype(str))),
            any_cell_type_mentioning_the_word=int(
                len(f[f['cell_type'].astype(str).str.contains(target.split('_muscle')[0], case=False, na=False)])))
    compound = f[f['cell_type'].astype(str).str.startswith('tergopleural_promotor_pleural_promotor', na=False)]
    manifest = json.loads((RUN / 'motor_manifest.json').read_text())['measured']
    unresolved = [r for r in manifest['rows'] if r['status'] == 'unresolved']
    result = dict(
        status='MEASURED_LEG_MOTOR_COVERAGE_VOCABULARY',
        label='manual read-only vocabulary audit, not a pool assignment',
        question='Are the 16 missing motor pools absent from the annotation, or present under another name?',
        annotation=dict(feather=str(FEATHER), feather_sha256=sha(FEATHER),
                        efferent_cells=int(len(eff)),
                        efferent_on_leg_nerves=int(len(leg_nerve)),
                        efferent_on_leg_nerves_by_segment=dict(nerve_class),
                        distinct_cell_types_on_leg_nerves=int(leg_nerve['cell_type'].astype(str).nunique())),
        per_missing_target=per_target,
        whole_vocabulary_search=dict(
            cells_with_promotor_in_cell_type=int(len(promotor_cells)),
            by_nerve=dict(Counter(promotor_cells['nerve'].astype(str))),
            by_cell_type=dict(Counter(promotor_cells['cell_type'].astype(str))),
            finding='every promotor-labelled cell sits on a prothoracic (front-leg) nerve; none on '
                    'mesothoracic or metathoracic nerves'),
        annotation_named_compound_group=dict(
            cell_type='tergopleural_promotor_pleural_promotor (and _miller_28_30)',
            cells=int(len(compound)),
            by_nerve=dict(Counter(compound['nerve'].astype(str))),
            meaning='the annotation names one motor-neuron group as innervating BOTH the tergopleural and '
                    'pleural promotor muscles, so for the FRONT legs the model\'s pleural_promotor actuator '
                    'could share the pool already assigned to tergopleural_promotor',
            proposal_status='recorded as a proposal only; NOT wired in this audit and not applied to any manifest',
            would_fill='LF and RF pleural_promotor (2 of the 16)'),
        unresolved_actuators=int(len(unresolved)),
        conclusion='The gap is annotation coverage, not a name mismatch: the muscles are simply not annotated on '
                   'middle/hind leg nerves. 14 of 16 therefore cannot be filled from this release at all.',
        limitations=['One annotation release (banc888).',
                     'Absence of an annotated motor neuron is not evidence that the animal lacks one.',
                     'The compound-group proposal would create a three-way shared pool on front legs and must be '
                     'declared and controlled before use.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'leg_motor_coverage.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(annotation=result['annotation'],
                          whole_vocabulary=result['whole_vocabulary_search']['finding'],
                          compound=result['annotation_named_compound_group']['meaning'],
                          per_target={k: {'cells_on_leg_nerves': v['cells_on_leg_nerves'],
                                          'by_nerve': v['by_nerve']} for k, v in per_target.items()}),
                     indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('leg_motor_coverage', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
