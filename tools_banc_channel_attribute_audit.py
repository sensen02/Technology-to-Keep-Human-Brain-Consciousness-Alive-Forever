"""Read-only runtime-root annotation audit. 手工研究原型，不是成品.

Annotation variation is not tuning evidence. Correlation is not a causal
restoring-feedback test. No roots, sensory currents or connectivity are changed.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd
from run_sensory_tuning_annotation_audit import exact_groups, unique_join, sha, text, polarity

HERE = Path(__file__).resolve().parent
META = HERE/'data/banc888/banc_888_meta.feather'
LOOKUP = HERE/'outputs/neural_rebuild/sensory_channel_lookup.json'
RUNS = HERE/'outputs/neural_muscle_closure'
OUT = HERE/'outputs/model_branches/channel_attribute_audit'
COLUMNS = ['side', 'nerve', 'neuromere', 'region', 'root_region', 'cell_sub_class',
           'cell_type', 'manc_cell_type', 'cell_function', 'cell_function_detailed',
           'body_part_sensory', 'peripheral_target_type', 'manual_cluster', 'cluster'] + [
           f'seed_{i:02d}' for i in range(1, 15)]


def normalized(value):
    # Arrow-backed strings preserve NA under astype(str); test missingness first.
    if pd.isna(value):
        return None
    return text(value)


def column_summary(values):
    values = list(values)
    counts = Counter(v for x in values if (v := normalized(x)) is not None)
    return {'nonempty': sum(counts.values()), 'missing': len(values)-sum(counts.values()),
            'unique_nonempty': len(counts), 'counts': dict(sorted(counts.items())),
            'two_groups_at_least_two_cells': sum(n >= 2 for n in counts.values()) >= 2}


def joint_terms(annotations):
    # Literal annotation terms only, not a root-to-mechanical-joint assignment.
    s = ' '.join(v or '' for v in annotations.values()).lower()
    return sorted(set(re.findall(r'(?<![a-z])(coxa|coxal|trochanter|trochanteral|femur|femoral|tibia|tibial|tarsus|tarsal)(?![a-z])', s)))


def review_flags(channel, annotations):
    flags = []
    target = annotations.get('peripheral_target_type') or ''
    function = annotations.get('cell_function_detailed') or ''
    if channel == 'campaniform_strain' and ('hair_plate' in target or function == 'joint_angle'):
        flags.append('campaniform_channel_vs_hairplate_or_joint_angle_annotation')
    if channel == 'bristle_touch' and function:
        flags.append('bristle_nonempty_function_requires_modality_review')
    if channel == 'hairplate_angle' and annotations.get('cell_type') == 'CS':
        flags.append('hairplate_channel_with_CS_type_label_requires_review')
    if target == 'orphan' or not target:
        flags.append('missing_or_orphan_peripheral_target')
    return flags


def audit():
    pointer_path = RUNS/'current_run.json'
    pointer = json.loads(pointer_path.read_text())
    bundle = RUNS/pointer['path']
    identity_path, manifest_path = bundle/'neural_identity.json', bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest_path) != pointer['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    if sha(identity_path) != manifest['files']['neural_identity.json']['sha256']:
        raise ValueError('Identity hash mismatch')
    inputs = [pointer_path, manifest_path, identity_path, META, LOOKUP, Path(__file__),
              HERE/'run_sensory_tuning_annotation_audit.py']
    hashes = {str(p): sha(p) for p in inputs}
    identity = json.loads(identity_path.read_text())
    groups = exact_groups(identity, json.loads(LOOKUP.read_text()))
    if not groups:
        raise ValueError('Empty sensory audit')
    df = pd.read_feather(META, use_threads=False)
    joined = unique_join(df['banc_888_id'], {r['root_id'] for r in groups})
    columns = [c for c in COLUMNS if c in df]
    cells = []
    for group in groups:
        row = df.iloc[joined[group['root_id']]]
        annotations = {c: normalized(row[c]) for c in columns}
        cells.append(dict(group, annotations=annotations,
                          review_flags=review_flags(group['channel'], annotations),
                          literal_joint_terms=joint_terms(annotations),
                          polarity_lexical_status=polarity(annotations.values()),
                          confirmed_root_specific_joint=None,
                          confirmed_tuning_polarity=None))
    table = {}
    for channel in sorted({r['channel'] for r in cells}):
        table[channel] = {}
        for leg in sorted({r['leg'] for r in cells if r['channel'] == channel}):
            subset = [r for r in cells if (r['channel'], r['leg']) == (channel, leg)]
            table[channel][leg] = {'n_cells': len(subset),
                'columns': {c: column_summary(r['annotations'][c] for r in subset) for c in columns},
                'polarity_lexical_status': dict(Counter(r['polarity_lexical_status'] for r in subset)),
                'literal_joint_terms': dict(Counter(t for r in subset for t in r['literal_joint_terms']))}
    if hashes != {str(p): sha(p) for p in inputs}:
        raise ValueError('Inputs changed during audit')
    result = {'status': 'EXACT_RUNTIME_ANNOTATION_AUDIT_NOT_TUNING_ASSIGNMENT',
              'label': '手工研究原型，不是成品',
              'tier_neurons': len(identity['tier_root_ids']), 'sensory_cells': len(cells),
              'join_key': 'banc_888_id', 'joined_unique_roots': len(joined),
              'columns_examined': columns, 'per_channel': table, 'cells': cells,
              'input_sha256': hashes,
              'limits': [
                  'Annotation category splits, SNpp numbers and seed labels do not establish tuning.',
                  'Literal body-segment terms do not identify a mechanical hinge or signed response.',
                  'FeCO claw/hook/club femur-tibia association is literature-level organ evidence, not a per-root polarity crosswalk.',
                  'Hairplate subtype/plate-to-joint and campaniform field/orientation require an external verified root crosswalk.',
                  'No position coordinate is treated as a peripheral soma or converted to anatomical units.',
                  'No dynamical stability or standing success is tested; no simulation input is changed.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'channel_attribute_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'runtime_sensory_evidence.csv').open('w', newline='') as stream:
        names = ['root_id', 'tier_index', 'leg', 'channel', 'polarity_lexical_status', 'literal_joint_terms']+columns
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for r in cells:
            writer.writerow({**{k: r[k] for k in names[:5]},
                             'literal_joint_terms': ';'.join(r['literal_joint_terms']), **r['annotations']})
    flagged = [r for r in cells if r['review_flags']]
    (OUT/'annotation_review_queue.json').write_text(json.dumps({
        'label': '手工研究原型，不是成品',
        'meaning': 'Conflicts or missing evidence requiring review; NOT automatic reclassification.',
        'n_roots': len(flagged),
        'flag_counts': dict(Counter(f for r in flagged for f in r['review_flags'])),
        'cells': flagged}, ensure_ascii=False, indent=2)+'\n')
    plot(result)
    print(json.dumps({'sensory_cells': len(cells), 'per_channel': {k: sum(v['n_cells'] for v in legs.values()) for k, legs in table.items()}, 'output': str(OUT)}, indent=2))
    return result


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    channels = list(result['per_channel'])
    legs = ['lf', 'rf', 'lm', 'rm', 'lh', 'rh']
    a = np.array([[result['per_channel'][c].get(l, {}).get('n_cells', 0) for l in legs] for c in channels])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(a, cmap='Blues', aspect='auto')
    for i in range(len(channels)):
        for j in range(len(legs)):
            ax.text(j, i, str(a[i, j]), ha='center', va='center', color='white' if a[i,j] > a.max()/2 else 'black')
    ax.set_xticks(range(6), legs)
    ax.set_yticks(range(len(channels)), channels)
    ax.set_title('Exact runtime sensory roots joined to BANC v888 (n='+str(result['sensory_cells'])+')')
    fig.colorbar(im, ax=ax, label='Cells, not sensing strength')
    fig.text(.5, .025, 'MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT\nAnnotation coverage only; polarity and root-specific joints unresolved.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT/'runtime_sensory_coverage.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    audit()
