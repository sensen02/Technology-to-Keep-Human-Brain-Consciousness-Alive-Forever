"""Bounded read-only annotation audit. No network, tier build, or polarity assignment."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import gzip
import hashlib
import json
import math
import re
import signal
from collections import Counter
from pathlib import Path
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/neural_muscle_closure'
FEATHER = HERE/'data/banc888/banc_888_meta.feather'
LOOKUP = HERE/'outputs/neural_rebuild/sensory_channel_lookup.json'
CSV = HERE/'data/flywire/banc_neurons.csv.gz'
MISSING = {'', 'na', 'nan', 'none', 'null', '<na>'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def text(value):
    if value is None: return None
    value = str(value).strip()
    return None if value.lower() in MISSING else value


def root(value):
    if isinstance(value, (float, bool)) or not str(value).isdecimal():
        raise ValueError('Root ID must be an exact decimal string/integer')
    return str(int(value))


def polarity(values):
    """Lexical evidence only; SNpp codes, side, shape and position are NOT polarity."""
    s = ' '.join(text(v) or '' for v in values).lower()
    flex = bool(re.search(r'(?<![a-z])flex(?:ion|or|ing)?(?![a-z])|屈曲', s))
    ext = bool(re.search(r'(?<![a-z])extens(?:ion|or|ing)?(?![a-z])|伸展', s))
    if flex and ext: return 'both_terms_requires_review'
    if flex: return 'flexion_term_requires_review'
    if ext: return 'extension_term_requires_review'
    return 'no_explicit_polarity_tag' if s.strip() else 'missing_annotation'


def summarize(values):
    values = list(values)
    c = Counter(v for x in values if (v := text(x)) is not None)
    return {'rows': len(values), 'nonempty': sum(c.values()), 'missing': len(values)-sum(c.values()),
            'unique_nonempty': len(c), 'examples_top8': dict(c.most_common(8)),
            'polarity_classifications': dict(Counter(polarity([v]) for v in values))}


def exact_groups(identity, lookup):
    roots = list(map(root, identity['tier_root_ids']))
    if len(set(roots)) != len(roots): raise ValueError('Duplicate tier root')
    result = []
    seen = set()
    for leg, channels in identity['sensory_groups'].items():
        for channel, indices in channels.items():
            for i in indices:
                if type(i) is not int or not 0 <= i < len(roots) or i in seen:
                    raise ValueError('Invalid/overlapping sensory index')
                seen.add(i)
                rid = roots[i]
                if lookup['leg_channel_by_root'].get(rid) != [leg, channel]:
                    raise ValueError('Stored group disagrees with current lookup')
                result.append({'tier_index': i, 'root_id': rid, 'leg': leg, 'channel': channel})
    expected = {r for r in roots if r in lookup['leg_channel_by_root']}
    if {r['root_id'] for r in result} != expected: raise ValueError('Incomplete stored groups')
    return result


def unique_join(ids, targets):
    positions = {}
    for i, value in enumerate(ids):
        rid = root(value)
        if rid in targets:
            if rid in positions: raise ValueError('Duplicate annotation join root: '+rid)
            positions[rid] = i
    if set(positions) != set(targets): raise ValueError('Missing annotation join roots')
    return positions


def audit(report):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc
    pointer = OUT/'current_run.json'
    current = json.loads(pointer.read_text())
    bundle = OUT/current['path']
    identity_path = bundle/'neural_identity.json'
    manifest_path = bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == current['manifest_sha256']
    assert sha(identity_path) == manifest['files']['neural_identity.json']['sha256']
    paths = [pointer, manifest_path, identity_path, FEATHER, LOOKUP, CSV,
             HERE/'engine/embodied/sensory_channels.py', HERE/'run_neural_muscle_closure.py',
             HERE/'tools_export_sensory_channel_lookup.py', Path(__file__),
             HERE/'test_sensory_tuning_annotation_audit.py']
    before = {str(p): sha(p) for p in paths}
    identity = json.loads(identity_path.read_text())
    lookup = json.loads(LOOKUP.read_text())
    groups = exact_groups(identity, lookup)
    focused = [r for r in groups if r['channel'] in ('claw_position','hook_direction')]
    with pa.memory_map(str(FEATHER),'r') as src: schema = ipc.open_file(src).schema
    d = pd.read_feather(FEATHER, use_threads=False)
    targets = {r['root_id'] for r in groups}
    joined = unique_join(d['banc_888_id'], targets)
    annotation_cols = [c for c in d.columns if not (c in ('banc_888_id','root_id','supervoxel_id','nucleus_id') or re.fullmatch(r'root_\d+',c))]
    local = d[d.cell_sub_class.fillna('').str.contains('claw_chordotonal|hook_chordotonal', regex=True)]
    exact = d.iloc[[joined[r['root_id']] for r in focused]]
    csv_rows = {}
    with gzip.open(CSV,'rt',newline='') as stream:
        reader = csv.DictReader(stream)
        csv_headers = reader.fieldnames
        for row in reader:
            rid = root(row['Root ID'])
            if rid in targets:
                if rid in csv_rows: raise ValueError('Duplicate raw CSV root')
                csv_rows[rid] = row
    assert set(csv_rows) == targets
    csv_cols = [c for c in csv_headers if c != 'Root ID']
    records = []
    for rec in focused:
        row = d.iloc[joined[rec['root_id']]]
        raw = csv_rows[rec['root_id']]
        evidence = {c:text(row[c]) for c in annotation_cols}
        records.append(dict(rec, feather_zero_based_row=joined[rec['root_id']],
                            annotations=evidence, raw_csv_annotations={c:text(raw[c]) for c in csv_cols},
                            polarity_classification=polarity(list(evidence.values())+[raw[c] for c in csv_cols]),
                            assigned_polarity=None))
    counts = Counter(r['channel'] for r in groups)
    per_leg = {leg:dict(Counter(r['channel'] for r in groups if r['leg']==leg)) for leg in identity['sensory_groups']}
    numeric = [f.name for f in schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)]
    tuning_cols = [c for c in d.columns if re.search(r'target.?angle|preferred.?angle|tuning|gain|tonic|firing.?rate|flexion|extension',c,re.I)]
    result = {'status':'READ_ONLY_LOCAL_ANNOTATION_AUDIT_NOT_BIOLOGICAL_POLARITY_TEST',
        'current_run':current, 'identity_path':str(identity_path), 'join_key':'banc_888_id',
        'source_sha256_before':before, 'source_sha256_after':{str(p):sha(p) for p in paths},
        'schema':{f.name:str(f.type) for f in schema}, 'feather_rows':len(d),
        'tier_neurons':len(identity['tier_root_ids']), 'exact_groups_total':len(groups),
        'exact_groups_counts':dict(counts), 'exact_groups_per_leg':per_leg,
        'exact_groups':groups, 'all_sensory_roots_joined_once':len(joined),
        'local_claw_hook_rows':len(local), 'local_claw_hook_subclass_counts':local.cell_sub_class.value_counts().to_dict(),
        'exact_claw_hook_rows':len(focused), 'exact_claw_hook_polarity_counts':dict(Counter(r['polarity_classification'] for r in records)),
        'local_claw_hook_columns':{c:summarize(local[c]) for c in annotation_cols},
        'exact_claw_hook_columns':{c:summarize(exact[c]) for c in annotation_cols},
        'raw_csv_headers':csv_headers,
        'raw_csv_exact_claw_hook_columns':{c:summarize(csv_rows[r['root_id']][c] for r in focused) for c in csv_cols},
        'requested_column_presence':{name:name in d.columns for name in ('finecelltype','morphologylabel','sidenames','lineage','hemilineage','cell_type','side')},
        'numeric_fields':{c:summarize(exact[c]) for c in numeric},
        'candidate_tuning_parameter_columns':tuning_cols,
        'per_cell_target_angle_gain_tonic_available':bool(tuning_cols),
        'numeric_interpretation':'Morphometric sizes, connectivity/side indices, mitochondria and NT scores are not measured sensory angle/gain/tonic tuning. position/root_position/root_position_nm are anatomical coordinates.',
        'per_root_claw_hook':records,
        'mapping_availability':'Coarse claw(position)/hook(direction) and leg identities plus uninterpreted SNpp type codes; no explicit root-level flexion/extension mapping established by these local annotations.',
        'limitations':['Lexical absence does not prove biological absence of opposing subpopulations.',
                      'SNpp codes and cross-dataset matches require a validated external physiological crosswalk; no morphology/side/ID/random split was used.',
                      'Existing channel current/angle/velocity normalizers are declared model settings, not per-cell physiological measurements.',
                      'No core model edits, network rebuild, literature-derived root assignment, or polarity assignment.'],
        'memory_guard':dict(report())}
    assert result['source_sha256_before'] == result['source_sha256_after'], 'Inputs changed during audit'
    result['all_sources_unchanged'] = True
    if not math.isfinite(result['memory_guard']['previous_soft_limit_mb']):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    dest = OUT/'sensory_tuning_annotation_audit.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('tier_neurons','exact_groups_total','exact_groups_counts','exact_groups_per_leg','exact_claw_hook_rows','exact_claw_hook_polarity_counts','candidate_tuning_parameter_columns','memory_guard')},indent=2))
    return result


def main():
    def deadline(*_): raise TimeoutError('60-second annotation audit bound exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    try:
        with mg.guarded_run('sensory_tuning_annotation_audit',need_mb=512,cap_mb=2048,wait_s=45) as report:
            audit(report)
    finally: signal.alarm(0)

if __name__ == '__main__': main()
