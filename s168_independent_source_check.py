#!/usr/bin/env python3
"""S168 independent pure-data provenance check; no engine imports or edits."""
import csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path('/run/media/sensen/Data2/cell_wound_prototype')
S = Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z')
F = Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z/BOUND_INPUT_FREEZE.json')
OUT = P / 'S168_INDEPENDENT_SOURCE_CHECK.json'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def csvrows(p):
    with open(p, newline='') as f: return list(csv.DictReader(f))
freeze = json.loads(F.read_text())
records = {r['source']: r for r in freeze['records']}
def bound(source):
    r = records[source]; p = Path(r['frozen'])
    assert sha(p) == r['sha256']
    return p
manifest = bound('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z/MANIFEST.json')
npz = bound(str(P / 'data/flywire/banc_connectome.npz'))
raw = P / 'data/flywire/banc_neurons.csv.gz'
meta = P / 'data/banc888/banc_888_meta.feather'
inputs = [F, manifest, npz, raw, meta, S/'GRAPH_90_MUSCLE_MAPPING.csv', S/'GRAPH_ALL_NEURONS.csv']
hashes = {str(p):sha(p) for p in inputs}
z = np.load(npz, allow_pickle=False)
roots = [str(int(x)) for x in z['root_ids']]
assert len(roots) == len(set(roots))
global_by_root = {r:i for i,r in enumerate(roots)}
tier = csvrows(S/'GRAPH_ALL_NEURONS.csv')
tier_by_root = {r['root_id']: r for r in tier}
assert len(tier_by_root) == 17099
for r in tier:
    assert roots[int(r['global_dataset_index'])] == r['root_id']
    assert int(r['local_index']) < 17099
mapping = csvrows(S/'GRAPH_90_MUSCLE_MAPPING.csv')
m = json.loads(manifest.read_text())
missing = [r for r in mapping if r['mapping_status'].startswith('UNMAPPED')]
assert len(missing) == 16
with gzip.open(raw,'rt') as f: old = list(csv.DictReader(f))
old_by_root = {r['Root ID']:r for r in old}
df = pd.read_feather(meta).fillna('')
# Arrow identifier columns are strings. Never cast root IDs through float.
assert all(isinstance(x,str) for x in df['root_id'])
new = df.to_dict('records')
body = {'F':'front_leg','M':'middle_leg','H':'hind_leg'}
def legmatch(r, leg):
    return r['side'] == ('left' if leg[0]=='L' else 'right') and r['body_part_effector'] == body[leg[1]]
def evidence(r):
    root = r['root_id']; t = tier_by_root.get(root)
    o = old_by_root.get(root,{})
    return {'root_id':root,'root_888':r['root_888'], 'full_frozen_dataset_index':global_by_root.get(root),
            'tier_local_index':int(t['local_index']) if t else None,
            'motor_identity':{'v888_super_class':r['super_class'],'v888_cell_class':r['cell_class'],
                              'raw_super_class':o.get('Super Class'),'raw_class':o.get('Class')},
            'muscle_evidence':{'peripheral_target_type':r['peripheral_target_type'],
                               'body_part_effector':r['body_part_effector'],'side':r['side'],'nerve':r['nerve'],
                               'cell_type_alias_not_sufficient_alone':r['cell_type']},
            'raw_primary_type':o.get('Primary Cell Type'), 'raw_alternative_types':o.get('Alternative Cell Type(s)')}
rows = []
for r in missing:
    i = int(r['actuator_index']); mr=m['rows'][i]
    assert mr['actuator'] == r['actuator'] and mr['pool_root_ids'] == [] and mr['tier_root_ids'] == [] and mr['local_indices'] == []
    target=r['muscle_target']; leg=r['leg']; needle=target.removesuffix('_muscle')
    exact=[x for x in new if x['super_class']=='motor' and x['peripheral_target_type']==target and legmatch(x,leg)]
    # Record all lexical leads, including nonmotor records; these are NOT authorized sources.
    leads=[x for x in new if legmatch(x,leg) and (needle in str(x['cell_type']) or needle in str(x['peripheral_target_type']))]
    full=[x for x in exact if x['root_id'] in global_by_root]
    present=[x for x in full if x['root_id'] in tier_by_root]
    rows.append({'actuator_index':i,'actuator':r['actuator'],'leg':leg,'muscle_target':target,
                 'manifest_status':mr['status'],'original_declared_pool_root_ids':[],
                 'actual_motor_named_leg_muscle_sources':[evidence(x) for x in exact],
                 'n_exact_v888_sources':len(exact),'n_exact_full_frozen_sources':len(full),'n_exact_tier_sources':len(present),
                 'lexical_alias_leads_not_innervation_authorization':[evidence(x) for x in leads],
                 'classification':'NO_ACTUAL_ANNOTATED_SOURCE_IN_INSPECTED_SNAPSHOTS' if not exact else 'EXACT_SOURCE_FOUND',
                 'safe_minimal_repair_authorized':False})
assert all(r['n_exact_v888_sources']==0 for r in rows)
# The named-target universe is retained, so other-leg evidence cannot silently become a source.
targets={r['muscle_target'] for r in missing}
universe=[evidence(x) for x in new if x['peripheral_target_type'] in targets]
assert hashes == {str(p):sha(p) for p in inputs}
result={'scope':'S168/P1-P3 independent offline root/muscle provenance; no engine/neural/optimization, no mapping/tracker edits',
        'input_sha256':hashes,'counts':{'full_frozen_neurons':len(roots),'tier_neurons':len(tier),'v888_annotations':len(new),'raw_csv_annotations':len(old),'unmapped_actuators':len(rows)},
        'root_handling':'Raw CSV/feather IDs stay decimal strings; frozen int64 IDs use exact int then str; never float',
        'tier_global_local_root_check':'PASS all 17099 rows; full dataset root uniqueness PASS',
        'candidates':rows,'actual_target_annotation_universe_across_all_legs':universe,
        'conclusion_zh':'16项原pool均空，并非已声明源被17099裁掉。全153962冻结数据与188508行v888原始注释中，未找到这些缺失项对应的motor+明确腿/侧+peripheral_target_type来源；LF/RF组合type别名有现存tier roots，但真实target只tergopleural，不能借别名补pleural。其余相同肌肉只在其他腿有注释，不能复制。未授权任何补映射；仅限已检查快照，不声称真实生物中没有该肌肉神经。',
        'motor_identity_vs_innervation':'motor/leg_motor_neuron confirms motor identity only, not target muscle. Alias-only/other-leg matches are not innervation evidence.',
        'execution':{'model_load':0,'forward':0,'physics_step':0,'neural_step':0,'optimization':0,
                     'prior_schema_inspection_failure':'venv_body pandas feather inspection exit1 missing pyarrow; switched to existing venv with pyarrow, no installation'},
        'input_hashes_unchanged_after_check':True,'script_sha256':sha(Path(__file__))}
# Output is new and deliberately separate from primary producer archive.
assert not OUT.exists(), f'refuse overwrite {OUT}'
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'candidates':len(rows),'exact_sources':sum(r['n_exact_v888_sources'] for r in rows),'alias_leads':sum(len(r['lexical_alias_leads_not_innervation_authorization']) for r in rows),'verified':True},ensure_ascii=False))
