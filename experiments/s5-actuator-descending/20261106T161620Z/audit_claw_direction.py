import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'claw_direction';O.mkdir(parents=True,exist_ok=True)
B=Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz');I=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz');L=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
bank=np.load(B,allow_pickle=False);ids=np.load(I,allow_pickle=False);lookup=json.loads(L.read_text());aud=json.loads((ROOT/'neural_tendon/SENSORY_AUDIT.json').read_text());cells=aud['channels']['claw_position']['ids'];fields=[k for k in bank.files if k.startswith('ann_')];rows=[]
for local in cells:
 g=int(ids['global_index'][local]);runtime=int(ids['root_ids'][local]);assert lookup['leg_channel_by_root'][str(runtime)]==['lf','claw_position']
 row=dict(local_index=local,global_index=g,runtime_root888=runtime,bank_root=int(bank['root_ids'][g]),annotations={k:str(bank[k][g]) for k in fields});rows.append(row)
assert all(r['annotations']['ann_class']=='chordotonal_organ_neuron' and r['annotations']['ann_flow']=='afferent' and r['annotations']['ann_function']=='position,proprioception' for r in rows)
report=dict(importance='核心必需',rows=rows,bank_root_equals_runtime_count=sum(r['runtime_root888']==r['bank_root'] for r in rows),mapping_basis='saved runtime global_index into bank annotation arrays; runtime888 identity separately checked against channel snapshot; do not join changed root IDs directly',available_fields=fields,direction_terms=[dict(local_index=r['local_index'],field=k,value=v) for r in rows for k,v in r['annotations'].items() if any(t in v.lower() for t in ['flex','exten','tuning','preferred'])],scope='audit available frozen annotation fields only; absence of labels does not mean biological cells lack tuning',hashes={str(k):hashlib.sha256(k.read_bytes()).hexdigest() for k in [Path(__file__),B,I,L]})
(O/'ANNOTATION_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(cells=len(rows),direction_terms=report['direction_terms'],bank_root_equals_runtime_count=report['bank_root_equals_runtime_count'],distinct_annotations={k:sorted(set(r['annotations'][k] for r in rows)) for k in fields})))
