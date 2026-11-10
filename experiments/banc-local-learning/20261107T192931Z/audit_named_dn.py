"""Read-only source-label candidates for future DN interface; not brain training."""
from pathlib import Path
import csv,json,re,hashlib
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/named-dn')
SOURCE=ROOT/'experiments/flybody-retarget/20261107T182940Z/neural-interface'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cache=ROOT/'data/flywire/banc_connectome.npz'
with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in ['root_ids','pre','post','syn','ann_super_class','ann_primary_type','ann_nt_verified','ann_class']}
with np.load(SOURCE/'LOCAL_MODULE_CANDIDATE.npz') as z:c={k:z[k] for k in ['root_ids','cache_indices','pre_local','post_local','synapse_counts','pure_verified_label_mask']}
with np.load(SOURCE/'GROUP_CACHE_INDICES.npz') as z:dn=z['descending_boundary'];vnc=z['vnc_intrinsic']
lookup={int(x):i for i,x in enumerate(c['cache_indices'])};names=['DNa01','DNa02','DNb02','DNb05','DNb06','DNg13','DNg34','DNp09','MDN'];rx=re.compile(r'(?<![A-Za-z0-9])('+ '|'.join(names) +r')(?![A-Za-z0-9])')
fullout=np.bincount(a['pre'],minlength=len(a['root_ids']));vout=np.bincount(a['pre'][np.isin(a['post'],vnc)],minlength=len(a['root_ids']));localout=np.bincount(c['pre_local'],minlength=20000);strictout=np.bincount(c['pre_local'][c['pure_verified_label_mask']],minlength=20000)
rows=[]
for i,t in enumerate(a['ann_primary_type']):
    hit=rx.findall(str(t))
    if not hit:continue
    j=lookup.get(i)
    rows.append(dict(root_id_string=str(a['root_ids'][i]),cache_index=i,local_index='' if j is None else j,matched_names='|'.join(sorted(set(hit))),source_type=str(t),source_superclass=str(a['ann_super_class'][i]),source_class=str(a['ann_class'][i]),source_nt=str(a['ann_nt_verified'][i]),is_descending_boundary=bool(i in dn),full_outgoing_edges=int(fullout[i]),full_outgoing_to_VNC_edges=int(vout[i]),local_outgoing_edges=0 if j is None else int(localout[j]),strict_local_outgoing_edges=0 if j is None else int(strictout[j])))
OUT.mkdir(parents=True,exist_ok=False)
with (OUT/'CANDIDATES.csv').open('w') as f:
    fields=list(rows[0]) if rows else ['root_id_string','matched_names'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
result=dict(status='SOURCE_LABEL_CANDIDATES_ONLY_NOT_FUNCTIONAL_OR_CROSS_DATASET_ID_CONFIRMATION',counts={name:sum(name in r['matched_names'].split('|') for r in rows) for name in names},matched_cells=len(rows),strict_outgoing_cells=sum(r['strict_local_outgoing_edges']>0 for r in rows),all_candidates_in_DN=all(r['is_descending_boundary'] for r in rows),source_sha256={str(p):sha(p) for p in [cache,SOURCE/'LOCAL_MODULE_CANDIDATE.npz',Path(__file__)]},sources=['https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/','https://www.nature.com/articles/s41586-024-07523-9'],limits='Cache primary labels only; match does not establish cross-dataset root identity, laterality, receptors, physiology or DN function in this model. No injection/training/physics. DNa02/DNg13 are not routine forward speed commands per published Cell text. Nature fulltext redirect blocked; initial search summary only for DNp09/MDN.',neural_steps=0,physical_steps=0)
with (OUT/'RESULT.json').open('w') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
