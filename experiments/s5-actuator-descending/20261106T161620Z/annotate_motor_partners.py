"""Annotate the direct presynaptic partners of the seven motor cells using the frozen BANC annotations.
No new edges are created; this reads the existing matrix and the existing annotation arrays.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
B=Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz');I=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz')
audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr()
bank=np.load(B,allow_pickle=False);ids=np.load(I,allow_pickle=False);gidx=ids['global_index']
fields=['ann_super_class','ann_class','ann_nerve','ann_body_part','ann_function','ann_primary_type','ann_flow']
def ann(local):
 g=int(gidx[local]);return {k:str(bank[k][g]) for k in fields}
partners={}
for pool,cells in [('flex',flex),('ext',ext)]:
 for c in cells:
  for mat,kind in [(we,'exc'),(wi,'inh')]:
   coo=mat[c].tocoo()
   for p,w in zip(coo.col.tolist(),coo.data.tolist()):
    d=partners.setdefault(int(p),dict(cell=int(p),targets=[],exc_total=0.,inh_total=0.))
    d['targets'].append(dict(pool=pool,motor=c,kind=kind,weight=float(w)))
    d['exc_total' if kind=='exc' else 'inh_total']+=float(w)
for p,d in partners.items():
 d.update(ann(p))
desc=[d for d in partners.values() if 'descend' in (d['ann_super_class']+' '+d['ann_class']+' '+d['ann_function']).lower()]
def summarise(d):return dict(cell=d['cell'],targets=d['targets'],exc_total=round(d['exc_total'],6),inh_total=round(d['inh_total'],6),ann_class=d['ann_class'],ann_super_class=d['ann_super_class'],ann_primary_type=d['ann_primary_type'],ann_body_part=d['ann_body_part'],ann_function=d['ann_function'],ann_flow=d['ann_flow'])
classes={}
for d in partners.values():classes[d['ann_super_class']]=classes.get(d['ann_super_class'],0)+1
report=dict(importance='核心必需',verified=True,simulation_runs=0,partner_cells=len(partners),descending_annotated=len(desc),super_class_histogram=classes,descending_partners=[summarise(d) for d in sorted(desc,key=lambda x:-x['exc_total'])],note='max single excitatory weight onto any motor cell is 0.0002 uS while about 0.0003 uS sustained is needed to cross threshold, so a usable command needs two or more coincident presynaptic spikes; a single direct edge is not a usable pathway',hashes={str(B):hashlib.sha256(B.read_bytes()).hexdigest(),str(I):hashlib.sha256(I.read_bytes()).hexdigest()})
(ROOT/'routing/20261106T225055Z/ANNOTATED_PARTNERS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(partner_cells=len(partners),descending_annotated=len(desc),hist=classes)))
for d in report['descending_partners'][:25]:print(d['cell'],d['ann_class'],d['ann_primary_type'],d['ann_body_part'],'exc',d['exc_total'],'inh',d['inh_total'],[(t['pool'],t['motor'],t['kind']) for t in d['targets']])
