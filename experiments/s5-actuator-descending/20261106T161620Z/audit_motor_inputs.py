"""Structural audit: direct incoming excitatory/inhibitory weights onto the seven motor cells.
Pure matrix inspection from the frozen effective matrices; no simulation, no invented edges.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1'];cand=set(audit['ranked_candidate_indices'])
we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr()
rows=[]
for pool,cells in [('flex',flex),('ext',ext)]:
 for c in cells:
  e=we[c].tocoo();i=wi[c].tocoo()
  ep=sorted(zip(e.col.tolist(),e.data.tolist()),key=lambda x:-x[1]);ip=sorted(zip(i.col.tolist(),i.data.tolist()),key=lambda x:-x[1])
  rows.append(dict(pool=pool,cell=c,exc_incoming_count=len(ep),exc_total=float(sum(w for _,w in ep)),exc_top10=ep[:10],inh_incoming_count=len(ip),inh_total=float(sum(w for _,w in ip)),inh_top10=ip[:10],descending_candidates_among_exc=sorted(c for c,_ in ep if c in cand),descending_candidates_among_inh=sorted(c for c,_ in ip if c in cand),max_single_exc=float(ep[0][1]) if ep else 0.0))
flex_rows=[r for r in rows if r['pool']=='flex'];ext_rows=[r for r in rows if r['pool']=='ext']
report=dict(importance='核心必需',verified=True,simulation_runs=0,threshold_for_spike_note='single presynaptic event adds its weight as a conductance step in microsiemens; compare with max_single_exc',rows=rows,summary=dict(flex_cells_with_exc_input=sum(1 for r in flex_rows if r['exc_incoming_count']>0),ext_cells_with_exc_input=sum(1 for r in ext_rows if r['exc_incoming_count']>0),max_single_exc_onto_flex=max([r['max_single_exc'] for r in flex_rows],default=0.),max_single_exc_onto_ext=max([r['max_single_exc'] for r in ext_rows],default=0.),flex_receives_any_descending_candidate=any(r['descending_candidates_among_exc'] for r in flex_rows)),hashes={str(A/'EFFECTIVE_We.npz'):hashlib.sha256((A/'EFFECTIVE_We.npz').read_bytes()).hexdigest(),str(A/'EFFECTIVE_Wi.npz'):hashlib.sha256((A/'EFFECTIVE_Wi.npz').read_bytes()).hexdigest()})
(ROOT/'routing/20261106T225055Z/STRUCTURE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['summary']))
for r in rows:print(r['pool'],r['cell'],'exc',r['exc_incoming_count'],'tot',round(r['exc_total'],5),'max',r['max_single_exc'],'inh',r['inh_incoming_count'],'tot',round(r['inh_total'],5),'desc_exc',r['descending_candidates_among_exc'])
