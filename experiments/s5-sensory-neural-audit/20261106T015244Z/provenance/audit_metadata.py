"""Read-only metadata audit; no project imports, neural/physics stepping or rebuilding."""
import json, hashlib, gzip, resource
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s5-sensory-neural-audit/20261106T015244Z/provenance'
assert O.resolve().is_relative_to((D/'s5-sensory-neural-audit/20261106T015244Z').resolve())
O.mkdir(parents=True,exist_ok=True)
def dump(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
B=D/'s5-banc888-physical-handover/20261101T220018Z'
A=D/'s5-route-b-ablation/20261108T060000Z'
RE=D/'s5-bristle-edge-rebuild/20261104T050000Z'
C=D/'s3-banc888-runtime-admission/20261101T220018Z'
files=[R/p for p in ['AGENTS.md','GOAL_ROADMAP.zh-CN.md','engine/neural_cond.py','engine/neck_cut_data.py','engine/embodied/adapters.py','engine/embodied/sensory_channels.py','build_banc_cns.py','flywire_download.py','data/flywire/banc_connectome.npz','data/flywire/banc_connections_princeton.csv.gz','data/flywire/banc_neurons.csv.gz','data/banc888/banc_888_meta.feather','experiments/s5-neural-translator/20261105T200021Z/runtime.py','experiments/s5-neural-translator/20261105T200021Z/translator.py','tools/export_banc888_interface_candidates.py','tools/prepare_banc888_runtime_admission.py']]
files += [B/'IDENTITIES.npz',B/'RUN/MAPPING.json',C/'VERSION_CONTRACT.json',C/'SENSORY_LOOKUP_SNAPSHOT888.json',C/'MN_POOLS_SNAPSHOT888.json',RE/'EXECUTED_rebuild_edges_with_bristle_signfill.py',RE/'REBUILD.json',A/'EXECUTED_build_ablation_networks.py',A/'FROZEN_VERDICT.json']
files += [A/'ARMS/A'/f'ACTUAL_{k}.npz' for k in ['We','Wi']]+[RE/f'REBUILT_{k}.npz' for k in ['We','Wi']]
before={str(p):sha(p) for p in files}
z=np.load(R/'data/flywire/banc_connectome.npz',allow_pickle=False)
roots=z['root_ids']; selected=np.load(B/'IDENTITIES.npz')['root_ids']; gi=np.searchsorted(roots,selected)
assert np.array_equal(roots[gi],selected)
res={'calls':{'neural_step':0,'physics_step':0,'model_import':0},'dataset_cells':len(roots),'selected_cells':len(selected),'unique_root_match':bool(len(np.unique(selected))==12000),'coverage':{}}
for col in ['ann_super_class','ann_class','ann_nt_verified']:
 v=z[col];res['coverage'][col]={'dataset':dict(Counter(v.tolist())),'selected':dict(Counter(v[gi].tolist()))}
pre=z['pre'];post=z['post'];loc=np.full(len(roots),-1,dtype=np.int32);loc[gi]=np.arange(12000)
inside=(loc[pre]>=0)&(loc[post]>=0)
res['dataset_pairs']=len(pre);res['within_selected_pairs']=int(inside.sum());res['outside_selected_pairs']=int((~inside).sum())
pp=pre[inside];qq=post[inside];ss=z['syn'][inside];ver=z['ann_nt_verified'][pp]
res['inside_presynaptic_verified_labels']=dict(Counter(ver.tolist()))
res['matrices']={}
for k in ['We','Wi']:
 m=sparse.load_npz(A/'ARMS/A'/f'ACTUAL_{k}.npz');old=sparse.load_npz(RE/f'REBUILT_{k}.npz')
 assert (m!=old).nnz==0
 co=m.tocoo();keys=loc[pp].astype(np.int64)*12000+loc[qq];order=np.argsort(keys);mk=co.col.astype(np.int64)*12000+co.row;ii=np.searchsorted(keys[order],mk)
 assert np.array_equal(keys[order[ii]],mk)
 expected=np.minimum(ss[order[ii]].astype(float),8)*5e-4*.2
 assert np.array_equal(co.data,expected)
 res['matrices'][k]={'nnz':m.nnz,'source_pairs_all_exist':True,'weights_exact_min_syn8_times_5e_4_times_point2':True,'equal_rebuilt':True,'source_verified_labels':dict(Counter(ver[order[ii]].tolist())),'zero_in_degree':int((np.diff(m.indptr)==0).sum()),'zero_out_degree':int((np.bincount(co.col,minlength=12000)==0).sum())}
 del m,old,co,keys,order,mk,ii,expected
res['excluded_inside_pairs']=res['within_selected_pairs']-sum(v['nnz'] for v in res['matrices'].values())
res['raw_csv_headers']={}
for name in ['banc_connections_princeton.csv.gz','banc_neurons.csv.gz']:
 with gzip.open(R/'data/flywire'/name,'rt') as f:res['raw_csv_headers'][name]=f.readline().strip()
res['peak_rss_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
assert before=={str(p):sha(p) for p in files}
res['all_audited_sources_unchanged']=True
dump('GRAPH_METADATA.json',res)
dump('SOURCE_SHA256.json',{'importance':'核心必需','scope':'Current source byte identities; not proof of biological validity','files':[{'path':str(p),'resolved_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':before[str(p)]} for p in files]})
print(json.dumps({k:v for k,v in res.items() if k not in ['coverage','inside_presynaptic_verified_labels']},ensure_ascii=False))
