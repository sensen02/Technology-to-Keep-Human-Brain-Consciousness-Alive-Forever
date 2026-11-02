"""Offline explicit prediction/receptor-hypothesis graph; no runtime registration.
Only existing raw edges from a priori driven zero-output cells, no synapse invention.
"""
from pathlib import Path
import sys,json,hashlib,csv,collections
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1];sys.path.append(str(R/'venv/lib/python3.12/site-packages'))
import pandas as pd
P=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-nt-prediction-provenance/20261101T230101Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
O=Path('/home/sensen/Desktop/engine修复验收/s3-sourced-nt-candidate-graph/20261102T004326Z')
SCORE=.9;COUNT=20

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def reach(matrix,seeds,targets):
 adjacency=matrix.tocsc();seen=np.zeros(matrix.shape[0],bool);parent=np.full(matrix.shape[0],-1,dtype=int);dist=np.full(matrix.shape[0],-1,dtype=int)
 q=collections.deque(map(int,seeds));seen[list(seeds)]=True;dist[list(seeds)]=0
 while q:
  node=q.popleft()
  for dest in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]:
   if not seen[dest]:seen[dest]=True;parent[dest]=node;dist[dest]=dist[node]+1;q.append(int(dest))
 paths={}
 for target in targets:
  if not seen[target]:continue
  path=[int(target)]
  while parent[path[-1]]>=0:path.append(int(parent[path[-1]]))
  paths[str(int(target))]=path[::-1]
 return dict(reachable_nodes=int(seen.sum()),reachable_readout_MN=len(paths),shortest_path_lengths={key:int(dist[int(key)]) for key in paths},paths=paths)
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'RESULT.json').exists(), 'No completed graph may be overwritten'
 inputs=[Path(__file__),P/'OFFICIAL_banc_888_neurotransmitter_prediction_v2.csv',P/'DRIVEN_RECIPIENTS.csv',P/'INDEPENDENT_VERIFICATION.json',P/'OFFICIAL_SCHEMA.md',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',B/'IDENTITIES.npz',B/'RUN/MAPPING.json',R/'data/banc888/banc_888_meta.feather',R/'data/flywire/banc_connectome.npz']
 hashes={str(p):sha(p) for p in inputs}
 dump(O/('PREREGISTRATION_CACHED_ARRAYS.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(task='S3 explicitly assumed offline graph for motion-training readiness, not deployment or behavior',source_sha256=hashes,rules=dict(score_min=SCORE,count_min=COUNT,source_population='Previously audited nonzero-driven, zero modeled outgoing, non-readout nodes only; verified empty in both cache and current meta',raw_edges='Existing raw rows with both endpoints in original12000tier only',receptors='Explicit illustrative pure ACh excitatory, GABA/glutamate inhibitory, rest excluded; no claim receptor measurement',weight='Per raw row min(syn,8)*.0005*.2, then CSR duplicate sum; same as archived baseline',preserved='All baseline roots/weight values/support, known labels, unresolved muscles and interface; only previously excluded outgoing rows added'),comparisons='Baseline vs prediction-only added graph vs exact ablation of added graph; boolean reachability and raw witnesses only, no model/neuronal integration',runtime_allowed=False,limits='Classifier score is not calibrated receptor sign certainty. Selection does not prove optimality. Reachability ignores E/I dynamics, weights and body feasibility. No default graph or trust anchors changed.'))
 d=np.load(R/'data/flywire/banc_connectome.npz');ids=np.load(B/'IDENTITIES.npz');roots=ids['root_ids'];gidx=ids['global_index'];n=len(roots);assert n==12000 and np.array_equal(d['root_ids'][gidx],roots)
 frame=pd.read_feather(R/'data/banc888/banc_888_meta.feather');frame.index=frame['root_888'].astype('int64');meta=frame.loc[roots]
 verified=meta['neurotransmitter_verified'].fillna('').astype(str).to_numpy();cached=d['ann_nt_verified'][gidx].astype(str)
 We=sparse.load_npz(B/'ACTUAL_We.npz').tocsr();Wi=sparse.load_npz(B/'ACTUAL_Wi.npz').tocsr();out=np.asarray(We.getnnz(axis=0)+Wi.getnnz(axis=0)).ravel()
 recipients=list(csv.DictReader((P/'DRIVEN_RECIPIENTS.csv').open()));selected=[];labels={};provenance=[]
 for row in recipients:
  i=int(row['tier_index']);label=row['official_predicted'];score=float(row['official_score']) if row['official_score'] else -1;count=int(row['official_synapse_count'])
  if row['modeled_zero_output']=='1' and cached[i]=='' and verified[i]=='' and label in ['acetylcholine','gaba','glutamate'] and score>=SCORE and count>=COUNT:
   assert out[i]==0 and int(roots[i])==int(row['root_888']) and row['motor_readout']=='0'
   selected.append(i);labels[i]=label;provenance.append(dict(local_index=i,root_888=int(roots[i]),label=label,score=score,count=count))
 assert len(selected)==963
 synapses=d['syn'];rawpre=d['pre'];rawpost=d['post']
 local=np.full(len(d['root_ids']),-1,dtype=np.int64);local[gidx]=np.arange(n);prelocal=local[rawpre];postlocal=local[rawpost]
 selectmask=np.zeros(n,bool);selectmask[selected]=True;inside=(prelocal>=0)&(postlocal>=0);chosen=inside.copy();chosen[inside]=selectmask[prelocal[inside]]
 rows=np.flatnonzero(chosen);assert len(rows)==17264
 ex=[];inh=[]
 with (O/'ADDED_RAW_ROWS.csv').open('w',newline='') as f:
  wr=csv.writer(f);wr.writerow(['raw_row','pre_local','post_local','pre_root888','post_root888','synapses','predicted_NT','receptor_assumption','conductance_increment_uS'])
  for raw in rows:
   i=int(prelocal[raw]);j=int(postlocal[raw]);nt=labels[i];sign='EXC' if nt=='acetylcholine' else 'INH';syn=float(synapses[raw]);weight=min(syn,8)*.0005*.2
   assert syn>0 and weight>0
   wr.writerow([int(raw),i,j,int(roots[i]),int(roots[j]),syn,nt,sign,weight]);(ex if sign=='EXC' else inh).append((j,i,weight))
 def csr(entries):
  if not entries:return sparse.csr_matrix((n,n))
  post,pre,value=zip(*entries);a=sparse.coo_matrix((value,(post,pre)),shape=(n,n)).tocsr();a.sum_duplicates();a.sort_indices();return a
 Eadd=csr(ex);Iadd=csr(inh);assert We.multiply(Eadd).nnz==0 and Wi.multiply(Iadd).nnz==0
 Ec=(We+Eadd).tocsr();Ic=(Wi+Iadd).tocsr()
 assert ((Ec-Eadd)-We).nnz==0 and ((Ic-Iadd)-Wi).nnz==0
 for name,a in [('ADDED_We',Eadd),('ADDED_Wi',Iadd),('CANDIDATE_We',Ec),('CANDIDATE_Wi',Ic)]:sparse.save_npz(O/(name+'.npz'),a)
 np.savez_compressed(O/'IDENTITIES_AND_SELECTED.npz',root_ids=roots,global_index=gidx,selected_local=np.array(selected,dtype=np.int64))
 dump(O/'SELECTED_SOURCE_PROVENANCE.json',provenance)
 mapping=json.loads((B/'RUN/MAPPING.json').read_text());motor=sorted({i for row in mapping['rows'] for i in row['local_indices']});assert len(motor)==204
 base=(We+Wi).astype(bool);candidate=(Ec+Ic).astype(bool)
 baseline_reach=reach(base,selected,motor);candidate_reach=reach(candidate,selected,motor)
 counts=dict(selected_nodes=len(selected),selected_NT=dict(collections.Counter(labels.values())),raw_rows=len(rows),raw_synapse_sum=float(synapses[rows].sum()),added_exc_unique=Eadd.nnz,added_inh_unique=Iadd.nnz,base_exc_unique=We.nnz,base_inh_unique=Wi.nnz,candidate_exc_unique=Ec.nnz,candidate_inh_unique=Ic.nnz)
 dump(O/'BOOLEAN_REACHABILITY.json',dict(baseline=baseline_reach,candidate=candidate_reach,cut_added_edges_restores_baseline_exact=True,scope='Collective reachability from963selected roots; not each source reaches each MN. Path witnesses are local IDs with root lookup, not cell-function claims.'))
 dump(O/'RESULT.json',dict(status='OFFLINE_PREDICTION_RECEPTOR_HYPOTHESIS',runtime_allowed=False,source_sha256=hashes,counts=counts,baseline_reachable_readout=baseline_reach['reachable_readout_MN'],candidate_reachable_readout=candidate_reach['reachable_readout_MN'],all_existing_weights_untouched=True,known_verified_labels_overridden=0,source_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),calls=dict(body_models=0,tier_build=0,network=0,neural_step=0,body_step=0,training=0)))
 assert all(sha(Path(p))==h for p,h in hashes.items());print(json.dumps(counts,indent=2));print('MN topology',baseline_reach['reachable_readout_MN'],candidate_reach['reachable_readout_MN'])
if __name__=='__main__':main()
