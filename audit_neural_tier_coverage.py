"""Offline reconstruction of the exact default embodied tier, identities and directed paths."""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from engine.embodied.adapters import build_neural_tier,TierConfig
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-tier-coverage/20261028T070000Z');S=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
def hist(x):
 u,n=np.unique(x,return_counts=True);return dict(zip(u.tolist(),n.tolist()))
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'COVERAGE.json').exists();start=time.time();src=[Path(__file__),P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'data/flywire/banc_connectome.npz',S/'MAPPING.json',S/'NEURAL.npz'];hashes={str(x):sha(x) for x in src};dump(O/('RETRY_PREREGISTRATION.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Offline default TierConfig(background_mean_nA=0,background_sd_nA=0), exactly runtime build. No neural network instance/step or physics. Counts/path reachability, not a behavior or cell-identity claim.',source_hashes=hashes))
 b=build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0));a=b['data'];gi=b['global_index'];root=b['tier']['root_ids'];n=b['n'];we=b['edges']['We'].tocsr();wi=b['edges']['Wi'].tocsr();print('BUILT',n,'We',we.nnz,'Wi',wi.nnz,flush=True)
 # Conductance matrix is postsynaptic-row, presynaptic-column: transpose for outgoing traversal.
 from engine.neural_cond import ConductanceNetwork
 import inspect
 code=inspect.getsource(ConductanceNetwork.step);assert '@' in code or 'dot' in code
 adj=(we+wi).T.tocsr();adj.data[:]=1
 sensory=np.unique(np.concatenate([np.asarray(v,dtype=int) for v in b['sensory'].values()]));mapping=json.loads((S/'MAPPING.json').read_text());motors=np.unique([int(i) for r in mapping['rows'] for i in r['local_indices']]);assert all(set(int(root[i]) for i in r['local_indices'])<=set(r['pool_root_ids']) for r in mapping['rows'])
 def multibfs(seeds):
  dist=np.full(n,-1,dtype=int);dist[seeds]=0;front=np.asarray(seeds,dtype=int)
  for depth in range(1,n+1):
   nxt=np.unique(adj[front].indices);nxt=nxt[dist[nxt]<0]
   if not len(nxt):break
   dist[nxt]=depth;front=nxt
  return dist
 sd=multibfs(sensory);desc=np.asarray(b['descending_local'],dtype=int);dd=multibfs(desc);spikes=np.load(S/'NEURAL.npz')['spike_counts'];assert spikes.shape==(250,n)
 meta={k:np.asarray(a[k])[gi] for k in ['ann_flow','ann_super_class','ann_class','ann_nerve','ann_body_part','ann_function','ann_primary_type','ann_nt_verified']}
 # Identify literal names only; never infer DN identity from broad descending labels.
 mask=np.zeros(n,dtype=bool)
 for k in ['ann_primary_type','ann_class','ann_function']:
  mask|=np.array([any(tag in str(v).lower() for tag in ['dna02','dnp09','mdn','moonwalker']) for v in meta[k]])
 named=np.flatnonzero(mask);dr=[]
 for i in desc:
  dr.append(dict(local_index=int(i),root_id=int(root[i]),annotations={k:str(v[i]) for k,v in meta.items()},spikes_first15ms=int(spikes[:15,i].sum()),spikes_total=int(spikes[:,i].sum()),sensory_distance=int(sd[i]),motor_targets_reachable=int(np.sum(multibfs(np.array([i]))[motors]>=0))))
 rows=[]
 for r in mapping['rows']:
  ids=np.array(r['local_indices'],dtype=int);rows.append(dict(actuator_index=r['actuator_index'],actuator_name=r['actuator'],local_indices=ids,root_ids=[int(root[i]) for i in ids],sensory_distances=sd[ids],descending_distances=dd[ids],spikes_first15ms=int(spikes[:15,ids].sum()),spikes_total=int(spikes[:,ids].sum())))
 coverage=dict(neurons=n,dataset_neurons=len(a['root_ids']),edge_nonzero_excitatory=we.nnz,edge_nonzero_inhibitory=wi.nnz,sensory_cells=len(sensory),motor_unique_cells=len(motors),mapped_actuators=sum(bool(r['local_indices']) for r in mapping['rows']),descending_cells=len(desc),descending_spikes_first15ms=int(spikes[:15,desc].sum()),descending_spikes_total=int(spikes[:,desc].sum()),sensory_reachable_motors=int(np.sum(sd[motors]>=0)),descending_reachable_motors=int(np.sum(dd[motors]>=0)),histograms={k:hist(v) for k,v in meta.items() if k not in ['ann_nerve','ann_primary_type']},named_literature_label_matches=[dict(local_index=int(i),root_id=int(root[i]),annotations={k:str(v[i]) for k,v in meta.items()}) for i in named],full_dataset_named_matches={k:[dict(root_id=int(a['root_ids'][i]),label=str(a[k][i]),in_tier=bool(i in set(gi))) for i in np.flatnonzero(np.array([any(t in str(v).lower() for t in ['dna02','dnp09','mdn','moonwalker']) for v in a[k]]))] for k in ['ann_primary_type','ann_class','ann_function']},path_scope='Unsigned reachability of retained positive conductance edges; no proof of effective excitation or behavior')
 dump(O/'COVERAGE.json',coverage);dump(O/'DESCENDING_IDENTITIES.json',dr);dump(O/'MOTOR_PATHS.json',rows);np.savez_compressed(O/'TIER_IDENTITIES.npz',root_ids=root,global_index=gi,sensory_local=sensory,motor_local=motors,descending_local=desc,sensory_distance=sd,descending_distance=dd);dump(O/'BUILD_COVERAGE.json',b['coverage']);dump(O/'EXECUTION.json',dict(tier_builds=1,model_load=0,forward=0,body_steps=0,neural_steps=0,network_instances=0,exit_code=0,runtime_s=time.time()-start,source_unchanged=all(sha(Path(k))==v for k,v in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(coverage,indent=2),flush=True)
if __name__=='__main__':main()
