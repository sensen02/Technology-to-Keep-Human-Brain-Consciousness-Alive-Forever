"""Independent direct-cache graph fixture: no tier builder or production graph functions."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.sparse import csr_matrix,save_npz
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=Path('/home/sensen/Desktop/engine修复验收/s5-descending-graph-verification/20261028T073000Z')
C=P/'data/flywire/banc_connectome.npz'
T=Path('/home/sensen/Desktop/engine修复验收/s5-neural-tier-coverage/20261028T070000Z/TIER_IDENTITIES.npz')
IDS=np.array([720575941433155799,720575941566493282],dtype=np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def fingerprint(m):
 m=m.copy();m.sum_duplicates();m.sort_indices()
 h=hashlib.sha256()
 for x in [np.array(m.shape,dtype=np.int64),m.indptr.astype(np.int64),m.indices.astype(np.int64),m.data.astype(np.float64)]:h.update(x.tobytes())
 return h.hexdigest()
def main():
 start=time.time();O.mkdir(parents=True,exist_ok=False)
 sources={str(x):sha(x) for x in [C,T,Path(__file__),P/'engine/neck_cut_data.py',P/'engine/embodied/adapters.py']}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=sources,method='Independent manual cache filtering; no production build_edges/build_neural_tier or augmentation module imported',root_order='Exactly070 ordered12000, append DNp09 root433155799 then566493282',presynaptic_sign='pure acetylcholine excitatory; pure gaba/glutamate inhibitory; all other labels excluded',edge_weight='min(float64(syn),8)*0.0005, duplicates summed; row postsynaptic/column presynaptic',scope='Graph provenance/identity fixture only; not proof of neural behavior; helper runtime0.2 multiplier is separate'))
 with np.load(C) as z:a={k:z[k] for k in ['root_ids','pre','post','syn','ann_nt_verified','ann_primary_type']}
 with np.load(T) as z:old=z['root_ids'].copy();old_global=z['global_index'].copy()
 assert len(old)==12000 and len(set(old.tolist()))==12000
 np.testing.assert_array_equal(a['root_ids'][old_global],old)
 assert not np.isin(IDS,old).any()
 index={int(r):i for i,r in enumerate(a['root_ids'])};extra=np.array([index[int(r)] for r in IDS]);roots=np.r_[old,IDS];gi=np.r_[old_global,extra]
 assert all(a['ann_primary_type'][i]=='DNp09' for i in extra)
 assert all(a['ann_nt_verified'][i]=='acetylcholine' for i in extra)
 loc=np.full(len(a['root_ids']),-1,dtype=np.int64);loc[gi]=np.arange(len(gi));pre=loc[a['pre']];post=loc[a['post']];inside=(pre>=0)&(post>=0);label=a['ann_nt_verified'][a['pre']]
 matrices={};counts={}
 for name,labels in [('We',['acetylcholine']),('Wi',['gaba','glutamate'])]:
  mask=inside&np.isin(label,labels);w=np.minimum(a['syn'][mask].astype(np.float64),8.)*.0005
  mat=csr_matrix((w,(post[mask],pre[mask])),shape=(12002,12002));mat.sum_duplicates();mat.sort_indices();core=mat[:12000,:12000].tocsr();oldmask=mask&(pre<12000)&(post<12000);oldmat=csr_matrix((np.minimum(a['syn'][oldmask].astype(np.float64),8.)*.0005,(post[oldmask],pre[oldmask])),shape=(12000,12000));oldmat.sum_duplicates();oldmat.sort_indices();assert (core!=oldmat).nnz==0
  assert core.nnz==({'We':143175,'Wi':148811}[name])
  matrices[name]=mat;save_npz(O/('EXPECTED_'+name+'.npz'),mat)
  counts[name]=dict(old_nnz=int(core.nnz),new_nnz=int(mat.nnz),added_nnz=int(mat.nnz-core.nnz),canonical_sha256=fingerprint(mat),old_canonical_sha256=fingerprint(core),signed_source_cache_rows=int(mask.sum()),old_submatrix_exact=True)
 np.savez_compressed(O/'EXPECTED_TIER_IDENTITIES.npz',root_ids=roots,global_index=gi,descending_command_local=np.array([12000,12001]))
 result=dict(passed=True,counts=counts,old_neurons=12000,new_neurons=12002,appended=[dict(root_id=int(r),local_index=12000+j,global_index=int(extra[j]),type=str(a['ann_primary_type'][extra[j]]),verified_transmitter=str(a['ann_nt_verified'][extra[j]])) for j,r in enumerate(IDS)],source_hashes=sources,source_unchanged=all(sha(k)==v for k,v in sources.items()),runtime_s=time.time()-start,tier_builds=0,network_instances=0,neural_steps=0,model_loads=0,forward=0,body_steps=0,cache_loads_this_script=1,earlier_metadata_preflight_cache_loads=1)
 dump(O/'EXPECTED_GRAPH.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
