"""Actual directed topology audit; no neural/physical simulation or added edges."""
import json,hashlib
from pathlib import Path
from collections import deque
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/descending')
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz')
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
F=R/'data/flywire/banc_connectome.npz'
MAP=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/POOL_TO_JOINT.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'PATH_AUDIT.json').exists()
 inputs=[H,F,MAP,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',Path(__file__)]
 hashes={str(p):sha(p) for p in inputs}
 dump(O/'PROTOCOL.json',dict(importance='核心必需',hashes=hashes,max_path_edges=4,matrix_order='post,pre',selection='all descending with directed paths to BOTH antagonist pools <=4 edges; rank max distance then sum distance then local index; topology only, no functional claim'))
 roots=np.load(H)['root_ids']; z=np.load(F,allow_pickle=True)
 lookup={int(v):i for i,v in enumerate(z['root_ids'])}; rows=np.array([lookup[int(v)] for v in roots]); assert len(rows)==12000
 classes=z['ann_super_class'][rows].astype(str)
 desc=np.flatnonzero(classes=='descending')
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr(); wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr()
 assert we.shape==wi.shape==(12000,12000)
 graph=(we+wi).tocsr(); graph.eliminate_zeros()
 pools=[p for p in json.loads(MAP.read_text())['muscle_to_joint'] if p['joint']=='joint_LFTibia_pitch']
 targets={str(sign):sorted(set(v for p in pools if p['sign']==sign for v in p['local_indices'])) for sign in [-1,1]}
 def backwards(target):
  dist=np.full(12000,-1,int); nxt=np.full(12000,-1,int); queue=deque(target)
  for v in target:dist[v]=0
  while queue:
   post=queue.popleft()
   if dist[post]>=4:continue
   for pre in graph.indices[graph.indptr[post]:graph.indptr[post+1]]:
    if dist[pre]<0:dist[pre]=dist[post]+1;nxt[pre]=post;queue.append(int(pre))
  return dist,nxt
 trees={k:backwards(v) for k,v in targets.items()}
 records=[]
 for v in desc:
  paths={}
  for k,(dist,nxt) in trees.items():
   nodes=[int(v)]; edges=[]
   if dist[v]>=0:
    while nxt[nodes[-1]]>=0:
     pre=nodes[-1];post=int(nxt[pre]); e=float(we[post,pre]);i=float(wi[post,pre]);assert e>0 or i>0
     edges.append(dict(pre=pre,post=post,We=e,Wi=i));nodes.append(post)
    assert nodes[-1] in targets[k]
   paths[k]=dict(distance=int(dist[v]),local_indices=nodes if dist[v]>=0 else [],edges=edges)
  records.append(dict(local_index=int(v),banc_888_id=str(int(roots[v])),paths=paths))
 candidates=[x for x in records if all(p['distance']>=0 for p in x['paths'].values())]
 candidates.sort(key=lambda x:(max(p['distance'] for p in x['paths'].values()),sum(p['distance'] for p in x['paths'].values()),x['local_index']))
 assert hashes=={str(p):sha(p) for p in inputs}
 result=dict(importance='核心必需',descending_count=len(desc),central_brain_intrinsic_count=int((classes=='central_brain_intrinsic').sum()),targets=targets,all_targets_motor=bool(all(classes[v]=='motor' for vs in targets.values() for v in vs)),edges=dict(We=we.nnz,Wi=wi.nnz),candidates_both_pools=len(candidates),ranked_candidate_indices=[x['local_index'] for x in candidates],records=records,inputs_unchanged=True,interpretation='topological reachability only; no net effect or walking function established; no descending current applied')
 dump(O/'PATH_AUDIT.json',result)
 print(json.dumps({k:result[k] for k in ['descending_count','central_brain_intrinsic_count','candidates_both_pools','all_targets_motor','edges']}))
if __name__=='__main__':main()
