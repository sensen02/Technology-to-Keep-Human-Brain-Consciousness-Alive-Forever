"""Read-only S5 anatomical audit; no engine import, construction or simulation."""
from pathlib import Path
import json, hashlib, csv, gzip
from collections import Counter, deque
import numpy as np
import pyarrow.feather as feather
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s5-actuator-descending/20261106T161620Z/descending'
SRC=Path(__file__).resolve().parent
M=R/'data/banc888/banc_888_meta.feather'
ID=D/'s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz'
AB=D/'s5-route-b-ablation/20261108T060000Z/ARMS/A'
TI=D/'s5-model-correction/20261106T063313Z/sensory/CANDIDATES/TRANSLATOR_IDENTITIES.json'
POOL=D/'s5-joint-torque-plant/20261105T060730Z/POOL_TO_JOINT.json'
CON=R/'data/flywire/banc_connectome.npz'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def rows(p,rs,fields=None):
 rs=list(rs)
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 O.mkdir(parents=True,exist_ok=True)
 sources=[M,ID,TI,POOL,CON,AB/'ACTUAL_We.npz',AB/'ACTUAL_Wi.npz',R/'engine/neural_cond.py',R/'experiments/s5-model-correction/20261106T063313Z/gain_sweep/common.py',R/'experiments/s5-model-correction/20261106T063313Z/gain_sweep/offline.py',R/'experiments/s5-model-correction/20261106T063313Z/numerics/run.py',R/'experiments/s5-neural-translator/20261105T200021Z/runtime.py',R/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py']
 hashes={str(p):sha(p) for p in sources};dump(O/'INPUT_HASHES.json',hashes)
 ids=np.load(ID)['root_ids'].astype(str);assert len(ids)==len(set(ids))==12000
 meta=feather.read_table(M).to_pandas().fillna('');assert meta.banc_888_id.is_unique
 meta['annotation_row']=np.arange(len(meta));a=meta.set_index('banc_888_id',drop=False).loc[ids].copy();a.insert(0,'local_index',np.arange(12000))
 cells=json.loads(TI.read_text())['cells'];assert len(cells)==len(set(cells))==204
 a['readout204']=np.isin(np.arange(12000),cells);a.to_csv(O/'ACTIVE_12000_ANNOTATIONS.csv',index=False)
 desc=np.flatnonzero(a.super_class.to_numpy()=='descending');brain=np.flatnonzero(a.super_class.to_numpy()=='central_brain_intrinsic');motor=np.flatnonzero(a.super_class.to_numpy()=='motor');leg=np.flatnonzero(a.cell_function.to_numpy()=='leg_motor');targets=sorted(set(leg)|set(cells));T=set(targets)
 assert len(desc)==194
 E,I=[sparse.load_npz(AB/f'ACTUAL_{k}.npz').tocsr() for k in ['We','Wi']]
 assert E.shape==I.shape==(12000,12000) and E.nnz==180604 and I.nnz==148811
 assert np.all(E.data>0) and np.all(I.data>0)
 A=(E+I).tocsr();A.data[:]=1;A=A.astype(np.int64);C=A.tocsc()
 ec,ic=E.tocsc(),I.tocsc();adj=[];edge={}
 for u in range(12000):
  rr=[]
  for mat,sign in [(ec,'E'),(ic,'I')]:
   for k in range(mat.indptr[u],mat.indptr[u+1]):
    v=int(mat.indices[k]);w=float(mat.data[k]);rr.append((v,sign,w));edge[u,v,sign]=w
  adj.append(sorted(rr))
 indeg=np.diff(A.indptr);outdeg=np.diff(C.indptr)
 # A[post,pre] times presynaptic indicator; all signed simple paths of lengths 1..3.
 reverse_dist=np.full(12000,999,dtype=int);reverse_dist[targets]=0;q=deque(targets)
 while q:
  v=q.popleft()
  for u in A.indices[A.indptr[v]:A.indptr[v+1]]:
   if reverse_dist[u]>reverse_dist[v]+1:reverse_dist[u]=reverse_dist[v]+1;q.append(int(u))
 path_counts=Counter();pair_counts=Counter();direct=[];npaths=0
 with gzip.open(O/'PATHS_ALL_SIMPLE_LE3.csv.gz','wt',newline='') as f:
  w=csv.writer(f);w.writerow(['source_index','source_banc_888_id','target_index','target_banc_888_id','hops','edge_signs','net_sign_parity','local_index_path','banc_888_id_path','weights_uS'])
  def visit(src,path,signs,weights):
   nonlocal npaths
   u=path[-1];h=len(signs)
   if h and u in T:
    parity='E_even_I' if signs.count('I')%2==0 else 'I_odd_I'
    w.writerow([src,ids[src],u,ids[u],h,signs,parity,';'.join(map(str,path)),';'.join(ids[path]),';'.join(format(x,'.17g') for x in weights)])
    path_counts[h,signs]+=1;pair_counts[src,u,h,signs]+=1;npaths+=1
    if h==1:direct.append(dict(source_index=src,source_banc_888_id=ids[src],target_index=u,target_banc_888_id=ids[u],sign=signs,weight_uS=weights[0],readout204=u in cells,annotated_leg_motor=u in set(leg)))
   if h==3:return
   for v,s,g in adj[u]:
    if v not in path and reverse_dist[v]<=2-h:visit(src,path+[v],signs+s,weights+[g])
  for src in desc:visit(int(src),[int(src)],'',[])
 rows(O/'DIRECT_EDGES.csv',direct,['source_index','source_banc_888_id','target_index','target_banc_888_id','sign','weight_uS','readout204','annotated_leg_motor'])
 rows(O/'PAIR_SIGNED_PATH_COUNTS.csv',[dict(source_index=s,source_banc_888_id=ids[s],target_index=t,target_banc_888_id=ids[t],hops=h,edge_signs=sg,count=n) for (s,t,h,sg),n in sorted(pair_counts.items())])
 print('paths',npaths,flush=True)
 # Unbounded directed reachability distinguishes >3 from truly disconnected in active graph.
 reach=np.zeros((len(desc),12000),bool);dist=np.full((len(desc),12000),-1,dtype=np.int16)
 for k,s in enumerate(desc):
  dist[k,s]=0;q=deque([int(s)])
  while q:
   u=q.popleft()
   for v,_,_ in adj[u]:
    if dist[k,v]<0:dist[k,v]=dist[k,u]+1;q.append(v)
  reach[k]=dist[k]>=0
 np.savez_compressed(O/'DESCENDING_DISTANCE.npz',source_indices=desc,source_banc_888_ids=ids[desc],target_indices=np.arange(12000),target_banc_888_ids=ids,distance=dist)
 targetrows=[]
 for t in targets:
  rr=a.iloc[t].to_dict();rr.update(in_degree=int(indeg[t]),out_degree=int(outdeg[t]),descending_sources_direct=int((dist[:,t]==1).sum()),descending_sources_le3=int(((dist[:,t]>=1)&(dist[:,t]<=3)).sum()),descending_sources_any=int(reach[:,t].sum()),min_descending_distance=int(dist[:,t][dist[:,t]>=0].min()) if reach[:,t].any() else -1);targetrows.append(rr)
 rows(O/'TARGET_MOTOR_AUDIT.csv',targetrows)
 descrows=[]
 for k,s in enumerate(desc):
  rr=a.iloc[s].to_dict();rr.update(in_degree=int(indeg[s]),out_degree=int(outdeg[s]),direct_E=int(ec.indptr[s+1]-ec.indptr[s]),direct_I=int(ic.indptr[s+1]-ic.indptr[s]),targets_direct=int((dist[k,targets]==1).sum()),targets_le3=int(((dist[k,targets]>=1)&(dist[k,targets]<=3)).sum()),targets_any=int(reach[k,targets].sum()));descrows.append(rr)
 rows(O/'DESCENDING_194.csv',descrows)
 pools=json.loads(POOL.read_text())['muscle_to_joint'];joints=[]
 for j in sorted(set(p['joint'] for p in pools)):
  pr=[p for p in pools if p['joint']==j];plus=sorted({i for p in pr if p['sign']>0 for i in p['local_indices']});minus=sorted({i for p in pr if p['sign']<0 for i in p['local_indices']});both=[]
  for k,s in enumerate(desc):
   if plus and minus and np.any((dist[k,plus]>0)&(dist[k,plus]<=3)) and np.any((dist[k,minus]>0)&(dist[k,minus]<=3)):both.append(int(s))
  joints.append(dict(joint=j,leg=pr[0]['leg'],positive_indices=plus,negative_indices=minus,positive_ids=ids[plus].tolist(),negative_ids=ids[minus].tolist(),overlap=sorted(set(plus)&set(minus)),both_sign_descending_le3=both,both_sign_descending_ids=ids[both].tolist(),pool_annotations=[dict(actuator_name=p['actuator_name'],sign=p['sign'],local_indices=p['local_indices']) for p in pr],note='Signs from existing mechanical pool mapping, not inferred anatomical flexion labels; verify with actuator branch.'))
 dump(O/'JOINT_PATH_COVERAGE.json',joints)
 z=np.load(CON,allow_pickle=True);lookup={str(x):k for k,x in enumerate(z['root_ids'])};zi=np.array([lookup[x] for x in ids]);zs=np.asarray(z['ann_super_class'])[zi].astype(str)
 subgroup=[]
 for typ in sorted(set(a.iloc[desc].cell_type)):
  ss=[int(s) for s in desc if a.iloc[s].cell_type==typ];kk=[int(np.where(desc==s)[0][0]) for s in ss];subgroup.append(dict(cell_type=typ,indices=ss,banc_888_ids=ids[ss].tolist(),side=a.iloc[ss].side.tolist(),nt_predicted=a.iloc[ss].neurotransmitter_predicted.tolist(),cell_function=a.iloc[ss].cell_function.tolist(),cell_function_detailed=a.iloc[ss].cell_function_detailed.tolist(),leg_targets_le3=int(np.any((dist[kk][:,leg]>0)&(dist[kk][:,leg]<=3),axis=0).sum()),readout_targets_le3=int(np.any((dist[kk][:,cells]>0)&(dist[kk][:,cells]<=3),axis=0).sum()),functional_status='ANNOTATION_ONLY_NOT_VALIDATED_COMMAND'))
 dump(O/'DESCENDING_SUBGROUPS.json',subgroup)
 bset=set(brain);dset=set(desc);braininfo=dict(count=len(brain),nonisolated=int(((indeg+outdeg)[brain]>0).sum()),E_internal=int(E[brain][:,brain].nnz),I_internal=int(I[brain][:,brain].nnz),E_to_descending=int(E[desc][:,brain].nnz),I_to_descending=int(I[desc][:,brain].nnz),E_from_descending=int(E[brain][:,desc].nnz),I_from_descending=int(I[brain][:,desc].nnz),indices=brain.tolist(),banc_888_ids=ids[brain].tolist())
 dump(O/'BRAIN_INTRINSIC.json',braininfo)
 summary=dict(active_cells=len(ids),join_key='banc_888_id',matched=12000,root_id_join_matches=int(np.isin(ids,meta.root_id).sum()),super_class_counts=dict(Counter(a.super_class)),annotated_leg_motor=len(leg),motor_superclass=len(motor),readout_count=len(cells),readout_motor=int((a.iloc[cells].super_class=='motor').sum()),readout_leg_motor=int((a.iloc[cells].cell_function=='leg_motor').sum()),target_union=len(targets),descending=len(desc),connectome_superclass_matches=bool(np.array_equal(zs,a.super_class.to_numpy())),descending_by_class=int((a.cell_class=='descending_neuron').sum()),sensory_descending_indices=np.flatnonzero(a.super_class=='sensory_descending').tolist(),matrices={k:dict(path=str(AB/f'ACTUAL_{k}.npz'),sha256=sha(AB/f'ACTUAL_{k}.npz'),nnz=m.nnz,sum_uS=float(m.data.sum())) for k,m in [('We',E),('Wi',I)]},matrix_orientation='post,pre; adj[pre] lists postsynaptic targets',RH_release_rule='multiply RH postsynaptic We rows by 0.375 once; positive scaling preserves topology and signs; stored weights are pre-release baseline',path_bound=3,path_semantics='ALL directed simple paths lengths1..3; no repeated vertex; sign parity structural only, not physiological net effect',path_count=npaths,signed_path_counts=[dict(hops=h,edge_signs=s,count=n) for (h,s),n in sorted(path_counts.items())],direct_edges=len(direct),descending_isolated=[int(s) for s in desc if indeg[s]+outdeg[s]==0],descending_no_targets_any=[int(s) for k,s in enumerate(desc) if not reach[k,targets].any()],targets_unreachable_any=[int(t) for t in targets if not reach[:,t].any()],targets_unreachable_le3=[int(t) for t in targets if not ((dist[:,t]>0)&(dist[:,t]<=3)).any()],brain_intrinsic=braininfo,simulations=0,neural_step_calls=0,source_mutations=0)
 dump(O/'SUMMARY.json',summary)
 assert {str(p):sha(p) for p in sources}==hashes
 print(json.dumps({k:v for k,v in summary.items() if k in ['descending','annotated_leg_motor','motor_superclass','direct_edges','path_count','connectome_superclass_matches','descending_by_class','targets_unreachable_any']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
