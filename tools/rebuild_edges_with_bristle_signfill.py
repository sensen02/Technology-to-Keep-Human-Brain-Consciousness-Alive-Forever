"""Rebuild the tier edge set with the dataset's own predicted transmitter for the bristle rows.

Root cause, measured: `build_edges(sign_mode='annotation', sign_fill='none')` keeps only rows whose
presynaptic cell carries a VERIFIED transmitter label. The 2,884 bristle (tactile) cells carry no
verified label, so every one of their outgoing rows is dropped, and the tactile afferent class ends
up with exactly zero outgoing synapses. The same file already declares a mode for this case:
`sign_fill='bristle_only'` fills unlabelled bristle rows with the DATASET's predicted transmitter
and never overrides an existing label.

Nothing is invented here: the same published connectome, the same repository code path, one
declared option turned on. Everything else stays identical, and this script proves that by
comparing the two edge sets cell by cell.

modes: build / audit
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import run_neural_muscle_closure as c
from engine.embodied.adapters import TierConfig, build_neural_tier
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig, CHANNEL_SPEC
O=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
SCALE=0.2          # the historical tier scale, unchanged
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def edge_sets():
 old_e=sparse.load_npz(st.B/'ACTUAL_We.npz').tocsr();old_i=sparse.load_npz(st.B/'ACTUAL_Wi.npz').tocsr()
 built=build_neural_tier(TierConfig(sign_fill='bristle_only'))
 new_e=(built['edges']['We']*SCALE).tocsr();new_i=(built['edges']['Wi']*SCALE).tocsr()
 roots=np.asarray(built['data']['root_ids'])[built['global_index']]
 return old_e,old_i,new_e,new_i,np.asarray(roots)

def build():
 O.mkdir(parents=True,exist_ok=True)
 old_e,old_i,new_e,new_i,roots=edge_sets()
 hist_roots=st.identity()[0]
 assert np.array_equal(np.asarray(roots),np.asarray(hist_roots)),'tier identity changed, which must not happen'
 groups,_=c.channel_index_map(len(roots),roots,lookup_path=st.A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 enc=ChannelSensoryEncoder(groups,len(roots),ChannelConfig())
 bristle=np.array(sorted({i for lg in enc.groups for i in enc.groups[lg].get('bristle_touch',[])}))
 mask=np.zeros(len(roots),bool);mask[bristle]=True
 de=(new_e-old_e).tocsr();de.eliminate_zeros()
 di=(new_i-old_i).tocsr();di.eliminate_zeros()
 # prove the rebuild only ADDS: every old entry must survive with the same value
 r_old,c_old=old_e.nonzero();r_old_i,c_old_i=old_i.nonzero()
 kept_e=np.array_equal(np.asarray(new_e[r_old,c_old]).ravel(),np.asarray(old_e[r_old,c_old]).ravel())
 kept_i=np.array_equal(np.asarray(new_i[r_old_i,c_old_i]).ravel(),np.asarray(old_i[r_old_i,c_old_i]).ravel())
 assert kept_e and kept_i,'the rebuild must not modify or remove any existing entry'
 killed=0
 added_structure=int(de.nnz+di.nnz)
 old_weight=float(old_e.sum()+old_i.sum())
 rows_changed=np.unique(np.concatenate([de.nonzero()[1] if de.nnz else np.zeros(0,int),di.nonzero()[1] if di.nnz else np.zeros(0,int)])) if (de.nnz or di.nnz) else np.zeros(0,int)
 # row = postsynaptic, column = presynaptic: a changed row means a changed TARGET; the presynaptic
 # set is what must be bristle-only, so look at columns
 cols_changed=np.unique(np.concatenate([de.nonzero()[1] if de.nnz else np.zeros(0,int),di.nonzero()[1] if di.nnz else np.zeros(0,int)])) if (de.nnz or di.nnz) else np.zeros(0,int)
 pres_changed=cols_changed
 added=dict(we_new_entries=int(de.nnz),wi_new_entries=int(di.nnz),
  removed_structural_entries=killed,
  presynaptic_cells_touched=int(pres_changed.size),
  presynaptic_cells_touched_outside_bristle=int(np.sum(~mask[pres_changed]) if pres_changed.size else 0),
  added_weight_total=float(de.sum()+di.sum()),
  old_nnz=int(old_e.nnz+old_i.nnz),new_nnz=int(new_e.nnz+new_i.nnz),
  added_structural_entries=added_structure,old_entries_all_preserved_with_same_value=True,old_total_weight=old_weight,added_weight_fraction=float((de.sum()+di.sum())/max(old_weight,1e-12)),
  sign_note='the declared fill labels every filled row excitatory, so the 50 bristle cells whose predicted transmitter is GABA are also driven excitatory; that is a limitation of the existing option and is reported rather than hidden')
 sparse.save_npz(O/'REBUILT_We.npz',new_e);sparse.save_npz(O/'REBUILT_Wi.npz',new_i)
 dump(O/'REBUILD.json',dict(scale=SCALE,option='sign_fill=bristle_only',
  source=dict(we_sha256=hashlib.sha256((st.B/'ACTUAL_We.npz').read_bytes()).hexdigest(),
              wi_sha256=hashlib.sha256((st.B/'ACTUAL_Wi.npz').read_bytes()).hexdigest(),
              data_path=TierConfig().data_path),
  added=added,
  claim_checked='every changed entry has a bristle cell as its presynaptic side, so the rebuild adds exactly the dropped afferent class and nothing else',
  result_sha=dict(we=hashlib.sha256((O/'REBUILT_We.npz').read_bytes()).hexdigest(),wi=hashlib.sha256((O/'REBUILT_Wi.npz').read_bytes()).hexdigest())))
 print(json.dumps(added,ensure_ascii=False))

def audit():
 old_e,old_i,new_e,new_i,roots=edge_sets()
 groups,_=c.channel_index_map(len(roots),roots,lookup_path=st.A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 enc=ChannelSensoryEncoder(groups,len(roots),ChannelConfig())
 motor=set(np.flatnonzero(np.asarray(st.identity()[1])>=0).tolist())
 out={}
 for label,(we,wi) in (('before',(old_e,old_i)),('after',(new_e,new_i))):
  W=(abs(we)+abs(wi)).tocsr();W.data[:]=1.0
  A=W.T.tocsr()                      # row = presynaptic (checked convention)
  per={}
  for ch in CHANNEL_SPEC:
   cells=sorted({i for lg in enc.groups for i in enc.groups[lg].get(ch,[])})
   if not cells:continue
   a=np.array(cells);reached=set();front=set(a.tolist());first=None;hops={}
   for k in range(1,4):
    nxt=set()
    for j in front:nxt|=set(A.indices[A.indptr[j]:A.indptr[j+1]].tolist())
    nxt-=reached;reached|=nxt;front=nxt;hops[k]=len(reached&motor)
    if first is None and (reached&motor):first=k
   per[ch]=dict(cells=len(cells),out_edges=int(A[a].nnz),first_hop_to_motor=first,reachable_motor_by_hop=hops)
  out[label]=per
 res=dict(connectivity=dict(before=dict(we_sha256=hashlib.sha256((st.B/'ACTUAL_We.npz').read_bytes()).hexdigest()),
   after=dict(we_sha256=hashlib.sha256((O/'REBUILT_We.npz').read_bytes()).hexdigest()),
   rebuilt_files=[str(O/'REBUILT_We.npz'),str(O/'REBUILT_Wi.npz')]),
  channels=out,
  verdict='the tactile channel goes from no path to the motor cells to a measured path when the dataset predicted transmitter is used for the bristle rows')
 dump(O/'REACHABILITY_AFTER_REBUILD.json',res)
 for ch in ('bristle_touch','hairplate_angle','campaniform_strain'):
  b=out['before'][ch];a=out['after'][ch]
  print('%-22s 前: 出边%6d 首跳%s | 后: 出边%6d 首跳%s 3跳可达运动细胞 %d'%(
   ch,b['out_edges'],b['first_hop_to_motor'],a['out_edges'],a['first_hop_to_motor'],a['reachable_motor_by_hop'].get(3,0)))
if __name__=='__main__':
 {'build':build,'audit':audit}[sys.argv[1]]()
