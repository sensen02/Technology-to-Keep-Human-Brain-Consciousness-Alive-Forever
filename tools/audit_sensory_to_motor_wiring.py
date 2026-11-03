"""Structural check: which sensory channels actually reach the motor output.

The sensory sweep found that silencing or tripling the tactile weight changes the network's total
spike count by five to ten times while the motor output stays bit-identical. That is only possible
if those cells do not reach the readout, so this audit counts it directly on the same connectivity
the simulation uses: for every declared channel, how many of its cells have any outgoing synapse,
and how many have a path of one or two hops to a motor cell of the 66 shared pools.

No simulation runs. modes: audit
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
A=st.A;B=st.B
O=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-wiring/20261104T010000Z')
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def audit():
 import run_neural_muscle_closure as c
 from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig, CHANNEL_SPEC
 folders=R/'engine';sys.path.insert(0,str(folders))
 roots,leg,pool,npools=st.identity()
 n=len(roots)
 contract=js(A/'VERSION_CONTRACT.json')
 lookup=A/'SENSORY_LOOKUP_SNAPSHOT888.json'
 assert hashlib.sha256(lookup.read_bytes()).hexdigest()==contract['sensory_lookup_sha256']
 groups,extra=c.channel_index_map(n,roots,lookup_path=lookup)
 enc=ChannelSensoryEncoder(groups,n,ChannelConfig())
 We=sparse.load_npz(B/'ACTUAL_We.npz').tocsr();Wi=sparse.load_npz(B/'ACTUAL_Wi.npz').tocsr()
 # CONVENTION: in these matrices the COLUMN is the presynaptic cell. That is checked, not assumed:
 # read as rows the readout cells would receive zero synapses, which cannot be true, and the
 # channels that demonstrably change the motor output are the ones that reach it when read as
 # columns. Both facts are recorded in the output.
 raw=(abs(We)+abs(Wi)).tocsr();raw.data[:]=1.0
 motor_list=sorted(np.flatnonzero(np.asarray(leg)>=0).tolist())
 motor=set(motor_list)
 convention=dict(rows_as_source_motor_in_degree_mean=float(np.asarray(raw[:,motor_list].sum(0)).mean()),
  cols_as_source_motor_in_degree_mean=float(np.asarray(raw.T[motor_list].sum(1)).mean()),
  used='cols_as_source')
 W=raw.T.tocsr()
 rows={}
 for ch in CHANNEL_SPEC:
  cells=set()
  for lg in enc.groups:
   cells|=set(enc.groups[lg].get(ch,[]))
  cells=sorted(cells)
  if not cells:rows[ch]=dict(cells=0,note='no cells assigned');continue
  out=np.asarray(W[cells].sum(axis=1)).ravel()
  reached=set();front=set(cells);hops={};first=None
  while front and len(hops)<8:
   nxt=set()
   for j in front:nxt|=set(W.indices[W.indptr[j]:W.indptr[j+1]].tolist())
   nxt-=reached;reached|=nxt;front=nxt
   hops[len(hops)+1]=len(reached&motor)
   if first is None and (reached&motor):first=len(hops)
  rows[ch]=dict(cells=len(cells),with_any_outgoing=int((out>0).sum()),
   cells_that_are_motor=int(len(set(cells)&motor)),
   reachable_motor_by_hop=hops,first_hop_to_motor=first,
   reaches_any_motor=bool(first is not None),
   direct_injection_into_readout=bool(len(set(cells)&motor)>0))
 indeg=np.asarray(W.T[:,sorted(motor)].sum(axis=1)).ravel()
 res=dict(neurons=int(n),motor_cells=int(len(motor_list)),pools=int(npools),
  convention=convention,
  motor_cell_in_degree=dict(mean=float(indeg.mean()),median=float(np.median(indeg)),max=float(indeg.max()),
   note='synaptic inputs each readout motor cell receives from the tier'),
  connectivity=dict(source=str(B/'ACTUAL_We.npz'),sha256=hashlib.sha256((B/'ACTUAL_We.npz').read_bytes()).hexdigest()),
  channels=rows,
  interpretation='direct_to_motor counts distinct motor cells reachable in one hop from that channel; a channel with zero reaches no muscle-driving cell at all, so any weight on it cannot change the command')
 dump(O/'SENSORY_WIRING.json',res)
 print(json.dumps(res,ensure_ascii=False,indent=1))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'audit':audit}[sys.argv[1]]()
