"""Is there a legitimate motor readout downstream of the tactile component?

The wiring audit showed the tactile channel (2,884 cells, about 81% of the sensory cells) has no
path to any of the 204 readout motor cells, and that the tactile weight changes the motor command
by exactly zero. The only compliant fix is to select the readout from real anatomy, never to write
synapses. So this audit answers three questions against the published BANC annotations:

  1. what are the 204 current readout cells, by type
  2. what cell types does the tactile component actually reach
  3. does that component contain cells annotated as motor neurons, and are any of them outside the
     current readout

Source: BANC 888 metadata, CC BY 4.0, doi:10.7910/DVN/7WTH1N, file banc_888_meta.feather.
No simulation runs. modes: audit
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
import pandas as pd
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import run_neural_muscle_closure as c
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig, CHANNEL_SPEC
A=st.A;B=st.B
META=Path('/tmp/banc888_meta.feather')
O=Path('/home/sensen/Desktop/engine修复验收/s5-tactile-readout/20261104T030000Z')
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def audit():
 import pandas as pd
 roots,leg,pool,npools=st.identity()
 n=len(roots)
 lookup=A/'SENSORY_LOOKUP_SNAPSHOT888.json'
 groups,_=c.channel_index_map(n,roots,lookup_path=lookup)
 enc=ChannelSensoryEncoder(groups,n,ChannelConfig())
 raw=(abs(sparse.load_npz(B/'ACTUAL_We.npz'))+abs(sparse.load_npz(B/'ACTUAL_Wi.npz'))).tocsr();raw.data[:]=1.0
 W=raw.T.tocsr()                                  # column is presynaptic, checked in the wiring audit
 motor=set(np.flatnonzero(np.asarray(leg)>=0).tolist())
 meta=pd.read_feather(META)
 key='banc_888_id'
 if not set(map(int,roots)).issubset(set(meta[key].astype('int64'))):
  for alt in ('root_888','root_id'):
   if set(map(int,roots)).issubset(set(meta[alt].astype('int64'))):key=alt;break
 m=meta.set_index(meta[key].astype('int64'))
 def typed(idx):
  sub=m.loc[[int(roots[i]) for i in idx]]
  return dict(super_class=Counter(sub['super_class'].fillna('NA')).most_common(6),
   cell_class=Counter(sub['cell_class'].fillna('NA')).most_common(8),
   cell_type_top=Counter(sub['cell_type'].fillna('NA')).most_common(8),
   cell_function=Counter(sub['cell_function'].fillna('NA')).most_common(6),
   body_part_effector=Counter(sub['body_part_effector'].fillna('NA')).most_common(6))
 tactile=set()
 for lg in enc.groups:tactile|=set(enc.groups[lg].get('bristle_touch',[]))
 front=set(tactile);seen=set(tactile);hops={}
 for k in range(1,7):
  nxt=set()
  for j in front:nxt|=set(W.indices[W.indptr[j]:W.indptr[j+1]].tolist())
  nxt-=seen;seen|=nxt;front=nxt;hops[k]=len(nxt)
  if not nxt:break
 reached=seen-tactile
 sub=m.loc[[int(roots[i]) for i in sorted(reached)]]
 is_motor=sub['cell_class'].fillna('').str.contains('motor',case=False)|sub['cell_function'].fillna('').str.contains('motor',case=False)|sub['super_class'].fillna('').str.contains('motor',case=False)|sub['cell_type'].fillna('').str.contains('motor',case=False)
 cand=sorted(set(np.array(sorted(reached))[np.asarray(is_motor)] .tolist()))
 res=dict(source=dict(dataset='doi:10.7910/DVN/7WTH1N',file=META.name,license='CC BY 4.0',
   sha256=hashlib.sha256(META.read_bytes()).hexdigest(),join_key=key),
  network=dict(neurons=int(n),readout_motor_cells=int(len(motor)),pools=int(npools)),
  question1_current_readout_types=typed(sorted(motor)),
  question2_tactile_component=dict(tactile_cells=len(tactile),reached_cells=len(reached),cells_per_hop=hops,
   types=typed(sorted(reached))),
  question3_motor_annotated_candidates=dict(
   count=len(cand),
   inside_current_readout=int(len(set(cand)&motor)),
   outside_current_readout=int(len(set(cand)-motor)),
   annotation_basis='any of cell_class, cell_function, super_class, cell_type containing "motor" (case-insensitive), taken as published and not edited',
   sample=[dict(index=int(i),banc_id=int(roots[i]),cell_class=str(m.loc[int(roots[i]),'cell_class']),
     cell_type=str(m.loc[int(roots[i]),'cell_type']),cell_function=str(m.loc[int(roots[i]),'cell_function']),
     body_part_effector=str(m.loc[int(roots[i]),'body_part_effector']),
     neurotransmitter=str(m.loc[int(roots[i]),'neurotransmitter_predicted'])) for i in cand[:40]]),
  honesty='annotation strings are taken as published; a type containing "motor" is not by itself proof that the cell drives a muscle in this model, so candidates must be verified before any readout change')
 dump(O/'TACTILE_READOUT_CANDIDATES.json',res)
 print(json.dumps({k:(v if k!='question3_motor_annotated_candidates' else {kk:vv for kk,vv in v.items() if kk!='sample'}) for k,v in res.items() if k!='source'},ensure_ascii=False)[:2500])
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'audit':audit}[sys.argv[1]]()
