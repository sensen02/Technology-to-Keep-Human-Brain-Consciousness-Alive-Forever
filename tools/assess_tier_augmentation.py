"""Cost assessment for the only remaining route to the silent motor cells.

The silence is emergent recurrent dynamics, so the cells themselves cannot be revived by any readout.
The remaining structural lever is the tier: which cells of the 153962 neuron BANC graph get included.
This measures, without changing anything, how many of the silent motor cells' upstream partners in the
FULL graph are missing from the 12000 cell tier.

Two facts make augmentation cheaper than resampling: the tier is a fixed ordering, so appending cells
keeps every existing index valid, and the trained vector indexes actuators and their cell assignments,
which do not change. So if the missing partner count is small, augmentation preserves every result so
far instead of invalidating them.

modes: run
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s5-tier-augmentation-cost/20261107T180000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
CAUSE=Path('/home/sensen/Desktop/engine修复验收/s5-silent-motoneuron-cause/20261107T020000Z/SILENT_CAUSE.json')
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if hasattr(x,'item') else str(x)))+'\n')

def run():
 import dataclasses,importlib
 os.environ['PCR_OUT']=str(O)
 import train_per_cell_readout as t
 importlib.reload(t)
 t.make_set()
 import engine.embodied.adapters as A
 from engine.embodied.adapters import TierConfig
 orig=A.build_neural_tier
 A.build_neural_tier=lambda config=None,*a,**k: orig(dataclasses.replace(config or TierConfig(),sign_fill='bristle_only'),*a,**k)
 try:built=A.build_neural_tier()
 finally:A.build_neural_tier=orig
 tier=built['tier']
 ids=tier.get('root_ids') if isinstance(tier,dict) else None
 if ids is None:
  for k in ('selected_root_ids','root_ids','ids'):
   if isinstance(tier,dict) and k in tier:ids=tier[k];break
 ids=np.asarray(ids,dtype=np.int64)
 print('tier 细胞数',ids.size)
 tier_set=set(int(x) for x in ids)
 # 全图
 z=np.load('data/flywire/banc_connectome.npz',allow_pickle=True)
 pre=z['pre'];post=z['post'];syn=z['syn'].astype(float);roots=z['root_ids']
 print('全图 %d 神经元 / %d 边'%(roots.size,pre.size))
 # 静默运动细胞的 root id
 man=json.loads((t.SET/'RUN/MAPPING.json').read_text())['rows']
 cause=json.loads(CAUSE.read_text())
 silent_idx=[int(x['cell']) for x in cause['per_cell'] if x['spikes']==0]
 silent_roots=sorted({int(ids[k]) for k in silent_idx})
 print('静默运动细胞 %d 个（tier 索引），对应 root id %d 个'%(len(silent_idx),len(silent_roots)))
 ridx={int(r):i for i,r in enumerate(roots)}
 sel_post=[ridx[r] for r in silent_roots if r in ridx]
 present=syn>0
 mask=present&np.isin(post,np.asarray(sel_post,dtype=np.int64))
 pre_sel=pre[mask];w_sel=syn[mask]
 uniq,inv=np.unique(pre_sel,return_inverse=True)
 wt=np.zeros(uniq.size);np.add.at(wt,inv,w_sel)
 pre_roots=[int(roots[i]) for i in uniq]
 missing=[(r,float(w)) for r,w in zip(pre_roots,wt) if r not in tier_set]
 missing.sort(key=lambda x:-x[1])
 strong=[x for x in missing if x[1]>=10.0]      # the tier's own partner threshold
 out=dict(
  question='how many of the silent motor cells upstream partners in the full BANC graph are missing from the 12000 cell tier',
  full_graph=dict(neurons=int(roots.size),edges=int(pre.size)),
  tier_cells=int(ids.size),
  silent_motor_cells=len(silent_roots),
  partners_total=len(pre_roots),
  partners_in_tier=len(pre_roots)-len(missing),
  partners_missing=len(missing),
  partners_missing_with_10_or_more_synapses=len(strong),
  missing_synapse_mass=float(sum(w for _,w in missing)),
  top_missing_partners=[dict(root_id=r,synapses=w) for r,w in missing[:15]],
  augmentation_cost=dict(
   cells_to_append=len(strong),
   new_tier_size=int(ids.size+len(strong)),
   index_stability='appending at the end leaves indices 0 to 11999 unchanged, and the trained vector indexes actuators plus their own cell assignments, so every existing result stays valid',
   what_must_be_rebuilt=['the excitatory and inhibitory edge matrices at the larger size','the sensory group projections','the rebuilt bristle sign fill'],
   what_does_NOT_change=['the 204 motor cells and their 253 actuator assignments','the 24 pool modes','the 90 actuator indices','the replacement layer targets and sources'],
   retraining_needed='yes, but from the current best vector as the seed rather than from scratch'))
 dump(O/'AUGMENTATION_COST.json',out)
 print()
 print('静默细胞的全图上游伙伴: %d 个'%len(pre_roots))
 print('  已在 tier 内: %d'%out['partners_in_tier'])
 print('  缺失: %d（其中突触≥10 的 %d 个）'%(out['partners_missing'],out['partners_missing_with_10_or_more_synapses']))
 print('  缺失突触总量: %.0f'%out['missing_synapse_mass'])
 print('  → 追加 %d 个细胞即可，tier 变为 %d'%(out['augmentation_cost']['cells_to_append'],out['augmentation_cost']['new_tier_size']))
if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 {'run':run}[sys.argv[1]]()
