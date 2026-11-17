"""Exact existing training arrays; paired sampler metadata, new12validation configs."""
from common import *
legacy=module('_paired_original_dataequations',ORIGINAL/'data.py');MOTIFS=legacy.MOTIFS;arrays=legacy.arrays
def training_meta():return read(OLD/'DATASET.json')['splits']['train']['configs']
def pair_table():
 meta=training_meta();groups=(('left_pulse','right_pulse'),('left_to_right','right_to_left'),('noodor','equal'));table=[]
 for a,b in groups:
  for variant in ('T0','T1','T2','T3'):
   pair=[next(i for i,c in enumerate(meta) if c['motif']==motif and c['id']==variant) for motif in (a,b)];table.append(pair)
 return table
def expand_pairs(ids,table=None):
 table=pair_table() if table is None else table
 return np.asarray([episode for pairid in ids for episode in table[int(pairid)]],dtype=np.int64)
def split(which):
 if which=='train':return arrays(OLD/'train.npz'),training_meta()
 rows=[legacy.episode(m,c) for m in MOTIFS for c in read(HERE/'PREREG.json')['data']['validation_variants']]
 return dict(receptors=np.stack([r[0] for r in rows],axis=1),targets=np.stack([r[1] for r in rows],axis=1),weights=np.stack([r[2] for r in rows],axis=1),concentrations=np.stack([r[3] for r in rows],axis=1),time=np.arange(1920)*.005),[r[4] for r in rows]
def training_data():
 a=arrays(HERE/'train.npz');return {k:torch.from_numpy(v) for k,v in a.items() if k in ('receptors','targets','weights')}
