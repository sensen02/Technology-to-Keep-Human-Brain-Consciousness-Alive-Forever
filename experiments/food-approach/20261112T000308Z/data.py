"""Exact original inputs/labels, only8pulse offline point weights differ."""
from common import *
legacy_data=module('_segmented_frozen_dataequations',OLD/'data.py');MOTIFS=legacy_data.MOTIFS
arrays=legacy_data.arrays
def segment_masks(time,cfg):
 eps=1e-10;on=cfg['on_s'];off=cfg['off_s'];t=time
 ON=(t>=on+.2-eps)&(t<off-eps)
 TRANS=((t>=on-eps)&(t<on+.2-eps))|((t>=off-eps)&(t<off+.5-eps))
 REC=(t>=off+.5-eps)&(t<off+1.-eps)
 BASE=~(ON|TRANS|REC)
 return np.stack((ON,TRANS,REC,BASE),axis=1)
def reweight_train():
 a=arrays(OLD/'train.npz');meta=read(OLD/'DATASET.json')['splits']['train']['configs'];oldweights=a['weights'].copy();segment=np.full((1920,24),-1,dtype=np.int64)
 for j,cfg in enumerate(meta):
  if cfg['motif'] not in ('left_pulse','right_pulse'):continue
  mask=segment_masks(a['time'],cfg)
  if not np.all(mask.sum(1)==1):raise ValueError('segments not mutuallyexclusive/complete')
  masses=np.array([.30,.20,.40,.10]);counts=mask.sum(0);nonempty=counts>0;masses=masses*nonempty;masses/=masses.sum();weights=np.zeros(1920)
  for k in range(4):
   if counts[k]:weights[mask[:,k]]=masses[k]/counts[k]
  if abs(weights.sum()-1)>1e-12:raise ValueError('episodeweightmass not1')
  a['weights'][:,j]=weights;segment[:,j]=np.argmax(mask,axis=1)
 a['segment_id']=segment
 return a,meta,oldweights
def split(which):
 if which=='train':
  a,meta,_=reweight_train();return a,meta
 rows=[legacy_data.episode(m,c) for m in MOTIFS for c in read(HERE/'PREREG.json')['data']['validation_variants']]
 return dict(receptors=np.stack([r[0] for r in rows],axis=1),targets=np.stack([r[1] for r in rows],axis=1),weights=np.stack([r[2] for r in rows],axis=1),concentrations=np.stack([r[3] for r in rows],axis=1),time=np.arange(1920)*.005),[r[4] for r in rows]
def training_data():
 a=arrays(HERE/'train.npz');return {k:torch.from_numpy(v) for k,v in a.items() if k in ('receptors','targets','weights')}
