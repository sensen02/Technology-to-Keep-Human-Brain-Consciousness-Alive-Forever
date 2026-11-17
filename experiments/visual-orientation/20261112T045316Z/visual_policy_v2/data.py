"""Reuse old24bitexact; append only actualwall/occlusion training schedules."""
import types
from common import *
legacy=module('_wallvision_old_schedule',OLD/'data.py');MOTIFS=legacy.MOTIFS;CAPTURE=legacy.CAPTURE;DECISION=legacy.DECISION;USED=legacy.USED
POSE=dict(legacy.POSE,V2='pose5',V3='pose6');_g=dict(legacy.__dict__);_g['POSE']=POSE
_episode=types.FunctionType(legacy.episode.__code__,_g,legacy.episode.__name__,legacy.episode.__defaults__,legacy.episode.__closure__)
def arrays(name):
 with np.load(HERE/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def masks(on,off):
 eps=1e-9;active=(DECISION>=on+.2-eps)&(DECISION<off-eps);trans=((DECISION>=on-eps)&(DECISION<on+.2-eps))|((DECISION>=off-eps)&(DECISION<off+.5-eps));rec=(DECISION>=off+.5-eps)&(DECISION<off+1.-eps);return np.stack((active,trans,rec,~(active|trans|rec)))
def new_training_episode(cfg,motif,manifest):
 ids,target,w,meta=_episode(cfg,motif,manifest)
 if motif in ('left_occluded','right_occluded'):
  groups=masks(meta['released_on_s'],meta['released_off_s'])
  if not np.all(groups.sum(0)==1):raise ValueError('wallrecoverysegmentsmutuallyexclusive')
  w=sum(mass*g/g.sum() for mass,g in zip((.3,.2,.4,.1),groups))
 elif motif=='occluder_only':
  scene=np.asarray([manifest['rows'][int(i)]['stimulus'] for i in ids[USED]]);wall=scene=='occluder_only';off=scene=='off'
  if not wall.any() or not off.any() or np.any(~(wall|off)):raise ValueError('realoffthenwallrequired')
  w=.5*wall/wall.sum()+.5*off/off.sum();meta['released_wall_s']=float(DECISION[np.flatnonzero(wall)[0]])
  if not np.all(off[:np.flatnonzero(wall)[0]]) or np.any(target!=0):raise ValueError('purewalltarget0/realprewall')
 return ids,target,w,meta

def split(which,manifest):
 pr=read(HERE/'PREREG.json')
 if which=='train':
  with np.load(OLD/'train.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  oldmeta=read(OLD/'DATASET.json')['splits']['train']['rows'];extra=[new_training_episode(c,m,manifest) for m in ('left_occluded','right_occluded','occluder_only') for c in pr['train_configs']]
  for k,j in [('frame_ids',0),('target',1),('weights',2)]:a[k]=np.concatenate((a[k],np.stack([r[j] for r in extra],1)),1)
  return a,oldmeta+[r[3] for r in extra]
 configs=pr['validation_configs'];motifs=MOTIFS if which=='validation' else ('left_occluded','right_occluded','occluder_only','fixed_left_while_world_flips');rows=[_episode(c,m,manifest) for m in motifs for c in configs]
 return dict(frame_ids=np.stack([r[0] for r in rows],1),target=np.stack([r[1] for r in rows],1),weights=np.stack([r[2] for r in rows],1),capture_time=CAPTURE,decision_time=DECISION,used_capture_index=USED),[r[3] for r in rows]
def pair_table():return legacy.pair_table()+[[24+i,28+i] for i in range(4)]+[[32,33],[34,35]]
def expand(ids):return [v for i in ids for v in pair_table()[int(i)]]
def batch(rgb,data,indices):return legacy.batch(rgb,data,indices)
