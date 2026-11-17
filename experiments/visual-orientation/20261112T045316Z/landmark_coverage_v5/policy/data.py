"""720staticposeclamp descriptions; uniqueconfig+pose lookup, no duplicateimagebuffers."""
import types
from common import *
legacy=module('_coverage_original_schedule_equations',V1/'data.py');MOTIFS=legacy.MOTIFS;CAPTURE=legacy.CAPTURE;DECISION=legacy.DECISION;USED=legacy.USED

def variants(which):
 pr=read(HERE/'PREREG.json');configs=pr['train_configs'] if which=='train' else pr['validation_configs'];count=20 if which=='train' else 4;prefix='trainpose' if which=='train' else 'valpose'
 return [dict(c,pose_id=prefix+str(j),variant_index=i*count+j) for i,c in enumerate(configs) for j in range(count)]
def episode(cfg,motif,manifest,training=False):
 # Clone originalpurefunction with this exactgeometry+pose mapping; do not silently reuse onepose.
 g=dict(legacy.__dict__);g['POSE']={cfg['id']:cfg['pose_id']};fn=types.FunctionType(legacy.episode.__code__,g,legacy.episode.__name__,legacy.episode.__defaults__,legacy.episode.__closure__);ids,target,w,meta=fn(cfg,motif,manifest)
 if training and motif in ('left_occluded','right_occluded'):
  groups=masks(meta['released_on_s'],meta['released_off_s'])
  if not np.all(groups.sum(0)==1):raise ValueError('coveragefourgroupoverlap')
  w=sum(mass*group/group.sum() for mass,group in zip((.3,.2,.4,.1),groups))
 if training and motif=='occluder_only':
  scenes=np.array([manifest['rows'][int(i)]['stimulus'] for i in ids[USED]]);box=scenes=='occluder_only';off=scenes=='off'
  if not off.any() or not box.any() or np.any(~(off|box)):raise ValueError('actualpreoff/postbox')
  w=.5*off/off.sum()+.5*box/box.sum();meta['released_box_s']=float(DECISION[np.flatnonzero(box)[0]])
 meta.update(variant_index=cfg['variant_index'],coverage_staticpose=True,not_independent_motion_trajectory=True);return ids,target,w,meta
def masks(on,off):
 eps=1e-9;active=(DECISION>=on+.2-eps)&(DECISION<off-eps);trans=((DECISION>=on-eps)&(DECISION<on+.2-eps))|((DECISION>=off-eps)&(DECISION<off+.5-eps));rec=(DECISION>=off+.5-eps)&(DECISION<off+1-eps);return np.stack((active,trans,rec,~(active|trans|rec)))
def split(which,manifest):
 configurations=variants('train' if which=='train' else 'validation');motifs=MOTIFS+('left_occluded','right_occluded','occluder_only') if which=='train' else MOTIFS if which=='validation' else ('left_occluded','right_occluded','occluder_only','fixed_left_while_world_flips');records=[episode(c,m,manifest,which=='train') for m in motifs for c in configurations]
 return dict(frame_ids=np.stack([r[0] for r in records],1),target=np.stack([r[1] for r in records],1),weights=np.stack([r[2] for r in records],1),capture_time=CAPTURE,decision_time=DECISION,used_capture_index=USED),[r[3] for r in records]
def arrays(name):
 with np.load(HERE/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pair_table():
 return [[160+i,240+i] for i in range(80)]+[[320+i,400+i] for i in range(80)]+[[i,80+i] for i in range(80)]+[[480+i,560+i] for i in range(80)]+[[640+i,640+i+1] for i in range(0,80,2)]
def expand(ids):return [episode for pairid in ids for episode in pair_table()[int(pairid)]]
def batch(rgb,data,indices):return legacy.batch(rgb,data,indices)
def pair_coverage(history):
 counts=np.bincount(np.asarray(history,dtype=np.int64).reshape(-1),minlength=360) if len(history) else np.zeros(360,dtype=np.int64)
 return dict(declared_pair_count=360,pair_selection_counts=counts.tolist(),unseen_pair_ids=np.flatnonzero(counts==0).tolist(),minimum_selections=int(counts.min()),maximum_selections=int(counts.max()),no_fill_in_training_or_budget_extension=True)
