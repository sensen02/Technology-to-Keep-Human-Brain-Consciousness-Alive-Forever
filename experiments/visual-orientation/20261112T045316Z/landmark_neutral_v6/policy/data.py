"""Original720actualhistories exact plus independent episode_scale4/1; no reweight normalization."""
import types,copy
from common import *
legacy=module('_neutral_original_v1_schedule',V1/'data.py');MOTIFS=legacy.MOTIFS;CAPTURE=legacy.CAPTURE;DECISION=legacy.DECISION;USED=legacy.USED

def variants():
 configs=read(HERE/'PREREG.json')['validation_configs'];return [dict(c,pose_id='neutralvalpose'+str(j),variant_index=i*4+j) for i,c in enumerate(configs) for j in range(4)]
def episode(cfg,motif,manifest):
 g=dict(legacy.__dict__);g['POSE']={cfg['id']:cfg['pose_id']};fn=types.FunctionType(legacy.episode.__code__,g,legacy.episode.__name__,legacy.episode.__defaults__,legacy.episode.__closure__);ids,target,w,meta=fn(cfg,motif,manifest);meta.update(variant_index=cfg['variant_index'],neutral_objective_newvalidation=True,not_independent_motion_trajectory=True);return ids,target,w,meta
def split(which,manifest):
 if which=='train':
  with np.load(OLD/'train.npz',allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
  meta=copy.deepcopy(read(OLD/'DATASET.json')['splits']['train']['rows']);out['episode_scale']=np.asarray([4. if r['motif'] in ('empty','center','occluder_only') else 1. for r in meta],dtype=np.float64)
  for r,s in zip(meta,out['episode_scale']):r['objective_group_scale']=float(s)
  return out,meta
 configurations=variants();motifs=MOTIFS if which=='validation' else ('left_occluded','right_occluded','occluder_only','fixed_left_while_world_flips');records=[episode(c,m,manifest) for m in motifs for c in configurations]
 return dict(frame_ids=np.stack([r[0] for r in records],1),target=np.stack([r[1] for r in records],1),weights=np.stack([r[2] for r in records],1),capture_time=CAPTURE,decision_time=DECISION,used_capture_index=USED,episode_scale=np.ones(len(records),dtype=np.float64)),[r[3] for r in records]
def arrays(name):
 with np.load(HERE/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pair_table():return [[160+i,240+i] for i in range(80)]+[[320+i,400+i] for i in range(80)]+[[i,80+i] for i in range(80)]+[[480+i,560+i] for i in range(80)]+[[640+i,640+i+1] for i in range(0,80,2)]
def expand(ids):return [x for i in ids for x in pair_table()[int(i)]]
def batch(rgb,data,indices):
 # Return original three tensors unchanged, then separate episode scale; no normalization cancellation.
 x,y,w=legacy.batch(rgb,data,indices);return x,y,w,torch.from_numpy(data['episode_scale'][indices])
def pair_coverage(history):
 count=np.bincount(np.asarray(history,dtype=np.int64).reshape(-1),minlength=360) if len(history) else np.zeros(360,dtype=np.int64)
 return dict(declared_pair_count=360,pair_selection_counts=count.tolist(),unseen_pair_ids=np.flatnonzero(count==0).tolist(),minimum_selections=int(count.min()),maximum_selections=int(count.max()),no_fill_in_training_or_budget_extension=True)
