"""Schedules contain actual library IDs; visible direction labels stay offline."""
from common import *
MOTIFS=('empty','center','left_pulse','right_pulse','left_to_right','right_to_left')
POSE={'T0':'pose0','T1':'pose1','T2':'pose2','T3':'pose0','V0':'pose3','V1':'pose4'}
CAPTURE=-.02+np.arange(481)*.02;DECISION=np.arange(1920)*.005;USED=np.arange(1920)//4
def scene_at(motif,t,cfg):
 if t<cfg['on_s'] or motif=='empty':return 'off'
 if motif=='center':return 'center' if t<cfg['off_s'] else 'off'
 if motif in ('left_pulse','right_pulse'):return motif.split('_')[0] if t<cfg['off_s'] else 'off'
 if motif=='left_to_right':return 'left' if t<cfg['switch_s'] else 'right'
 if motif=='right_to_left':return 'right' if t<cfg['switch_s'] else 'left'
 if motif in ('left_occluded','right_occluded'):return motif.split('_')[0] if t<cfg['off_s'] else 'occluded_'+motif.split('_')[0]
 if motif=='occluder_only':return 'occluder_only'
 if motif=='fixed_left_while_world_flips':return 'left'  # First released frame onwards; stale capture stream explicit control.
 raise ValueError(motif)
def episode(cfg,motif,manifest):
 lookup={r['stimulus']:r for r in manifest['rows'] if r['config_id']==cfg['id'] and r['pose_id']==POSE[cfg['id']]}
 ids=np.asarray([lookup['off' if t<0 or (motif=='fixed_left_while_world_flips' and t<cfg['on_s']) else scene_at(motif,float(t),cfg)]['frame_id'] for t in CAPTURE],dtype=np.int64)
 rows=[manifest['rows'][int(i)] for i in ids[USED]]
 target=np.asarray([.8 if r['stimulus']=='left' else -.8 if r['stimulus']=='right' else 0. for r in rows]);w=np.ones(1920)/1920
 nonzero=np.flatnonzero(target!=0);on=None if not len(nonzero) else float(DECISION[nonzero[0]]);off=None;switch=None
 if motif.endswith('pulse') or motif.endswith('occluded'):
  neutral=np.flatnonzero((DECISION>=cfg['off_s'])&(target==0));off=None if not len(neutral) else float(DECISION[neutral[0]])
 if motif in ('left_to_right','right_to_left'):
  sign=1 if motif.startswith('left') else -1;opposite=np.flatnonzero(sign*target<0);switch=None if not len(opposite) else float(DECISION[opposite[0]])
 if motif.endswith('pulse'):
  if on is None or off is None:raise ValueError('pulse must have actual released on/off')
  eps=1e-9;active=(DECISION>=on+.2-eps)&(DECISION<off-eps);trans=((DECISION>=on-eps)&(DECISION<on+.2-eps))|((DECISION>=off-eps)&(DECISION<off+.5-eps));recovery=(DECISION>=off+.5-eps)&(DECISION<off+1.-eps);base=~(active|trans|recovery);masks=np.stack((active,trans,recovery,base))
  if not np.all(masks.sum(0)==1):raise ValueError('disjoint visual timing weights')
  w=sum(mass*mask/mask.sum() for mass,mask in zip((.3,.2,.4,.1),masks))
 else:
  yes=abs(target)>=.05;no=~yes
  if yes.any() and no.any():w=.5*yes/yes.sum()+.5*no/no.sum()
 meta=dict(config_id=cfg['id'],pose_id=POSE[cfg['id']],motif=motif,world_on_s=cfg['on_s'],world_off_s=cfg['off_s'],world_switch_s=cfg['switch_s'],released_on_s=on,released_off_s=off,released_switch_s=switch,static_head_clamp=True)
 return ids,target,w,meta
def split(which,manifest):
 p=read(HERE/'PREREG.json');configs=p['train_configs'] if which=='train' else p['validation_configs'];motifs=MOTIFS if which!='controls' else ('left_occluded','right_occluded','occluder_only','fixed_left_while_world_flips')
 records=[episode(c,m,manifest) for m in motifs for c in configs]
 return dict(frame_ids=np.stack([r[0] for r in records],1),target=np.stack([r[1] for r in records],1),weights=np.stack([r[2] for r in records],1),capture_time=CAPTURE,decision_time=DECISION,used_capture_index=USED),[r[3] for r in records]
def arrays(name):
 with np.load(HERE/(name+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pair_table():return [[8+i,12+i] for i in range(4)]+[[16+i,20+i] for i in range(4)]+[[i,4+i] for i in range(4)]
def expand(ids):return [v for i in ids for v in pair_table()[int(i)]]
def batch(rgb,data,indices):
 return torch.from_numpy(rgb[data['frame_ids'][:,indices]]),torch.from_numpy(data['target'][:,indices]),torch.from_numpy(data['weights'][:,indices])
