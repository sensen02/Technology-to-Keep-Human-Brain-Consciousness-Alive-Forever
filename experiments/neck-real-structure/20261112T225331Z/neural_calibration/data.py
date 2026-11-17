"""Parent-only actual archived-X238 subset extraction, never synthesis from actions/geometry."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'DATASET.json').exists():raise FileExistsError('No blind dataset overwrite')
 bp,b,digest=binding();groups={};hashes={};meta=[]
 for split,names in [('train',TRAIN),('development',DEV)]:
  body=[];cue=[];teacher=[]
  for condition in names:
   folder=BODY/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');paths=[folder/n for n in ('RESULT.json','VERIFY.json','CAPTURES.json','neural_decisions.npz')]
   if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(paths[0]):raise ValueError('Actual archived-body audit required')
   verify(r['source_hashes']);hashes.update(r['source_hashes']);hashes.update({str(p):sha(p) for p in paths})
   cap=read(folder/'CAPTURES.json')
   with np.load(folder/'neural_decisions.npz',allow_pickle=False) as z:
    for key in ('X238','raw_turn','time','image_capture_id','used_stimulus_code'):
     if key not in z.files:raise ValueError('Missing actual archived field '+key)
    if z['X238'].shape!=(2880,238) or z['raw_turn'].shape!=(2880,):raise ValueError('Actual complete archive2880rows required')
    raw=z['X238'][960:2880].copy();t=z['time'][960:2880].copy();ids=z['image_capture_id'][960:2880].copy();codes=z['used_stimulus_code'][960:2880].copy();target=z['raw_turn'][960:2880].copy()
   if np.max(abs(t-(5.8+np.arange(1920)*.005)))>1e-8:raise ValueError('Actual autonomous clock contract')
   for k,fid in enumerate(ids):
    c=cap[int(fid)]
    if ('off','left','right','center')[int(codes[k])]!=c['stimulus'] or c['release_time_s']>t[k]+1e-8 or c['capture_time_s']+.02>t[k]+1e-8:raise ValueError('Experimental cue must derive alreadyreleased actual stimulus only')
   x=raw[:,MASK]
   if x.shape!=(1920,111) or not np.isfinite(x).all() or not np.isfinite(target).all():raise ValueError('Finite actual111body features required')
   body.append(x);cue.append(np.asarray([0.,1.,-1.,0.])[codes.astype(np.int64)][:,None]);teacher.append(target[:,None]);meta.append(dict(split=split,condition=condition,rows=1920,source=str(folder),cue='explicit released taskinstruction; not native sensory signal',teacher_is_loss_only=True,archived_physical_trajectory_not_new_holdout=True))
  groups[split]=dict(body111=np.stack(body,1),cue=np.stack(cue,1),teacher=np.stack(teacher,1))
 mean=groups['train']['body111'].mean((0,1));std=np.maximum(groups['train']['body111'].std((0,1)),1e-6)
 for name,value in groups.items():np.savez_compressed(HERE/(name+'.npz'),**value,mean=mean,std=std)
 desc=dict(source_binding=str(bp),source_binding_sha256=digest,source_hashes=hashes,outputs={name:dict(path=str(HERE/(name+'.npz')),sha256=sha(HERE/(name+'.npz'))) for name in groups},rows=meta,allowed_raw_indices=MASK,normalization_from_train_only=True,physical_steps=0,model_forward_calls=0)
 save(HERE/'DATASET.json',desc);print('ACTUAL_ARCHIVED_X238_TRAIN4_DEV2_PREPARED_NO_PHYSICS')
def arrays(split):
 import numpy as np
 with np.load(HERE/(split+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def tensors(data,indices=None):
 import torch
 keys=('body111','cue','teacher');return tuple(torch.from_numpy(data[k] if indices is None else data[k][:,indices]) for k in keys)
if __name__=='__main__':raise SystemExit('Use parent explicitflag orchestration; import main only afterflag')
