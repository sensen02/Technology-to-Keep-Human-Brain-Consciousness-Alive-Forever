"""Reuses original720allarrays exact; separate240scale4/480scale1 metadata only."""
from data import *
def main():
 cpu1();rgb,m=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('datasetalreadycreated')
 splits={}
 for name,n in (('train',720),('validation',48),('controls',32)):
  d,rows=split(name,m)
  if d['frame_ids'].shape!=(481,n) or d['target'].shape!=(1920,n) or d['episode_scale'].shape!=(n,) or np.max(abs(d['weights'].sum(0)-1))>1e-12:raise ValueError('originalpointweights/timingschema')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**d);splits[name]=dict(path=str(path),sha256=sha(path),rows=rows,sequences=n,used_actual_frame_ids=sorted(set(d['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=splits,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),original_training_input_histories_unchanged=True,episode_scale_separate=True,pointweights_remain_normalized1=True,scale4_neutral_episodes=240,scale1_other_episodes=480,RGB_NOT_COPIED=True,static_head_not_navigation=True,not_independent720_motion_trajectories=True,all_scene_labels_offline_only=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train720_sameinputs=True,neutral_scale4=240,validation48=True,controls32=True,actualRGB672=True,physics_steps=0)))
if __name__=='__main__':main()
