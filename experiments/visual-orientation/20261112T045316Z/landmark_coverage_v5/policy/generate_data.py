"""Onlyactual616frame IDs andtargetsweights, not720newphysicaltrajectories."""
from data import *
def main():
 cpu1();rgb,manifest=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('datasetalreadygenerated')
 splits={}
 for name,count in (('train',720),('validation',48),('controls',32)):
  data,rows=split(name,manifest)
  if data['frame_ids'].shape!=(481,count) or data['target'].shape!=(1920,count) or np.max(abs(data['weights'].sum(0)-1))>1e-12:raise ValueError('coveragefullhistoryschema')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**data);splits[name]=dict(path=str(path),sha256=sha(path),rows=rows,sequences=count,used_actual_frame_ids=sorted(set(data['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=splits,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),RGB_NOT_COPIED=True,static_head_not_navigation=True,not_independent720_motion_trajectories=True,actual_train_background_poses=20,actual_train_configpose_combinations=80,validation_newpose_indices_not_independent_newtrajectory=True,all_scene_labels_offline_only=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train720=True,validation48=True,controls32=True,actualRGB616=True,Core_constructions=0)))
if __name__=='__main__':main()
