"""Parent-only schedule generation from independently rendered library; no fake images."""
from data import *
def main():
 cpu1();rgb,manifest=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('dataset already frozen')
 meta={}
 for name,n in (('train',24),('validation',12),('controls',8)):
  a,rows=split(name,manifest)
  if a['frame_ids'].shape!=(481,n) or a['target'].shape!=(1920,n):raise ValueError('visual timing schema')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);meta[name]=dict(path=str(path),sha256=sha(path),rows=rows,sequences=n,used_actual_frame_ids=sorted(set(a['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=meta,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),RGB_NOT_COPIED=True,unused_library_frames_not_extra_episodes=True,static_head_not_navigation=True,all_scene_labels_offline_only=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train24=True,validation12=True,controls8=True,unique_rendered_pairs=len(rgb),physics_steps=0)))
if __name__=='__main__':main()
