"""Generate only ID/target/weight schedules from actual42RGB; no synthetic images."""
from data import *
def main():
 cpu1();rgb,m=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('dataset alreadygenerated')
 metadata={}
 for name,count in (('train',36),('validation',12),('controls',8)):
  a,rows=split(name,m)
  if a['frame_ids'].shape!=(481,count) or a['target'].shape!=(1920,count):raise ValueError('full far schedule shape')
  if not np.all(np.isfinite(a['weights'])) or np.max(abs(a['weights'].sum(0)-1))>1e-12:raise ValueError('episodeweights sum1')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);metadata[name]=dict(path=str(path),sha256=sha(path),rows=rows,sequences=count,used_actual_frame_ids=sorted(set(a['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=metadata,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),RGB_NOT_COPIED=True,new_far_domain=True,near_body_failures_unchanged=True,all_scene_labels_offline_only=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train36=True,validation12=True,controls8=True,actualRGBpairs=42,physics_steps=0)))
if __name__=='__main__':main()
