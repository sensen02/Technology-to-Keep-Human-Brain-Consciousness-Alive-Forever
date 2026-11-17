"""Generate72descriptors referencing56actual frames; no augmented pixels stored."""
from data import *
def main():
 cpu1();rgb,m=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('datasetalreadycreated')
 splits={}
 for name,n in (('train',72),('validation',12),('controls',8)):
  a,metadata=split(name,m)
  if a['frame_ids'].shape!=(481,n) or a['target'].shape!=(1920,n) or a['mirror_flags'].shape!=(n,):raise ValueError('mirror schedule shape')
  if not np.all(np.isfinite(a['weights'])) or np.max(abs(a['weights'].sum(0)-1))>1e-12:raise ValueError('weights sum1')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);splits[name]=dict(path=str(path),sha256=sha(path),rows=metadata,sequences=n,used_actual_frame_ids=sorted(set(a['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=splits,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),RGB_NOT_COPIED=True,digital_mirror_descriptors=36,actual_raw_RGB_pairs=56,digital_mirror_not_new_exposure=True,oldV4V5_development=True,all_scene_labels_offline_only=True,static_head_not_navigation=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train72=True,validation12=True,controls8=True,actualRawPairs56=True,digitalAugmentationNotExposure=True,physical_steps=0)))
if __name__=='__main__':main()
