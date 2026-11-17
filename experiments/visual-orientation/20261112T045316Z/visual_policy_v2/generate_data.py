"""Parent-only imageID schedules; no image fabrication or additional rendering."""
from data import *
def main():
 cpu1();rgb,manifest=library()
 if (HERE/'DATASET.json').exists():raise FileExistsError('datasetalreadyfrozen')
 oldmeta=read(OLD/'DATASET.json');verify({str(OLD/'train.npz'):oldmeta['splits']['train']['sha256']});meta={}
 for name,n in (('train',36),('validation',12),('controls',8)):
  a,rows=split(name,manifest)
  if a['frame_ids'].shape!=(481,n) or a['target'].shape!=(1920,n) or np.max(abs(a['weights'].sum(0)-1))>1e-12:raise ValueError('new36/12/8visualschema')
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);meta[name]=dict(path=str(path),sha256=sha(path),rows=rows,sequences=n,used_actual_frame_ids=sorted(set(a['frame_ids'].ravel().tolist())))
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=meta,library=str(LIB/'LIBRARY.json'),library_sha256=sha(LIB/'LIBRARY.json'),rgb_sha256=sha(LIB/'RGB.npz'),parent_training_data=str(OLD/'train.npz'),parent_training_data_sha256=sha(OLD/'train.npz'),old_validation_now_development=True,RGB_NOT_COPIED=True,static_head_not_navigation=True,Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train36=True,validation12=True,controls8=True,actualRGBpairs=len(rgb),physics_steps=0)))
if __name__=='__main__':main()
