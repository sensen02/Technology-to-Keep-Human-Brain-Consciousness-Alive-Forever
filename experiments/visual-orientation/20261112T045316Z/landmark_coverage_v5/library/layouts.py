"""Stdlib fixedgeometry/actualpose expansion; no images or policy operations."""
def library_configs(pr):
 lib=pr['library'];rows=[]
 for split,geometries,poses,prefix in [('train',lib['train_geometries'],lib['train_pose_indices'],'trainpose'),('validation',lib['validation_geometries'],lib['validation_pose_indices'],'valpose')]:
  for geom in geometries:
   for j,index in enumerate(poses):rows.append(dict(geom,split=split,pose_id=prefix+str(j),saved_pose_index=index))
 return rows
