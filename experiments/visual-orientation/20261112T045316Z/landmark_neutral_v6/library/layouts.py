"""Stdlibpredeclarednew56configposeexpansion only."""
def new_configs(pr):
 return [dict(g,pose_id=pr['library']['pose_ids'][i],saved_pose_index=index,split='validation') for g in pr['library']['new_geometries'] for i,index in enumerate(pr['library']['new_pose_indices'])]
