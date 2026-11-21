"""Prescribed moving exterior for the existing keyframed fly, used only for scene generation."""
import sys
from pathlib import Path
import bpy
import numpy as np
out=Path(sys.argv[-1]); scene=bpy.context.scene
legs=('LF','LM','LH','RF','RM','RH')
names=[f'{leg} | {part}' for leg in legs for part in ('femur','tibia','tarsus')]
dots=[f'Tracking dot | {leg}_{part}' for leg in legs for part in ('femur_tibia','tibia_tarsus','tarsus_tip')]
objects=[bpy.data.objects[n] for n in names+dots]
positions=[];orient=[];sizes=[]
for frame in range(450,521):
    scene.frame_set(frame)
    loc=[];quat=[];scale=[]
    for o in objects:
        t,q,s=o.matrix_world.decompose()
        loc.append(list(t));quat.append(list(q));scale.append(list(s))
    positions.append(loc);orient.append(quat);sizes.append(scale)
np.savez_compressed(out/'exterior_keyframes.npz',pos=positions,quat=orient,scale=sizes)
