"""Reopen the delivered Blender scene and verify reconstructed camera transforms."""
import json,sys
from pathlib import Path
import bpy
from mathutils import Matrix,Vector

folder=Path(sys.argv[-1])
scene=bpy.context.scene
poses=json.loads((folder/'estimated.json').read_text())
assert scene.render.fps==500
for k,value in poses.items():
    if value is None:continue
    scene.frame_set(int(k))
    for name in 'LR':
        eye=bpy.data.objects[f'Optical reconstruction {name}']
        est=value['eyes'][name]
        assert (eye.matrix_world.translation*1000-Vector(est['pos_mm'])).length<.0002
        q=eye.matrix_world.to_quaternion()
        other=Matrix(est['rot']).to_quaternion()
        assert abs(q.dot(other))>1-2e-6
print('Verified both camera transforms on every solved frame.')
