import bpy, json,sys
from pathlib import Path
from mathutils import Matrix
src,out=map(Path,sys.argv[-2:])
out.mkdir(exist_ok=True)
scene=bpy.context.scene
for name in ('07   Optical test target | 2.8 mm','08b   Color markers | 18 landmarks'):
    bpy.data.collections[name].hide_render=True
info=json.loads((src/'input.json').read_text())
W,H=info['frame_size']
scene.render.resolution_x,scene.render.resolution_y=W,H
scene.render.resolution_percentage=100
scene.cycles.samples=4
scene.render.threads_mode='FIXED'
scene.render.threads=5
scene.view_settings.view_transform='Standard'
scene.view_settings.exposure=-4
scene.cycles.samples=12
for name,p in info['projection'].items():
    cam=bpy.data.objects[name]
    k=Matrix(p) @ cam.matrix_world @ Matrix.Diagonal((1.,-1.,-1.,1.))
    cam.data.lens=k[0][0]/W*cam.data.sensor_width
    cam.data.shift_x=(W/2-k[0][2])/W
    cam.data.shift_y=(k[1][2]-H/2)/W
    scene.camera=cam
    scene.render.filepath=str(out/(name+'.png'))
    bpy.ops.render.render(write_still=True)
