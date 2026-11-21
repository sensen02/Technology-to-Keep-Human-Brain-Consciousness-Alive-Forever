"""External gauge with known separations. Render with and without the removable balls."""
import colorsys
import json
from pathlib import Path
import runpy
import sys

import bpy
from mathutils import Matrix

source,out = map(Path, sys.argv[sys.argv.index('--')+1:])
out.mkdir(parents=True, exist_ok=True)
info = json.loads((source/'input.json').read_text())
scene = bpy.context.scene
for group in ('07   Optical test target | 2.8 mm','08b   Color markers | 18 landmarks'):
    bpy.data.collections[group].hide_render=True
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,-.0007,1.00955))
plate=bpy.context.object
plate.dimensions=(.009,.007,.0001)
mat=bpy.data.materials.new('Matte gauge surface')
mat.diffuse_color=(.30,.25,.20,1)
plate.data.materials.append(mat)
locations=[(-.0016,0.),(0.,0.),(.0016,0.),(-.0016,-.0014),(0.,-.0014),(.0016,-.0014)]
names=[n for n in info['colors'] if n.endswith('_tarsus_tip')]
gaps_um=[180,0,25,25,180,0]
balls=[]
for name,(x,y),gap in zip(names,locations,gaps_um):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=.000066, location=(x,y,1.0096+.000066+gap/1e6))
    ball=bpy.context.object
    ball.data.materials.append(bpy.data.materials['Synthetic reference | '+name])
    balls.append(ball)
W,H=info['frame_size']
scene.render.resolution_x,scene.render.resolution_y=W,H
scene.render.resolution_percentage=100
scene.render.threads_mode='FIXED'
scene.render.threads=6
scene.view_settings.view_transform='Standard'
scene.cycles.samples=8
scene.cycles.use_denoising=False
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.18
for ob in bpy.data.collections['08   Lighting'].objects:
    if ob.type=='LIGHT':
        ob.data.energy*=.74
(out/'marked').mkdir(exist_ok=True)
for name,p in info['projection'].items():
    cam=bpy.data.objects[name]
    intr=Matrix(p) @ cam.matrix_world @ Matrix.Diagonal((1.,-1.,-1.,1.))
    cam.data.lens=intr[0][0]/W*cam.data.sensor_width
    cam.data.shift_x=(W/2-intr[0][2])/W
    cam.data.shift_y=(intr[1][2]-H/2)/W
    scene.camera=cam
    scene.render.filepath=str(out/'marked'/('0_'+name+'.png'))
    bpy.ops.render.render(write_still=True)
for ball in balls:
    ball.hide_render=True
sys.argv[-1]=str(out/'empty')
runpy.run_path(str(Path(__file__).with_name('capture.py')),run_name='__main__')
(out/'gauge_reference.json').write_text(json.dumps(dict(zip(names,gaps_um)),indent=2))
