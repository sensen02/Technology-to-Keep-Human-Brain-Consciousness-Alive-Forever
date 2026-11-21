"""Capture a removable projected speckle pattern, with an empty chamber and its glass intact."""
import json
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

source, output = map(Path, sys.argv[sys.argv.index('--') + 1:])
output.mkdir(parents=True, exist_ok=True)
info = json.loads((source / 'input.json').read_text())
scene = bpy.context.scene
for group in ('07   Optical test target | 2.8 mm', '08b   Color markers | 18 landmarks'):
    bpy.data.collections[group].hide_render = True
for lamp in bpy.data.lights:
    lamp.energy = 0
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
# A physical perforated slide between a point light and the sand. Dimensions in metres.
lamp = bpy.data.lights.new('Calibration point source', 'POINT')
lamp.energy = .022
lamp.shadow_soft_size = .000009
light = bpy.data.objects.new(lamp.name, lamp)
scene.collection.objects.link(light)
light.location = (0, 0, 1.035)
verts, faces = [], []
rng = random.Random(9801)
pitch = .000086
for i in range(-48, 48):
    for j in range(-48, 48):
        if rng.random() < .5:
            k = len(verts)
            verts.extend([(x*pitch, y*pitch, 1.022) for x,y in
                          [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]])
            faces.append((k,k+1,k+2,k+3))
slide = bpy.data.meshes.new('Perforated slide')
slide.from_pydata(verts, [], faces)
slide.update()
slide_ob = bpy.data.objects.new(slide.name, slide)
scene.collection.objects.link(slide_ob)
scene.render.resolution_x, scene.render.resolution_y = info['frame_size']
scene.render.resolution_percentage = 100
scene.render.threads_mode = 'FIXED'
scene.render.threads = 6
scene.view_settings.view_transform = 'Standard'
scene.cycles.samples = 32
scene.cycles.use_denoising = False
W,H = info['frame_size']
for name,p in info['projection'].items():
    if not name.startswith('H'):
        continue
    cam = bpy.data.objects[name]
    intr = Matrix(p) @ cam.matrix_world @ Matrix.Diagonal((1.,-1.,-1.,1.))
    cam.data.lens = intr[0][0] / W * cam.data.sensor_width
    cam.data.shift_x = (W/2 - intr[0][2])/W
    cam.data.shift_y = (intr[1][2] - H/2)/W
    scene.camera = cam
    scene.render.filepath = str(output / (name+'.png'))
    bpy.ops.render.render(write_still=True)
