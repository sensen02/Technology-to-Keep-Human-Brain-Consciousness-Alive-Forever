import bpy,sys
from pathlib import Path
scene=bpy.context.scene
folder=Path(sys.argv[-1]);folder.mkdir(exist_ok=True)
scene.cycles.samples=6;scene.cycles.use_denoising=False
scene.render.threads_mode='FIXED';scene.render.threads=6
for n in [40,213,300]:
    scene.frame_set(n)
    for c in sorted(bpy.data.collections['06   Recording cameras | 500 fps'].objects,key=lambda o:o.name):
        scene.camera=c;scene.render.filepath=str(folder/f'{n}_{c.name}.png')
        bpy.ops.render.render(write_still=True)
