"""Render world exposure from the optical pose. The scoring mode is a separate invocation."""
import json,os,sys
from pathlib import Path
from math import radians,tan
import bpy
from mathutils import Matrix,Vector

out=Path(sys.argv[-1])
check=os.environ.get('EYE_REFERENCE')=='1'
pilot=os.environ.get('EYE_PILOT')=='1'
rig=json.loads((out/'rig.json').read_text())
poses=json.loads((out/('reference.json' if check else 'estimated.json')).read_text())
scene=bpy.context.scene
bpy.data.collections['11  Head tracking landmarks'].hide_render=True
for i in rig['omit_eye_view']:
    ob=bpy.data.objects.get(f'P1 segment {i}')
    if ob:ob.hide_render=True
h,w=rig['image_hw']
scene.render.resolution_y=h;scene.render.resolution_x=w;scene.render.resolution_percentage=100
scene.render.threads_mode='FIXED';scene.render.threads=7;scene.cycles.samples=12;scene.cycles.use_denoising=True
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=-4
scene.cycles.seed=112
scene.cycles.use_animated_seed=False
cams={}
for eye in rig['eyes']:
    side=eye['name']
    data=bpy.data.cameras.new(f'Optical reconstruction {side}')
    data.sensor_fit='VERTICAL';data.sensor_height=32.
    data.lens=data.sensor_height/(2*tan(radians(rig['fov_y_deg']/2)))
    data.clip_start=.00001;data.clip_end=8.
    ob=bpy.data.objects.new(data.name,data);scene.collection.objects.link(ob);cams[side]=ob
    ob.rotation_mode='QUATERNION'
    previous=None
    for frame,pose in sorted(poses.items(),key=lambda kv:int(kv[0])):
        if pose is None:continue
        if check:
            r=Matrix(pose['rot'])
            p=Vector(pose['pos_mm'])+r@Vector(eye['pos_mm'])
            q=(r@Matrix(eye['rot'])).to_quaternion()
        else:
            p=Vector(pose['eyes'][side]['pos_mm'])
            q=Matrix(pose['eyes'][side]['rot']).to_quaternion()
        ob.location=p/1000
        if previous is not None:q.make_compatible(previous)
        ob.rotation_quaternion=q;previous=q.copy()
        for attribute in ('location','rotation_quaternion'):
            ob.keyframe_insert(attribute,frame=int(frame))
        ob['valid_frame_policy']='Only frames listed as valid in estimated.json carry measured poses'
folder=out/('eye_reference' if check else 'eye_pilot' if pilot else 'eyes')
folder.mkdir(exist_ok=True)
for frame,pose in sorted(poses.items(),key=lambda kv:int(kv[0])):
    if pose is None:continue
    n=int(frame)
    if check and n not in {1,127,211,315,316,317,318,319}:continue
    if pilot and n!=211:continue
    scene.frame_set(n)
    for side,cam in cams.items():
        scene.camera=cam
        scene.render.filepath=str(folder/f'{n}_{side}.png')
        bpy.ops.render.render(write_still=True)
if not (check or pilot):
    scene.camera=cams['L'];scene.frame_set(211)
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'reconstructed_eyes.blend'),relative_remap=False)
