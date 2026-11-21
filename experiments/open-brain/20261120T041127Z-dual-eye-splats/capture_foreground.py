"""Capture separated body photographs at each sampled instant for time-varying 3DGS."""
import json,math,sys
from pathlib import Path
from mathutils import Vector
import bpy

out=Path(sys.argv[-3]);shard=int(sys.argv[-2]);shards=int(sys.argv[-1])
prev=Path(__file__).resolve().parent.parent/'20261120T034750Z-dual-eye-observer'
poses=json.loads((prev/'estimated.json').read_text())
rig=json.loads((prev/'rig.json').read_text())
origin=Vector((.001,-.00045,1.0089690685272217));scale=.01
scene=bpy.context.scene
for ob in bpy.data.objects:
    if ob.type not in {'MESH','CURVE','SURFACE','FONT','META'}:continue
    if ob.name.startswith('P1 segment '):
        ob.hide_render=int(ob.name.split()[-1]) in rig['omit_eye_view']
    else:ob.hide_render=True
camdata=bpy.data.cameras.new('Calibrated fly survey')
camera=bpy.data.objects.new(camdata.name,camdata);scene.collection.objects.link(camera);scene.camera=camera
hw=224;fov=66.
focal=hw/(2*math.tan(math.radians(fov)/2))
camdata.sensor_fit='VERTICAL';camdata.sensor_height=32.;camdata.lens=camdata.sensor_height*focal/hw
camdata.clip_start=.00003
scene.render.resolution_x=hw;scene.render.resolution_y=hw;scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.render.threads_mode='FIXED';scene.render.threads=5;scene.cycles.samples=5;scene.cycles.use_denoising=True
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=-4
for k in list(poses)[shard::shards]:
    p=poses[k]
    scene.frame_set(int(k))
    target=Vector(p['pos_mm'])-.92*Vector(p['forward'])-Vector((0,0,.28))
    target/=1000
    folder=out/k
    (folder/'images').mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(12):
        theta=i*math.pi/6
        at=target+Vector((.0036*math.cos(theta),.0036*math.sin(theta),(.0002,.00135,.0025)[i%3]))
        q=(target-at).to_track_quat('-Z','Y')
        camera.location=at;camera.rotation_mode='QUATERNION';camera.rotation_quaternion=q
        a=q.to_matrix().to_4x4();a.translation=(at-origin)/scale
        path=f'images/view_{i:02}.png'
        rows.append(dict(file_path=path,transform_matrix=[list(x) for x in a],fl_x=focal,fl_y=focal,cx=hw/2,cy=hw/2,w=hw,h=hw))
        scene.render.filepath=str(folder/path)
        bpy.ops.render.render(write_still=True)
    (folder/'pose.json').write_text(json.dumps(dict(target=list((target-origin)/scale),frames=rows),indent=2))
