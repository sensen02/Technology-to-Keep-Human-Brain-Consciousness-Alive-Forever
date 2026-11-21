"""Capture calibrated RGB scan photos. Export camera metadata only."""
import json,math,sys
from pathlib import Path
from mathutils import Matrix,Vector
import bpy

out=Path(sys.argv[-3]);shard=int(sys.argv[-2]);shards=int(sys.argv[-1])
eye_source=Path(__file__).resolve().parent.parent/'20261120T034750Z-dual-eye-observer'
origin=Vector((.001,-.00045,1.0089690685272217))
scale=.01
scene=bpy.context.scene
for name in ['07   Optical test target | 2.8 mm','08b   Color markers | 18 landmarks']:
    bpy.data.collections[name].hide_render=True
data=bpy.data.cameras.new('Calibrated RGB survey')
camera=bpy.data.objects.new(data.name,data);scene.collection.objects.link(camera);scene.camera=camera
data.sensor_fit='VERTICAL';data.sensor_height=32.;data.clip_start=.00004;data.clip_end=8
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.render.threads_mode='FIXED';scene.render.threads=6
scene.cycles.samples=10;scene.cycles.use_denoising=True;scene.cycles.seed=307
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=-4
scene.frame_set(213)
rows=[]
def view(split,xyz,rot,fov,hw,key):
    transform=rot.to_4x4();transform.translation=Vector([v/1000 for v in xyz])/scale
    rows.append(dict(split=split,file_path=f'images/{key}.png',transform_matrix=[list(v) for v in transform],
                     raw_world_xyz_mm=list(xyz),fov_y=fov,h=hw[0],w=hw[1],
                     fl_y=hw[0]/(2*math.tan(math.radians(fov)/2)),fl_x=hw[0]/(2*math.tan(math.radians(fov)/2)),
                     cx=hw[1]/2,cy=hw[0]/2))
def offset_view(xyz,yaw,pitch,fov,split,key,hw=(300,320)):
    a,p=map(math.radians,(yaw,pitch))
    axis=Vector((math.cos(a)*math.cos(p),math.sin(a)*math.cos(p),math.sin(p)))
    rot=axis.to_track_quat('-Z','Y').to_matrix()
    view(split,xyz,rot,fov,hw,key)

sites=[(-2.95,-1.3,.91),(-2.50,1.45,1.06),(-.85,1.70,.87),(.05,-1.50,1.10),
       (1.9,1.6,1.18),(2.7,-1.5,.88),(3.6,1.4,1.015),(.7,1.9,1.4)]
for j,xyz in enumerate(sites):
    for n,heading in enumerate([26,47,70,95,-27,-49,-73,-98]):
        offset_view(xyz,heading,[-11.,0.,11.][(j+n)%3],157,'train',f'low_{j}_{n}',(360,340))

out.mkdir(exist_ok=True)
(out/'images').mkdir(exist_ok=True)
if shard==0:
    (out/'calibration.json').write_text(json.dumps(dict(origin_m=list(origin),unit_m=scale,frames=rows),indent=2))
for i,row in enumerate(rows):
    if i%shards!=shard:continue
    transform=Matrix(row['transform_matrix'])
    transform.translation=transform.translation*scale+origin
    camera.matrix_world=transform
    data.lens=data.sensor_height/(2*math.tan(math.radians(row['fov_y'])/2))
    scene.render.resolution_x=row['w'];scene.render.resolution_y=row['h']
    scene.render.filepath=str(out/row['file_path'])
    bpy.ops.render.render(write_still=True)
