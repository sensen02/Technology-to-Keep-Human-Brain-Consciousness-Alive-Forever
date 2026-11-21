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

sites=[(-4,-2.2,1.35),(-2,2.7,1.9),(.0,-2.65,1.35),(.8,2.4,1.85),(3.5,-2.65,1.7),
       (4.0,2.6,1.3),(6.0,-4.0,3.0),(-5.,4.4,3.8),(-15,-12,5.),(18,15,5.)]
for j,xyz in enumerate(sites):
    for n in range(8):
        offset_view(xyz,45*n+(j%3-1)*5,(-13,4,12)[(j+n)%3],(116,134,151)[(j+n)%3],
                    'train',f'near_{j:02}_{n}')
# Wider camera baselines constrain distant plants and walls.
for j,xyz in enumerate([(-50,-38,12),(-42,42,16),(55,-22,13),(65,43,22),(0,0,40),(-8,4,8)]):
    for n in range(6):
        offset_view(xyz,60*n+15,-25 if n%3==0 else -4,122,'train',f'wide_{j:02}_{n}')
for i,xyz in enumerate([(-1.4,-1.4,1.12),(2.,1.3,1.16),(-3.,.4,1.48),(5.,-.9,1.3)]):
    for j,heading in enumerate((60,-62)):
        offset_view(xyz,heading+5*i,-3+2*i,157,'holdout',f'hold_{i}_{j}',(400,352))
poses=json.loads((eye_source/'estimated.json').read_text())
for k in [1,127,211,315,317,318]:
    for side in 'LR':
        eye=poses[str(k)]['eyes'][side]
        xyz=Vector(eye['pos_mm'])-origin*1000
        view('eyecheck',xyz,Matrix(eye['rot']),157,(512,450),f'eye_{k}_{side}')

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
