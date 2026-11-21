"""Viewer for the measured walking run. Cameras/observer use separate read-only recordings."""
from pathlib import Path
import json,struct,sys
import bpy
import numpy as np
from mathutils import Matrix,Vector
src,out=map(Path,sys.argv[-2:])
scene=bpy.context.scene
for group in ('07   Optical test target | 2.8 mm','08b   Color markers | 18 landmarks'):
    col=bpy.data.collections[group];col.hide_render=True;col.hide_viewport=True
record=np.load(out/'insert/optical_scene.npz')
mesh=np.load(out/'insert/render_mesh.npz')
geo=np.load(out/'geometry.npz')
origin=geo['world_origin_m']
tr=record['transforms'];nt=len(tr)
group=bpy.data.collections.new('10  Loaded P1 walking trial')
scene.collection.children.link(group)
def material(name,color):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);return m
cuticle=material('P1 body surface',(.19,.085,.037))
lookup={g:i for i,g in enumerate(record['gids'])}
parts={}
for g,vertices,faces in zip(mesh['gids'],np.split(mesh['vertices'],np.cumsum(mesh['vcounts'])[:-1]),np.split(mesh['faces'],np.cumsum(mesh['fcounts'])[:-1])):
    v=bpy.data.meshes.new(f'P1 segment {g}')
    v.from_pydata(vertices/1000,[],faces);v.update()
    ob=bpy.data.objects.new(v.name,v);group.objects.link(ob);parts[g]=ob
    ob.data.materials.append(cuticle);ob.rotation_mode='QUATERNION'
    last=None
    for n,s in enumerate(tr[:,lookup[g]]):
        ob.location=origin+s[:3]/1000;q=Matrix(s[3:].reshape(3,3)).to_quaternion()
        if last is not None:q.make_compatible(last)
        ob.rotation_quaternion=q;last=q.copy()
        for key in ('location','rotation_quaternion'): ob.keyframe_insert(key,frame=n+1)
for g,name in zip(mesh['foot_ids'],'LF LM LH RF RM RH'.split()):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=.000066)
    ob=bpy.context.object;ob.name='Observed marker '+name;ob.parent=parts[g]
    ob.location=(0,0,0)
    ink=bpy.data.materials['Synthetic reference | '+name+'_tarsus_tip']
    for node in ink.node_tree.nodes:
        if node.type=='EMISSION':node.inputs['Strength'].default_value=16.
    ob.data.materials.append(ink)
shape=np.load(out/'insert/tile_motion.npz')['depth_um']
x=geo['x_edges_mm'];y=geo['y_edges_mm'];h=geo['height_mm']
top=bpy.data.objects['Removable sand replica | 8 x 4.5 mm']
top.hide_render=True;top.hide_viewport=True
vertices=[];faces=[]
for j,i in np.ndindex(shape.shape[1:]):
    tile=h[j:j+2,i:i+2];z=tile.mean()
    cx=(x[i]+x[i+1])/2;cy=(y[j]+y[j+1])/2
    sx=np.diff(tile,axis=1).mean()/.25;sy=np.diff(tile,axis=0).mean()/.25
    off=len(vertices)
    for dx,dy in ((-.125,-.125),(.125,-.125),(.125,.125),(-.125,.125)):
        vertices.append(np.array([cx+dx,cy+dy,z+dx*sx+dy*sy])/1000)
    faces.append((off,off+1,off+2,off+3))
v=bpy.data.meshes.new('Loaded replica cells');v.from_pydata(vertices,[],faces);v.update()
surface=bpy.data.objects.new(v.name,v);group.objects.link(surface);surface.location=origin
surface.data.materials.append(top.data.materials[0])
def animate_mesh(ob,filename,baseline,per_cell):
    with (out/filename).open('wb') as f:
        f.write(struct.pack('<12siiffi',b'POINTCACHE2\0',1,len(baseline),0.,1.,nt))
        for frame in shape:
            a=baseline.copy();a[:,2]-=np.repeat(frame.ravel(),per_cell)/1e6
            f.write(np.asarray(a,dtype='<f4').tobytes())
    mod=ob.modifiers.new('500 Hz measured surface motion','MESH_CACHE')
    mod.cache_format='PC2';mod.filepath='//'+filename;mod.frame_start=1
animate_mesh(surface,'surface.pc2',np.array(vertices),4)
dots=bpy.data.objects['576 optical fiducials at rest']
# Recenter large world coordinates before caching micrometre-scale motion.
vertices=np.array([list(v.co) for v in dots.data.vertices])-origin
for vertex,p in zip(dots.data.vertices,vertices):vertex.co=p
dots.location=origin
animate_mesh(dots,'fiducials.pc2',vertices,8)
info=json.loads((src/'input.json').read_text())
W,H=info['frame_size']
for name,p in info['projection'].items():
    cam=bpy.data.objects[name];k=Matrix(p)@cam.matrix_world@Matrix.Diagonal((1.,-1.,-1.,1.))
    cam.data.lens=k[0][0]/W*cam.data.sensor_width;cam.data.shift_x=(W/2-k[0][2])/W;cam.data.shift_y=(k[1][2]-H/2)/W
scene.camera=bpy.data.objects['H02_NW']
scene.render.resolution_x,scene.render.resolution_y=W,H
scene.render.resolution_percentage=100;scene.render.fps=500;scene.frame_start=1;scene.frame_end=nt
scene.frame_set(213);scene.view_settings.exposure=-4
scene['trial']='Frame 1 = 2 ms; axial displacements in metres, native scale'
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/'walking_sensor.blend'))
