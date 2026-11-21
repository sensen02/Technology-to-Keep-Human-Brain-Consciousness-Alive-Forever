"""Derive a removable insert from the same visible sand. Leave original project intact."""
from pathlib import Path
import sys
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
out=Path(sys.argv[-1])
sand=bpy.data.objects['Continuous sand and soil height mesh']
origin=np.array([-.0009,-.00045])
bounds=np.array([[-2,6],[-2.25,2.25]])/1000+origin[:,None]
world=sand.matrix_world.copy()
bm=bmesh.new()
bm.from_mesh(sand.data)
bm.transform(world)
tree=BVHTree.FromBMesh(bm)
def depth(x,y):
    return tree.ray_cast(Vector((x,y,1.02)),Vector((0,0,-1)))[0].z
mid=depth(*origin)
pitch=.25
xx=np.arange(-2,6+pitch/2,pitch)
yy=np.arange(-2.25,2.25+pitch/2,pitch)
h=np.array([[1000*(depth(origin[0]+x/1000,origin[1]+y/1000)-mid) for x in xx] for y in yy])
np.savez(out/'geometry.npz',x_edges_mm=xx,y_edges_mm=yy,height_mm=h,world_origin_m=[*origin,mid])
for axis in range(2):
    for limit in bounds[axis]:
        point=[0.,0.,0.]; point[axis]=limit
        normal=[0.,0.,0.]; normal[axis]=1.
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                              plane_co=point,plane_no=normal,dist=1e-9)
def inside(f):
    c=f.calc_center_median()
    return all(bounds[i,0]<c[i]<bounds[i,1] for i in range(2))
sheet=bm.copy()
for mesh,selected in ((sheet,False),(bm,True)):
    bmesh.ops.delete(mesh,geom=[f for f in mesh.faces if inside(f)==selected],context='FACES')
    mesh.transform(world.inverted())
top=sand.copy()
top.data=sand.data.copy()
top.name='Removable sand replica | 8 x 4.5 mm'
sand.users_collection[0].objects.link(top)
for mesh,obj in ((sheet,top),(bm,sand)):
    mesh.to_mesh(obj.data); mesh.free()
# Keep original procedural material coordinates after cutting the shared mesh.
top.data.use_auto_texspace=False
original=bpy.data.objects['Continuous sand and soil height mesh']
top.data.texspace_location=original.data.texspace_location
top.data.texspace_size=original.data.texspace_size
top['mechanics']='geometry.npz supplies the unloaded surface to the separate force-floor simulator'
bpy.ops.wm.save_as_mainfile(filepath=str(out/'small_chamber_sensor.blend'))
