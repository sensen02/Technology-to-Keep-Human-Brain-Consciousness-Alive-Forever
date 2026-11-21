"""Add the underside fiducials and glass support to the derived Blender assembly."""
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
out=Path(bpy.path.abspath('//'))
geo=np.load(out/'geometry.npz')
x=geo['x_edges_mm'];y=geo['y_edges_mm'];z=geo['height_mm'];origin=geo['world_origin_m']
group=bpy.data.collections.new('09  Force floor | underside readout')
bpy.context.scene.collection.children.link(group)
def mesh(name,vertices,faces,color):
    data=bpy.data.meshes.new(name);data.from_pydata(vertices,[],faces);data.update()
    ob=bpy.data.objects.new(name,data);group.objects.link(ob)
    mat=bpy.data.materials.new(name);mat.diffuse_color=(*color,1);ob.data.materials.append(mat)
    return ob
v=[];faces=[]
for j,i in np.ndindex(len(y)-1,len(x)-1):
    cx=(x[i]+x[i+1])/2;cy=(y[j]+y[j+1])/2
    cz=z[j:j+2,i:i+2].mean()-.055
    first=len(v)
    for a in np.arange(8)*np.pi/4:
        v.append(origin+np.array([cx+.006*np.cos(a),cy+.006*np.sin(a),cz])/1000)
    faces.append(tuple(range(first,first+8)))
mesh('576 optical fiducials at rest',v,faces,(.91,.63,.20))
v=[origin+np.array([xx,yy,zz])/1000 for zz in (-.43,-.40) for yy in (y[0],y[-1]) for xx in (x[0],x[-1])]
base=mesh('Transparent carrier',v,[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)],(.80,.94,.98))
mat=base.data.materials[0];mat.use_nodes=True
shader=mat.node_tree.nodes.get('Principled BSDF')
shader.inputs['Transmission Weight'].default_value=1.;shader.inputs['Roughness'].default_value=.04
for sign in (-1,1):
    data=bpy.data.cameras.new('Bottom optical axis');data.type='ORTHO';data.ortho_scale=.009
    data.clip_start=.00002
    ob=bpy.data.objects.new(f'Bottom readout axis {sign:+d}',data);group.objects.link(ob)
    aim=origin+np.array([.002,0,-.0001])
    ob.location=Vector(aim+np.array([sign*.005,0,-.005]))
    ob.rotation_euler=(Vector(aim)-ob.location).to_track_quat('-Z','Y').to_euler()
group['readout']='Orthographic axes and unloaded fiducials. Dynamic microscope images are synthesized by acquire.py.'
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/'small_chamber_sensor.blend'))
