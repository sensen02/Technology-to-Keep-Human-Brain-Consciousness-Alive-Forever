"""Evaluation only. Run after freezing the camera predictions."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

src,folder=map(Path,sys.argv[sys.argv.index('--')+1:])
digest=hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest()
assert digest==(folder/'predictions.sha256').read_text().split()[0]
rows=list(csv.DictReader((folder/'predictions.csv').open()))
truth=json.loads((src/'truth.json').read_text())
xy=[tuple(truth[r['frame']][r['leg']+'_tarsus_tip'][:2]) for r in rows]
region=[(min(v[i] for v in xy)-.003,max(v[i] for v in xy)+.003) for i in range(2)]
verts,faces=[],[]
dg=bpy.context.evaluated_depsgraph_get()
for ob in bpy.data.collections['04   Sand, soil & vegetation'].all_objects:
    if ob.type!='MESH' or ob.hide_render:
        continue
    obj=ob.evaluated_get(dg)
    corners=[obj.matrix_world@Vector(p) for p in obj.bound_box]
    if any(max(p[i] for p in corners)<lo or min(p[i] for p in corners)>hi
           for i,(lo,hi) in enumerate(region)):
        continue
    mesh=obj.to_mesh()
    base=len(verts)
    verts.extend(obj.matrix_world@v.co for v in mesh.vertices)
    faces.extend(tuple(base+i for i in poly.vertices) for poly in mesh.polygons)
    obj.to_mesh_clear()
tree=BVHTree.FromPolygons(verts,faces)
def below(x,y):
    point,normal,_,_=tree.ray_cast(Vector((x,y,1.020)),Vector((0,0,-1)),.04)
    if point is None:
        raise RuntimeError('No reference surface')
    return point.z, abs(normal.z)
ref=[]
for r in rows:
    bpy.context.scene.frame_set(int(r['frame']))
    leg=r['leg']
    x,y,z=truth[r['frame']][leg+'_tarsus_tip']
    gz,nz=below(x,y)
    ground_pred_xy=below(float(r['toe_x_mm'])/1000,float(r['toe_y_mm'])/1000)[0]
    rod=bpy.data.objects[leg+' | tarsus'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    vertices=[rod.matrix_world@v.co for v in rod.data.vertices]
    rod_tree=BVHTree.FromPolygons(vertices,[tuple(p.vertices) for p in rod.data.polygons])
    intersects=bool(rod_tree.overlap(tree))
    rod_gap=min((v.z-below(v.x,v.y)[0])*1000 for v in vertices)
    ref.append(dict(frame=int(r['frame']),leg=leg,ground_at_estimated_xy_mm=ground_pred_xy*1000,
                    marker_gap_mm=(z-gz)*1000*nz-.066,tarsus_min_vertex_vertical_mm=rod_gap,tarsus_surface_intersection=intersects))
(folder/'geometry_reference.json').write_text(json.dumps(ref,indent=2))
