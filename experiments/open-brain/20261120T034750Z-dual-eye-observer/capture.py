"""Render synchronized exterior RGB and write a sealed reference for later scoring."""
import json,os,sys
from pathlib import Path
import bpy
from mathutils import Matrix,Vector,Euler

cal,out=map(Path,sys.argv[-2:])
rig=json.loads((out/'rig.json').read_text())
old=json.loads(cal.read_text())
scene=bpy.context.scene
head=bpy.data.objects[f"P1 segment {rig['head_geom']}"]
collection=bpy.data.collections.new('11  Head tracking landmarks')
scene.collection.children.link(collection)
for i,(pos,color) in enumerate(zip(rig['tags_mm'],rig['colors'])):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=rig['tag_radius_mm']/1000)
    ob=bpy.context.object
    for col in list(ob.users_collection):col.objects.unlink(ob)
    collection.objects.link(ob)
    ob.name=f'Head tag {i+1}';ob.parent=head;ob.location=Vector(pos)/1000
    mat=bpy.data.materials.new(ob.name);mat.use_nodes=True
    def linear(c):
        f=c/255
        return (f+.055)**2.4/1.055**2.4 if f>.04045 else f/12.92
    tree=mat.node_tree
    shader=tree.nodes.new('ShaderNodeEmission')
    shader.inputs[0].default_value=(*map(linear,color),1);shader.inputs[1].default_value=16
    tree.links.new(shader.outputs[0],tree.nodes.get('Material Output').inputs[0])
    ob.data.materials.append(mat)
for ob in bpy.data.objects:
    if ob.name.startswith('Observed marker '):ob.hide_render=True
# Independent head turns on copies of the same body posture.
scene.frame_set(213)
parts={ob:ob.matrix_world.copy() for ob in bpy.data.objects if ob.name.startswith('P1 segment ')}
pivot=head.matrix_world.translation.copy()
for frame,ang in [(316,(0,0,0)),(317,(0,0,25)),(318,(0,0,-25)),(319,(14,16,0))]:
    change=Euler([a*3.141592653589793/180 for a in ang]).to_matrix().to_4x4()
    around=Matrix.Translation(pivot)@change@Matrix.Translation(-pivot)
    for ob,pose in parts.items():
        gid=int(ob.name.split()[-1])
        ob.matrix_world=around@pose if gid in rig['head_children'] else pose
        for prop in ('location','rotation_quaternion'):ob.keyframe_insert(prop,frame=frame)
scene.frame_end=319
# Preserve physical floor height for added control frames by clamping the cache range.
w,h=old['frame_size']
scene.render.resolution_x=w;scene.render.resolution_y=h
scene.render.resolution_percentage=75
scene.render.threads_mode='FIXED';scene.render.threads=7
scene.cycles.samples=4
scene.cycles.use_denoising=False
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=-4
frames=[1,22,43,64,85,106,127,148,169,190,211,232,253,274,295,315,316,317,318,319]
projections={name:[[v*(.75 if row<2 else 1) for v in values] for row,values in enumerate(p)] for name,p in old['projection'].items()}
(out/'observations.json').write_text(json.dumps(dict(frames=frames,native_fps=500,projections=projections,hw=[336,504]),indent=2))
scene.frame_set(211);scene.camera=bpy.data.objects['H02_NW']
# Keep original PC2 files available when opening the derived scene.
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/'tagged_head.blend'),relative_remap=False)
images=out/'external';images.mkdir(exist_ok=True)
ref={}
preflight=os.environ.get('HEAD_PREFLIGHT')=='1'
for k in ([211,317] if preflight else frames):
    scene.frame_set(k)
    ref[k]={'pos_mm':[1000*v for v in head.matrix_world.translation],
            'rot':[list(r) for r in head.matrix_world.to_3x3()]}
    for name in projections:
        scene.camera=bpy.data.objects[name]
        scene.render.filepath=str(images/f'{k}_{name}.png')
        bpy.ops.render.render(write_still=True)
(out/'reference.json').write_text(json.dumps(ref,indent=2))
