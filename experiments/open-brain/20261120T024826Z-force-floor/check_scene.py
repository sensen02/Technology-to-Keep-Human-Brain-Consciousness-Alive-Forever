import sys
from pathlib import Path
import bpy
import numpy as np
out=Path(sys.argv[-1]);scene=bpy.context.scene
tr=np.load(out/'insert/optical_scene.npz');m=np.load(out/'insert/render_mesh.npz')
origin=np.load(out/'geometry.npz')['world_origin_m']
pose=np.load(out/'insert/tile_motion.npz')['depth_um']
diff=[]
for idx in (0,41,212,314):
    scene.frame_set(idx+1)
    ob=bpy.data.objects['Loaded replica cells']; ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    raw=np.array([v.co.z for v in ob.data.vertices]);moved=np.array([v.co.z for v in ev.data.vertices])
    diff.append(float(np.max(abs((raw-moved)*1e6-np.repeat(pose[idx].ravel(),4)))))
    for g,name in zip(m['foot_ids'],'LF LM LH RF RM RH'.split()):
        expected=origin+tr['transforms'][idx,np.where(tr['gids']==g)[0][0],:3]/1000
        tip=bpy.data.objects['Observed marker '+name]
        assert np.linalg.norm(np.array(tip.matrix_world.translation)-expected)<1e-6
print('surface residual µm',diff,flush=True)
assert max(diff)<.02
assert scene.render.fps==500 and scene.frame_end==315
scene.frame_set(213)
scene.cycles.samples=10;scene.render.threads_mode='FIXED';scene.render.threads=5
scene.render.filepath=str(out/'WALK_FRAME.png')
bpy.ops.render.render(write_still=True)
