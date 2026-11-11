"""Six saved state camera previews, no simulation."""
from pathlib import Path
import runpy
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
ns=runpy.run_path(str(HERE/'render_wide.py'),run_name='preview_import')['namespace']
mj=ns['mj']
def blocked(*args,**kwargs):raise RuntimeError('integration forbidden')
mj.mj_step=mj.mj_step1=mj.mj_step2=blocked
run=HERE.parent/'continuation3000/RECOVERY_EVAL_speed085'
import json
identity=json.loads((run/'IDENTITY.json').read_text())
core=ns['Core'](adhesion=False);ns['verify_runtime'](identity,core)
with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
 q=z['qpos'];v=z['qvel'];com=z['com'];t=z['time']
cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=14.;cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=(com.min(axis=0)+com.max(axis=0))/2
option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;option.geomgroup[:]=1
out=HERE/'WIDE_PREVIEW';out.mkdir(exist_ok=False)
renderer=mj.Renderer(core.plant,height=480,width=640)
montage=Image.new('RGB',(1280,1440))
for j,i in enumerate((0,1000,5780,5800,7300,15380)):
 core.d.qpos[:]=q[i];core.d.qvel[:]=v[i];core.d.time=t[i];mj.mj_forward(core.plant,core.d)
 assert np.array_equal(core.d.qpos,q[i]) and np.array_equal(core.d.qvel,v[i])
 renderer.update_scene(core.d,camera=cam,scene_option=option)
 image=Image.fromarray(renderer.render());ImageDraw.Draw(image).text((10,10),f'ACTUAL saved t={t[i]:.3f} / wide fixed camera14',fill='white')
 image.save(out/f'preview_{i:05d}.png');montage.paste(image,((j%2)*640,(j//2)*480))
renderer.close();montage.save(out/'MONTAGE.png');ns['verify_runtime'](identity,core)
print(out/'MONTAGE.png')
