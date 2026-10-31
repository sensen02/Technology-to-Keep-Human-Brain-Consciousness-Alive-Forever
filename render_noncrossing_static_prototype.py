"""Static front/top views of candidate footprint; never integrates physics."""
from pathlib import Path
import os,json
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from flygym.compose.fly.musculoskeletal import _load_mjcf
O=Path('/home/sensen/Desktop/engine修复验收/s5-noncrossing-static-prototype/20261028T041000Z')
def main():
 m=_load_mjcf('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z/INTEGRATED_MODEL.xml').compile();d=mujoco.MjData(m);d.qpos[:]=np.load(O/'STATIC_STANCE.npz')['qpos'];mujoco.mj_forward(m,d)
 mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71;m.geom_rgba[mesh,3]=1;m.vis.global_.offwidth=max(800,m.vis.global_.offwidth);m.vis.global_.offheight=max(700,m.vis.global_.offheight)
 opt=mujoco.MjvOption();opt.geomgroup[:]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
 renderer=mujoco.Renderer(m,height=700,width=800);panels=[]
 for label,az,el in [('FRONT',180,-12),('TOP',90,-89)]:
  cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,1.];cam.distance=9.;cam.azimuth=az;cam.elevation=el;renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im);dr.rectangle((0,0,799,45),fill='white');dr.text((10,10),'STATIC IK PROTOTYPE / '+label+' / NOT WALKING',fill='black');panels.append(np.asarray(im))
 renderer.close();imageio.imwrite(O/'STATIC_FRONT_TOP.png',np.concatenate(panels,axis=1));(O/'RENDER_CHECK.json').write_text(json.dumps(dict(model_load=1,forward=1,physics_steps=0,neural_steps=0,mesh_count=71,all_meshes_opaque=True,views=['front','top']),indent=2)+'\n');print(O/'STATIC_FRONT_TOP.png')
if __name__=='__main__':main()
