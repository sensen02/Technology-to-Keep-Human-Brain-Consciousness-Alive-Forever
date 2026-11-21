"""Native-state comparison of old and repaired controller; no simulated steps."""
from common import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

items=((PARENT/'HELD/block015_1723','Old: stopping lift FAIL'),
       (HERE/'DEV_ROUTING/block015_1723','Repaired: same case PASS'),
       (HERE/'DEV_ROUTING/block030_181','Repaired: 0.30mm FAIL retained'))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
bodies=[];sources={};fig,axes=plt.subplots(3,4,figsize=(16,9))
for row,(folder,label) in enumerate(items):
 m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
 with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 sources.update({str(folder/n):sha(folder/n) for n in ('TRACE.npz','BODY.mjb','RESULT.json')})
 m.mat_reflectance[:]=0
 for g in range(m.ngeom):
  body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
  if '/lm_' in body:m.geom_matid[g]=-1;m.geom_rgba[g]=[1.,.15,.05,1.]
 renderer=mujoco.Renderer(m,height=320,width=480);bodies.append((m,d,a,renderer))
 for col,t in enumerate((1.25,1.3362,1.36,1.50)):
  tick=round(t/DT)-1;d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d)
  cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][tick,0],0,.85];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
  renderer.update_scene(d,cam);axes[row,col].imshow(renderer.render());axes[row,col].axis('off');axes[row,col].set_title(f'{label} | {t:.4f}s',fontsize=9)
fig.suptitle('Saved physical states. Red = left middle leg (render-only color); no pose changes.')
fig.tight_layout();fig.savefig(HERE/'STOP_REPAIR_STATES.png',dpi=140);plt.close(fig)
video=HERE/'STOP_REPAIR.mp4';writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=7,macro_block_size=16);count=0
try:
 for tick in range(0,17000,50):
  panels=[]
  for (_,label),(m,d,a,renderer) in zip(items,bodies):
   d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d)
   cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][tick,0],0,.85];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
   renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
   draw.rectangle((0,0,480,64),fill=(20,20,20));draw.text((6,3),label,font=font,fill='white')
   draw.text((6,25),f't={(tick+1)*DT:.4f}s  0.125x | red=left middle',font=font,fill='white');panels.append(np.asarray(im))
  writer.append_data(np.concatenate(panels,axis=1));count+=1
finally:
 writer.close()
 for m,d,a,renderer in bodies:renderer.close()
fig,axes=plt.subplots(2,2,figsize=(12,8))
for folder,label in items[:2]:
 m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
 foot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'nmf/lm_tarsus5')
 with np.load(folder/'TRACE.npz',allow_pickle=False) as z:q=z['qpos'].copy();pos=z['position'].copy();initial=z['initial_position'].copy()
 footz=[]
 for pose in q:d.qpos[:]=pose;mujoco.mj_kinematics(m,d);footz.append(d.xpos[foot,2])
 t=(np.arange(len(q))+1)*DT;j=np.flatnonzero(m.jnt_limited);angles=q[:,m.jnt_qposadr[j]]
 ex=np.maximum(0,np.maximum(m.jnt_range[j,0]-angles,angles-m.jnt_range[j,1])).max(1)
 axes[0,0].plot(t,footz,label=label);axes[0,1].plot(t,pos[:,2]/initial[2]);axes[1,0].plot(t,ex);axes[1,1].plot(t,pos[:,0])
axes[0,0].set(ylabel='Left middle tarsus5 height (mm)');axes[0,0].legend(fontsize=8)
axes[0,1].set(ylabel='Thorax height / initial');axes[0,1].axhline(.7,ls='--',color='k')
axes[1,0].set(ylabel='Joint excess (rad)');axes[1,0].axhline(.02,ls='--',color='k')
axes[1,1].set(ylabel='Thorax x (mm)')
for ax in axes.flat:ax.set_xlabel('Time (s)');ax.axvspan(1.3,1.7,alpha=.08,color='gray');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'STOP_REPAIR_METRICS.png',dpi=150);plt.close(fig)
write(HERE/'MEDIA.json',dict(video=str(video),sha256=sha(video),frames=count,fps=25,duration_s=13.6,
 playback=.125,physical_sample_interval_s=.005,source_sha256=sources,physical_steps=0,
 render_only_changes='floor reflectance zero; left middle geoms red; no pose or geometry change',includes_old_and_retained_failure=True))
print('Rendered',count,'native-state comparison frames')
