"""Same-run native state rendering; includes the held-out failure."""
from shared import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
items=(('flat_1151','flat: PASS'),('block015_1151','0.15mm: PASS'),('block030_1151','0.30mm: PASS'),
       ('block015_1723','0.15mm: FAIL height/joint'))
bodies=[];sources={};fig,axes=plt.subplots(4,4,figsize=(16,12))
for row,(label,title) in enumerate(items):
    folder=HERE/'HELD'/label
    m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
    with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    for name in ('TRACE.npz','BODY.mjb','RESULT.json'):sources[str(folder/name)]=sha(folder/name)
    m.mat_reflectance[:]=0.;renderer=mujoco.Renderer(m,height=320,width=480)
    bodies.append((m,d,a,renderer))
    for col,t in enumerate((.25,.60,.95,1.50)):
        index=round(t/DT)-1;d.qpos[:]=a['qpos'][index];mujoco.mj_kinematics(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][index,0],0,.8];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
        renderer.update_scene(d,cam);axes[row,col].imshow(renderer.render());axes[row,col].axis('off')
        axes[row,col].set_title(f'{title} | t={t:.2f}s',fontsize=10)
fig.suptitle('Student sensory ANN + fitted readouts/local CPG. Saved physical states; teacher calls = 0.',fontsize=13)
fig.tight_layout();fig.savefig(HERE/'STUDENT_STATES.png',dpi=130);plt.close(fig)
video=HERE/'STUDENT_HELD.mp4';count=0
writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=7,macro_block_size=16)
try:
 for tick in range(0,17000,50):
    panels=[]
    for (_,title),(m,d,a,renderer) in zip(items,bodies):
        d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][tick,0],0,.8];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
        renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,480,61),fill=(20,20,20));draw.text((6,3),f'Student ANN / local CPG | {title}',font=font,fill='white')
        draw.text((6,26),f't={(tick+1)*DT:.4f}s   playback 0.125x',font=font,fill='white')
        panels.append(np.asarray(im))
    writer.append_data(np.concatenate(panels,axis=1));count+=1
finally:
 writer.close()
 for m,d,a,renderer in bodies:renderer.close()
fig,axes=plt.subplots(2,2,figsize=(12,8))
for path in sorted((HERE/'HELD').glob('*/TRACE.npz')):
    with np.load(path,allow_pickle=False) as a:
        m=mujoco.MjModel.from_binary_path(str(path.parent/'BODY.mjb'));j=np.flatnonzero(m.jnt_limited)
        q=a['qpos'][:,m.jnt_qposadr[j]];ex=np.maximum(0,np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1])).max(1)
        t=(np.arange(len(q))+1)*DT
        axes[0,0].plot(t,a['position'][:,0],label=path.parent.name)
        axes[0,1].plot(t,a['position'][:,2]/a['initial_position'][2])
        axes[1,0].plot(t,ex);axes[1,1].plot(t,np.linalg.norm(a['qvel'][:,:2],axis=1))
axes[0,0].set(ylabel='Thorax x (mm)');axes[0,0].legend(fontsize=7)
axes[0,1].set(ylabel='Height / initial height');axes[0,1].axhline(.7,ls='--',color='k')
axes[1,0].set(ylabel='Actual joint limit excess (rad)');axes[1,0].axhline(.02,ls='--',color='k')
axes[1,1].set(ylabel='Root horizontal speed (mm/s)');axes[1,1].axvspan(1.5,1.7,color='gray',alpha=.2)
for ax in axes.flat:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'STUDENT_METRICS.png',dpi=140);plt.close(fig)
write(HERE/'MEDIA.json',dict(video=str(video),video_sha256=sha(video),frames=count,fps=25,
 physical_sample_interval_s=.005,playback_speed=.125,expected_duration_s=count/25,
 source_sha256=sources,new_physical_steps=0,control='stateless sensory ANN, fitted motor/pad readouts, explicit local CPG',
 render_only_change='floor reflectance zero; no trajectory or geometry changes',includes_held_failure=True))
print('rendered',count,'saved-state frames')
