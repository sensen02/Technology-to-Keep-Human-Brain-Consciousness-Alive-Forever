"""Read-only render of same-run saved native states, no physical integration."""
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from body_model import HERE,json,write,sha,Path

labels=('flat','block015','block030')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
sources=[];bodies=[];fig,axes=plt.subplots(3,4,figsize=(16,9))
for row,label in enumerate(labels):
    folder=HERE/'FORMAL'/f'terminal_{label}_383';sources.append(folder/'TRACE.npz')
    m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
    data=np.load(folder/'TRACE.npz',allow_pickle=False)
    m.mat_reflectance[:]=0. # render-only removal of floor reflections
    renderer=mujoco.Renderer(m,height=320,width=480)
    bodies.append((m,d,data,renderer))
    for col,t in enumerate((.25,.60,.95,1.50)):
        index=round(t/.0001)-1;d.qpos[:]=data['qpos'][index];d.qvel[:]=data['qvel'][index];mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[float(data['position'][index,0]),0,.8];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
        renderer.update_scene(d,cam);frame=renderer.render()
        axes[row,col].imshow(frame);axes[row,col].set_title(f'{label} | t={(index+1)*.0001:.2f}s',fontsize=10);axes[row,col].axis('off')
fig.suptitle('Qualified teacher, limited canonical body, active pads. Native saved states, not ANN.',fontsize=14)
fig.tight_layout();fig.savefig(HERE/'TEACHER_STATES.png',dpi=150);plt.close(fig)
video=HERE/'TEACHER_QUALIFICATION.mp4'
writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=7,macro_block_size=16)
count=0
try:
 for tick in range(0,17000,50):
    panels=[]
    for label,(m,d,data,renderer) in zip(labels,bodies):
        d.qpos[:]=data['qpos'][tick];d.qvel[:]=data['qvel'][tick];mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[float(data['position'][tick,0]),0,.8];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
        renderer.update_scene(d,cam);frame=renderer.render();im=Image.fromarray(frame)
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,480,58),fill=(20,20,20))
        draw.text((8,4),f'{label}: teacher / finite joint guards / pads',font=font,fill='white')
        draw.text((8,28),f't={(tick+1)*.0001:.4f}s   playback 0.125x',font=font,fill='white')
        panels.append(np.asarray(im))
    writer.append_data(np.concatenate(panels,axis=1));count+=1
finally:
 writer.close()
 for m,d,data,renderer in bodies:renderer.close();data.close()
fig,axes=plt.subplots(2,2,figsize=(12,8))
for label in labels:
    for seed in (383,941):
        with np.load(HERE/'FORMAL'/f'terminal_{label}_{seed}'/'TRACE.npz',allow_pickle=False) as data:
            t=(np.arange(len(data['position']))+1)*.0001
            axes[0,0].plot(t,data['position'][:,0],label=f'{label}, {seed}')
            axes[0,1].plot(t,data['position'][:,2])
            axes[1,0].plot(t,np.linalg.norm(data['pad_qfrc'][:,:3],axis=1))
            axes[1,1].plot(t,data['nonfoot'])
axes[0,0].set(ylabel='Thorax x (FlyGym mm)');axes[0,0].legend(fontsize=7)
axes[0,1].set(ylabel='Thorax height (FlyGym mm)')
axes[1,0].set(ylabel='Pad generalized root-force norm (native)')
axes[1,1].set(ylabel='Non-foot contact force (native)')
for ax in axes.flat:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'QUALIFICATION_METRICS.png',dpi=150);plt.close(fig)
write(HERE/'MEDIA.json',dict(video=str(video),video_sha256=sha(video),frames=count,fps=25,
         physical_sample_interval_s=.005,playback_speed=.125,expected_duration_s=count/25,
         source_sha256={str(p):sha(p) for p in sources},new_physical_steps=0,
         render_only_change='Floor reflection removed; original geometry/poses unchanged.'))
print('Rendered',count,'frames and two figures from saved physical state; no new steps')
