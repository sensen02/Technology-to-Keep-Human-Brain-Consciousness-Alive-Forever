"""Render saved diagnostic state only, without new dynamics."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from context import HERE

names=['flat_teacher_119','block015_teacher_119','block015_oracle015_119','block015_student_119',
       'block030_teacher_119','block030_oracle015_119','block030_oracle15_119','block030_student_119']
fig,axes=plt.subplots(2,2,figsize=(12,8))
for name in names:
    with np.load(HERE/'RUNS'/name/'TRACE.npz',allow_pickle=False) as a:
        t=np.arange(len(a['position']))*.0001
        axes[0,0].plot(t,a['position'][:,0],label=name)
        axes[0,1].plot(t,a['position'][:,2],label=name)
        axes[1,0].plot(t,np.max(np.abs(a['teacher']-a['rhythmic']),axis=1),label=name)
        axes[1,1].plot(t,a['nonfoot_force'],label=name)
for ax,title in zip(axes.flat,('Thorax x (FlyGym mm)','Thorax z (FlyGym mm)','Needed correction, max joint (rad)','Non-foot force (native)')):
    ax.set(xlabel='Simulation time (s)',ylabel=title);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(HERE/'REACHABILITY.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(13,10))
for row,name in enumerate(('block015_teacher_119','block015_oracle015_119','block030_teacher_119','block030_student_119')):
    m=mujoco.MjModel.from_binary_path(str(HERE/'RUNS'/name/'BODY.mjb'));d=mujoco.MjData(m)
    with np.load(HERE/'RUNS'/name/'TRACE.npz',allow_pickle=False) as a:
        renderer=mujoco.Renderer(m,height=320,width=480)
        for col,t in enumerate((.45,.85,1.5)):
            frame=min(round(t/.001),len(a['qpos'])-1);d.qpos[:]=a['qpos'][frame];mujoco.mj_forward(m,d)
            cam=mujoco.MjvCamera();cam.lookat[:]=[float(d.qpos[0]),0,.7];cam.distance=8.;cam.azimuth=105;cam.elevation=-24
            renderer.update_scene(d,cam);im=renderer.render()
            axes[row,col].imshow(im);axes[row,col].set_title(f'{name}\nt={frame*.001+.0001:.4f}s',fontsize=8);axes[row,col].axis('off')
        renderer.close()
fig.suptitle('Saved physical states. Teacher / privileged oracle / ANN diagnostic; no new training.',fontsize=11)
fig.tight_layout();fig.savefig(HERE/'PHYSICAL_STATES.png',dpi=150);plt.close(fig)
print('Saved REACHABILITY.png and PHYSICAL_STATES.png; rendering only')

fig,axes=plt.subplots(1,2,figsize=(12,4))
amplitude=json.loads((HERE/'AMPLITUDE_RESULT.json').read_text())['results']
for leg in ('lf','lm','lh','rf','rm','rh'):
    rows=[r for r in amplitude if r['leg']==leg]
    axes[0].plot([r['bound_rad'] for r in rows],[r['hold_lift_mean_native'] for r in rows],marker='o',label=leg)
axes[0].axhline(.15,color='black',linestyle='--',label='Declared lift target')
axes[0].set(xlabel='Per-joint corrective amplitude (rad)',ylabel='Measured relative foot lift (FlyGym mm)')
axes[0].legend(fontsize=7)
for mode in ('off','on','release'):
    with np.load(HERE/f'pad_{mode}_dt0.0001.npz',allow_pickle=False) as a:
        axes[1].plot(a['trace'][:,0],a['trace'][:,2],label=mode)
axes[1].set(xlabel='Simulation time (s)',ylabel='Single-pad vertical displacement (FlyGym mm)')
axes[1].legend()
zoom=axes[1].inset_axes([.12,.16,.40,.35])
with np.load(HERE/'pad_on_dt0.0001.npz',allow_pickle=False) as a:
    zoom.plot(a['trace'][:,0],a['trace'][:,2],color='orange')
zoom.axhline(-.02,color='black',linestyle='--');zoom.set_ylim(-.12,.01);zoom.set_title('ON: drift exceeds 0.02',fontsize=7)
zoom.tick_params(labelsize=6)
fig.tight_layout();fig.savefig(HERE/'LIFT_AND_PAD.png',dpi=150);plt.close(fig)
