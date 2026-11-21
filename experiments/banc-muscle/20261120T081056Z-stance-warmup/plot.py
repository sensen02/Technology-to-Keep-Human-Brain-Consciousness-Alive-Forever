"""Render the solved posture and actual release trajectory at selected native frames."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from solve import HERE, body, mj
from warm_release import HANDOFF

def plot():
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),layout='constrained')
    m=body(1.)
    # The canonical XML hides most meshes to expose one leg. Show all surfaces in this renderer only.
    m.geom_rgba[1:,3]=1
    m.vis.headlight.ambient[:]=.45
    m.vis.headlight.diffuse[:]=.7
    with np.load(HERE/'results/warm_release.npz') as trace, mj.Renderer(m,height=290,width=450) as renderer:
        d=mj.MjData(m)
        camera=mj.MjvCamera()
        camera.azimuth=70
        camera.elevation=-18
        camera.distance=6.4
        camera.lookat[:]=[-.65,0,1.3]
        for ax,n,label in zip(axes[0],(0,500,1200),('Release','50 ms after release','120 ms after release'),strict=True):
            d.qpos[:]=trace['qpos'][HANDOFF+n]
            mj.mj_forward(m,d)
            opt=mj.MjvOption()
            opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
            opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR]=False
            opt.sitegroup[:]=0
            renderer.update_scene(d,camera=camera,scene_option=opt)
            ax.imshow(renderer.render())
            ax.set_axis_off()
            ax.set_title(label)
    for mode,label in [('warm_release','Warmed BANC'),('cold_release','Cold BANC'),
                       ('motor_cut','Motor cut'),('teacher_hold','Teacher held')]:
        with np.load(HERE/'results'/(mode+'.npz')) as tr:
            t=np.arange(len(tr['qpos']))*.1-500
            feet=tr['loads'][:,:6].sum(axis=1)
            for ax,y in zip(axes[1],(tr['torso'][:,0]/tr['torso'][HANDOFF,0],feet,tr['loads'][:,-1]),strict=True):
                ax.plot(t,y,label=label,lw=1.25)
                ax.set_xlim(-15,115)
                ax.axvline(0,linewidth=.5,color='0.6')
                ax.set_xlabel('Time from release (ms)')
    for ax,title in zip(axes[1],('COM height / initial','Foot load / body weight','Nonfoot load / body weight'),strict=True):
        ax.set_title(title)
    axes[1,0].legend(fontsize=8)
    fig.savefig(HERE/'STANCE_RELEASE.png',dpi=130)

if __name__=='__main__':
    plot()
