"""Read-only close inspection of both retained final-policy failures."""
from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(16,7))
for row,(group,label,center) in enumerate((('DEV2','block030_263',1.0452),('HELD','block015_1723',1.3362))):
    folder=HERE/group/label;m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
    with np.load(folder/'TRACE.npz',allow_pickle=False) as z:q=z['qpos'].copy();pos=z['position'].copy()
    m.mat_reflectance[:]=0.;renderer=mujoco.Renderer(m,height=320,width=480)
    for col,offset in enumerate((-.035,-.010,0.,.035)):
        tick=round((center+offset)/DT)-1;d.qpos[:]=q[tick];mujoco.mj_kinematics(m,d)
        cam=mujoco.MjvCamera();cam.lookat[:]=[pos[tick,0],0,.75];cam.distance=7.;cam.azimuth=105.;cam.elevation=-20.
        renderer.update_scene(d,cam);axes[row,col].imshow(renderer.render());axes[row,col].axis('off')
        axes[row,col].set_title(f'{group} {label}: FAIL | {(tick+1)*DT:.4f}s',fontsize=9)
    renderer.close()
fig.suptitle('Retained failures: actual joint overshoot. Held 0.15mm dip occurs during stopping after crossing.')
fig.tight_layout();fig.savefig(HERE/'FAILURE_STATES.png',dpi=130);plt.close(fig)
print('saved failure states, no integration')
