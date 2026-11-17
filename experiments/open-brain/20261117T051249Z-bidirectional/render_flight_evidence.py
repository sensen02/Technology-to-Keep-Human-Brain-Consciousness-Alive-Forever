"""Render the recorded physical poses at matched times around flight interruption."""
from pathlib import Path
import matplotlib.pyplot as plt
from dm_control.mujoco.engine import MovableCamera
import numpy as np

from free_flight import make_env

HERE = Path(__file__).resolve().parent
source = HERE / 'FREE_FLIGHT_QUAL'
env = make_env()
env.reset()
p = env.physics
for idx in range(p.model.ngeom):
    key = p.model.id2name(idx, 'geom') or ''
    if 'ghost' in key:
        p.model.geom_rgba[idx,3] = 0
cam = MovableCamera(p, height=240, width=360)
fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.1), layout='constrained')
for row, name in enumerate(('intact','cut_restore_short')):
    folder = source/name
    trace = np.load(folder/'TRACE.npy',allow_pickle=False)
    poses = np.load(folder/'QPOS.npy',allow_pickle=False)
    for column, t in enumerate((.065,.101,.25)):
        index = int(np.argmin(abs(trace[:,0]-t)))
        p.data.qpos[:] = poses[index]
        p.forward()
        cam.set_pose(lookat=trace[index,1:4], distance=1.1, azimuth=115, elevation=-12)
        ax=axes[row,column]
        pixels=cam.render()
        ax.imshow(pixels)
        label='Intact' if row==0 else 'Brief cut + restore'
        ax.set_title(f'{label}  ·  {trace[index,0]*1000:.0f} ms')
        ax.axis('off')
fig.savefig(HERE/'FLIGHT_POSES.png',dpi=150)
