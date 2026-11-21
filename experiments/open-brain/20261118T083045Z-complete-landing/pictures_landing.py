"""Render poses and plot the paired physical landing trace."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from dm_control.mujoco.engine import MovableCamera

from land import COLUMNS, build_env


def read(folder):
    values = np.load(folder / 'TRACE.npy', allow_pickle=False)
    return dict(zip(COLUMNS, values.T, strict=True))


def plot(folder, output):
    active, cut = (read(folder / case) for case in ('intact', 'cut'))
    fig, grid = plt.subplots(2, 2, figsize=(11.5, 7.4), layout='constrained')
    teal, slate = '#0f766e', '#64748b'
    for label, d, color in [('DNp10 intact', active, teal), ('DNp10 cut', cut, slate)]:
        t = d['time_s'] * 1000
        grid[0, 0].plot(t, d['z_cm'] * 10, color=color, label=label)
        grid[0, 1].plot(t, np.hypot(d['vx_cm_s'], d['vy_cm_s']) * 10, color=color, label=label)
    t = active['time_s'] * 1000
    grid[1, 0].step(t, active['feet'], where='post', color=teal, label='Feet in contact')
    grid[1, 0].plot(t, active['nonfoot_contacts'], color='#a4381c', label='Nonfoot floor contacts')
    grid[1, 1].plot(t, active['wing_mix'], color='#3b5480', label='Wing deployment')
    grid[1, 1].plot(t, active['dnp10_left'], color='#874894', label='DNp10 VNC readout (left)')
    for cell in grid.ravel():
        cell.axvline(421.4, color='#a0a9b5', ls=':', linewidth=1.)
        cell.grid(alpha=.2)
        cell.legend(frameon=False, loc='best')
        cell.set_xlabel('Simulation time (ms)')
    grid[0, 0].set(title='Takeoff, cruise and landing', ylabel='CoM height (mm)')
    grid[0, 1].set(title='Braking before touchdown', ylabel='Horizontal speed (mm/s)')
    grid[1, 0].set(title='Ground contact', ylabel='Contact count', xlim=(600, 900))
    grid[1, 1].set(title='Post-neck command and wing folding', ylabel='Normalized readout', xlim=(400, 875))
    fig.savefig(output / 'LANDING.png', dpi=150)
    plt.close(fig)


def render(folder, output):
    active = folder / 'intact'
    data = {key: d[3::4] for key, d in read(active).items()}
    poses = np.load(active / 'QPOS.npy', allow_pickle=False)
    facts = json.loads((active / 'RESULT.json').read_text())
    touch = facts['landing_observations']['first_contact_s']
    assert touch is not None
    env = build_env()
    env.reset()
    phy = env.physics
    for i in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(i, 'geom') or ''):
            phy.model.geom_rgba[i, 3] = 0.
    phy.model.site_rgba[:, 3] = 0.
    frame_camera = MovableCamera(phy, height=420, width=672)

    def frame(index):
        phy.data.qpos[:] = poses[index]
        phy.forward()
        center = [data[name][index] for name in ('x_cm', 'y_cm', 'z_cm')]
        frame_camera.set_pose(lookat=center, distance=.78, azimuth=122, elevation=18)
        canvas = Image.fromarray(frame_camera.render())
        paint = ImageDraw.Draw(canvas)
        paint.rectangle((0, 0, 671, 28), fill='#152636')
        copy = (f"{data['time_s'][index]*1000:6.1f} ms   "
                f"height {10*center[2]:4.1f} mm   "
                f"feet {int(data['feet'][index])}   "
                f"nonfoot contacts {int(data['nonfoot_contacts'][index])}")
        paint.text((12, 8), copy, fill='white')
        return canvas

    keypoints = (.075, .390, touch - .063, touch, touch + .040, .995)
    fig, grid = plt.subplots(2, 3, figsize=(12, 5.8), layout='constrained')
    for axis, time in zip(grid.ravel(), keypoints, strict=True):
        idx = int(np.argmin(abs(data['time_s'] - time)))
        axis.imshow(frame(idx))
        axis.axis('off')
    fig.savefig(output / 'LANDING_POSES.png', dpi=145)
    plt.close(fig)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
           '-pixel_format', 'rgb24', '-video_size', '672x420', '-framerate', '30',
           '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output / 'LANDING.mp4')]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for idx in range(9, len(poses), 10):
            encoder.stdin.write(frame(idx).tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    plot(args.run, args.output)
    render(args.run, args.output)
