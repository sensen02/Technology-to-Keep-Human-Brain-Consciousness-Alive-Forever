"""Render the recorded takeoff chronologically and plot native contact and flight."""
import argparse
import json
from pathlib import Path
import subprocess

from dm_control.mujoco.engine import MovableCamera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from takeoff import PARENT
from legged_fixture import build_env

HERE = Path(__file__).resolve().parent
RUN = HERE / 'FORMAL17'
CASES = ('takeoff_probe', 'takeoff_cut', 'fluid_off', 'legs_only')
NAMES = ('Intact', 'DNp11 cut', 'Fluid off', 'Legs only')
COLORS = ('#027d73', '#51617b', '#c2410c', '#784293')


def trace(folder):
    facts = json.loads((folder / 'RESULT.json').read_text())
    array = np.load(folder / 'TRACE.npy', allow_pickle=False)
    return dict(zip(facts['columns'], array.T, strict=True))


def plot():
    fig, panels = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    for case, label, color in zip(CASES, NAMES, COLORS, strict=True):
        data = trace(RUN / case)
        t = data['time_s'] * 1000
        panels[0, 0].plot(t, data['z_cm'] * 10, lw=1.4, color=color, label=label)
        panels[0, 1].plot(t, data['x_cm'] * 10, lw=1.4, color=color, label=label)
    intact = trace(RUN / 'takeoff_probe')
    t = intact['time_s'] * 1000
    panels[1, 0].step(t, intact['feet'], where='post', color=COLORS[0], label='Feet in contact')
    panels[1, 0].plot(t, intact['wing_mix'], color='#285f9d', label='Wing pattern mix')
    panels[1, 0].plot(t, intact['teacher_mix'], color='#88429e', label='FlyBody policy mix')
    panels[1, 0].axvline(81.3, color='#778294', ls=':', label='First DNp11 arrival')
    panels[1, 1].plot(t, np.rad2deg(intact['pitch_rad']), color='#285f9d', label='Pitch')
    panels[1, 1].plot(t, np.rad2deg(intact['roll_rad']), color='#9d4c15', label='Roll')
    panels[1, 1].plot(t, np.rad2deg(intact['yaw_rad']), color=COLORS[0], label='Yaw')
    panels[0, 0].set(title='Continuous ground-to-air takeoff', ylabel='CoM height (mm)')
    panels[0, 1].set(title='Forward travel', ylabel='Position (mm)')
    panels[1, 0].set(title='Push-off, wing deployment and flight control', ylabel='Contacts / blend', xlim=(75, 165))
    panels[1, 1].set(title='Body attitude', ylabel='Angle (°)')
    for panel in panels.ravel():
        panel.set_xlabel('Time (ms)')
        panel.grid(alpha=.18)
        panel.legend(frameon=False, loc='best', fontsize=9)
    fig.savefig(HERE / 'TAKEOFF.png', dpi=155)
    plt.close(fig)


def render(video):
    data = trace(RUN / 'takeoff_probe')
    poses = np.load(RUN / 'takeoff_probe/QPOS.npy', allow_pickle=False)
    data = {key: vec[3::4] for key, vec in data.items()}
    env = build_env()
    env.reset()
    phy = env.physics
    for idx in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(idx, 'geom') or ''):
            phy.model.geom_rgba[idx, 3] = 0.
    phy.model.site_rgba[:, 3] = 0.
    camera = MovableCamera(phy, height=420, width=672)

    def frame(index):
        phy.data.qpos[:] = poses[index]
        phy.forward()
        focus = [data[axis][index] for axis in ('x_cm', 'y_cm', 'z_cm')]
        camera.set_pose(lookat=focus, distance=.75, azimuth=119, elevation=12)
        rgb = camera.render()
        canvas = Image.fromarray(rgb)
        paint = ImageDraw.Draw(canvas)
        paint.rectangle((0, 0, 671, 28), fill='#14212d')
        msg = (f"{data['time_s'][index]*1000:6.1f} ms    "
               f"height {focus[2]*10:4.1f} mm    "
               f"feet in contact {int(data['feet'][index])}")
        paint.text((12, 8), msg, fill='white')
        return canvas

    fig, panels = plt.subplots(2, 3, figsize=(12, 5.8), layout='constrained')
    for panel, time in zip(panels.ravel(), (75, 99, 114, 135, 200, 385), strict=True):
        index = int(np.argmin(abs(data['time_s'] - time / 1000)))
        panel.imshow(frame(index))
        panel.axis('off')
    fig.savefig(HERE / 'TAKEOFF_POSES.png', dpi=145)
    plt.close(fig)
    if video:
        output = HERE / 'TAKEOFF.mp4'
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '672x420', '-framerate', '30',
               '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ffmpeg:
            for index in range(9, len(poses), 10):
                ffmpeg.stdin.write(frame(index).tobytes())
            ffmpeg.stdin.close()
            assert ffmpeg.wait() == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    plot()
    render(args.video)
