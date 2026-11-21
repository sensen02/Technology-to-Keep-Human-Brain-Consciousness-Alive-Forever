"""Render the physical body replay and show the contact/food records together."""
import argparse
from pathlib import Path
import subprocess

from dm_control.mujoco.engine import MovableCamera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from perching import build_scene
from verify_flight_fruit import array, read, series


def plot(root, case, meal_path, out):
    facts, active = series(case)
    _, cut = series(root / 'CUT')
    _, blind = series(root / 'BLIND')
    meal = array(meal_path, 'MEAL')
    no_food = array(root / 'MEAL_BLOCKED', 'MEAL')
    t = active['time_s']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3), layout='constrained')
    for label, trace, color in [('Intact', active, '#067468'), ('DNp10 cut', cut, '#b75717'),
                                ('Vision masked', blind, '#56647c')]:
        axes[0, 0].plot(trace['x_cm'] * 10., trace['z_cm'] * 10., label=label, c=color,
                        ls='--' if label == 'Vision masked' else '-')
    geo = facts['geometry']
    axes[0, 0].fill_between(10 * np.array([geo['x_cm'] - geo['radius_cm'],
        geo['x_cm'] + geo['radius_cm']]), 0, 10 * geo['top_cm'], color='#ca4138', alpha=.3,
        label='Solid fruit')
    axes[0, 0].set(xlabel='Distance from launch (mm)', ylabel='Body height (mm)', xlim=(0, 115),
                   title='Flight and landing')
    eye = array(case, 'EYE')
    axes[0, 1].step(eye[:, 0], eye[:, 1], label='Red target width (px)', color='#ce4b37')
    paired_dnp10 = np.minimum(active['dnp10_left'], active['dnp10_right'])
    right = axes[0, 1].twinx()
    right.plot(t, paired_dnp10, color='#6a448d', label='Paired VNC readout')
    right.set_ylim(0, 1.35)
    axes[0, 1].set(xlim=(.22, .6), ylabel='Width (pixels)', title='Eye trigger and arriving landing drive')
    a, b = axes[0, 1].get_legend_handles_labels()
    c, d = right.get_legend_handles_labels()
    axes[0, 1].legend(a + c, b + d, loc='lower right', frameon=False, fontsize=9)
    axes[1, 0].step(t, active['fruit_feet'], label='Feet on fruit top', color='#067468')
    axes[1, 0].step(t, active['floor_feet'], label='Feet on ground', color='#56647c', alpha=.8)
    axes[1, 0].fill_between(t, 0, 6, where=active['taste_contact'] > 0, color='#f4ba61',
                            alpha=.32, label='Fruit in mouth reach')
    axes[1, 0].set(title='Support and taste', ylabel='Contact count', ylim=(-.25, 6.55))
    axes[1, 1].plot(meal[:, 0], meal[:, 5], c='#067468', label='Intact cumulative intake')
    axes[1, 1].plot(meal[:, 0], meal[:, 7], c='#3b71ad', label='Crop content')
    axes[1, 1].plot(no_food[:, 0], no_food[:, 5], c='#b75717', ls='--', label='Intake with feeding blocked')
    axes[1, 1].set(title='Transfer from fruit to body', ylabel='Food amount (simulation units)')
    for i, ax in enumerate(axes.ravel()):
        ax.grid(alpha=.18)
        if i:
            ax.set_xlabel('Simulation time (s)')
        if i != 1:
            ax.legend(loc='best', frameon=False, fontsize=9)
    fig.savefig(out / 'FLIGHT_FRUIT.png', dpi=164)
    plt.close(fig)


def render(case, meal_path, out):
    facts, trace = series(case)
    poses = array(case, 'QPOS')
    meal = array(meal_path, 'MEAL')
    obs = {key: values[3::4] for key, values in trace.items()}
    geo = facts['geometry']
    env = build_scene(geo['x_cm'], geo['y_cm'])
    env.reset()
    phy = env.physics
    for i in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(i, 'geom') or ''):
            phy.model.geom_rgba[i, 3] = 0.
    phy.model.site_rgba[:, 3] = 0.
    floor = phy.model.name2id('groundplane', 'geom')
    phy.model.geom_size[floor, :2] = 25.  # Extend the plane in the presentation camera only.
    view = MovableCamera(phy, height=448, width=768)
    onset = read(meal_path)['first_intake_s']

    def frame(index):
        phy.data.qpos[:] = poses[index]
        phy.forward()
        x, y, z = [obs[f'{axis}_cm'][index] for axis in 'xyz']
        t = obs['time_s'][index]
        touch = facts['first_fruit_contact_s']
        if t < .23:
            center, distance = (x + .15, y, max(z, .20)), 1.55
        elif t < touch - .19:
            center, distance = ((x + geo['x_cm']) / 2, y, .40), 5.2
        else:
            center, distance = (x + .11, y, max(z - .06, .32)), 1.54
        view.set_pose(lookat=center, distance=distance, azimuth=121, elevation=-23 if t < touch else -43)
        canvas = Image.fromarray(view.render())
        paint = ImageDraw.Draw(canvas)
        paint.rectangle((0, 0, 767, 36), fill='#153044')
        consumed = float(np.interp(t, meal[:, 0], meal[:, 5]))
        phase = ('Intake' if onset is not None and t >= onset else
                 'Stable perch' if obs['perch_ready'][index] else
                 'Contact' if t >= touch else 'Approach' if t >= facts['landing_s'] else 'Flight')
        paint.text((16, 11), f'{t:5.3f} s   |   {phase}   |   Feet on fruit {int(obs["fruit_feet"][index])}/6'
                            f'   |   Intake {consumed:.3f}', fill='white')
        return canvas

    times = (.106, facts['visual_trigger_s'] + .012, facts['first_fruit_contact_s'] - .024,
             facts['first_all_six_s'] + .075, max(onset or .94, .94), 1.36)
    fig, grid = plt.subplots(2, 3, figsize=(12.4, 5.6), layout='constrained')
    for t, ax in zip(times, grid.ravel(), strict=True):
        index = int(np.argmin(abs(obs['time_s'] - t)))
        ax.imshow(frame(index))
        ax.axis('off')
    fig.savefig(out / 'FLIGHT_FRUIT_FRAMES.png', dpi=156)
    plt.close(fig)

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-video_size', '768x448', '-framerate', '30', '-i', '-', '-an', '-threads', '2',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out / 'FLIGHT_FRUIT.mp4')]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ffmpeg:
        for index in range(35, len(poses), 36):
            ffmpeg.stdin.write(frame(index).tobytes())
        ffmpeg.stdin.close()
        assert ffmpeg.wait() == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('meal', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    plot(args.case.resolve().parent, args.case, args.meal, args.output)
    render(args.case, args.meal, args.output)
