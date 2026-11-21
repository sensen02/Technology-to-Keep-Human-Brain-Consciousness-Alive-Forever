"""Plot measured flight and render the recorded MuJoCo poses for visual QA."""
from pathlib import Path

from dm_control.mujoco.engine import MovableCamera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fly import make_env
from legged_fixture import build_env

HERE = Path(__file__).resolve().parent
COLORS = {'sham': '#64748b', 'left': '#0f766e', 'right': '#b45309',
          'amp_low': '#5b21b6', 'amp_high': '#be185d', 'combined': '#0ea5e9'}


def trace(folder, condition):
    return np.load(HERE / folder / condition / 'TRACE.npy', allow_pickle=False)


def flight_plot():
    fig, ax = plt.subplots(2, 2, figsize=(10.8, 7.3), layout='constrained')
    for case in ('sham', 'left', 'right', 'combined'):
        t = trace('VALID_29', case)
        label = case.replace('_', ' ')
        ax[0, 0].plot(t[:, 1] * 10, t[:, 2] * 10, color=COLORS[case], label=label)
        ax[0, 1].plot(t[:, 0] * 1000, np.rad2deg(t[:, 4]), color=COLORS[case], label=label)
        ax[1, 0].plot(t[:, 0] * 1000, t[:, 3] * 10, color=COLORS[case], lw=1., label=label)
    for case in ('sham', 'amp_low', 'amp_high'):
        t = trace('VALID_29', case)
        use = (t[:, 0] >= .140) & (t[:, 0] <= .156)
        ax[1, 1].plot(t[use, 0] * 1000, np.rad2deg(t[use, 8]),
                     color=COLORS[case], label=case.replace('_', ' '))
    dry = trace('VALID_NO_FLUID', 'sham')
    ax[1, 0].plot(dry[:, 0] * 1000, dry[:, 3] * 10, ':k', label='Fluid disabled')
    ax[0, 0].set(xlabel='Forward distance (mm)', ylabel='Side displacement (mm)', title='Measured flight path')
    ax[0, 1].set(xlabel='Time (ms)', ylabel='Body yaw (°)', title='DNp03 steering')
    ax[1, 0].set(xlabel='Time (ms)', ylabel='CoM height (mm)', title='Aerodynamic support')
    ax[1, 1].set(xlabel='Time (ms)', ylabel='Left wing stroke (°)', title='DNg02_a actual joint motion')
    ax[0, 1].axvspan(80, 190, alpha=.09, color='#94a3b8')
    for axis in ax.ravel():
        axis.grid(alpha=.18)
        axis.legend(frameon=False, loc='best', fontsize=9)
    fig.savefig(HERE / 'AIRBORNE.png', dpi=150)
    plt.close(fig)


def picture(env, rows, name):
    env.reset()
    phy = env.physics
    for idx in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(idx, 'geom') or ''):
            phy.model.geom_rgba[idx, 3] = 0.
    for idx in range(phy.model.nsite):
        phy.model.site_rgba[idx, 3] = 0.
    cam = MovableCamera(phy, height=250, width=400)
    fig, ax = plt.subplots(len(rows), 3, figsize=(11.5, 3.33 * len(rows)),
                           layout='constrained', squeeze=False)
    for row, (label, path, t_ms, stride) in enumerate(rows):
        t = np.load(path / 'TRACE.npy', allow_pickle=False)[stride - 1::stride]
        poses = np.load(path / 'QPOS.npy', allow_pickle=False)
        assert len(t) == len(poses)
        for col, time in enumerate(t_ms):
            index = int(np.argmin(abs(t[:, 0] - time * .001)))
            phy.data.qpos[:] = poses[index]
            phy.forward()
            cam.set_pose(lookat=t[index, 1:4], distance=.72, azimuth=117, elevation=16)
            ax[row, col].imshow(cam.render())
            ax[row, col].set_title(f'{label}  ·  {t[index, 0]*1000:.1f} ms')
            ax[row, col].axis('off')
    fig.savefig(HERE / name, dpi=140)
    plt.close(fig)


def transitions_plot():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for case, color, style in [('probe', '#0f766e', '-'), ('cut', '#b45309', '--')]:
        launch = trace('TRANSITION_A', f'takeoff_{case}')
        ax[0].plot(launch[:, 0] * 1000, launch[:, 3] * 10, style, color=color, label=case)
        ax[1].step(launch[:, 0] * 1000, launch[:, 4], where='post', color=color, linestyle=style, label=case)
        landing = trace('TRANSITION_B', f'landing_{case}')
        ax[2].step(landing[:, 0] * 1000, landing[:, 5], where='post', color=color, linestyle=style, label=case)
    ax[0].set(title='DNp11: center-of-mass lift', xlabel='Time (ms)', ylabel='CoM height (mm)', xlim=(75, 150))
    ax[1].set(title='DNp11: loss of all support', xlabel='Time (ms)', ylabel='Feet in contact', xlim=(97, 113))
    ax[2].set(title='DNp10: near-floor capture', xlabel='Time (ms)', ylabel='Nonfoot contacts', xlim=(0, 60))
    for axis in ax:
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    fig.savefig(HERE / 'TRANSITIONS.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    flight_plot()
    picture(make_env(), [(label, HERE / 'VALID_29' / case, (65, 145, 260), 1)
                         for label, case in [('DNp03 left', 'left'), ('DNp03 right', 'right')]],
            'AIRBORNE_POSES.png')
    transitions_plot()
    picture(build_env(), [
        ('DNp11', HERE / 'TRANSITION_A/takeoff_probe', (77, 105, 147), 4),
        ('DNp10', HERE / 'TRANSITION_B/landing_probe', (3, 15, 75), 4),
        ('DNp10 cut', HERE / 'TRANSITION_B/landing_cut', (3, 15, 75), 4),
    ], 'TRANSITION_POSES.png')
