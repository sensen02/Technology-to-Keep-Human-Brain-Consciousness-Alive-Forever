"""Show matched views of the original and repaired terminal pose."""
import importlib.util
from pathlib import Path
import sys

from dm_control.mujoco.engine import MovableCamera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inspect_pose import FLIGHT, build_scene
from verify_stow import HERE, load, read


def still(phy, pose):
    phy.data.qpos[:] = pose
    phy.forward()
    center = phy.named.data.subtree_com['walker/'].copy()
    camera = MovableCamera(phy, height=340, width=560)
    camera.set_pose(lookat=center, distance=.74, azimuth=121, elevation=-33)
    rgb = camera.render()
    return rgb


def compare_visual(out):
    case = HERE / 'NEAR'
    geometry = read(case)['geometry']
    env = build_scene(geometry['x_cm'], geometry['y_cm'])
    env.reset()
    phy = env.physics
    for i in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(i, 'geom') or ''):
            phy.model.geom_rgba[i, 3] = 0
    phy.model.site_rgba[:, 3] = 0
    facts = np.load(HERE / 'NEAR_POSES.npz', allow_pickle=False)
    t = facts['time_s']
    index = round(1.36 / .0002) - 1
    fig = plt.figure(figsize=(10.7, 6.6), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    cases = (('Original: wings held below the abdomen', FLIGHT / 'NEAR', '#b6562c'),
             ('Corrected: wings stowed above the abdomen', case, '#067468'))
    for i, (label, folder, color) in enumerate(cases):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(still(phy, load(folder, 'QPOS')[index]))
        ax.set_title(label, color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
    ax = fig.add_subplot(grid[1, :])
    for label, name, color in [('Original', 'old_error_deg', '#b6562c'),
                               ('Corrected', 'corrected_error_deg', '#067468')]:
        ax.plot(t, np.abs(facts[name]).max(axis=1), color=color, label=label)
    ax.axhline(2, color='#64748b', ls='--', label='Stow tolerance')
    ax.set(xlim=(.69, 1.42), xlabel='Simulation time (s)', ylabel='Largest wing angle\nerror (degrees)')
    ax.grid(alpha=.22)
    ax.legend(loc='upper right', frameon=False)
    fig.savefig(out / 'WING_STOW_COMPARISON.png', dpi=155)
    plt.close(fig)
    env.close()


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir()
    compare_visual(out)
    module = importlib.util.spec_from_file_location('fruit_flight_pictures', FLIGHT / 'pictures.py')
    movie = importlib.util.module_from_spec(module)
    module.loader.exec_module(movie)
    movie.render(HERE / 'NEAR', HERE / 'MEAL_NEAR', out)
