"""Render the eye evidence, odor field and actual teacher body trajectories."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import PowerNorm
import numpy as np

from fruit_field import FRUIT, FRUIT_RADIUS, X, Y, Wall


def main(path):
    with np.load(path/'FORMAL'/'SENSORS.npz', allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    fig = plt.figure(figsize=(11.7, 6.9), layout='constrained')
    grid = fig.add_gridspec(2, 4, height_ratios=[.55, 1.])
    stations = ('open', 'behind_wall', 'past_left_edge', 'past_right_edge')
    titles = ('Open', 'Behind wall', 'Past left edge', 'Past right edge')
    for i, (name, title) in enumerate(zip(stations, titles, strict=True)):
        panel = fig.add_subplot(grid[0, i])
        panel.imshow(np.hstack(data[name]))
        panel.set_title(title, fontsize=11)
        panel.axis('off')
    ax = fig.add_subplot(grid[1, :2])
    intake = fig.add_subplot(grid[1, 2:])
    field = ax.pcolormesh(X, Y, data['odor_wall'].T, cmap='YlGnBu', shading='nearest',
                         norm=PowerNorm(gamma=.55))
    fig.colorbar(field, ax=ax, label='Odor (engineering units)', pad=.015)
    wall = Wall()
    ax.add_patch(Rectangle((wall.x-wall.half_width, -wall.half_length),
                           2*wall.half_width, 2*wall.half_length, facecolor='#444451'))
    ax.add_patch(Circle(FRUIT[:2], FRUIT_RADIUS, edgecolor='#8a1425', facecolor='#f46c58', zorder=3))
    ax.scatter(0., 0., s=45, facecolor='white', edgecolor='#182c3e', zorder=4)
    for directory, color, label in (('WALL_L','#d25832','Teacher, left detour'),
                                    ('WALL_R','#6d3cb1','Teacher, right detour')):
        with np.load(path/directory/'TRACE463.npz', allow_pickle=False) as source:
            t = source['trace']
            xy = t[:, 1:3]
        ax.plot(*xy.T, color=color, label=label, lw=2.4)
        first = np.flatnonzero(t[:, 8] > 0.)[0]
        ax.scatter(*xy[first], facecolor='white', edgecolor=color, s=55, zorder=5)
        intake.plot(t[:, 0], t[:, 8], color=color, label=label, lw=2.4)
    with np.load(path/'OPEN_R'/'TRACE137.npz', allow_pickle=False) as source:
        t = source['trace']
        intake.plot(t[:, 0], t[:, 8], color='#386d71', label='Teacher, open', lw=2.)
    ax.set(xlim=(-1.7, 10.7), ylim=(-6, 6), xlabel='World x (mm)', ylabel='World y (mm)',
           title='Odor field and physical teacher paths')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_aspect('equal')
    intake.set(xlim=(0, 5), ylim=(0, 1.04), xlabel='Time (s)', ylabel='Cumulative intake (units)',
               title='Intake requires mouth contact')
    intake.spines[['right', 'top']].set_visible(False)
    intake.grid(axis='y', alpha=.25)
    intake.legend(loc='upper left', fontsize=9)
    fig.savefig(path/'FRUIT.png', dpi=151)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
