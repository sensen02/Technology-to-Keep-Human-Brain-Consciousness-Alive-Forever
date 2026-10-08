"""viz.py -- rendering helpers for the CPM self-organisation simulations.

Everything here plots state that comes out of an actual run.  Nothing is
synthesised.  Figures are written with matplotlib's Agg backend so they can be
produced headless.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

# Default palette: index 0 = medium, then per cell type, then marker colours.
MEDIUM_RGB = (1.0, 1.0, 1.0)


def type_image(cpm, colors, state_colors=None, state_of=None):
    """RGB image of the lattice.

    `colors` maps cell type -> (r,g,b); `state_of` maps cell id -> state key and
    `state_colors` maps state key -> (r,g,b).  States override types.
    """
    dim = cpm.dim
    lat = cpm.lat.reshape((cpm.Ly, cpm.Lx) if dim == 2 else (cpm.Lz, cpm.Ly, cpm.Lx))
    st = cpm.ctype[lat]
    img = np.zeros(lat.shape + (3,), dtype=np.float64)
    img[:] = MEDIUM_RGB
    for t, col in colors.items():
        if t == 0:
            continue
        img[st == t] = col
    if state_of is not None and state_colors:
        for key, col in state_colors.items():
            ids = np.fromiter(state_of.get(key, ()), dtype=np.int64)
            if ids.size == 0:
                continue
            mask = np.isin(lat, ids)
            img[mask] = col
    return np.clip(img, 0, 1)


def save_lattice_png(path, cpm, colors, title="", state_of=None,
                     state_colors=None, legend=None, dpi=140):
    img = type_image(cpm, colors, state_colors, state_of)
    fig, ax = plt.subplots(figsize=(6.2, 6.2 * img.shape[0] / img.shape[1]), dpi=dpi)
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor="0.3", label=l)
                           for l, c in legend],
                  loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_panels_png(path, panels, ncols=None, figsize=(13, 8), dpi=140,
                    suptitle=None):
    """`panels` is a list of dicts: {"img": ..., "title": str, "cmap": optional,
    "colorbar": bool, "origin": optional}.

    IMPORTANT: `origin` defaults to "upper" (the imshow default), which is
    correct for an RGB array that was read back from an already-rendered PNG --
    such an image is already in display orientation and must NOT be flipped
    again.  Pass origin="lower" explicitly for raw data arrays whose row 0 is
    the bottom of the physical domain (e.g. a concentration field).
    """
    n = len(panels)
    ncols = ncols or min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, panels):
        origin = p.get("origin", "upper")
        if p.get("cmap"):
            im = ax.imshow(p["img"], cmap=p["cmap"], interpolation="nearest",
                           origin=origin)
            if p.get("colorbar"):
                fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(p["img"], interpolation="nearest", origin=origin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p.get("title", ""), fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_metric_png(path, series, xlabel, ylabel, title="", dpi=140,
                    figsize=(7.5, 4.6), logy=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, label=label, lw=1.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_gif(path, frames_png, duration_ms=180):
    """Combine already-rendered PNG frames into an animated GIF."""
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames_png]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=True)
    return path


def connected_components(n_nodes, edges):
    """Union-find over int node ids.  Returns (labels, n_components)."""
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {}
    labels = np.zeros(n_nodes, dtype=np.int64)
    for i in range(n_nodes):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels, len(roots)
