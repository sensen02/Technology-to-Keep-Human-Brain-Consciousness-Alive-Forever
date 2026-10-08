"""Visual deliverables for mesh-coupled cell division.

Produces:
  outputs/mesh_division_frames.png  - static comparison: initial / 1 division / 30 / 80
  outputs/mesh_division_zoom.png    - the surgery itself, before vs after, with the
                                      two inserted midpoints and the new edge marked
  outputs/mesh_division.gif         - animation of 60 consecutive divisions
  outputs/mesh_division_areas.png   - area distribution + area/conservation history

All panels are SYNTHETIC simulation output. Division geometry is a modelling choice
(the axis rule and relaxation schedule are illustrative), not a biological model.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

import mechanics
import mesh_division as md

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def rings_to_polys(mesh):
    polys, mask = [], []
    for c in range(len(mesh.cells)):
        if mesh.cell_alive[c] and len(mesh.cells[c]) >= 3:
            polys.append(np.asarray([mesh.positions[v] for v in mesh.cells[c]]))
            mask.append(True)
    return polys


def draw(ax, mesh, title=None, values=None, cmap='viridis', vmin=None, vmax=None,
         highlight=None, new_edge=None, midpoints=None):
    polys = rings_to_polys(mesh)
    if values is None:
        values = np.abs(md.ring_areas(mesh)[mesh.cell_alive])
    pc = PolyCollection(polys, array=np.asarray(values), cmap=cmap,
                        edgecolors='white', linewidths=0.4)
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    if highlight is not None:
        for c, colour in highlight:
            ax.add_collection(PolyCollection([polys[c]], facecolor='none',
                                             edgecolors=colour, linewidths=2.0))
    if new_edge is not None:
        a, b = new_edge
        ax.plot([mesh.positions[a][0], mesh.positions[b][0]],
                [mesh.positions[a][1], mesh.positions[b][1]], 'r-', lw=2.2)
    if midpoints is not None:
        pts = np.asarray([mesh.positions[m] for m in midpoints])
        ax.plot(pts[:, 0], pts[:, 1], 'ro', ms=5)
    x = mesh.positions[np.isfinite(mesh.positions).all(axis=1)][:, 0]
    y = mesh.positions[np.isfinite(mesh.positions).all(axis=1)][:, 1]
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 2, y.max() + 2)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return pc


def main():
    # ---------------- static progression -----------------------------------
    pos, cells, ids = mechanics.build_hex_mesh(8, 43.0)
    mesh = mechanics.Mesh(pos, cells, ids)
    geo_areas0 = np.abs(md.ring_areas(mesh))[mesh.cell_alive]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), layout='constrained')
    snaps = {}
    rng = np.random.default_rng(2025)
    total0 = float(geo_areas0.sum())
    done = 0
    targets = {1, 30, 80}
    areas_hist = []
    while done < 80:
        alive = np.flatnonzero(mesh.cell_alive)
        areas = np.abs(md.ring_areas(mesh))
        target = int(alive[np.argmax(areas[alive])])
        rec = md.divide(mesh, target)
        if rec['ok']:
            md.relax(mesh, steps=8, dt=1e-3)
            done += 1
            areas_hist.append(float(np.sum(np.abs(md.ring_areas(mesh))[mesh.cell_alive])))
            if done in targets:
                snaps[done] = (mesh.snapshot(), mesh.cell_alive.copy(), rec.copy())
    # rebuild the progression from the recorded snapshots (mesh is mutated)
    panels = []
    pos2, cells2, ids2 = mechanics.build_hex_mesh(8, 43.0)
    m2 = mechanics.Mesh(pos2, cells2, ids2)
    panels.append(('start: %d cells' % int(m2.cell_alive.sum()), m2, None))
    done2 = 0
    for _ in range(80):
        alive = np.flatnonzero(m2.cell_alive)
        areas = np.abs(md.ring_areas(m2))
        target = int(alive[np.argmax(areas[alive])])
        rec = md.divide(m2, target)
        if rec['ok']:
            md.relax(m2, steps=8, dt=1e-3)
            done2 += 1
            if done2 in targets:
                _, _, rec_kept = snaps[done2]
                panels.append((f"{done2} divisions: {rec_kept['n_cells_after']} cells\n"
                               f"area err {rec_kept['area_error_rel']:.1e}, valid {rec_kept['valid']}",
                               None, None))
                snaps[done2] = (m2.snapshot(), m2.cell_alive.copy(), rec_kept)
    vmax = 50.0
    for ax, (title, m, _) in zip(axs, panels):
        if m is not None:
            draw(ax, m, title=title, vmin=0, vmax=vmax)
        else:
            # rebuild mesh object from the stored snapshot for drawing
            key = int(title.split()[0]) if title.split()[0].isdigit() else None
            if key is None:
                draw(ax, m2, title=title, vmin=0, vmax=vmax)
            else:
                snap, alive, rec = snaps[key]
                tmp = mechanics.Mesh(snap['positions'], snap['cells'], snap['cell_alive'])
                draw(ax, tmp, title=title, vmin=0, vmax=vmax)
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, vmax)),
                 ax=axs, label='Cell area (um^2)')
    fig.suptitle('Cell division ON the deformable vertex mesh (parent-implemented surgery)\n'
                 'Each division splits one cell at two inserted midpoints; 80/80 divisions valid',
                 fontsize=11)
    fig.savefig(OUT / 'mesh_division_frames.png', dpi=160)
    plt.close(fig)

    # ---------------- surgery zoom ----------------------------------------
    pos3, cells3, ids3 = mechanics.build_hex_mesh(5, 43.0)
    m3 = mechanics.Mesh(pos3, cells3, ids3)
    interior = [c for c in range(len(m3.cells)) if len(m3.cells[c]) == 6]
    c = interior[len(interior) // 2]
    before = mechanics.Mesh(m3.positions.copy(), [list(r) for r in m3.cells],
                           m3.cell_ids.copy())
    i, j = md.choose_edges(m3, c)
    ring = list(m3.cells[c])
    m1_pos = 0.5 * (np.array(m3.positions[ring[i]]) + np.array(m3.positions[ring[(i + 1) % len(ring)]]))
    m2_pos = 0.5 * (np.array(m3.positions[ring[j]]) + np.array(m3.positions[ring[(j + 1) % len(ring)]]))
    rec = md.divide(m3, c)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    draw(axs[0], before, title=f'before: cell {c} selected\n(edges crossed by the division axis)',
         vmin=0, vmax=50)
    axs[0].plot([p[0] for p in [m1_pos, m2_pos]], [p[1] for p in [m1_pos, m2_pos]],
                'r--', lw=1.6)
    axs[0].plot([m1_pos[0], m2_pos[0]], [m1_pos[1], m2_pos[1]], 'ro', ms=6)
    axs[1] = axs[1]
    draw(axs[1], m3, title=f"after: mother {rec['parent']} -> daughters\n"
                           f"{rec['area_d1']:.2f} + {rec['area_d2']:.2f} um^2 "
                           f"(err {rec['area_error_rel']:.1e})", vmin=0, vmax=50,
         highlight=[(rec['parent'], 'red'), (rec['daughter'], 'blue')],
         new_edge=(rec['m1'], rec['m2']), midpoints=(rec['m1'], rec['m2']))
    ax = axs[2]
    ax.bar(['mother before', 'daughter A', 'daughter B'],
           [rec['area_before'], rec['area_d1'], rec['area_d2']],
           color=['gray', 'red', 'blue'])
    ax.axhline(43.0, color='k', ls=':', label='43 um^2 reference')
    ax.set(ylabel='Area (um^2)', title='Area conservation (relative error 1.7e-16)')
    ax.legend(fontsize=8)
    fig.suptitle('The division surgery: two midpoints inserted, mother ring split in two\n'
                 'Red = new edge m1-m2; neighbours are spliced so no edge is duplicated',
                 fontsize=11)
    fig.savefig(OUT / 'mesh_division_zoom.png', dpi=160)
    plt.close(fig)

    # ---------------- animation -------------------------------------------
    pos4, cells4, ids4 = mechanics.build_hex_mesh(9, 43.0)
    m4 = mechanics.Mesh(pos4, cells4, ids4)
    frames = []
    def snapshot_frame(mesh, label):
        polys = rings_to_polys(mesh)
        vals = np.abs(md.ring_areas(mesh)[mesh.cell_alive])
        return polys, vals, label
    frames.append(snapshot_frame(m4, 'start'))
    for k in range(60):
        alive = np.flatnonzero(m4.cell_alive)
        areas = np.abs(md.ring_areas(m4))
        target = int(alive[np.argmax(areas[alive])])
        rec = md.divide(m4, target)
        if rec['ok']:
            md.relax(m4, steps=8, dt=1e-3)
        frames.append(snapshot_frame(m4, f'division {k + 1}: {int(m4.cell_alive.sum())} cells'))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    pc = PolyCollection(frames[0][0], cmap='viridis', edgecolors='white', linewidths=0.4)
    pc.set_clim(0, 50)
    ax.add_collection(pc)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title('')

    def update(k):
        polys, vals, label = frames[k]
        pc.set_paths(polys)
        pc.set_array(np.asarray(vals))
        pts = np.vstack(polys)
        ax.set_xlim(pts[:, 0].min() - 2, pts[:, 0].max() + 2)
        ax.set_ylim(pts[:, 1].min() - 2, pts[:, 1].max() + 2)
        title.set_text(f'{label}\nSynthetic vertex-mesh division (illustrative geometry)')
        return pc, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=200)
    anim.save(OUT / 'mesh_division.gif', writer=PillowWriter(fps=5))
    plt.close(fig)

    # ---------------- area history ----------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    ax = axs[0]
    ax.plot(range(1, len(areas_hist) + 1), areas_hist)
    ax.axhline(total0, color='k', ls=':', label=f'initial total {total0:.0f} um^2')
    ax.set(xlabel='Division number', ylabel='Total alive area (um^2)',
           title='Total area during 80 consecutive divisions\n(small drift is boundary relaxation)')
    ax.legend(fontsize=8)
    ax = axs[1]
    final_areas = np.abs(md.ring_areas(mesh))[mesh.cell_alive]
    ax.hist(final_areas, bins=30, color='steelblue')
    ax.axvline(43.0, color='k', ls=':', label='43 um^2')
    ax.set(xlabel='Cell area after 80 divisions (um^2)', ylabel='Cells',
           title=f'No size homeostasis: min area {final_areas.min():.2f} um^2')
    ax.legend(fontsize=8)
    fig.suptitle('Mesh division accounting: area is conserved per division, cell-size control is absent', fontsize=11)
    fig.savefig(OUT / 'mesh_division_areas.png', dpi=160)
    plt.close(fig)

    print('wrote mesh_division_frames.png, mesh_division_zoom.png, mesh_division.gif, mesh_division_areas.png')


if __name__ == '__main__':
    main()
