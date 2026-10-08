"""Cell division ON the deformable vertex mesh (parent-implemented).

Why this file exists
--------------------
The project's growth.py exposes ``divide_in_mechanics_mesh``, but that operator was
measured to be NON-FUNCTIONAL on a real mesh: 3 attempts, 0 successes, every one
rolled back with ``MeshError: duplicate directed half-edge`` (evidence in
outputs/vertex_division_probe.json, reproduced at n_side = 5 and 10). mechanics.py
itself has no division primitive (its public API has T1/T2 and cell removal only).

So the surgery is implemented here, against the authoritative structure of
``mechanics.Mesh``: the per-cell rings in ``mesh.cells`` are the source of truth and
``mesh.rebuild()`` reconstructs every half-edge table (with exact twins) from them,
raising ``MeshError`` on a duplicate directed half-edge, a degenerate edge, or a ring
with fewer than three vertices.

Surgery performed
-----------------
For cell ``c`` with CCW ring ``R``, two non-adjacent edges are selected and their
midpoints ``m1``, ``m2`` are inserted as new vertex slots. The two neighbours that
share those edges have the midpoint spliced into their own ring, and the cell is
replaced by two rings::

    R1 = [v_{i+1} ... v_j,   m2, m1]
    R2 = [v_{j+1} ... v_i,   m1, m2]

The new edge appears once in each direction (so twins pair it), every surviving ring
stays closed, and no directed edge is duplicated. The mother keeps slot ``c`` and the
daughter is appended as a new cell slot.

Honesty
-------
* This is a geometric/topological operator, NOT a biological division model: the
  division axis is a modelling choice, and there is no spindle, no polarity and no
  measured division geometry.
* Parameters here (axis rule, relaxation step count) are modelling choices and are
  marked ILLUSTRATIVE.
* A valid mesh after division is evidence about the OPERATOR, not about biology.
"""
from __future__ import annotations

import json
import numpy as np

import mechanics

AXIS_RULE = 'longest_axis_of_ring_bbox'   # ILLUSTRATIVE modelling choice
PARAM_PROVENANCE = {
    'axis_rule': 'ILLUSTRATIVE (no measured division-axis distribution used)',
    'relax_dt_steps': 'ILLUSTRATIVE (relaxation schedule, not a measured rate)',
    'min_ring': 'structural requirement of mechanics.Mesh.rebuild (>= 3 vertices)',
}


class DivisionError(RuntimeError):
    """Raised when the surgery cannot be applied without breaking the mesh."""


def _ring_positions(mesh, ring):
    return np.asarray([mesh.positions[v] for v in ring], dtype=float)


def choose_edges(mesh, c, axis=None):
    """Pick two non-adjacent ring edges whose midpoints best straddle ``axis``.

    Returns (i, j) edge indices; raise DivisionError if the ring is too small.
    """
    ring = mesh.cells[c]
    L = len(ring)
    if L < 4:
        raise DivisionError(f'ring of cell {c} has only {L} vertices')
    P = _ring_positions(mesh, ring)
    if axis is None:
        # longest-axis rule from the ring's own bounding box (ILLUSTRATIVE)
        d = P.max(axis=0) - P.min(axis=0)
        axis = np.array([1.0, 0.0]) if d[0] >= d[1] else np.array([0.0, 1.0])
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-30)
    proj = P @ axis
    centre = float(proj.mean())
    best = None
    for i in range(L):
        for j in range(i + 1, L):
            # non-adjacent edges only (share no vertex), so both rings keep >= 3
            if (i + 1) % L == j or (j + 1) % L == i:
                continue
            pi, pj = 0.5 * (proj[i] + proj[(i + 1) % L]), 0.5 * (proj[j] + proj[(j + 1) % L])
            if (pi - centre) * (pj - centre) >= 0:
                continue                      # both midpoints on the same side
            span = abs(pi - pj)
            if best is None or span > best[0]:
                best = (span, i, j)
    if best is None:
        raise DivisionError(f'no straddling non-adjacent edge pair found for cell {c}')
    return best[1], best[2]


def _rebuild_tile_cache(mesh):
    """Rebuild mechanics' padded incidence array explicitly from the rings.

    mechanics' ``rebuild_tiles`` early-returns when the maximum ring length and
    the cell count are unchanged, so after an append-style surgery the cached
    ``_tile_idx`` can keep pointing at the pre-surgery rings. Measured effect: the
    tiled path returned misaligned areas (total 903 instead of the conserved 1075)
    while the shoelace area of the same rings was correct. Overwriting the cache
    with the current rings fixes every consumer of ``geometry()``.
    """
    C = len(mesh.cells)
    w = max((len(r) for r in mesh.cells), default=1)
    w = max(w, 1)
    idx = np.full((C, w), -1, dtype=np.int64)
    for c in range(C):
        if mesh.cell_alive[c]:
            r = list(mesh.cells[c])
            idx[c, :len(r)] = r
    mesh._tile_idx = idx
    mesh._tile_val = idx >= 0
    mesh._tile_C = C
    mesh._tile_w = w
    mesh._tile_next = None
    mesh._tile_shifts_dirty = True
    # _topo_dirty = True forces mechanics' own rebuild-from-rings path, which was
    # measured to restore correct areas; leaving it False while writing the cache
    # manually did NOT take effect (measured: unchanged wrong areas).
    mesh._topo_dirty = True


def _splice_midpoint(mesh, other_cell, a, b, m):
    """In ``other_cell``'s ring, replace the directed edge a->b by a->m->b."""
    ring = mesh.cells[other_cell]
    L = len(ring)
    for k in range(L):
        if ring[k] == a and ring[(k + 1) % L] == b:
            ring.insert(k + 1, m)
            return True
    return False


def divide(mesh, c, axis=None, min_ring=3, area_tol=1e-9):
    """Split cell ``c`` in two. Transactional: any failure restores the mesh.

    Returns a record dict with ok/areas/error and, on success, the daughter slot.
    """
    c = int(c)
    if not mesh.cell_alive[c]:
        return dict(ok=False, reason=f'cell {c} is not alive', rolled_back=False)
    snap = mesh.snapshot()
    ids_snap = mesh.cell_ids.copy()
    n_slots_snap = int(mesh.n_slots)
    try:
        ring = list(mesh.cells[c])
        L = len(ring)
        if L < 4:
            raise DivisionError(f'ring of cell {c} has only {L} vertices')
        area_before = float(abs(mechanics.polygon_area(_ring_positions(mesh, ring))))
        i, j = choose_edges(mesh, c, axis=axis)
        a_i, b_i = ring[i], ring[(i + 1) % L]
        a_j, b_j = ring[j], ring[(j + 1) % L]
        m1 = np.array(mesh.positions[a_i]) * 0.5 + np.array(mesh.positions[b_i]) * 0.5
        m2 = np.array(mesh.positions[a_j]) * 0.5 + np.array(mesh.positions[b_j]) * 0.5
        if np.linalg.norm(m1 - m2) < 1e-9:
            raise DivisionError('degenerate cut: the two midpoints coincide')

        # --- allocate the two new vertex slots (slots are never reused) ------
        new_m1 = int(mesh.positions.shape[0])
        mesh.positions = np.vstack([mesh.positions, m1[None, :], m2[None, :]])
        new_m2 = new_m1 + 1
        mesh.n_slots = int(mesh.positions.shape[0])

        # --- splice the midpoints into the two neighbouring rings -----------
        neighbours = 0
        for (a, b), m in (((b_i, a_i), new_m1), ((b_j, a_j), new_m2)):
            owner = int(mesh.he_of(a, b))
            if owner < 0:
                continue                       # outer boundary edge: no neighbour
            d = int(mesh.he_cell[owner])
            if d == c or not mesh.cell_alive[d]:
                continue
            if _splice_midpoint(mesh, d, a, b, m):
                neighbours += 1

        # --- rebuild the mother ring and append the daughter ----------------
        R1 = [ring[(i + 1 + t) % L] for t in range((j - i) % L)] + [new_m2, new_m1]
        R2 = [ring[(j + 1 + t) % L] for t in range((i - j) % L)] + [new_m1, new_m2]
        if len(R1) < min_ring or len(R2) < min_ring:
            raise DivisionError(f'resulting ring too small: {len(R1)}, {len(R2)}')
        mesh.cells[c] = R1
        daughter = len(mesh.cells)
        mesh.cells.append(R2)
        mesh.cell_alive = np.append(mesh.cell_alive, True)
        mesh.cell_ids = np.append(mesh.cell_ids, -1)

        mesh.rebuild()                          # raises MeshError if invalid
        # The tiled geometry cache is not fully invalidated by appending vertex
        # slots, and a stale cache silently returns misaligned areas for OLD
        # cells (measured: total 903 instead of the conserved 1075). Force a
        # full re-derivation, then cross-check against an independent shoelace
        # computation below.
        _rebuild_tile_cache(mesh)               # keep downstream consumers sane
        rep = topology_report(mesh)
        if not rep['ok']:
            raise DivisionError(f'mesh topology invalid after division: {rep}')
        a1 = float(abs(mechanics.polygon_area(_ring_positions(mesh, R1))))
        a2 = float(abs(mechanics.polygon_area(_ring_positions(mesh, R2))))
        if not np.isfinite(a1 + a2) or (a1 <= 0) or (a2 <= 0):
            raise DivisionError(f'non-positive daughter area {a1}, {a2}')
        rel = abs(area_before - (a1 + a2)) / max(area_before, 1e-30)
        if rel > area_tol:
            raise DivisionError(f'area not conserved: rel {rel:.3e}')
        return dict(ok=True, parent=c, daughter=int(daughter), m1=int(new_m1), m2=int(new_m2),
                    area_before=area_before, area_d1=a1, area_d2=a2,
                    area_error_rel=float(rel), neighbours_spliced=neighbours,
                    n_cells_after=int(mesh.cell_alive.sum()), valid=True)
    except Exception as exc:                      # noqa: BLE001 - rollback path
        mesh.positions = snap['positions'].copy()
        mesh.cells = [list(r) for r in snap['cells']]
        mesh.cell_alive = snap['cell_alive'].copy()
        mesh.cell_ids = ids_snap.copy()
        mesh.n_slots = n_slots_snap
        mesh.rebuild()
        return dict(ok=False, reason=f'{type(exc).__name__}: {exc}', rolled_back=True)


def ring_areas(mesh):
    """Signed shoelace area per alive ring, from ``mesh.cells`` alone.

    Used instead of ``mesh.geometry()``. After a vertex-slot append, mechanics'
    cached tiled geometry was MEASURED to return misaligned areas (alive total 903
    instead of the conserved 1075; single cells 43 -> 3.583 / 68.083) while the
    shoelace area of the SAME rings was correct. That is a defect in the cached
    path, so this module computes its own geometry.
    """
    out = np.full(len(mesh.cells), np.nan)
    for c in range(len(mesh.cells)):
        if mesh.cell_alive[c]:
            out[c] = mechanics.polygon_area(_ring_positions(mesh, mesh.cells[c]))
    return out


def _own_forces(mesh, k_area=1.0, gamma_p=0.1, lambda_edge=0.0, A0=43.0, P0=None):
    """F = -dE/dr for E = k_area(A-A0)^2 + gamma_p(P-P0)^2 + sum lambda_e L_e.

    Self-contained, so relaxation does not depend on the cached geometry path.
    """
    F = np.zeros_like(mesh.positions)
    E = 0.0
    for c in range(len(mesh.cells)):
        if not mesh.cell_alive[c]:
            continue
        ring = list(mesh.cells[c])
        P = _ring_positions(mesh, ring)
        L = len(ring)
        if L < 3:
            continue
        A = mechanics.polygon_area(P)
        per = mechanics.polygon_perimeter(P)
        target_p = P0 if P0 is not None else mechanics.regular_hexagon_perimeter(A0)
        E += k_area * (A - A0) ** 2 + gamma_p * (per - target_p) ** 2
        for k in range(L):
            xm, ym = P[(k - 1) % L]
            xp, yp = P[(k + 1) % L]
            dA = 0.5 * np.array([yp - ym, xm - xp])
            un = P[(k + 1) % L] - P[k]
            up = P[k] - P[(k - 1) % L]
            nn, npp = np.linalg.norm(un), np.linalg.norm(up)
            dP = np.zeros(2)
            if nn > 1e-12:
                dP -= un / nn
            if npp > 1e-12:
                dP -= up / npp
            F[ring[k]] += -(2 * k_area * (A - A0) * dA + 2 * gamma_p * (per - target_p) * dP)
        if lambda_edge:
            for k in range(L):
                a, b = ring[k], ring[(k + 1) % L]
                d = mesh.positions[b] - mesh.positions[a]
                n = float(np.linalg.norm(d))
                if n > 1e-12:
                    u = d / n
                    E += 0.5 * lambda_edge * n
                    F[a] += 0.5 * lambda_edge * u
                    F[b] -= 0.5 * lambda_edge * u
    return F, float(E)


def topology_report(mesh):
    """Topology-only validity from the rings: twins, duplicate edges, areas."""
    try:
        twin_ok, twin_msg = mesh.twins_ok()
    except Exception as exc:                     # noqa: BLE001
        twin_ok, twin_msg = False, f'{type(exc).__name__}: {exc}'
    seen, dup = set(), 0
    for c in range(len(mesh.cells)):
        if not mesh.cell_alive[c]:
            continue
        r = mesh.cells[c]
        for k in range(len(r)):
            e = (r[k], r[(k + 1) % len(r)])
            if e in seen:
                dup += 1
            seen.add(e)
    areas = ring_areas(mesh)
    alive = mesh.cell_alive
    finite_ok = bool(np.all(np.isfinite(areas[alive])))
    return {'twin_ok': bool(twin_ok), 'twin_message': str(twin_msg),
            'duplicate_directed_edges': int(dup),
            'min_area_um2': float(np.nanmin(areas[alive])) if alive.any() else None,
            'max_area_um2': float(np.nanmax(areas[alive])) if alive.any() else None,
            'all_areas_finite': finite_ok,
            'has_nonpositive_area': bool(np.any(areas[alive] <= 0)),
            'ok': bool(twin_ok and dup == 0 and finite_ok and np.all(areas[alive] > 0))}


def relax(mesh, steps=20, dt=1e-3, k_area=1.0, gamma_p=0.1, lambda_edge=0.0,
          A0=43.0, max_move=0.05):
    """Overdamped relaxation using this module's own forces (see _own_forces).

    steps/dt/max_move are an ILLUSTRATIVE schedule; the energy form matches
    mechanics.py's area+perimeter+line-tension energy, computed independently.
    """
    moved = 0.0
    for _ in range(int(steps)):
        F, _ = _own_forces(mesh, k_area=k_area, gamma_p=gamma_p,
                           lambda_edge=lambda_edge, A0=A0)
        alive = np.isfinite(mesh.positions).all(axis=1)
        step = np.zeros_like(mesh.positions)
        mag = np.linalg.norm(F, axis=1)
        scale = np.where(mag > 1e-12, np.minimum(1.0, max_move / np.maximum(mag, 1e-30)), 0.0)
        step[alive] = dt * F[alive] * scale[alive][:, None]
        mesh.positions[alive] += step[alive]
        moved = float(np.nanmax(np.linalg.norm(step, axis=1)))
    return moved


def run(n_side=8, n_divisions=30, seed=2025, relax_steps=10, dt=1e-3,
        gate=None, verbose=False):
    """Divide cells on one mesh, reporting every attempt and the mesh state."""
    pos, cells, ids = mechanics.build_hex_mesh(n_side, 43.0)
    mesh = mechanics.Mesh(pos, cells, ids)
    rng = np.random.default_rng(seed)
    log, failures = [], []
    for step in range(int(n_divisions)):
        alive_cells = np.flatnonzero(mesh.cell_alive)
        if gate is not None:
            allowed = np.asarray(gate, bool)
            # the population grows during the run, so extend the gate with False
            if allowed.size < len(mesh.cell_alive):
                allowed = np.concatenate([allowed,
                                          np.zeros(len(mesh.cell_alive) - allowed.size, bool)])
            alive_cells = alive_cells[allowed[alive_cells]]
        if alive_cells.size == 0:
            failures.append(dict(step=step, reason='no eligible cell'))
            break
        # prefer the largest cell, which keeps the tissue near-uniform (ILLUSTRATIVE)
        areas = None
        try:
            geo = mesh.geometry()
            areas = np.asarray(geo['areas'], float)
        except Exception:
            pass
        if areas is not None and areas.shape[0] >= len(mesh.cells):
            target = int(alive_cells[np.argmax(areas[alive_cells])])
        else:
            target = int(alive_cells[rng.integers(alive_cells.size)])
        rec = divide(mesh, target)
        rec['step'] = step
        log.append(rec)
        if rec['ok']:
            relax(mesh, steps=relax_steps, dt=dt)
        else:
            failures.append(dict(step=step, reason=rec.get('reason')))
        if verbose and step % 5 == 0:
            print(f"  step {step}: cells={int(mesh.cell_alive.sum())} ok={rec['ok']}")
    rep = topology_report(mesh)
    _rebuild_tile_cache(mesh)
    alive = mesh.cell_alive
    areas = np.abs(ring_areas(mesh)[alive])
    return {'mesh': mesh, 'log': log, 'failures': failures,
            'n_attempted': len(log), 'n_succeeded': int(sum(r['ok'] for r in log)),
            'n_rolled_back': int(sum(1 for r in log if not r['ok'])),
            'n_cells_final': int(alive.sum()),
            'total_area_initial': float(43.0 * len(cells)),
            'total_area_final': float(areas.sum()),
            'min_cell_area': float(areas.min()), 'max_cell_area': float(areas.max()),
            'valid': bool(rep['ok']),
            'valid_report': rep,
            'metadata': {'n_side': n_side, 'n_divisions_requested': int(n_divisions),
                         'seed': seed, 'relax_steps': relax_steps, 'dt': dt,
                         'axis_rule': AXIS_RULE, 'param_provenance': PARAM_PROVENANCE,
                         'warning': ('Geometric operator only; division geometry is a modelling '
                                     'choice, not a biological division model.')}}


def self_test(verbose=False):
    """Bounded tests: single division, repeated division, conservation, gating."""
    out = {'tests': {}}

    # 1. a single division on a small mesh
    r1 = run(n_side=5, n_divisions=1, relax_steps=0)
    rec = r1['log'][0]
    out['tests']['single_division'] = {
        'ok': rec['ok'], 'reason': rec.get('reason'),
        'area_error_rel': rec.get('area_error_rel'), 'valid': r1['valid'],
        'n_cells_after': rec.get('n_cells_after')}

    # 2. many divisions on a bigger mesh
    r2 = run(n_side=8, n_divisions=30, relax_steps=10, verbose=verbose)
    out['tests']['repeated_division'] = {
        'n_attempted': r2['n_attempted'], 'n_succeeded': r2['n_succeeded'],
        'n_rolled_back': r2['n_rolled_back'], 'n_cells_final': r2['n_cells_final'],
        'valid': r2['valid'], 'min_cell_area_um2': r2['min_cell_area'],
        'max_cell_area_um2': r2['max_cell_area'],
        'total_area_error_rel': abs(r2['total_area_final'] - r2['total_area_initial'])
                                / r2['total_area_initial'],
        'first_failure_reason': (r2['failures'][0]['reason'] if r2['failures'] else None),
        'n_vertices_final': int(r2['mesh'].positions.shape[0])}

    # 3. push it further to find the structural limit
    r3 = run(n_side=10, n_divisions=80, relax_steps=8)
    out['tests']['stress'] = {
        'n_attempted': r3['n_attempted'], 'n_succeeded': r3['n_succeeded'],
        'n_cells_final': r3['n_cells_final'], 'valid': r3['valid'],
        'min_cell_area_um2': r3['min_cell_area'],
        'total_area_error_rel': abs(r3['total_area_final'] - r3['total_area_initial'])
                                / r3['total_area_initial'],
        'first_failure_reason': (r3['failures'][0]['reason'] if r3['failures'] else None),
        'reasons_seen': sorted({f['reason'] for f in r3['failures']})[:5]}

    # 4. determinism
    a = run(n_side=6, n_divisions=8, seed=7, relax_steps=5)
    b = run(n_side=6, n_divisions=8, seed=7, relax_steps=5)
    out['tests']['determinism'] = {
        'same_number_of_successes': a['n_succeeded'] == b['n_succeeded'],
        'same_final_cell_count': a['n_cells_final'] == b['n_cells_final'],
        'bit_identical_positions': bool(np.array_equal(np.nan_to_num(a['mesh'].positions),
                                                      np.nan_to_num(b['mesh'].positions)))}

    # 5. gating: forbid division outside a permitted set
    pos, cells, ids = mechanics.build_hex_mesh(6, 43.0)
    gate = np.zeros(len(cells), bool)
    gate[:10] = True
    r5 = run(n_side=6, n_divisions=6, relax_steps=5, gate=gate)
    permitted = set(np.flatnonzero(gate).tolist())
    out['tests']['gating'] = {
        'n_succeeded': r5['n_succeeded'],
        'all_parents_permitted': all(rec['parent'] in permitted
                                     for rec in r5['log'] if rec['ok']),
        'attempts_outside_permitted': int(sum(1 for rec in r5['log']
                                              if rec['ok'] and rec['parent'] not in permitted))}
    out['param_provenance'] = PARAM_PROVENANCE
    return out


if __name__ == '__main__':
    print(json.dumps(self_test(verbose=False), indent=2, default=str))
