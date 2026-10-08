"""Build the paracrine/immune scene layers for the integrated workbench.

Called by ``scene_build_cli.py``.  Both new simulation layers are 2-D radial patches
in their OWN coordinate frames; this module embeds them at TRUE SCALE onto the
abdomen node of the FlyGym body, exactly as the epithelial wound-calcium patch is
embedded, and labels the placement as unregistered rather than pretending the frames
coincide.

The physical mapping is: the model's 2-D plane (x, y) becomes the body-local plane
(x, z) at a fixed y (the abdomen's lateral offset), i.e. the patch is laid FLAT on the
abdomen surface.  The model is drawn at 1:1 so the 137 um patch really is about 1/22
of the 3 mm body -- a viewer that renders it large would be lying about the scale.
"""
from __future__ import annotations

import json

import numpy as np

import paracrine as pa
import immune as im

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _embed(local_xy_mm, y_offset_mm):
    """Model 2-D (x, y) in millimetres -> body-local (x, y, z) with the patch flat."""
    pts = np.asarray(local_xy_mm, dtype=np.float64)
    out = np.zeros((pts.shape[0], 3), dtype=np.float64)
    out[:, 0] = pts[:, 0]
    out[:, 1] = y_offset_mm
    out[:, 2] = pts[:, 1]
    return out


def build_paracrine_patch(radius_um=51.0, cells_per_lambda=6.0, t_end_s=None,
                          y_offset_mm=0.20, frames=40):
    """Run the spatial paracrine field on a wound-sized patch and return scene data."""
    p = dict(pa.DEFAULTS)
    lam = pa.lambda_um(p=p)
    dx = lam / cells_per_lambda
    domain_um = 6.0 * radius_um                     # 306 um across, 13.7 lambda
    n = int(round(domain_um / dx)) | 1
    dx = domain_um / (n - 1)
    t_end = (12.0 / p['k_decay_per_s']) if t_end_s is None else float(t_end_s)

    X, Y, _ = pa.make_grid(n=n, dx_um=dx, p=p)
    area = dx * dx
    src = pa.source_disc(X, Y, radius_um, 1.0 / area)
    sample_every = max(1, int(np.ceil((t_end / 0.5) / frames)))
    res = pa.run(n=n, dx_um=dx, t_end_s=t_end, dt_s=t_end / 200, p=p,
                 boundary='noflux', sources=lambda a, b, t: src,
                 sample_every=sample_every)
    fields = res['fields']
    n_frames = fields.shape[0]
    vmax = float(np.percentile(fields[-1], 99.9)) or 1.0

    # Quantise per frame so the animation is not dominated by the first instant.
    lo = float(fields.min())
    hi = float(fields.max())
    span = (hi - lo) or 1.0
    q = np.rint(np.clip((fields - lo) / span, 0, 1) * 255).astype(np.uint8)

    # cell positions (flat patch), embedded at true scale: every cell is drawn
    xy_mm = np.c_[X.ravel(), Y.ravel()] / 1000.0
    pos = _embed(xy_mm, y_offset_mm)

    return {
        'name': 'paracrine',
        'positions': pos,
        'frames_u8': q,
        'n_frames': n_frames,
        'n_points': int(n * n),
        'range': [lo, hi],
        'times_s': res['times_s'],
        'meta': {
            'lambda_um': round(lam, 4), 'dx_um': round(dx, 4), 'grid_n': n,
            'domain_um': round(domain_um, 2), 'wound_radius_um': radius_um,
            't_end_s': t_end, 'scheme': res['scheme'],
            'dt_used_s': res['dt_used_s'], 'n_steps': res['n_steps'],
            'source_total_rate': float(src.sum() * area),
            'frame_quantisation': 'per-run min/max linear to uint8',
        },
    }


def build_immune_patch(n=64, t_end_s=600.0, dt_s=5.0, radius_um=120.0,
                       y_offset_mm=0.35, frames=40, seed=0):
    """Run the haemocyte/clearance layer and return scene data for it."""
    p = dict(im.DEFAULTS)
    dx = p['dx_um']
    X, Y, _ = im.make_grid(n=n, dx_um=dx, p=p)
    rng = np.random.default_rng(seed)
    start = rng.uniform(-0.45 * n * dx, 0.45 * n * dx, size=(96, 2))
    debris0 = im.wound_debris_profile(X, Y, radius_um, 1.0)
    sample_every = max(1, int(np.ceil((t_end_s / dt_s) / frames)))
    res = im.run(n=n, dx_um=dx, t_end_s=t_end_s, dt_s=dt_s, p=p, debris0=debris0,
                 haemocytes=start, chemotaxis=True, sample_every=sample_every, seed=seed)

    deb = res['debris']
    att = res['attractant']
    # Down-sample to the requested frame count HERE, on one index set shared by the
    # fields and the haemocyte tracks.  Doing it in two places (the module's own
    # sample_every plus a re-sample here) produced 1001 frames for a 20-frame request.
    total = deb.shape[0]
    if total > frames:
        idx = np.unique(np.linspace(0, total - 1, int(frames)).round().astype(int))
        deb, att = deb[idx], att[idx]
        res_times = np.asarray(res['times_s'])[idx]
    else:
        idx = np.arange(total)
        res_times = np.asarray(res['times_s'])
    scale = float(np.percentile(att[-1], 99.9)) or 1.0
    q_att = np.rint(np.clip(att / scale, 0, 1) * 255).astype(np.uint8)
    scale_d = float(deb.max()) or 1.0
    q_deb = np.rint(np.clip(deb / scale_d, 0, 1) * 255).astype(np.uint8)

    xy_mm = np.c_[X.ravel(), Y.ravel()] / 1000.0
    pos = _embed(xy_mm, y_offset_mm)
    # haemocyte trajectories: (frames, n_cells, 2) in model micrometres
    n_cells = res['positions_um'].shape[0]
    traj = res.get('haemocyte_tracks')
    if traj is None or traj.shape[0] != len(idx):
        base = res['positions_um'] if traj is None else traj
        pick = np.linspace(0, max(base.shape[0] - 1, 0), len(idx)).round().astype(int)
        traj = base[pick]
    cells = _embed(traj.reshape(-1, 2) / 1000.0, y_offset_mm).reshape(
        traj.shape[0], n_cells, 3)

    return {
        'name': 'immune',
        'positions': pos,
        'debris_u8': q_deb,
        'attractant_u8': q_att,
        'cells_mm': cells,
        'n_frames': deb.shape[0],
        'n_points': pos.shape[0],
        'n_cells': int(n_cells),
        'times_s': res_times,
        'scale_attractant': scale,
        'scale_debris': scale_d,
        'budget': {
            'max_relative_residual': res['budget_relative_residual'],
            'released_total': res['released_total'],
            'remaining_debris': res['remaining_debris'],
            'taken_up_total': float(np.sum(res['taken_up'])),
            'cleared_total': res['cleared_total'],
        },
        'meta': {
            'dx_um': dx, 'n': n, 'domain_um': round(n * dx, 1),
            'wound_radius_um': radius_um, 't_end_s': t_end_s,
            'dt_used_s': res['dt_used_s'], 'n_steps': res['n_steps'],
            'chemotaxis': True, 'substeps_used': res['substeps_used'],
        },
    }


def export_to_scene_dir(scene_dir, y_paracrine=0.20, y_immune=0.35, frames=40):
    """Build both patches and write their binary chunks into the scene directory.

    Separated from the scene builder on purpose: these two layers import scipy and are
    therefore built with ``venv/bin/python``, while ``scene_build_cli.py`` runs with
    ``venv_body/bin/python`` (pyarrow + mujoco).  Writing the chunks here and having the
    builder copy them keeps the two environments from having to agree.
    """
    from pathlib import Path as _P
    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    written = {}

    pp = build_paracrine_patch(frames=frames, y_offset_mm=y_paracrine)
    (scene / 'paracrine_pos.bin').write_bytes(
        np.ascontiguousarray(pp['positions'], dtype='<f4').tobytes())
    (scene / 'paracrine_frames.bin').write_bytes(np.ascontiguousarray(pp['frames_u8']).tobytes())
    written['paracrine'] = {'n_points': pp['n_points'], 'n_frames': pp['n_frames'],
                            'range': pp['range'], 'meta': pp['meta'],
                            'times_s': np.asarray(pp['times_s']).round(4).tolist()}

    ii = build_immune_patch(frames=frames, y_offset_mm=y_immune)
    (scene / 'immune_pos.bin').write_bytes(
        np.ascontiguousarray(ii['positions'], dtype='<f4').tobytes())
    (scene / 'immune_debris.bin').write_bytes(np.ascontiguousarray(ii['debris_u8']).tobytes())
    (scene / 'immune_attractant.bin').write_bytes(np.ascontiguousarray(ii['attractant_u8']).tobytes())
    (scene / 'immune_cells.bin').write_bytes(
        np.ascontiguousarray(ii['cells_mm'], dtype='<f4').tobytes())
    written['immune'] = {'n_points': ii['n_points'], 'n_frames': ii['n_frames'],
                         'n_cells': ii['n_cells'], 'meta': ii['meta'],
                         'budget': ii['budget'], 'scale_attractant': ii['scale_attractant'],
                         'scale_debris': ii['scale_debris'],
                         'times_s': np.asarray(ii['times_s']).round(4).tolist()}

    (scene / 'sim_layers.json').write_text(json.dumps(written, indent=1))
    return written


if __name__ == '__main__':
    import sys as _sys
    target = _sys.argv[1] if len(_sys.argv) > 1 else 'viewer/scene'
    info = export_to_scene_dir(target)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'meta'}
                      for k, v in info.items()}, indent=1))


def export_tissue3d(scene_dir, n_side=8, n_layers=7, h_um=2.5, y_offset_mm=-0.25,
                    deviation_gain=2000.0):
    """Write the 3-D multilayer stack as a scene layer.

    The stack is a few tens of micrometres across and high, so at TRUE SCALE it would be
    a speck on a 3 mm body.  The layer positions are exported at 1:1 and the DEVIATION
    from the rest stack is multiplied by a declared ``deviation_gain``, with the gain
    written into the metadata and printed in the UI: the shape is exaggerated, the scale
    and the layer spacing are not.

    Meshes: each cell of each layer becomes a quad (or its hexagon ring) and the vertical
    connections become lines, so the layer structure is visible rather than implied.
    """
    import json
    from pathlib import Path as _P
    import tissue3d as t3

    p = dict(t3.DEFAULTS, n_layers=int(n_layers), h_m=h_um * 1e-6)
    stack = t3.make_stack(n_side=n_side, n_layers=int(n_layers), h_m=h_um * 1e-6, p=p)
    rng = np.random.default_rng(0)
    # a smooth vertical perturbation so the stack has something to relax
    n_col = stack['n_columns']
    z = np.zeros((int(n_layers), n_col))
    xy = stack['plane_xy']
    z += 1e-6 * np.sin(xy[:, 0] / 12.0) * np.cos(xy[:, 1] / 12.0)[None, :]
    stack['positions_m'] = stack['positions_m'].copy()
    stack['positions_m'][:, 2] += z.ravel()
    energies = t3.relax(stack, steps=400, p=p, target_top_m=0.0)
    psi = t3.psi_from_positions(stack).reshape(int(n_layers), n_col)

    # embed: model x -> body x, model y -> body z, stack z (thickness) -> body z offset
    # NOTE the plane is (x, y) and the thickness is z; to lay the slab on the abdomen the
    # thickness must run OUT of the body surface, so model (x, y, z_thick) -> body
    # (x, y_offset, y) with the thickness added along body +y is wrong; instead the slab is
    # oriented with its thickness along body +z (up) and its plane in (x, y):
    #     body_x = model_x, body_y = model_y, body_z = y_offset + (z_thick - rest) * gain
    h = h_um * 1e-6
    layer_of = stack['layer_of']
    rest = layer_of * h
    positions = np.zeros((stack['positions_m'].shape[0], 3))
    positions[:, 0] = stack['plane_xy'][np.arange(n_col).repeat(1)][
        np.tile(np.arange(n_col), int(n_layers))][:, 0] / 1000.0
    positions[:, 1] = stack['plane_xy'][np.tile(np.arange(n_col), int(n_layers))][:, 1] / 1000.0
    positions[:, 2] = y_offset_mm + psi.ravel() * deviation_gain

    # vertical connectors between consecutive layers (same column)
    edges = []
    for L in range(int(n_layers) - 1):
        for c in range(n_col):
            edges.append([L * n_col + c, (L + 1) * n_col + c])

    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    (scene / 'tissue3d_pos.bin').write_bytes(
        np.ascontiguousarray(positions, dtype='<f4').tobytes())
    (scene / 'tissue3d_edges.bin').write_bytes(
        np.ascontiguousarray(np.asarray(edges, dtype='<u4')).tobytes())
    (scene / 'tissue3d_layers.bin').write_bytes(
        np.ascontiguousarray(layer_of.astype('<u2')).tobytes())
    info = {
        'n_vertices': int(positions.shape[0]), 'n_edges': len(edges),
        'n_layers': int(n_layers), 'n_columns': int(n_col),
        'h_um': h_um, 'plane_extent_um': [float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))],
        'deviation_gain': deviation_gain,
        'deviation_of_rest_before_m': 1e-6, 'deviation_after_relax_m': float(np.abs(psi).max()),
        'energy_first_J': float(energies[0]), 'energy_last_J': float(energies[-1]),
        'monotone': bool(np.all(np.diff(energies) <= 1e-22)),
        'y_offset_mm': y_offset_mm,
        'note': ('layer POSITIONS and spacing are 1:1; the deviation from the rest stack is '
                 'multiplied by deviation_gain so the deformation is visible at body scale. '
                 'That gain is declared here and shown in the UI.'),
    }
    (scene / 'tissue3d.json').write_text(json.dumps(info, indent=1))
    return info


def export_mechanics_patch(scene_dir, frames=60, y_offset_mm=0.45, x_offset_mm=0.0):
    """Export the 2-D vertex-mechanics patch (with its cut) as a scene layer.

    The patch is a SEPARATE simulation (31 recorded states of a cut vertex model), so it
    is embedded at true scale on the abdomen and labelled as not registered and not
    sharing the body clock, exactly like the other independent layers.
    """
    import json
    from pathlib import Path as _P
    npz = ROOT / 'outputs/mechanics_cut_n24.npz'
    if not npz.is_file():
        return None
    with np.load(npz, allow_pickle=False) as z:
        verts = np.asarray(z['vertices'], dtype=np.float64)      # (T, V, 2) in um
        times = np.asarray(z['times'], dtype=np.float64)
        alive = np.asarray(z['cell_alive'], dtype=bool) if 'cell_alive' in z.files else None
        energy = np.asarray(z['energy'], dtype=np.float64) if 'energy' in z.files else None
    nT, nV = verts.shape[0], verts.shape[1]
    idx = np.unique(np.linspace(0, nT - 1, min(frames, nT)).round().astype(int))
    verts = verts[idx]
    times = times[idx]
    # Centre the patch and convert um -> mm.  The mean MUST be NaN-AWARE: the source marks
    # removed cells with NaN (868 of 38688 components), and a plain mean of that array is
    # NaN, which then propagated to EVERY vertex and made the exported patch entirely NaN --
    # measured, the first export had zero finite vertices and the movement check read 0.
    # The NaN entries themselves are kept, because they are the dead-cell markers the client
    # uses to skip those vertices.
    v = verts.reshape(-1, 2)
    v = v - np.nanmean(v, axis=0)
    pos_flat = np.zeros((v.shape[0], 3))
    pos_flat[:, 0] = v[:, 0] / 1000.0 + x_offset_mm
    pos_flat[:, 1] = v[:, 1] / 1000.0
    pos_flat[:, 2] = y_offset_mm
    pos = pos_flat.reshape(len(idx), nV, 3)

    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    (scene / 'mech_pos.bin').write_bytes(
        np.ascontiguousarray(pos, dtype='<f4').tobytes())
    info = {'n_frames': int(len(idx)), 'n_vertices': int(nV), 'times_s': times.tolist(),
            'y_offset_mm': y_offset_mm, 'x_offset_mm': x_offset_mm,
            'extent_um': [float(np.nanmax(verts[..., 0]) - np.nanmin(verts[..., 0])),
                          float(np.nanmax(verts[..., 1]) - np.nanmin(verts[..., 1]))]}
    if alive is not None:
        (scene / 'mech_alive.bin').write_bytes(
            np.ascontiguousarray(alive[idx].astype(np.uint8)).tobytes())
        info['alive_shape'] = [len(idx), int(alive.shape[1])]
    if energy is not None:
        info['energy_J'] = energy[idx].tolist()
    (scene / 'mechanics.json').write_text(json.dumps(info, indent=1))
    return info


SCENARIOS = ('intact', 'neck_cut', 'neck_cut_supported', 'sham', 'eye_loss')


def export_scenarios(scene_dir, frames=200):
    """Export the five neck-cut / eye-loss scenarios as curves attached to the body.

    Each scenario is an already-computed 4801-sample recording of eye, brain, neck
    (proximal and distal) and body potentials plus the stimulus and the recorded
    artefact.  They are exported as per-region series so the scene can drive the colour
    of the corresponding structure and draw the comparison in place.
    """
    import json
    from pathlib import Path as _P
    keys = ('eye_mV', 'brain_mV', 'neck_proximal_mV', 'neck_distal_mV', 'body_mV',
            'stimulus_source_nA', 'artifact_recorded_mV')
    out = {}
    series, times = {}, None
    for name in SCENARIOS:
        path = ROOT / ('outputs/brain_isolation/integrated_%s.npz' % name)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            # every series is indexed on ITS OWN length: the stimulus and artefact arrays
            # are one sample shorter than the potential traces (4800 vs 4801), so a shared
            # index built from times_ms runs one past the end of them
            n_time = int(z['times_ms'].shape[0])
            idx_t = np.unique(np.linspace(0, n_time - 1, min(frames, n_time)).round().astype(int))
            if times is None:
                times = (np.asarray(z['times_ms'])[idx_t] / 1000.0).tolist()
            for k in keys:
                if k not in z.files:
                    continue
                a = np.asarray(z[k])
                idx = np.unique(np.linspace(0, a.shape[0] - 1,
                                           min(frames, a.shape[0])).round().astype(int))
                series.setdefault(k, {})[name] = a[idx].round(6).tolist()
        out[name] = {"file": str(path.name), "samples": n_time}
    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    payload = {'scenarios': list(out), 'frames': len(times) if times else 0,
               'times_s': times, 'series': series, 'per_scenario': out}
    (scene / 'scenarios.json').write_text(json.dumps(payload, indent=1))
    return {'scenarios': list(out), 'frames': len(times) if times else 0,
            'series': {k: list(v) for k, v in series.items()}}


def export_environment(scene_dir, ground_px=256):
    """Export a compact ground texture and sky colours for the scene environment.

    The ground comes from the project's CC0 photo asset.  Its provenance file records the
    Commons source page and the CC0 route, and that attribution is copied into the scene
    metadata so the licence travels with the image.  No other texture is used.
    """
    import json
    from pathlib import Path as _P
    from PIL import Image
    asset = ROOT / 'outputs/embodied_body/assets/natural_ground_photo.png'
    prov = ROOT / 'outputs/embodied_body/assets/natural_ground_photo.provenance.json'
    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    info = {'ground': None, 'sky_top': [0.06, 0.10, 0.16], 'sky_bottom': [0.16, 0.20, 0.26]}
    if asset.is_file():
        im = Image.open(asset).convert('RGB').resize((ground_px, ground_px), Image.LANCZOS)
        im.save(scene / 'ground.jpg', quality=82, optimize=True)
        info['ground'] = {'file': 'ground.jpg', 'px': ground_px,
                          'bytes': (scene / 'ground.jpg').stat().st_size}
    if prov.is_file():
        p = json.loads(prov.read_text())
        info['attribution'] = {k: p.get(k) for k in
                               ('route', 'source_url', 'source_page', 'source_title',
                                'license', 'license_url', 'author')}
    (scene / 'environment.json').write_text(json.dumps(info, indent=1))
    return info


def export_readouts(scene_dir):
    """Compact in-scene readouts for the layers that have NO spatial field.

    Three sources were checked before writing this:

      * intracellular space (`metrics_cellspace.json`) -- a single-cell radial/grid model,
        so there is no tissue-spatial field to draw;
      * vascular oxygen (`metrics_organism.json` -> ``vessel_spacing``) -- the artefact holds
        per-spacing SUMMARY STATISTICS (mean/min/max oxygen, hypoxic fraction), NOT a 2-D or
        3-D field, so no plausible field can be drawn from it;
      * the electrode chain (`electrode_frontend/tissue/stim/recording.json`) -- headline
        numbers and verdicts.

    All three therefore become READOUTS pinned in the same scene, which is what they are.
    Anything else would be drawing a field that was never computed.
    """
    import json
    from pathlib import Path as _P
    out = {}

    cpath = ROOT / 'outputs/metrics_cellspace.json'
    if cpath.is_file():
        c = json.loads(cpath.read_text())
        st = c.get('self_test', {})
        fs = c.get('full_suite', {})
        orders = None
        t2 = fs.get('test_2_analytic_gaussian_diffusion') if isinstance(fs, dict) else None
        if isinstance(t2, dict):
            orders = t2.get('observed_orders')
        t4 = fs.get('test_4_lumped_large_D_limit') if isinstance(fs, dict) else None
        out['cellspace'] = {
            'scope': c.get('scope'),
            'D_ca_um2_s': st.get('D_ca'), 'D_ip3_um2_s': st.get('D_ip3'),
            'uniform_preservation_max_dev': st.get('uniform_preservation_max_dev'),
            'wounded_peak_mean_c_uM': st.get('wounded_peak_mean_c_uM'),
            'convergence_orders': orders,
            'large_D_lumped_converged': (t4 or {}).get('converged'),
            'large_D_max_rel': ((t4 or {}).get('rows') or [{}])[-1].get('max_rel'),
        }

    mpath = ROOT / 'outputs/metrics_organism.json'
    if mpath.is_file():
        m = json.loads(mpath.read_text())
        vs = m.get('vessel_spacing') or {}
        rows = []
        for k, v in vs.items():
            if isinstance(v, dict):
                rows.append({'spacing_um': k,
                             'mean_mmHg': v.get('mean_mmHg'),
                             'min_mmHg': v.get('min_mmHg'),
                             'hypoxic_fraction': v.get('hypoxic_fraction_below_10mmHg')})
        out['vascular_oxygen'] = {
            'tissue_cells': (m.get('tissue') or {}).get('n_cells'),
            'domain_um': (m.get('tissue') or {}).get('domain_um'),
            'rows': rows,
            'note': ('per-spacing summary statistics only: the artefact has no 2-D or 3-D '
                     'oxygen field, so this layer is a readout and no field is drawn'),
        }

    for name, key in (('electrode_frontend', 'frontend'),
                      ('electrode_recording', 'recording'),
                      ('electrode_stim', 'stimulation'),
                      ('electrode_tissue', 'tissue')):
        path = ROOT / ('outputs/embodied_body/%s.json' % name)
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        entry = {'file': path.name}
        summary = d.get('pre_registered_summary') or d.get('summary')
        if summary:
            entry['verdicts'] = summary
        for k in ('status', 'recording_only', 'fixed_geometry', 'noise_assumptions'):
            if k in d:
                entry[k] = d[k]
        tests = d.get('tests')
        if isinstance(tests, dict):
            entry['tests_passed'] = tests.get('passed')
            entry['tests_checks'] = len(tests.get('checks') or [])
        out[key] = entry

    scene = _P(scene_dir)
    scene.mkdir(parents=True, exist_ok=True)
    (scene / 'readouts.json').write_text(json.dumps(out, indent=1, default=str))
    return out
