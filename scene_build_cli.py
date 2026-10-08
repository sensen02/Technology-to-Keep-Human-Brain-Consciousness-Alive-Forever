"""CLI: build the single-scene payload from real artefacts.

Run with the environment that has numpy + pyarrow:
    PYTHONPATH=<root>/vendor/pylibs <root>/venv_body/bin/python scene_build_cli.py

Writes viewer/scene/*.bin + viewer/scene/manifest.json. Nothing else is
touched. Every layer records its provenance, units, clock and byte size, and a
transform audit proves the CNS map is rigid and distance preserving.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import scene_build as sb

ROOT = sb.ROOT
SCENE = sb.SCENE

# payload budget (the viewer already refuses >35 MiB in one file; this is the
# per-scene budget across all chunks)
BUDGET_SOMAS = 40000
BUDGET_EDGES = 24000
BUDGET_NEURON_ACTIVITY = BUDGET_SOMAS
FRAMES = 100


def leg_layout(contact_positions_mm, tarsus_rest_mm=None, frame=0):
    """Name the six recorded contacts by matching them to the body's own feet.

    The episode stores six contacts in a fixed order but never names them. Rather
    than assume an order, each contact is matched to the nearest tarsus tip of
    the FlyGym body model (lf/lm/lh/rf/rm/rh tarsus5) and the match margin is
    reported. Measured margins in this recording are ~0.10 mm for the correct
    foot against ~1.14-1.94 mm for the next nearest, i.e. unambiguous.
    """
    p = np.asarray(contact_positions_mm, dtype=np.float64)   # (frames, 6, 3)
    mean = p[frame]
    if not tarsus_rest_mm:
        # fall back to a geometric grouping, clearly labelled as such
        side = np.where(mean[:, 1] >= 0, 'l', 'r')
        order = np.argsort(-mean[:, 0])
        rank, names = {}, ['hind', 'middle', 'front']
        for s in ('l', 'r'):
            rows = [i for i in order if side[i] == s]
            for k, row in enumerate(rows):
                rank[int(row)] = s + '_' + (names[k] if k < len(names) else 'x%d' % k)
        return [rank[i] for i in range(len(mean))], mean, {'method': 'geometric fallback', 'margins_mm': None}
    labels, margins = [], []
    for row in mean:
        d = sorted(((float(np.linalg.norm(np.asarray(v) - row)), k) for k, v in tarsus_rest_mm.items()))
        labels.append(d[0][1])
        margins.append([round(d[0][0], 4), round(d[1][0], 4)])
    if len(set(labels)) != len(labels):
        raise RuntimeError('contact-to-foot matching is not one-to-one: %s' % labels)
    return labels, mean, {'method': 'nearest FlyGym tarsus5 tip at frame 0',
                          'margins_mm': margins,
                          'margin_note': 'first value = distance to the matched foot, '
                                         'second = distance to the next nearest'}


def tarsus_tips(npz_path, frame=0):
    """The six tarsus5 tip positions at one body frame, keyed by leg name.

    The comparison must use the SAME frame as the contacts: this recording walks
    ~145 mm, so an average over all frames sits far from the body and would match
    every contact to one arbitrary foot.
    """
    import json as _json
    from pathlib import Path as _Path
    body = _json.loads((_Path(ROOT) / 'viewer/scene/body.json').read_text())
    node_names = [n['name'] for n in body['nodes']]
    positions = np.asarray(body['frames'], dtype=np.float64)[frame]
    tips = {}
    for side in ('l', 'r'):
        for seg in ('f', 'm', 'h'):
            node = 'nmf/%s%s_tarsus5' % (side, seg)
            if node in node_names:
                tips['%s_%s' % (side, seg)] = positions[node_names.index(node)]
    if len(tips) != 6:
        raise RuntimeError('expected six tarsus5 tips, found %d: %s' % (len(tips), sorted(tips)))
    return tips


def build_legs(npz_path, ix):
    with np.load(npz_path, allow_pickle=False) as z:
        t = np.asarray(z['truth/time_s'])[ix]
        cp_all = np.asarray(z['truth/contact_positions_mm'], dtype=np.float64)
        present_all = np.asarray(z['truth/contact_present'])
        contact_all = np.asarray(z['truth/contact_positions_mm'], dtype=np.float64)
        cp = cp_all[ix]
        present = present_all[ix]
        force = np.asarray(z['truth/contact_forces'], dtype=np.float64)[ix]
        act = np.asarray(z['truth/actuator_forces'], dtype=np.float64)[ix]
        phase = np.asarray(z['truth/cpg_phase_rad'], dtype=np.float64)[ix]
        magnitude = np.asarray(z['truth/cpg_magnitude'], dtype=np.float64)[ix]
        desc = np.asarray(z['observed/descending_rate_hz'], dtype=np.float64)[ix]
        desc_l = np.asarray(z['observed/descending_rate_left_hz'], dtype=np.float64)[ix]
        desc_r = np.asarray(z['observed/descending_rate_right_hz'], dtype=np.float64)[ix]
        speed_scale = np.asarray(z['commands/speed_scale'], dtype=np.float64)[ix]
        turn = np.asarray(z['commands/turn'], dtype=np.float64)[ix]
    # leg order comes from the recorded rest positions of the contacts themselves
    names, _, match_info = leg_layout(contact_all, tarsus_tips(npz_path, frame=0), frame=0)
    leg_index = np.tile(np.arange(len(names), dtype=np.uint8), (len(cp), 1))

    # unique contact sites: exact recorded positions, compacted. The sentinel
    # row holds "not in contact" and carries no position.
    flat = np.round(contact_all.reshape(-1, 3), 6)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(contact_all.shape[0], contact_all.shape[1])[ix]
    sentinel = len(uniq)
    site_index = np.where(present, inverse, sentinel).astype(np.uint16)
    contact_table = np.vstack([uniq.astype(np.float64),
                               np.array([[np.nan, np.nan, np.nan]])])

    magnitude_n = np.linalg.norm(force, axis=2)              # (frames, 6)
    norm = np.where(magnitude_n[..., None] > 0, magnitude_n[..., None], 1.0)
    direction = force / norm
    return {
        'names': names,
        'times_s': t,
        'leg_index': leg_index,
        'present': present,
        'site_index': site_index,
        'contact_table': contact_table,
        'force_magnitude': magnitude_n.astype(np.float32),
        'force_direction': direction.astype(np.float32),
        'actuator_forces': act.astype(np.float32),
        'cpg_phase_rad': phase.astype(np.float32),
        'cpg_magnitude': magnitude.astype(np.float32),
        'descending_hz': desc.astype(np.float32),
        'descending_left_hz': desc_l.astype(np.float32),
        'descending_right_hz': desc_r.astype(np.float32),
        'speed_scale': speed_scale.astype(np.float32),
        'turn': turn.astype(np.float32),
        'unit_note': 'contact force in model units (as recorded); direction unitless',
        'match_info': match_info,
    }


def z_contacts(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        return np.asarray(z['truth/contact_positions_mm'], dtype=np.float64)

def build_electrodes(cns_local_mm, somas_um, reg, in_cns_mask):
    """Electrode array at the FIXED 7 um / 20 um geometry, single source.

    Placement: a square 20 um lattice centred on the VNC landmark, oriented in
    the CNS local axes. Membership ("which channels see which neuron") is
    recomputed here for THIS geometry, because the stored access map was built
    at the older 100 um / 10 um geometry and must not be reused.
    """
    pitch = sb.ELECTRODE_PITCH_UM / 1000.0
    side = 32                                   # 1024 candidate sites
    grid = (np.arange(side) - (side - 1) / 2.0) * pitch
    gx, gy = np.meshgrid(grid, grid)
    vnc = np.asarray(reg['local_landmarks_mm']['vnc_leg_nerve'], dtype=np.float64)
    centres = np.c_[gx.ravel() + vnc[0], gy.ravel() + vnc[1], np.full(gx.size, vnc[2])]

    local = cns_local_mm[in_cns_mask]
    if len(local) == 0:
        raise RuntimeError('no CNS somas available for electrode membership')
    radius = sb.ELECTRODE_CAPTURE_RADIUS_UM / 1000.0
    cell = radius
    keys = np.floor(local / cell).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    membership = []
    span = int(np.ceil(radius / cell))
    for c in centres:
        base = np.floor(c / cell).astype(np.int64)
        hits = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    got = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if got:
                        hits.extend(got)
        if hits:
            rows = np.asarray(hits)
            d = np.linalg.norm(local[rows] - c, axis=1)
            rows = rows[d <= radius]
            membership.append(np.sort(rows))
        else:
            membership.append(np.zeros(0, dtype=np.int64))
    counts = np.array([len(m) for m in membership], dtype=np.int64)
    keep = counts > 0
    return {
        'positions_local_mm': centres[keep].astype(np.float64),
        'candidate_sites': int(side * side),
        'kept_channels': int(keep.sum()),
        'kept_index': np.where(keep)[0].astype(np.int64),
        'membership': [membership[i] for i in range(len(membership)) if keep[i]],
        'diameter_um': sb.ELECTRODE_DIAMETER_UM,
        'pitch_um': sb.ELECTRODE_PITCH_UM,
        'capture_radius_um': sb.ELECTRODE_CAPTURE_RADIUS_UM,
        'counts': counts[keep],
        'body_node': sb.CNS_BODY_NODE,
    }


def overlap_histogram(membership, n_somas):
    """How many channels capture each soma: the overlap that forces source separation."""
    if not membership:
        return {}
    hit = np.bincount(np.concatenate(membership).astype(np.int64), minlength=n_somas)
    vals, cnt = np.unique(hit[hit > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def write_chunk(name, payload: bytes, chunks, *, dtype, shape, units, extra=None):
    path = SCENE / name
    path.write_bytes(payload)
    entry = {'file': name, 'dtype': dtype, 'shape': list(shape), 'bytes': len(payload), 'units': units}
    if extra:
        entry.update(extra)
    chunks[name] = entry
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', default=str(ROOT / 'outputs/embodied_body/loop_episode_neural_modulated_seed0.npz'))
    ap.add_argument('--somas', default=str(ROOT / 'data/banc/somas_v1.parquet'))
    ap.add_argument('--connectome', default=str(ROOT / 'data/flywire/banc_connectome.npz'))
    ap.add_argument('--body-data', default=str(ROOT / 'viewer/data.js'))
    ap.add_argument('--frames', type=int, default=FRAMES)
    ap.add_argument('--somas-budget', type=int, default=BUDGET_SOMAS)
    ap.add_argument('--edges-budget', type=int, default=BUDGET_EDGES)
    args = ap.parse_args()

    scene_dir = SCENE
    scene_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    layers = []
    chunk_index = {}

    # ---------------------------------------------------------------- body
    body_src = json.loads(Path(args.body_data).read_text().split('=', 1)[1].rstrip().rstrip(';'))
    body = body_src['body']
    ix = sb.samples(len(body['frames']), args.frames)
    body_export = {
        'nodes': body['nodes'], 'links': body['links'], 'meshes': body['meshes'],
        'frames': np.asarray(body['frames'], dtype=np.float64)[ix].round(6).tolist(),
        'rotations_wxyz': np.asarray(body['rotations_wxyz'], dtype=np.float64)[ix].round(7).tolist(),
        'times_s': np.asarray(body['times_s'], dtype=np.float64)[ix].round(6).tolist(),
        'row_mapping_audit': body['row_mapping_audit'],
    }
    (scene_dir / 'body.json').write_text(json.dumps(body_export, separators=(',', ':')))
    layers.append({'id': 'body', 'label_zh': '身体 / 运动骨架 + 网格', 'kind': 'json',
                   'file': 'body.json', 'body_node': 0, 'space': 'world_mm',
                   'clock': 'body', 'frames': len(body_export['frames']),
                   'class': 'SIMULATED body replay (FlyGym/MuJoCo)',
                   'source': args.episode,
                   'source_sha256': sb.sha256_file(Path(args.episode)),
                   'units': 'mm, s', 'registration': None})

    # ---------------------------------------------------------------- CNS
    print('loading BANC somas ...', flush=True)
    soma_root, soma_um = sb.load_banc_somas(Path(args.somas))
    print('  somas', soma_um.shape, 'RSS %.0f MiB' % sb.bounded_rss_mib(), flush=True)
    print('loading connectome ...', flush=True)
    conn = sb.load_connectome(Path(args.connectome))
    reg = sb.cns_registration(soma_root, soma_um, conn, len(conn['root']))
    cns_local = sb.transform_cns(soma_um, reg)
    print('  RSS %.0f MiB' % sb.bounded_rss_mib(), flush=True)

    soma_of_neuron = sb.join_somas(soma_root, conn['root'])
    have = soma_of_neuron >= 0
    neuron_rows = np.where(have)[0]
    pick_local = sb.stratified_subsample(
        np.array([str(conn['ann_body'][r]) + '|' + str(conn['ann_super'][r]) for r in neuron_rows]),
        args.somas_budget, seed=0)
    rows = neuron_rows[pick_local]
    local = cns_local[soma_of_neuron[rows]]

    body_part_uniques, body_part_codes = np.unique(
        np.array([str(conn['ann_body'][r]) for r in rows]), return_inverse=True)
    super_uniques, super_codes = np.unique(
        np.array([str(conn['ann_super'][r]) for r in rows]), return_inverse=True)
    flow_uniques, flow_codes = np.unique(
        np.array([str(conn['ann_flow'][r]) for r in rows]), return_inverse=True)

    n_somas = len(local)
    write_chunk('somas_pos.bin', sb.pack_f32(local), chunk_index,
                dtype='float32', shape=(n_somas, 3), units='body-local mm')
    write_chunk('somas_class.bin',
                np.stack([super_codes, flow_codes, body_part_codes], axis=1).astype(np.uint8).tobytes(),
                chunk_index, dtype='uint8', shape=(n_somas, 3), units='class code indices')
    root_ids = np.asarray(conn['root'][rows], dtype=np.uint64)
    write_chunk('somas_root_id.bin', root_ids.astype('<u8').tobytes(), chunk_index,
                dtype='uint64', shape=(n_somas,), units='BANC root id (exact)')

    layers.append({
        'id': 'cns_somas', 'label_zh': '脑 + 腹神经索 神经元体（真实坐标）', 'kind': 'bin',
        'chunks': ['somas_pos.bin', 'somas_class.bin', 'somas_root_id.bin'],
        'body_node': sb.CNS_BODY_NODE, 'space': 'body_local_mm', 'clock': 'static',
        'count': int(n_somas), 'class': 'MEASURED coordinates (EM segmentation)',
        'source': args.somas, 'source_sha256': sb.sha256_file(Path(args.somas)),
        'units': 'body-local mm',
        'registration': reg,
        'sampling': {'method': 'stratified by body_part|super_class, seed 0',
                     'pool': int(len(neuron_rows)),
                     'pool_note': 'connectome neurons that have a soma row: 79.5% of 153962',
                     'chosen': int(n_somas)},
        'label_tables': {'super_class': super_uniques.tolist(), 'flow': flow_uniques.tolist(),
                         'body_part': body_part_uniques.tolist()},
    })

    # ------------------------------------------------------------- edges
    print('sampling edges ...', flush=True)
    # membership test: both ends present in the chosen soma subset
    lookup = np.full(len(conn['root']), -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows), dtype=np.int64)
    pre_ids = conn['root'][conn['pre']]
    post_ids = conn['root'][conn['post']]
    pre_sub = lookup[conn['pre']]
    post_sub = lookup[conn['post']]
    keep = (pre_sub >= 0) & (post_sub >= 0) & (pre_sub != post_sub)
    kept = np.where(keep)[0]
    if len(kept) > args.edges_budget:
        rng = np.random.default_rng(1)
        kept = np.sort(rng.choice(kept, size=args.edges_budget, replace=False))
    edges = np.stack([pre_sub[kept], post_sub[kept]], axis=1).astype(np.uint32)
    syn_counts = conn['syn'][kept].astype(np.float32)
    write_chunk('edges_index.bin', sb.pack_u32(edges), chunk_index,
                dtype='uint32', shape=edges.shape, units='soma row indices')
    write_chunk('edges_syn.bin', sb.pack_f32(syn_counts), chunk_index,
                dtype='float32', shape=(len(kept),), units='synapse count')
    layers.append({'id': 'cns_edges', 'label_zh': '突触连接（抽样）', 'kind': 'bin',
                   'chunks': ['edges_index.bin', 'edges_syn.bin'],
                   'body_node': sb.CNS_BODY_NODE, 'space': 'body_local_mm', 'clock': 'static',
                   'count': int(len(kept)), 'class': 'MEASURED connectivity (sampled)',
                   'source': args.connectome, 'source_sha256': sb.sha256_file(Path(args.connectome)),
                   'units': 'count', 'registration': reg,
                   'sampling': {'method': 'uniform random over edges with both ends in the soma subset',
                                'available_in_subset': int(keep.sum()), 'chosen': int(len(kept)), 'seed': 1}})

    # ---------------------------------------------------------- activity
    print('propagating activity on the real connectome ...', flush=True)
    activity, act_meta = sb.propagate_activity(conn, len(conn['root']), args.frames,
                                               drive_neurons=2000, seed=0)
    activity_sub = activity[:, rows]
    q_bytes, scales = sb.pack_u8_quantised(activity_sub)
    write_chunk('activity_u8.bin', q_bytes, chunk_index, dtype='uint8',
                shape=(args.frames, n_somas), units='quantised response (0-255)')
    write_chunk('activity_scale.bin', sb.pack_f32(scales), chunk_index, dtype='float32',
                shape=(args.frames, 2), units='per-frame [min, span]')
    layers.append({'id': 'cns_activity', 'label_zh': '逐神经元活动（真实连通组上的传播）',
                   'kind': 'bin', 'chunks': ['activity_u8.bin', 'activity_scale.bin'],
                   'body_node': sb.CNS_BODY_NODE, 'space': 'body_local_mm',
                   'clock': 'cns_activity', 'frames': int(args.frames),
                   'class': 'ILLUSTRATIVE dynamics on MEASURED connectivity',
                   'source': args.connectome, 'source_sha256': sb.sha256_file(Path(args.connectome)),
                   'units': 'arbitrary response units (quantised)', 'registration': reg,
                   'assumptions': act_meta})

    # ------------------------------------------------------------- legs
    print('building per-leg layers ...', flush=True)
    legs = build_legs(Path(args.episode), ix)
    n_legs = len(legs['names'])
    write_chunk('leg_site_index.bin', legs['site_index'].astype('<u2').tobytes(), chunk_index,
                dtype='uint16', shape=legs['site_index'].shape, units='index into leg_contact_table')
    write_chunk('leg_contact_table.bin', sb.pack_f32(legs['contact_table']), chunk_index,
                dtype='float32', shape=legs['contact_table'].shape, units='world mm (last row = no contact)')
    write_chunk('leg_present.bin', legs['present'].astype(np.uint8).tobytes(), chunk_index,
                dtype='uint8', shape=legs['present'].shape, units='bool')
    write_chunk('leg_force.bin', sb.pack_f32(legs['force_magnitude']), chunk_index,
                dtype='float32', shape=legs['force_magnitude'].shape, units='model force units')
    write_chunk('leg_force_dir.bin', sb.pack_f32(legs['force_direction']), chunk_index,
                dtype='float32', shape=legs['force_direction'].shape, units='unit vector')
    write_chunk('leg_actuator.bin', sb.pack_f32(legs['actuator_forces']), chunk_index,
                dtype='float32', shape=legs['actuator_forces'].shape, units='model actuator units')
    write_chunk('leg_cpg.bin',
                np.stack([legs['cpg_phase_rad'], legs['cpg_magnitude']], axis=-1).astype('<f4').tobytes(),
                chunk_index, dtype='float32', shape=(*legs['cpg_phase_rad'].shape, 2),
                units='rad, dimensionless')
    layers.append({'id': 'leg_contacts', 'label_zh': '六足接触点/接触力/CPG', 'kind': 'bin',
                   'chunks': ['leg_site_index.bin', 'leg_contact_table.bin', 'leg_present.bin',
                              'leg_force.bin', 'leg_force_dir.bin', 'leg_actuator.bin', 'leg_cpg.bin'],
                   'body_node': 0, 'space': 'world_mm', 'clock': 'body', 'frames': int(args.frames),
                   'class': 'SIMULATED recorded FlyGym episode', 'source': args.episode,
                   'source_sha256': sb.sha256_file(Path(args.episode)),
                   'units': 'see chunk units', 'leg_names': legs['names'],
                   'foot_matching': legs['match_info'],
                   'note': 'contacts are recorded positions, not re-simulated'})

    # -------------------------------------------------------- descending
    write_chunk('descending.bin',
                np.stack([legs['descending_hz'], legs['descending_left_hz'],
                          legs['descending_right_hz'], legs['speed_scale'], legs['turn']],
                         axis=-1).astype('<f4').tobytes(), chunk_index, dtype='float32',
                shape=(len(legs['descending_hz']), 5),
                units='Hz, Hz, Hz, dimensionless, dimensionless')
    layers.append({'id': 'descending', 'label_zh': '下行指令 / 速度与转向指令', 'kind': 'bin',
                   'chunks': ['descending.bin'], 'body_node': sb.CNS_BODY_NODE,
                   'space': 'body_local_mm', 'clock': 'body', 'frames': int(args.frames),
                   'class': 'SIMULATED loop telemetry', 'source': args.episode,
                   'source_sha256': sb.sha256_file(Path(args.episode)),
                   'columns': ['descending_total_hz', 'descending_left_hz', 'descending_right_hz',
                               'speed_scale', 'turn'],
                   'units': 'Hz and dimensionless'})

    # ------------------------------------------------------- electrodes
    print('rebuilding electrode array at the fixed 7/20 geometry ...', flush=True)
    in_cns = np.ones(len(cns_local), dtype=bool)
    elec = build_electrodes(cns_local, soma_um, reg, in_cns)
    write_chunk('electrode_pos.bin', sb.pack_f32(elec['positions_local_mm']), chunk_index,
                dtype='float32', shape=elec['positions_local_mm'].shape, units='body-local mm')
    write_chunk('electrode_site_index.bin', elec['kept_index'].astype('<u4').tobytes(), chunk_index,
                dtype='uint32', shape=(len(elec['kept_index']),),
                units='candidate lattice site index (32x32 lattice, row-major)')
    layers.append({'id': 'electrodes', 'label_zh': '电极阵列（固定 7 µm / 20 µm）', 'kind': 'bin',
                   'chunks': ['electrode_pos.bin', 'electrode_site_index.bin'],
                   'body_node': sb.CNS_BODY_NODE, 'space': 'body_local_mm', 'clock': 'static',
                   'count': int(len(elec['positions_local_mm'])),
                   'class': 'ASSUMED device geometry (FIXED user input)',
                   'source': 'fixed user constraint', 'source_sha256': None,
                   'units': 'body-local mm',
                   'geometry': {'diameter_um': elec['diameter_um'], 'pitch_um': elec['pitch_um'],
                                'capture_radius_um': elec['capture_radius_um'],
                                'capture_radius_class': 'DECLARED, not measured',
                                'candidate_sites': elec['candidate_sites'],
                                'sites_kept': elec['kept_channels'],
                                'lattice': '32x32, 20 um pitch, centred on the VNC landmark, '
                                           'sites with no soma inside the declared radius dropped'},
                   'notice': ('Geometry is a fixed user input, never searched. Capture radius is '
                              'declared. Membership counts are recomputed for THIS geometry; the '
                              'stored access map at 100/10 um is NOT reused.')})

    # ------------------------------------------------------ electrode cns overlay
    counts = elec['counts']
    write_chunk('electrode_counts.bin', counts.astype('<u4').tobytes(), chunk_index,
                dtype='uint32', shape=(len(counts),), units='somas within the declared capture radius')
    write_chunk('electrode_membership.bin',
                np.concatenate(elec['membership']).astype('<u4').tobytes() if len(counts) else b'',
                chunk_index, dtype='uint32', shape=(int(counts.sum()),), units='rows into the soma table order')
    write_chunk('electrode_membership_offsets.bin',
                np.cumsum(np.r_[0, counts]).astype('<u4').tobytes(), chunk_index, dtype='uint32',
                shape=(len(counts) + 1,), units='offsets into electrode_membership.bin')
    layers.append({'id': 'electrode_membership', 'label_zh': '电极↔神经元 成员数（本几何重算）',
                   'kind': 'bin', 'chunks': ['electrode_counts.bin', 'electrode_membership.bin',
                                             'electrode_membership_offsets.bin'],
                   'body_node': sb.CNS_BODY_NODE, 'space': 'body_local_mm', 'clock': 'static',
                   'class': 'GEOMETRIC recomputation at the fixed geometry',
                   'source': args.somas, 'source_sha256': sb.sha256_file(Path(args.somas)),
                   'units': 'count',
                   'summary': {'non_empty_channels': int((counts > 0).sum()),
                               'total_channels': int(len(counts)),
                               'max_per_channel': int(counts.max() if len(counts) else 0),
                               'min_per_channel': int(counts.min() if len(counts) else 0),
                               'multi_channel_somas': int(
                                   (np.bincount(np.concatenate(elec['membership']).astype(np.int64),
                                                minlength=len(rows)) > 1).sum()) if len(counts) else 0,
                               'overlap_histogram': overlap_histogram(elec['membership'], len(rows)),
                               'channel_pitch_um': elec['pitch_um'],
                               'capture_radius_um': elec['capture_radius_um'],
                               'compare_stored_access_map': 'NOT reused: stored map was built at '
                                                            '100 um / 10 um with 0 channel collisions'}})

    # ------------------------------------------------------ calcium (wound model)
    # Independent epithelial wound simulation. It has no registration to the body
    # and its 137.4 um patch is ~44x smaller than the 3 mm body, so it is placed
    # at TRUE scale on the abdomen and labelled as a separate model. Scaling it up
    # to "look right" would fabricate anatomical meaning it does not have.
    ca = body_src.get('calcium')
    if ca:
        ca_pos = np.asarray(ca['positions_um'], dtype=np.float64) / 1000.0   # um -> mm
        ca_vals = np.asarray(ca['values'], dtype=np.float64)
        n_ca = ca_pos.shape[0]
        write_chunk('calcium_pos.bin', sb.pack_f32(ca_pos), chunk_index, dtype='float32',
                    shape=(n_ca, 3), units='model mm (patch plane, true scale)')
        vmin = float(np.nanmin(ca_vals)); vmax = float(np.nanmax(ca_vals))
        span = (vmax - vmin) or 1.0
        q = np.clip((ca_vals - vmin) / span, 0, 1)
        write_chunk('calcium_u8.bin',
                    np.rint(q * 255).astype(np.uint8).tobytes(), chunk_index, dtype='uint8',
                    shape=ca_vals.shape, units='quantised cytosolic Ca')
        abd = [i for i, n in enumerate(body['nodes']) if 'abdomen' in n['name']]
        layers.append({'id': 'calcium_epithelium',
                       'label_zh': '上皮伤口钙（独立模型·真实尺度·未配准）', 'kind': 'bin',
                       'chunks': ['calcium_pos.bin', 'calcium_u8.bin'],
                       'body_node': int(abd[0]) if abd else 0, 'space': 'model_mm',
                       'clock': 'calcium', 'frames': int(ca_vals.shape[0]), 'count': int(n_ca),
                       'class': 'SIMULATED wound epithelium (independent model)',
                       'source': args.body_data,
                       'source_sha256': sb.sha256_file(Path(args.body_data)),
                       'units': ca['units'], 'value_range_uM': [vmin, vmax],
                       'extent_um': float(ca_pos[:, 0].max() - ca_pos[:, 0].min()) * 1000.0,
                       'registration': {'method': 'attached to the abdomen node, NOT registered',
                                        'notice': ('Patch drawn at true scale (137.4 um) on the '
                                                   'abdomen; no anatomical registration and no '
                                                   'shared clock with the CNS or the body.')}})

    # ----------------------------------------------------- slow physiology ledgers
    slow_path = ROOT / 'outputs/embodied_body/slowphys_report.json'
    if slow_path.is_file():
        slow = json.loads(slow_path.read_text())
        ledgers = slow.get('ledgers', {})
        layers.append({'id': 'physiology_ledgers',
                       'label_zh': '慢生理账本（O2 / 海藻糖 / 行为段）', 'kind': 'values',
                       'body_node': sb.CNS_BODY_NODE, 'space': 'in_scene_readout',
                       'clock': 'independent',
                       'class': 'SIMULATED bookkeeping (values recorded by the module)',
                       'source': str(slow_path), 'source_sha256': sb.sha256_file(slow_path),
                       'units': 'nmol, s',
                       'values': {
                           'oxygen_nmol': ledgers.get('O2', {}),
                           'trehalose_nmol': ledgers.get('trehalose', {}),
                           'epochs': slow.get('behaviour', {}).get('epochs'),
                           'sleep_implemented': slow.get('sleep', {}).get('implemented'),
                           'sleep_status': slow.get('sleep', {}).get('status'),
                       }})

    # ------------------------------------------- simulation layers (paracrine,
    # immune).  These chunks are WRITTEN BY scene_layers_sim.py with venv/bin/python
    # (they need scipy), while this builder runs under venv_body (pyarrow + mujoco).
    # The builder therefore registers them rather than computing them.
    sim_meta_path = SCENE / 'sim_layers.json'
    if sim_meta_path.is_file():
        sim = json.loads(sim_meta_path.read_text())
        for name, fname, shape, units in (
                ('paracrine_pos.bin', 'paracrine_pos.bin', None, 'body-local mm'),
                ('paracrine_frames.bin', 'paracrine_frames.bin', None, 'quantised ligand'),
                ('immune_pos.bin', 'immune_pos.bin', None, 'body-local mm'),
                ('immune_debris.bin', 'immune_debris.bin', None, 'quantised debris'),
                ('immune_attractant.bin', 'immune_attractant.bin', None, 'quantised attractant'),
                ('immune_cells.bin', 'immune_cells.bin', None, 'body-local mm')):
            path = SCENE / fname
            if not path.is_file():
                continue
            chunk_index[fname] = {'file': fname, 'bytes': path.stat().st_size,
                                  'units': units}
        pp = sim.get('paracrine', {})
        if pp:
            chunk_index['paracrine_pos.bin'].update(
                {'dtype': 'float32', 'shape': [pp['n_points'], 3]})
            chunk_index['paracrine_frames.bin'].update(
                {'dtype': 'uint8', 'shape': [pp['n_frames'], pp['n_points']]})
            layers.append({'id': 'paracrine_field',
                           'label_zh': '旁分泌配体场（空间版·真实尺度·未配准）', 'kind': 'bin',
                           'chunks': ['paracrine_pos.bin', 'paracrine_frames.bin'],
                           'body_node': 12, 'space': 'body_local_mm', 'clock': 'ligand',
                           'frames': int(pp['n_frames']), 'count': int(pp['n_points']),
                           'class': 'SIMULATED reaction-diffusion field (independent model)',
                           'source': 'scene_layers_sim.py -> paracrine.py',
                           'units': 'arbitrary ligand concentration (quantised)',
                           'registration': {
                               'method': 'laid flat on the abdomen at TRUE SCALE, not registered',
                               'notice': ('the 306 um patch is drawn at 1:1, i.e. about '
                                          '1/10 of the body length; its own coordinate '
                                          'frame is separate from the CNS and the body')},
                           'model': pp['meta']})
        ii = sim.get('immune', {})
        if ii:
            chunk_index['immune_pos.bin'].update(
                {'dtype': 'float32', 'shape': [ii['n_points'], 3]})
            chunk_index['immune_debris.bin'].update(
                {'dtype': 'uint8', 'shape': [ii['n_frames'], ii['n_points']]})
            chunk_index['immune_attractant.bin'].update(
                {'dtype': 'uint8', 'shape': [ii['n_frames'], ii['n_points']]})
            chunk_index['immune_cells.bin'].update(
                {'dtype': 'float32', 'shape': [ii['n_frames'], ii['n_cells'], 3]})
            layers.append({'id': 'immune_field',
                           'label_zh': '血细胞与碎片清除（真实尺度·未配准）', 'kind': 'bin',
                           'chunks': ['immune_pos.bin', 'immune_debris.bin',
                                      'immune_attractant.bin', 'immune_cells.bin'],
                           'body_node': 12, 'space': 'body_local_mm', 'clock': 'immune',
                           'frames': int(ii['n_frames']), 'count': int(ii['n_points']),
                           'class': 'SIMULATED agent field (independent model; not fly immunity)',
                           'source': 'scene_layers_sim.py -> immune.py',
                           'units': 'quantised debris and attractant; cells in body-local mm',
                           'registration': {
                               'method': 'laid flat on the abdomen at TRUE SCALE, not registered',
                               'notice': ('320 um patch at 1:1 with 96 agents; the material '
                                          'budget closes but nothing here is calibrated to a '
                                          'real haemocyte population')},
                           'budget': ii['budget'], 'model': ii['meta']})

    # ------------------------------------------- mechanics patch (mechanics_cut_n24.npz)
    mech_path = SCENE / 'mechanics.json'
    if mech_path.is_file():
        mech = json.loads(mech_path.read_text())
        fp = SCENE / 'mech_pos.bin'
        if fp.is_file():
            chunk_index['mech_pos.bin'] = {'file': 'mech_pos.bin', 'bytes': fp.stat().st_size,
                                           'dtype': 'float32',
                                           'shape': [mech['n_frames'], mech['n_vertices'], 3],
                                           'units': 'body-local mm'}
            layers.append({'id': 'mechanics_patch',
                           'label_zh': '二维顶点力学（独立仿真·真实尺度·未配准）', 'kind': 'bin',
                           'chunks': ['mech_pos.bin'],
                           'body_node': 12, 'space': 'body_local_mm', 'clock': 'mechanics',
                           'frames': int(mech['n_frames']), 'count': int(mech['n_vertices']),
                           'class': 'SIMULATED vertex mechanics (independent 2-D model)',
                           'source': 'outputs/mechanics_cut_n24.npz',
                           'units': 'body-local mm (source um, 1:1)',
                           'registration': {'method': 'laid on the abdomen at true scale, not registered',
                                            'notice': '31 recorded states of a cut 2-D vertex model; separate clock'},
                           'model': {'extent_um': mech['extent_um'], 'times_s': mech['times_s'],
                                     'energy_J': mech.get('energy_J')}})

    # ------------------------------------------------- scenarios (five arms)
    sc_path = SCENE / 'scenarios.json'
    if sc_path.is_file():
        sc = json.loads(sc_path.read_text())
        layers.append({'id': 'scenarios',
                       'label_zh': '断颈/眼损五臂对比（眼-脑-颈-体电位）', 'kind': 'values',
                       'body_node': sb.CNS_BODY_NODE, 'space': 'in_scene_readout',
                       'clock': 'independent', 'frames': int(sc.get('frames') or 0),
                       'class': 'SIMULATED scenario recordings (five arms, same protocol)',
                       'source': 'outputs/brain_isolation/integrated_*.npz',
                       'units': 'mV for potentials, nA for the stimulus',
                       'arms': sc.get('scenarios'),
                       'series_keys': list((sc.get('series') or {}).keys()),
                       'series': sc.get('series'),
                       'times_s': sc.get('times_s'),
                       'note': ('per-region traces drive the colour of the corresponding '
                                'structure; the artefact and stimulus traces are included so '
                                'the recording is not mistaken for a clean signal')})

    # ------------------------------------------------------- environment
    env_path = SCENE / 'environment.json'
    if env_path.is_file():
        env = json.loads(env_path.read_text())
        if env.get('ground'):
            fp = SCENE / env['ground']['file']
            chunk_index[env['ground']['file']] = {
                'file': env['ground']['file'], 'bytes': fp.stat().st_size,
                'dtype': 'image', 'shape': [env['ground']['px'], env['ground']['px']],
                'units': 'JPEG texture'}
            layers.append({'id': 'environment',
                           'label_zh': '场景环境（地面贴图·CC0 署名随附）', 'kind': 'json',
                           'file': env['ground']['file'],
                           'chunks': [env['ground']['file']],
                           'body_node': 0, 'space': 'world', 'clock': 'static',
                           'class': 'MEASURED texture (CC0 photograph, attribution carried)',
                           'source': (env.get('attribution') or {}).get('source_page'),
                           'units': 'dimensionless texture',
                           'attribution': env.get('attribution'),
                           'sky_top': env.get('sky_top'), 'sky_bottom': env.get('sky_bottom')})

    # --------------------------------------------------------- readouts
    ro_path = SCENE / 'readouts.json'
    if ro_path.is_file():
        ro = json.loads(ro_path.read_text())
        layers.append({'id': 'layer_readouts',
                       'label_zh': '胞内空间 / 血管氧 / 电极链（场内读数，无空间场）',
                       'kind': 'values', 'body_node': 0, 'space': 'in_scene_readout',
                       'clock': 'static',
                       'class': 'MIXED: measured summary statistics and module self-tests',
                       'source': 'outputs/metrics_cellspace.json, metrics_organism.json, '
                                 'outputs/embodied_body/electrode_*.json',
                       'units': 'uM, mmHg, verdict counts',
                       'values': ro,
                       'note': ('these three layers have NO spatial field in the artefacts: '
                                'the intracellular model is single-cell and the oxygen '
                                'artefact holds per-spacing summary statistics only, so they '
                                'are readouts rather than drawn fields')})

    # ------------------------------------------- 3-D multilayer stack (tissue3d.py)
    t3_path = SCENE / 'tissue3d.json'
    if t3_path.is_file():
        t3 = json.loads(t3_path.read_text())
        for fname, dtype, shape, units in (
                ('tissue3d_pos.bin', 'float32', [t3['n_vertices'], 3], 'body-local mm'),
                ('tissue3d_edges.bin', 'uint32', [t3['n_edges'], 2], 'vertex index pairs'),
                ('tissue3d_layers.bin', 'uint16', [t3['n_vertices']], 'layer index')):
            fp = SCENE / fname
            if not fp.is_file():
                continue
            chunk_index[fname] = {'file': fname, 'bytes': fp.stat().st_size,
                                  'dtype': dtype, 'shape': shape, 'units': units}
        layers.append({'id': 'tissue_3d',
                       'label_zh': '三维多层组织（棱柱堆叠·厚度变形已放大）', 'kind': 'bin',
                       'chunks': ['tissue3d_pos.bin', 'tissue3d_edges.bin',
                                  'tissue3d_layers.bin'],
                       'body_node': 12, 'space': 'body_local_mm', 'clock': 'static',
                       'count': int(t3['n_vertices']),
                       'class': 'SIMULATED 3-D prism stack (thickness direction verified)',
                       'source': 'scene_layers_sim.py -> tissue3d.py',
                       'units': 'body-local mm for positions; layer index is dimensionless',
                       'registration': {
                           'method': 'laid on the abdomen, plane 1:1, NOT registered',
                           'notice': ('the plane geometry comes from mechanics.build_hex_mesh, '
                                      'the layer the project validated in 2-D; the cells are '
                                      'prisms, not free 3-D shapes, and the thickness '
                                      'deviation is multiplied by a declared gain for '
                                      'visibility')},
                       'model': t3})

    # ------------------------------------------------------------- manifest
    total = sum(c['bytes'] for c in chunk_index.values())
    transform_audit = transform_check(soma_um, cns_local, reg)
    manifest = {
        'schema': 'workbench-scene-2',
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'build_seconds': round(time.time() - started, 1),
        'peak_rss_mib': round(sb.bounded_rss_mib(), 1),
        'frame': {'name': 'FlyGym world millimetres; body-local layers attach to a body node',
                  'body_node_names': [n['name'] for n in body['nodes']],
                  'anterior_axis': '-x (verified: most -x segments are the mouthparts)',
                  'up_axis': '+z', 'lateral_axis': 'y'},
        'clock': {'primary': 'body', 'times_s': body_export['times_s'], 'frames': len(body_export['frames']),
                  'independent': {'cns_activity': 'own clock, deterministic leaky response'}},
        'budget': {'bytes_total': total, 'limit_bytes': 30 * 1024 ** 2,
                   'per_layer_limit_bytes': 6 * 1024 ** 2},
        'chunks': chunk_index,
        'layers': layers,
        'transform_audit': transform_audit,
        'notices': [
            'One frame, one clock, layered: independent simulations are labelled, not merged.',
            'CNS placement is by named nerve-root landmarks, NOT anatomical registration.',
            'Brain/VNC subdivision is a display split; no measured boundary exists.',
            'Neural activity is an engineering response model on real connectivity, not measurement.',
            'Electrodes: 7 um / 20 um fixed; capture radius declared; no spike sorting implemented.',
        ],
    }
    over = [l['id'] for l in layers if l.get('kind') == 'bin'
            and sum(chunk_index[c]['bytes'] for c in l.get('chunks', [])) > manifest['budget']['per_layer_limit_bytes']]
    manifest['budget']['layers_over_per_layer_limit'] = over
    if total > manifest['budget']['limit_bytes']:
        raise MemoryError('scene payload over budget: %d bytes' % total)
    (scene_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'bytes_total': total, 'layers': len(layers), 'somas': n_somas,
                      'edges': int(len(kept)), 'peak_rss_mib': manifest['peak_rss_mib'],
                      'build_seconds': manifest['build_seconds']}, indent=1))
    print('transform audit:', json.dumps(transform_audit, indent=1))


def transform_check(soma_um, cns_local, reg, n=1500):
    """Prove the CNS map is rigid and distance preserving.

    Compares the full pairwise distance matrix of a bounded random subset:
    source converted to mm, destination as stored. A rigid map plus float32
    export precision must agree within 1e-5 mm (10 nm).
    """
    rng = np.random.default_rng(3)
    m = min(n, len(soma_um))
    idx = rng.choice(len(soma_um), size=m, replace=False)
    src = soma_um[idx] / 1000.0
    dst = cns_local[idx]
    ds = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    dd = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)
    iu = np.triu_indices(m, 1)
    err = np.abs(ds[iu] - dd[iu])
    return {'points_checked': int(m), 'pairs_checked': int(len(err)),
            'max_pairwise_distance_error_mm': float(err.max()),
            'mean_pairwise_distance_error_mm': float(err.mean()),
            'scale': float(reg['scale']), 'tolerance_mm': 1e-5,
            'rigid': bool(err.max() <= 1e-5)}


if __name__ == '__main__':
    main()
