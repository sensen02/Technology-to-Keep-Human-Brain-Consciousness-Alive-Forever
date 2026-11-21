"""Test the reference project's frozen BANC cervical circuit in a registered injury volume."""

import argparse
from collections import Counter
import importlib.util
import json
from math import ceil
from pathlib import Path
import sys

import numpy as np

from ..modules.array_damage import RadialInsertion
from ..modules.banc_neck import load_fixed_neck
from ..modules.conduction_damage import path_damage
from ..modules.electrode_observation import AxonProbe, ElectrodeObservation
from ..world import Grid
from .acute_implant import ACUTE_DAMAGE, SHAFT, AcuteWorld

UM = 1e-6


def subdivide(points, factor):
    """Increase observation samples without replacing original bends by chords."""
    pieces = points[:-1, None] + (points[1:] - points[:-1])[:, None] * (np.arange(factor) / factor)[None, :, None]
    return np.vstack((pieces.reshape(-1, 3), points[-1]))


def _test_shaft(points, offset_um):
    mid = len(points) // 2
    pos = points[mid]
    tangent = points[mid + 1, :2] - points[mid - 1, :2]
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    center = pos.copy()
    center[:2] += normal * offset_um * UM
    return np.array([center - [0, 0, 9 * UM], center + [0, 0, 9 * UM]])


def _world(neck, shaft):
    model = RadialInsertion()
    if not np.allclose(shaft[0, :2], shaft[1, :2], atol=1e-12) or shaft[1, 2] <= shaft[0, 2]:
        raise ValueError('this study expects one increasing BANC z shaft')
    xyz = np.concatenate([p.points_m for p in neck] + [shaft])
    h = 3.2 * UM
    low = np.floor((xyz.min(axis=0) - 6 * UM) / h) * h
    high = np.ceil((xyz.max(axis=0) + 6 * UM) / h) * h
    grid = Grid(tuple(np.rint((high-low)/h).astype(int)), h, tuple(low))
    damage = {}

    def select(g, level):
        x, y, z = g.centers_m()
        halo = model.spatial_threshold_radius_m(model.onset_Pa) + np.sqrt(3) * g.spacing_m
        return ((x - shaft[0, 0])**2 + (y - shaft[0, 1])**2 <= halo**2) & (
            (z >= shaft[0, 2] - g.spacing_m) & (z <= shaft[1, 2] + g.spacing_m))

    def materials(g, labels):
        if g.spacing_m > .201 * UM:
            return labels
        x, y, z = g.centers_m()
        radius2 = (x-shaft[0, 0])**2 + (y-shaft[0, 1])**2
        behind = shaft[1, 2] - z
        offset = np.clip(behind, 0, model.radius_m)
        reached = (z >= shaft[0, 2]) & (behind >= 0)
        solid = reached & (radius2 < offset * (2 * model.radius_m - offset))
        energy = model.spatial_strain_energy(np.sqrt(radius2), behind)
        severity = np.where(reached & ~solid, np.clip((energy-model.onset_Pa) /
                                                     (model.failure_Pa-model.onset_Pa), 0, 1), 0)
        ids = np.flatnonzero(severity)
        if len(ids):
            damage[g] = (ids, severity.ravel()[ids])
        labels[solid] = SHAFT
        return labels

    world = AcuteWorld(grid, 0, levels=(0, 2, 4), select=select, materialize=materials,
                       block_cells=4, max_cells=3_000_000)
    world.set_acute_history({key: damage[world.block_grid(key)] for key in world.block_keys
                             if world.block_grid(key) in damage})
    return world


def _summarize_route(world, path, points, field):
    effect = path_damage(world, points, field, blocked_material=SHAFT)
    return dict(root=path.root, cell_type=path.cell_type, side=path.side, direction=path.direction,
                available=effect.block_at_m is None, gain=effect.gain,
                delay_s=effect.delay_s(path.speed_m_s) if effect.block_at_m is None else None,
                block_at_m=effect.block_at_m, injury_integral_m=effect.injury_integral_m)


def _electrode_sweep(world, path):
    points = path.points_m
    start = points[len(points)//4]
    shift = np.array([3e-6, 0, 0])
    sites = np.array([start + shift, start + 100 * shift, points[-1] + shift])
    result = []
    for factor in (1, 2, 4, 8):
        sampled = subdivide(points, factor)
        obs = ElectrodeObservation(sites, [AxonProbe(path.root, sampled, speed_m_s=path.speed_m_s,
                                                     activation_V=.0001)], sensor_patches=(0, 2),
                                   damage_field=ACUTE_DAMAGE)
        obs.attach(world, 'electrode')
        obs.pulse(world, 0, 1, -2e-7, 0)
        result.append(dict(points=len(sampled), events=obs.events))
    return result


def _runtime_replay(source, catalog, path, by_root, use_damage):
    """Feed the voxel-derived cuts/gains into the upstream fixed conductance circuit."""
    runtime_path = Path(source['runtime'])
    spec = importlib.util.spec_from_file_location('banc_fixed_neck_runtime', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    circuit = runtime.FixedCircuit(catalog)
    cut = np.array([not by_root[root]['available'] if use_damage else False for root in circuit.port_roots])
    gains = np.array([by_root[root]['gain'] if use_damage else 1.0 for root in circuit.port_roots])
    if use_damage:
        circuit.neck = [m.multiply(gains[None, :]).tocsr() for m in circuit.neck]
        circuit.neck_delay[:] = [ceil(by_root[root]['delay_s'] * 1000 / circuit.dt_ms - 1e-12)
                                 if by_root[root]['available'] else tick
                                 for root, tick in zip(circuit.port_roots, circuit.neck_delay)]
        circuit._pending = np.zeros((int(circuit.neck_delay.max())+1, 1, len(circuit.ports)), dtype=bool)
    i = circuit.port_index[path.root]
    drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
    eon = np.zeros((1, len(circuit.ports)), dtype=bool)
    arriving, emitted = [], []
    target_columns = [circuit.index[target] for target, _ in path.outgoing]
    peak = 0.0
    for tick in range(int(circuit.neck_delay.max()) + 30):
        eon[0, i] = tick == 0
        source_events, arrival, _, _ = circuit.step(drive, cut, eon_output=eon)
        if source_events[0, i]:
            emitted.append(tick)
        if arrival[0, i]:
            arriving.append(tick)
        peak = max(peak, float(circuit.last_conductance[:, :, target_columns].max(initial=0)))
    return dict(launch_ticks=emitted, arrival_ticks=arriving, receiver_peak_conductance_nS=peak,
                cut=bool(cut[i]), edge_gain=float(gains[i]), runtime_dt_ms=circuit.dt_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--root', default='720575941510475536')
    parser.add_argument('--diagnostic-offset-um', type=float, default=2.43)
    parser.add_argument('--shaft-um', type=float, nargs=6, help='entry and tip in BANC Cartesian micrometres')
    parser.add_argument('--runtime-replay', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    neck, catalog, _, source = load_fixed_neck(args.project)
    path = next(p for p in neck if p.root == args.root)
    shaft = (_test_shaft(path.points_m, args.diagnostic_offset_um) if args.shaft_um is None else
             np.array(args.shaft_um).reshape(2, 3) * UM)
    world = _world(neck, shaft)
    try:
        routes = [_summarize_route(world, p, p.points_m, ACUTE_DAMAGE) for p in neck]
        by_root = {row['root']:row for row in routes}
        sweep = []
        for factor in (1, 2, 4, 8):
            p = subdivide(path.points_m, factor)
            sweep.append(dict(points=len(p), **_summarize_route(world, path, p, ACUTE_DAMAGE)))
        specific = dict(shaft_only=_summarize_route(world, path, path.points_m, None),
                        injury=by_root[path.root], sampling=sweep, electrode_sampling=_electrode_sweep(world, path))
        if args.runtime_replay:
            if path.direction != 'DN':
                raise ValueError('the frozen Eon source replay uses a DN root')
            sys.dont_write_bytecode = True
            specific['frozen_runtime'] = {condition: _runtime_replay(source, catalog, path, by_root, damaged)
                                          for condition, damaged in [('intact', False), ('injured', True)]}
        report = dict(source=source, planes_nm=catalog['planes_nm'], route_count=len(neck),
                      direction_counts=dict(Counter(p.direction for p in neck)),
                      downstream_edges=sum(len(p.outgoing) for p in neck),
                      downstream_synapse_counts=sum(n for p in neck for _, n in p.outgoing),
                      active_voxels=world.active_cells, shaft_um=(shaft/UM).tolist(),
                      shaft_placement='diagnostic single shaft' if args.shaft_um is None else 'user supplied BANC coordinates',
                      blocked=sum(not row['available'] for row in routes), routes=routes, selected=specific)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('route_count', 'active_voxels', 'blocked', 'selected')}, indent=2))
    finally:
        world.close()


if __name__ == '__main__':
    main()
