"""Run whole-tissue physiology with resolved extracellular volumes and axon fragments near the array."""

import argparse
import ctypes as ct
from dataclasses import replace
import json
from math import ceil, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from ..composite_world import CompositeGpuWorld
from ..engine import FieldSpec, PorousDiffusion
from ..modules.ion_feedback import IonFeedbackConfig
from ..modules.synaptic_plasticity import PlasticityConfig
from ..modules.tissue import EXCITATORY, TISSUE, ImplantLayout, TissueFluid, build_anatomy, tissue_grid
from ..modules.tissue_dynamics import CHLORIDE, POTASSIUM, TissueDynamics
from ..native_transport import DEFAULT_BUILD, _load, process_vram_bytes

LEVELS = (0, 3, 5, 7)
BASE_M = 25.6e-6
AXONS = 2
RADIUS_M = 0.30e-6
NEAR_M = 1.6e-6
MICRO_RADIUS_M = 1.8e-6
MARKER = "ecs.marker"
INTRACELLULAR = "axon.marker"


def _points(grid):
    return np.stack(np.broadcast_arrays(*grid.centers_m()), axis=-1).reshape(-1, 3)


def _rasterizer():
    library = _load(DEFAULT_BUILD, "hip")
    kernel = library.omni_rasterize_segments
    p = ct.c_void_p
    kernel.argtypes = (p, p, ct.c_int, p, p, ct.c_double, ct.c_double, p, p)
    kernel.restype = None
    return kernel


class _Segments:
    def __init__(self, start, end, ids=None):
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.lower = np.minimum(self.start, self.end)
        self.upper = np.maximum(self.start, self.end)
        self.delta = self.end - self.start
        self.length2 = np.square(self.delta).sum(axis=1)
        self.ids = np.arange(len(start)) if ids is None else ids

    def rasterize(self, grid, radius_m, kernel):
        origin = np.asarray(grid.origin_m, dtype=np.float64)
        shape = np.asarray(grid.shape, dtype=np.int32)
        upper = origin + shape * grid.spacing_m
        selected = np.all((self.lower <= upper + radius_m) & (self.upper >= origin - radius_m), axis=1)
        segments = np.ascontiguousarray(np.column_stack((self.start[selected], self.end[selected])), dtype=np.float64)
        ids = np.ascontiguousarray(self.ids[selected], dtype=np.int32)
        owner = np.full(grid.shape, -1, dtype=np.int32)
        if len(segments):
            distance = np.full(grid.shape, radius_m**2)
            kernel(segments.ctypes.data, ids.ctypes.data, len(ids), origin.ctypes.data, shape.ctypes.data,
                   grid.spacing_m, radius_m, distance.ctypes.data, owner.ctypes.data)
        return owner

    def nearest(self, points, radius_m):
        low, high = points.min(axis=0) - radius_m, points.max(axis=0) + radius_m
        candidates = np.flatnonzero(np.all((self.lower <= high) & (self.upper >= low), axis=1))
        best = np.full(len(points), radius_m ** 2)
        owner = np.full(len(points), -1, dtype=np.int32)
        for index in candidates:
            rel = points - self.start[index]
            vector = self.delta[index]
            if self.length2[index] > 0:
                t = np.clip((rel[:, 0] * vector[0] + rel[:, 1] * vector[1] + rel[:, 2] * vector[2])
                            / self.length2[index], 0, 1)
                rel -= t[:, None] * vector
            distance = np.einsum("ij,ij->i", rel, rel)
            closer = distance < best
            best[closer] = distance[closer]
            owner[closer] = self.ids[index]
        return owner


class _NearArray:
    def __init__(self, axes, anatomy, micro_radius_m=MICRO_RADIUS_M, fine_tip_depth_m=0.0, near_m=NEAR_M):
        self.axes = axes
        self.micro_radius_m = micro_radius_m
        self.near_m = near_m
        self.somata = anatomy.position_m
        self.guides = _Segments(axes[:, 0], axes[:, 1])
        syn = anatomy.synapses
        self.contact_ids = np.flatnonzero(self.guides.nearest(syn.position_m, 3.2e-6) >= 0)
        self.anchors = np.concatenate((axes[:, 1, :], syn.position_m[self.contact_ids]))
        self.points = _Segments(self.anchors, self.anchors)
        end = np.concatenate((anatomy.axon_branch_m, syn.position_m))
        start = np.concatenate((anatomy.position_m[:anatomy.neuron_count], anatomy.axon_branch_m[syn.pre]))
        owner = np.r_[np.arange(anatomy.neuron_count), syn.pre]
        self.axons = _Segments(start, end, owner)
        self.fine_box = None
        self._rasterize = None
        if fine_tip_depth_m:
            tips = axes[:, 1, :]
            earlier = tips - (tips - axes[:, 0, :]) * (fine_tip_depth_m / tips[0, 2])
            corners = np.concatenate((tips, earlier))
            pad = np.array((11.5e-6, 11.5e-6, 0))
            self.fine_box = (corners.min(axis=0) - pad, corners.max(axis=0) + pad)
            self._rasterize = _rasterizer()

    def refine(self, grid, level):
        points = _points(grid)
        pad = sqrt(3) * grid.spacing_m / 2
        near_anchor = self.points.nearest(points, self.micro_radius_m + pad) >= 0
        chosen = points[:, 2] >= 0
        if self.fine_box is not None:
            lo, hi = self.fine_box
            half = grid.spacing_m / 2 - 1e-13
            near_anchor |= np.all((points + half > lo) & (points - half < hi), axis=1)
        if level == LEVELS[-1]:
            return (chosen & near_anchor).reshape(grid.shape)
        near_guide = self.guides.nearest(points, self.near_m + pad) >= 0
        return (chosen & (near_guide | near_anchor)).reshape(grid.shape)

    def materials(self, grid, labels):
        if grid.spacing_m > 0.21e-6:
            return labels
        if self._rasterize is None:
            owner = self.axons.nearest(_points(grid), RADIUS_M).reshape(grid.shape)
        else:
            owner = self.axons.rasterize(grid, RADIUS_M, self._rasterize)
        # Virtual somata continue to sample the voxel at their own coordinates.
        bound = np.asarray(grid.origin_m) + grid.spacing_m * np.asarray(grid.shape)
        inside = np.all((self.somata >= grid.origin_m) & (self.somata < bound), axis=1)
        local = np.floor((self.somata[inside] - grid.origin_m) / grid.spacing_m).astype(int)
        local = np.minimum(local, np.asarray(grid.shape) - 1)
        owner[tuple(local.T)] = -1
        active = (labels == TISSUE) & (owner >= 0)
        labels[active] = AXONS + owner[active]
        return labels


def _levels(world):
    counts = {}
    for key in world.block_keys:
        grid = world.block_grid(key)
        tag = f"{grid.spacing_m*1e6:g}"
        counts[tag] = counts.get(tag, 0) + int(np.prod(grid.shape))
    return counts


def _locate_probe(world, geometry, anatomy):
    contacts = anatomy.synapses
    order = np.argsort(np.linalg.norm(contacts.position_m[geometry.contact_ids]
                                     - geometry.axes[0, 1], axis=1))
    for edge in geometry.contact_ids[order]:
        position = contacts.position_m[edge]
        if (anatomy.cell_type[contacts.pre[edge]] == EXCITATORY
                and world.block_grid(world.block_at(position)).spacing_m < 0.21e-6):
            return position, int(edge)
    return geometry.axes[0, 1], None


def _seed(world, position, fluid):
    keys = [key for key in world.refined_blocks if world.block_grid(key).spacing_m < 0.21e-6]
    ecs_pulse, internal_pulse, seed_masks = 0.0, 0.0, {}
    nearest, distance2 = None, float("inf")
    for key in keys:
        grid = world.block_grid(key)
        right = np.asarray(grid.origin_m) + grid.spacing_m * np.asarray(grid.shape)
        if np.any(position < np.asarray(grid.origin_m) - 0.65e-6) or np.any(position > right + 0.65e-6):
            continue
        labels = world.block_materials(key)
        x, y, z = grid.centers_m()
        distance = (x - position[0])**2 + (y - position[1])**2 + (z - position[2])**2
        near = distance < (0.65e-6)**2
        mask = near & (labels == TISSUE)
        if mask.any():
            marker = np.where(mask, fluid.volume_fraction * 5.0, 0.0)
            world.set_block_densities(MARKER, key, marker)
            for ion in (POTASSIUM, CHLORIDE):
                values = world.block_densities(ion, key)
                world.set_block_densities(ion, key, values + marker)
            ecs_pulse += float(marker.sum()) * grid.cell_volume_m3
            local = np.unravel_index(np.where(mask, distance, np.inf).argmin(), grid.shape)
            if distance[local] < distance2:
                distance2 = distance[local]
                nearest = tuple(x + (i + 0.5) * grid.spacing_m for x, i in zip(grid.origin_m, local))
        immersed = near & (labels >= AXONS)
        if immersed.any():
            marker = np.where(immersed, 2.0, 0.0)
            world.set_block_densities(INTRACELLULAR, key, marker)
            internal_pulse += float(marker.sum()) * grid.cell_volume_m3
            seed_masks[key] = immersed
    if ecs_pulse <= 0:
        raise RuntimeError("the pulse must fall in a fine extracellular region")
    return ecs_pulse, internal_pulse, nearest, seed_masks


def _collect(world, output):
    by_level, axons, voxel_count = {}, set(), 0
    boxes = []
    for key in world.block_keys:
        grid = world.block_grid(key)
        material = world.block_materials(key)
        patch = grid.spacing_m * 1e6
        tag = f"{patch:g}"
        if patch < 0.21:
            ids, counts = np.unique(material, return_counts=True)
            axons.update(ids[ids >= AXONS])
            voxel_count += int(counts[ids >= AXONS].sum())
        marker = world.block_amounts(MARKER, key)
        amount = float(marker.sum())
        by_level[tag] = by_level.get(tag, 0.0) + amount
        if output:
            lower = np.asarray(grid.origin_m) * 1e6
            boxes.append((*lower, *(lower + patch * np.asarray(grid.shape)), patch, amount))
    return by_level, len(axons), voxel_count, boxes


def run(layouts, orientation, *, ms=5.0, step_ms=0.25, max_cells=9_000_000, budget_mib=768, capture=False,
        micro_radius_m=MICRO_RADIUS_M, compact_equilibrium=True, lesion=None, fine_tip_depth_m=0.0,
        reserve_mib=0, plasticity=None, ion_feedback=None):
    if not np.isfinite(ms) or ms <= 0 or not np.isfinite(step_ms) or step_ms <= 0:
        raise ValueError("simulation and world step must be finite and positive")
    if not np.isfinite(micro_radius_m) or micro_radius_m < 0.8e-6:
        raise ValueError("micro radius must be at least 0.8 µm")
    if not 0 <= fine_tip_depth_m <= layouts[orientation].normal_depth_m:
        raise ValueError("fine tip depth must be within the implant travel")
    started = perf_counter()
    grid, materials = tissue_grid(layouts, spacing_m=BASE_M)
    anatomy = build_anatomy(grid)
    axes = layouts[orientation].axes_m()
    geometry = _NearArray(axes, anatomy, micro_radius_m, fine_tip_depth_m)
    world = CompositeGpuWorld(grid, materials, levels=LEVELS, select=geometry.refine,
                              materialize=geometry.materials, max_cells=max_cells,
                              block_cells=4, budget_bytes=budget_mib * 2**20, reserve_bytes=reserve_mib * 2**20,
                              leaf_limit=128 if fine_tip_depth_m else None, stream_initial=bool(fine_tip_depth_m))
    fluid = TissueFluid()
    try:
        # Uniform nonreacting solutes stay at their exact resting equilibrium.
        fluid.attach_fields(world, materials, excluded_material_from=AXONS, active_solutes={"K", "Cl"},
                            compact_equilibrium=compact_equilibrium)
        medium = PorousDiffusion(TISSUE, fluid.volume_fraction, fluid.tortuosity, AXONS)
        world.add_field(MARKER, FieldSpec(diffusivity_m2_s=1.96e-9, porous=medium))
        world.add_field(INTRACELLULAR, FieldSpec(diffusivity_m2_s=.2e-9, isolated_materials_from=AXONS))
        if fine_tip_depth_m:
            world.initialize_device()
        biology = TissueDynamics(anatomy, fluid, plasticity=plasticity, first_axon_material=AXONS,
                                 ion_feedback=ion_feedback)
        world.add_module("tissue", biology)
        damaged = None
        if lesion is not None:
            damaged = TissueDynamics(anatomy, fluid, buffer_glia=False, plasticity=plasticity,
                                     first_axon_material=AXONS, disconnected_axons=lesion.get("axon_roots"))
            damaged.apply_lesion(neurons=lesion["neurons"], synapses=lesion["connections"])
            world.add_module("damaged_tissue", damaged)
        probe, edge = _locate_probe(world, geometry, anatomy)
        k0, cl0 = world.total(POTASSIUM), world.total(CHLORIDE)
        pulse, axon_pulse, sample_position, seed_masks = _seed(world, probe, fluid)
        count = ceil(ms / step_ms)
        spike_history, trace, lesion_trace = [], [], []
        syn = anatomy.synapses
        drive_edge = edge if lesion is None else int(lesion["test_edge"])
        source, target = ((int(syn.pre[drive_edge]), int(syn.post[drive_edge]))
                          if drive_edge is not None and drive_edge >= 0 else (None, None))
        prepared = perf_counter() - started
        started = perf_counter()
        for index in range(count):
            if source is not None:
                current = 1.5e-9 if world.time_s < .0025 else 0.0
                biology.set_current(source, current)
                if damaged is not None:
                    damaged.set_current(source, current)
            world.step(ms * 1e-3 * (index + 1) / count - world.time_s)
            spike_history.extend({"time_ms":float(event.time_s * 1e3), "neuron":event.data["neuron"]}
                                 for event in world.drain_events() if event.kind == "spike" and event.source == "tissue")
            trace.append((world.time_s * 1e3, world.sample_concentration(POTASSIUM, sample_position),
                          0.0 if target is None else biology.excitatory_S[target]))
            if damaged is not None and target is not None:
                lesion_trace.append(damaged.excitatory_S[target])
        elapsed = perf_counter() - started
        by_level, explicit_axons, axon_voxels, boxes = _collect(world, capture)
        marker_balance = abs(world.total(MARKER) / pulse - 1)
        axon_balance = abs(world.total(INTRACELLULAR) / axon_pulse - 1) if axon_pulse else None
        intracellular = biology.intracellular_salt_change_mol
        errors = [(world.total(POTASSIUM) + intracellular - k0 - pulse) / k0,
                  (world.total(CHLORIDE) + intracellular - cl0 - pulse) / cl0]
        if marker_balance > 1e-11 or (axon_balance is not None and axon_balance > 1e-11) or max(map(abs, errors)) > 1e-11:
            raise AssertionError("material balance failed")
        left_seed = axon_pulse - sum(float(world.block_amounts(INTRACELLULAR, key)[mask].sum())
                                     for key, mask in seed_masks.items())
        summary = {
            "orientation_deg": layouts[orientation].tilt_deg, "shafts": len(axes),
            "micro_radius_um": micro_radius_m * 1e6, "gpu_field_buffers": world.gpu_field_buffers,
            "fine_tip_depth_um": fine_tip_depth_m * 1e6,
            "fine_tip_box_um":None if geometry.fine_box is None else (np.asarray(geometry.fine_box) * 1e6).tolist(),
            "base_shape": grid.shape, "base_spacing_m": grid.spacing_m, "origin_m": grid.origin_m,
            "active_cells":world.active_cells, "voxels_by_spacing_um":_levels(world),
            "uniform_0_2_um_cells":int(np.prod(grid.shape, dtype=np.int64)) * (1 << LEVELS[-1])**3,
            "neurons": anatomy.neuron_count, "glia":len(anatomy.position_m) - anatomy.neuron_count,
            "synapses": len(syn.pre), "synapses_refining_near_guides": len(geometry.contact_ids),
            "axons_with_voxel_fragments":explicit_axons, "axon_voxels":axon_voxels,
            "probe_m":list(map(float,probe)), "driven_neuron":source, "target_neuron":target,
            "sample_synapse_spacing_um":None if edge is None else world.block_grid(world.block_at(probe)).spacing_m*1e6,
            "driven_synapse_edge":drive_edge,
            "presynaptic_soma_spacing_um":None if source is None else
                world.block_grid(world.block_at(anatomy.position_m[source])).spacing_m*1e6,
            "target_peak_conductance_nS":float(max(row[2] for row in trace)*1e9),
            "damaged_target_peak_conductance_nS":None if not lesion_trace else float(max(lesion_trace)*1e9),
            "spikes": spike_history, "exc_deliveries":int(biology.deliveries[0]),
            "marker_seed_mol":pulse, "marker_fraction_by_spacing_um":{k:v/pulse for k,v in by_level.items()},
            "marker_balance":marker_balance, "ion_balance_relative":list(map(float,errors)),
            "axon_marker_seed_mol":axon_pulse, "axon_marker_balance":axon_balance,
            "axon_marker_outside_seed_fraction":None if not axon_pulse else left_seed/axon_pulse,
            "gpu_peak_bytes":world.peak_gpu_bytes, "process_vram_bytes":process_vram_bytes(),
            "simulation_ms":ms, "prepare_s":prepared, "advance_s":elapsed,
        }
        if biology.plasticity is not None:
            summary["plastic_graph_active_edges"] = sum(c.phase == "active" for c in biology.plasticity.contacts)
        if biology.ion_exchange is not None:
            summary["neuron_released_mol"] = biology.ion_exchange.released_mol
            summary["neuron_recovered_mol"] = biology.ion_exchange.recovered_mol
            summary["glial_stored_mol"] = biology.glial_K_mol
        return summary, {"boxes":np.asarray(boxes), "axes_m":axes, "trace":np.asarray(trace),
                         "damaged_conductance_S":np.asarray(lesion_trace)}
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orientation", choices=("vertical","oblique"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--tilt-deg", type=float, default=30)
    parser.add_argument("--ms", type=float, default=5.0)
    parser.add_argument("--step-ms", type=float, default=0.25)
    parser.add_argument("--max-cells", type=int, default=9_000_000)
    parser.add_argument("--gpu-budget-mib", type=int, default=768)
    parser.add_argument("--gpu-reserve-mib", type=int, default=0)
    parser.add_argument("--fine-tip-depth-um", type=float, default=0)
    parser.add_argument("--micro-radius-um", type=float, default=MICRO_RADIUS_M * 1e6)
    parser.add_argument("--dense-equilibrium", action="store_true")
    parser.add_argument("--lesion", type=Path, help="neural failure masks from implant_damage")
    parser.add_argument("--plasticity", action="store_true", help="enable STDP and local contact changes")
    parser.add_argument("--ion-feedback", action="store_true", help="enable conservative spike release and recovery")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    vertical = ImplantLayout(count=args.count)
    layouts = vertical, replace(vertical, tilt_deg=args.tilt_deg)
    lesion = None
    if args.lesion is not None:
        with np.load(args.lesion, allow_pickle=False) as archive:
            lesion = {key: archive[key] for key in ("neurons", "connections", "test_edge", "axes_m")}
            if "axon_roots" in archive:
                lesion["axon_roots"] = archive["axon_roots"]
        np.testing.assert_allclose(lesion["axes_m"], layouts[int(args.orientation == "oblique")].axes_m(),
                                   atol=1e-14, err_msg="lesion geometry differs from the selected layout")
    result, scene = run(layouts, int(args.orientation == "oblique"), ms=args.ms, step_ms=args.step_ms,
                        max_cells=args.max_cells, budget_mib=args.gpu_budget_mib, capture=args.output_dir is not None,
                        micro_radius_m=args.micro_radius_um * 1e-6, compact_equilibrium=not args.dense_equilibrium,
                        lesion=lesion, fine_tip_depth_m=args.fine_tip_depth_um * 1e-6,
                        reserve_mib=args.gpu_reserve_mib, plasticity=PlasticityConfig() if args.plasticity else None,
                        ion_feedback=IonFeedbackConfig() if args.ion_feedback else None)
    data = json.dumps(result, indent=2) + "\n"
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{args.orientation}.json").write_text(data)
        np.savez_compressed(args.output_dir / f"{args.orientation}.npz", **scene)
    print(data, end="")


if __name__ == "__main__":
    main()
