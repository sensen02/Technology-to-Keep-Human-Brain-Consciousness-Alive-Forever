"""Measure model neural damage along quasistatic vertical and oblique insertions."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..modules.array_damage import RadialInsertion
from ..modules.tissue import EXCITATORY, ImplantLayout, build_anatomy, tissue_grid
from .hybrid_implant import BASE_M


def measure(anatomy, layout, mechanic, *, stations=65):
    start = perf_counter()
    axes = layout.axes_m()
    travel = float(np.linalg.norm(axes[0, 1] - axes[0, 0]))
    a = anatomy
    s = a.synapses
    xyz = a.position_m
    n = a.neuron_count
    onset, failure = mechanic.onset_Pa, mechanic.failure_Pa

    def point_threshold(points, threshold):
        return mechanic.first_depths(points, points, axes, threshold)

    soma_contact = point_threshold(xyz, onset)
    soma_failure = point_threshold(xyz, failure)
    synapse_contact = point_threshold(s.position_m, onset)
    synapse_failure = point_threshold(s.position_m, failure)
    axon_root = mechanic.first_depths(xyz[:n], a.axon_branch_m, axes, failure)
    axon_branch = mechanic.first_depths(a.axon_branch_m[s.pre], s.position_m, axes, failure)
    axon_failure = np.minimum(axon_root[s.pre], axon_branch)
    axon_owner = axon_root.copy()
    np.minimum.at(axon_owner, s.pre, axon_failure)
    dendrite = mechanic.first_depths(s.position_m, xyz[s.post], axes, failure)
    disconnected = np.minimum.reduce((soma_failure[s.pre], soma_failure[s.post],
                                      synapse_failure, axon_failure, dendrite))
    # Report all mechanisms separately and also the union, indexed by stable anatomy IDs.
    measurements = {
        "neuron_somata_strained": soma_contact[:n], "neuron_somata_failed": soma_failure[:n],
        "glial_somata_failed": soma_failure[n:], "synapses_strained": synapse_contact,
        "synapse_locations_failed": synapse_failure, "axon_branches_failed": axon_failure,
        "unique_axons_failed": axon_owner, "dendrite_segments_failed": dendrite,
        "connections_disconnected": disconnected,
    }
    first = {name: (float(values.min() * layout.normal_depth_m / travel * 1e6)
                    if np.any(np.isfinite(values)) else None)
             for name, values in measurements.items()}
    depths = []
    for station in np.linspace(0, travel, stations):
        row = {"normal_depth_um": float(station / travel * layout.normal_depth_m * 1e6),
               "travel_along_shaft_um": float(station * 1e6)}
        row.update({key:int(np.count_nonzero(values <= station)) for key, values in measurements.items()})
        depths.append(row)
    # A surviving presynaptic soma, a surviving postsynaptic soma and one failed axonal route.
    eligible = np.flatnonzero(np.isfinite(axon_failure) & ~np.isfinite(soma_failure[s.pre])
                             & ~np.isfinite(soma_failure[s.post]) & (a.cell_type[s.pre] == EXCITATORY))
    chosen = int(eligible[np.argmin(axon_failure[eligible])]) if len(eligible) else None
    result = {
        "tilt_deg":layout.tilt_deg, "shafts":len(axes), "model":"independent plane-strain rounded-tip contact",
        "neuron_count":n, "glia_count":len(xyz)-n, "synapse_count":len(s.pre),
        "parameters":asdict(mechanic), "failure_radius_um":mechanic.failure_radius_m() * 1e6,
        "first_normal_depth_um":first, "stations":depths, "elapsed_s":perf_counter()-start,
        "test_axonal_contact":None if chosen is None else {
            "edge":chosen, "pre":int(s.pre[chosen]), "post":int(s.post[chosen]),
            "axonal_failure_normal_depth_um":float(axon_failure[chosen] * layout.normal_depth_m / travel * 1e6)
        }
    }
    mask = {"neurons":np.isfinite(soma_failure[:n]), "glia":np.isfinite(soma_failure[n:]),
            "axon_roots":np.isfinite(axon_root),
            "glia_first_m":soma_failure[n:], "connections":np.isfinite(disconnected),
            "test_edge":np.asarray(-1 if chosen is None else chosen), "soma_xyz_m":xyz[:n],
            "synapse_xyz_m":s.position_m, "axes_m":axes, "soma_first_m":soma_failure[:n],
            "synapse_first_m":synapse_failure, "connection_first_m":disconnected}
    return result, mask


def entry_check(radius):
    """Solve initial surface contact in the existing finite-element specimen."""
    from ..mechanics import AxisymmetricIndentation, AxisymmetricSolid
    results = []
    for resolution in (10, 12):
        solid = AxisymmetricSolid(radius_m=3*radius, height_m=4*radius, radial_elements=3*resolution,
                                 axial_elements=4*resolution, shear_modulus_Pa=3e3, bulk_modulus_Pa=120e3,
                                 damage_length_m=.2*radius, damage_onset_Pa=400, full_damage_Pa=850,
                                 residual_stiffness_fraction=.8)
        solver = AxisymmetricIndentation(solid, radius, contact_stiffness_Pa_m=2.1e9*100e-6/radius)
        stations = [asdict(solver.advance(float(radius * factor))) for factor in np.linspace(0, .96, 25)]
        results.append({"elements_per_radius":resolution,"stations":stations})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--tilt-deg", type=float, default=30)
    parser.add_argument("--shaft-radius-um", type=float, default=2)
    parser.add_argument("--entry-fem", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    vertical = ImplantLayout(count=args.count)
    layouts = (vertical, replace(vertical, tilt_deg=args.tilt_deg))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.entry_fem:
        entries = entry_check(args.shaft_radius_um * 1e-6)
        (args.output_dir/"entry-fem.json").write_text(json.dumps(entries, indent=2, allow_nan=False) + "\n")
        return
    grid, _ = tissue_grid(layouts, spacing_m=BASE_M)
    anatomy = build_anatomy(grid)
    mechanic = RadialInsertion(radius_m=args.shaft_radius_um * 1e-6)
    for name, layout in zip(("vertical","oblique"), layouts):
        result, mask = measure(anatomy, layout, mechanic)
        (args.output_dir/f"{name}-damage.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(args.output_dir/f"{name}-damage.npz", **mask)
        print(name, result["stations"][-1], flush=True)


if __name__ == "__main__":
    main()
