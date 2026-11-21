"""Project a declared slit onto the three reconstructed BANC axon traces."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "20261118T044947Z-neck-insertion-assay"
FIXED = HERE.parent / "20261117T164207Z-fixed-neck" / "PARAMETERS.json"
OUTPUT = HERE / "GEOMETRY_FAULT.json"
SLIT_DEPTH_UM = 15.


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intersections(path, axial_um):
    """Return axial plane crossings as [transverse, axial, height], from front."""
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    matches = []
    for j, (a, b) in enumerate(zip(path[:-1], path[1:], strict=True)):
        if not min(a[1], b[1]) <= axial_um < max(a[1], b[1]):
            continue
        fraction = (axial_um - a[1]) / (b[1] - a[1])
        point = a + fraction * (b - a)
        linear = cumulative[j] + fraction * lengths[j]
        matches.append(dict(point_um=point.tolist(), front_fraction=float(linear / cumulative[-1])))
    return matches, float(cumulative[-1])


def main():
    summary = json.loads((PARENT / "MECHANICS.json").read_text())
    data_path = PARENT / "MECHANICS.npz"
    assert sha(data_path) == summary["output_sha"]
    parameters = json.loads(FIXED.read_text())
    dt = parameters["dt_ms"]
    axial = summary["chart"]["length_um"] / 2.
    rows = []
    with np.load(data_path, allow_pickle=False) as data:
        for j, root in enumerate(summary["roots"]):
            source = parameters["ports"][root]
            crossings, total = intersections(data[f"axon{j}"], axial)
            assert abs(total - source["length_um"]) < 1e-7 and len(crossings) == 1
            hit, = crossings
            direction = "AN" if j < 2 else "DN"
            entry_fraction = 1. - hit["front_fraction"] if direction == "AN" else hit["front_fraction"]
            gate = int(np.ceil(entry_fraction * source["propagation_ms"] / dt - 1e-12))
            assert 0 < gate < source["delay_ticks"]
            rows.append(dict(root=root, direction=direction, **hit,
                entrance_fraction=entry_fraction, cut_by_projected_slit=0. <= hit["point_um"][0] <= SLIT_DEPTH_UM,
                entry_to_fault_ticks=gate, total_delay_ticks=source["delay_ticks"]))
    result = dict(axial_um=axial, slit_depth_um=SLIT_DEPTH_UM, trace_intersections=rows,
        area_scope="3 reconstructed reference trajectories in a plane, slit extended through unresolved z",
        predicts_rupture=False, source_sha256={str(path):sha(path) for path in
            (FIXED, PARENT/"MECHANICS.json",data_path,Path(__file__).resolve())})
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print([(r["root"], r["point_um"][0], r["entry_to_fault_ticks"],
            r["total_delay_ticks"], r["cut_by_projected_slit"]) for r in rows])


if __name__ == "__main__":
    main()
