"""Bind DNg100 activity to archived BANC v888 axons and official VNC partners."""
import base64
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
VISUAL = HERE.parent / "20261116T023521Z"
NECK = PROJECT / "experiments/neck-real-structure"
GEOMETRY = NECK / "20261112T214803Z"
DOWNLOADS = GEOMETRY / "geometry_downloads"
OLD = NECK / "20261112T225331Z"
sys.path.insert(0, str(GEOMETRY / "geometry_sources"))
from validate_swc import parse_swc, sha, validate

ROOTS = ("720575941626500746", "720575941500851362")
META = PROJECT / "data/banc888/banc_888_meta.feather"
POINTS = DOWNLOADS / "neck_connective_y92500.parquet"
EDGE = VISUAL / "banc_888_edgelist_simple_v2.feather"
PRIOR = VISUAL / "OFFICIAL_GRAPH.json"
GEOMETRY_REFERENCE = OLD / "dnp09_geometry/720575941433155799_METADATA.json"
PORT_REFERENCE = VISUAL / "DNa02_BINDING.json"
BINDING = HERE / "DNg100_BINDING.json"
GRAPH = HERE / "DNg100_GRAPH.json"


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def matched_branch(swc, report, annotations):
    tree, _, _ = parse_swc(swc)
    front, rear = (report["plane_intersections"][str(y)] for y in report["plane_y_nm"])
    ahead = {cross["child"]: cross for cross in front}
    pairs = defaultdict(list)
    for cross in rear:
        node = cross["child"]
        while node != -1:
            if node in ahead:
                pairs[node].append(cross)
            node = tree[node][3]
    candidates = [(math.dist(cross["point_nm"], point), cross, pairs[child])
                  for child, cross in ahead.items() if child in pairs
                  for point in annotations]
    distance, first, subsequent = min(candidates, key=lambda item: item[0])
    if distance > 1500:
        raise ValueError(f"Continuous branch is {distance:.1f} nm from curated seed")
    return dict(distance_to_front_seed_nm=distance, front=first, rear=subsequent)


def main():
    data = feather.read_table(META, columns=["root_888", "banc_888_id", "cell_type",
                                              "region", "side", "super_class", "tract"])
    rows = defaultdict(list)
    for row in data.to_pylist():
        rows[row["root_888"]].append(row)
    point_table = parquet.read_table(POINTS)
    markers = point_table.filter(pc.equal(point_table["valid"], True)).to_pylist()
    planes = json.loads(GEOMETRY_REFERENCE.read_text())["planes"]
    prior = json.loads(PRIOR.read_text())
    edge_pin = next(item for item in prior["sources"] if item["path"] == str(EDGE))
    assert sha(EDGE) == edge_pin["sha256"] and prior["synapse_release"] == "v2"
    references = [pin(p) for p in (META, POINTS, GEOMETRY_REFERENCE, PRIOR, PORT_REFERENCE,
                                    DOWNLOADS / "banc_swc_skeletons.md", Path(__file__))]
    ports = []
    for side, root, resolution in zip(("left", "right"), ROOTS, ("skeleton", "l2"), strict=True):
        row, = rows[root]
        assert (row["cell_type"], row["side"], row["super_class"], row["tract"], row["banc_888_id"]) == (
            "DNg100", side, "descending", "cervical_connective", root)
        swc = HERE / f"{root}_{resolution}.swc"
        header = HERE / (f"{root}.headers" if resolution == "skeleton" else f"{root}_l2.headers")
        raw = header.read_text()
        generation, = re.findall(r"^x-goog-generation: (\d+)$", raw, re.M)
        md5, = re.findall(r"^x-goog-hash: md5=(\S+)$", raw, re.M)
        assert base64.b64encode(hashlib.md5(swc.read_bytes()).digest()).decode() == md5
        points = [[float(v)*scale for v, scale in zip(m["pt_position"], planes["voxel_size_nm"], strict=True)]
                  for m in markers if str(m["pt_root_id"]) == root]
        facts = dict(schema="BANC_V888_SWC_GEOMETRY_CHECK_V1", dataset="BANC", materialization=888,
                     root_id=root, coordinate_space="BANC_NATIVE", coordinate_units="nm", swc_sha256=sha(swc),
                     source_evidence=[*references, pin(header)], planes=planes,
                     version_identity_evidence=dict(root_888=root, verified=True, source_row=row))
        report = validate(swc, facts)
        assert report["is_single_tree"] and points and report["each_plane_intersected"]
        report.update(gcs_generation=generation, resolution=resolution, md5=md5,
                      continuity=matched_branch(swc, report, points), seed_tolerance_nm=1500)
        geometry = HERE / f"{root}_GEOMETRY.json"
        write(geometry, report)
        ports.append(dict(root_id=root, direction="DN", side=side, source_reviewed=True,
                          structural_evidence_status="SOURCE_GEOMETRY_REVIEWED", source_cell_type="DNg100",
                          evidence=[pin(swc), pin(geometry)]))
        print(side, resolution, report["continuity"]["distance_to_front_seed_nm"], flush=True)

    edges = []
    with pa.memory_map(str(EDGE), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            matching = batch.filter(pc.is_in(batch.column("pre"), value_set=pa.array(ROOTS)))
            for row in matching.select(["pre", "post", "count"]).to_pylist():
                partners = rows[row["post"]]
                if len(partners) == 1 and partners[0]["region"] == "ventral_nerve_cord":
                    assert row["count"] > 0
                    edges.append(row)
    assert {e["pre"] for e in edges} == set(ROOTS)
    write(GRAPH, dict(roots=ROOTS, source=edge_pin, edges=edges, release="BANC v888 v2",
                      target_count=len({edge["post"] for edge in edges}), physiological_gains_verified=False))
    reference = json.loads(PORT_REFERENCE.read_text())
    ports.extend(p for p in reference["ports"] if p["direction"] == "AN")
    write(BINDING, dict(schema=reference["schema"], scope=reference["scope"], ports=ports,
                        source_files=[*references, pin(GRAPH), edge_pin],
                        transport_config=reference["transport_config"],
                        AN_runtime="archival identity, inactive during this experiment"))
    print(len(edges), "real VNC edges", flush=True)


if __name__ == "__main__":
    main()
