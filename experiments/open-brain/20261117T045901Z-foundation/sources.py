"""Pin the two donors and derive the run's four source-qualified axons."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
UPSTREAM = HERE / "upstream"
COMMIT = "a3db62f9436074e485c0278290c2164ed6150808"
GIT_BLOBS = {
    "LICENSE": "d159169d1050894d3ea3b98e1c965c4058208fe1",
    "code/benchmark.py": "0c9cb3b39f4cb7f7529f795ae1c6c127d1aae197",
    "code/run_pytorch.py": "4a929ca500af56e94bfaf64e5d526da65568f332",
    "data/2025_Completeness_783.csv": "7bdf251b81e857b19f2cf664eeb2e31ba614151f",
    "data/2025_Connectivity_783.parquet": "d386555d1a5f40ebfa1380bcb05b1fab044855fd",
}
VISUAL = REPO / "experiments/visual-orientation"
STEER = VISUAL / "20261116T023521Z"
FORWARD = VISUAL / "20261116T170000Z"
META = REPO / "data/banc888/banc_888_meta.feather"
EDGES = STEER / "banc_888_edgelist_simple_v2.feather"
NECK_PLANES = (370000.0, 484000.0)
PORTS = (
    ("DNa02", "left", "720575941510475536", "720575940629327659"),
    ("DNa02", "right", "720575941456897005", "720575940604737708"),
    ("DNg100", "left", "720575941626500746", "720575940640978048"),
    ("DNg100", "right", "720575941500851362", "720575940647228468"),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_upstream():
    pinned = {}
    for name, expected in GIT_BLOBS.items():
        path = UPSTREAM / name
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                blob.update(chunk)
        assert blob.hexdigest() == expected, path
        pinned[name] = {"sha256": digest(path), "git_blob": expected}
    return pinned


def flywire_annotations(name):
    path = REPO / f"data/flywire/fafb_{name}.csv.gz"
    with gzip.open(path, "rt") as stream:
        return {row["root_id"]: row for row in csv.DictReader(stream)}


def verify_branch(path, front, rear):
    tree = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            index, _, _, y, _, _, parent = line.split()[:7]
            index = int(index)
            assert index not in tree, path
            tree[index] = int(parent), float(y)
    for child, plane in zip((front, rear), NECK_PLANES, strict=True):
        parent, y = tree[child]
        assert min(y, tree[parent][1]) <= plane < max(y, tree[parent][1]), path
    ancestor = rear
    seen = set()
    while ancestor != front:
        assert ancestor in tree and ancestor not in seen, path
        seen.add(ancestor)
        ancestor = tree[ancestor][0]


def make_binding(upstream_index):
    steer_graph = load(STEER / "OFFICIAL_GRAPH.json")
    forward_graph = load(FORWARD / "DNg100_GRAPH.json")
    edge_sha = digest(EDGES)
    assert edge_sha == "363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb"
    assert digest(META) == "86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098"
    meta = feather.read_table(META, columns=["banc_888_id", "side", "cell_type", "region",
                                            "fafb_match", "neurotransmitter_verified",
                                            "neurotransmitter_predicted"])
    by_root = {row["banc_888_id"]: row for row in meta.to_pylist() if row["banc_888_id"]}
    assert by_root
    types = flywire_annotations("consolidated_cell_types")
    classification = flywire_annotations("classification")
    records = []
    for cell_type, side, banc, wire in PORTS:
        row = by_root[banc]
        assert (row["cell_type"], row["side"], row["fafb_match"]) == (cell_type, side, wire)
        assert row["region"] == "central_brain"
        assert types[wire]["primary_type"] == cell_type
        assert (classification[wire]["side"], classification[wire]["super_class"]) == (side, "descending")
        directory = STEER if cell_type == "DNa02" else FORWARD
        binding = load(directory / f"{cell_type}_BINDING.json")
        port = next(p for p in binding["ports"] if p["root_id"] == banc)
        geometry_path = directory / f"{banc}_GEOMETRY.json"
        geometry = load(geometry_path)
        swc = next(e for e in port["evidence"] if e["path"].endswith(".swc"))
        swc_path = Path(swc["path"])
        assert digest(swc_path) == swc["sha256"] and geometry["root_id"] == banc
        if cell_type == "DNa02":
            crossing = load(STEER / "DNa02_CROSSING_CONTINUITY.json")[banc]["continuous_branch"]
            rear = crossing["posterior_child_on_same_path"][0]
        else:
            crossing = geometry["continuity"]
            rear = crossing["rear"][0]["child"]
        front = crossing["front"]["child"]
        verify_branch(swc_path, front, rear)
        verified = row["neurotransmitter_verified"]
        assert (verified or row["neurotransmitter_predicted"]) == "acetylcholine"
        records.append(dict(cell_type=cell_type, side=side, banc_root=banc, flywire_root=wire,
                            flywire_index=upstream_index[int(wire)], swc=str(swc_path),
                            swc_sha256=swc["sha256"], crossing_children=[front, rear],
                            transmitter_basis="verified" if verified else "predicted"))

    # The Feather table uses exact root strings. Restrict posts by the published region.
    roots = [row["banc_root"] for row in records]
    table = feather.read_table(EDGES, columns=["pre", "post", "count"], memory_map=True)
    key_type = table.schema.field("pre").type
    root_keys = [int(root) if pa.types.is_integer(key_type) else root for root in roots]
    selected = table.filter(pc.is_in(table["pre"], value_set=pa.array(root_keys, type=key_type)))
    official = {}
    for row in selected.to_pylist():
        pre, post = str(row["pre"]), str(row["post"])
        if by_root.get(post, {}).get("region") == "ventral_nerve_cord":
            assert (pre, post) not in official and row["count"] > 0
            official[pre, post] = row["count"]
    reviewed = {(x["pre"], x["post"]): x["synapses"] for x in steer_graph["connections"]}
    reviewed.update({(x["pre"], x["post"]): x["count"] for x in forward_graph["edges"]})
    assert official == reviewed and len(official) == 2791
    targets = sorted({post for _, post in official})
    target_indices = {root: index for index, root in enumerate(targets)}
    source_indices = {root: index for index, root in enumerate(roots)}
    counts = np.zeros((len(roots), len(targets)), dtype=np.int32)
    for (pre, post), count in official.items():
        counts[source_indices[pre], target_indices[post]] = count
    for record, row in zip(records, counts, strict=True):
        record["vnc_targets"] = int(np.count_nonzero(row))
        record["vnc_synapses"] = int(row.sum())
    return records, targets, counts
