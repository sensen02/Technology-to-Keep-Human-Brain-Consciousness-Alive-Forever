"""Pin the publication's BANC v888 v2 edge table and derive bilateral DN output."""
import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOTS = ("720575941510475536", "720575941456897005")
SOURCE = HERE / "banc_888_edgelist_simple_v2.feather"
METADATA = PROJECT / "data/banc888/banc_888_meta.feather"
CONTRACT = HERE / "OFFICIAL_GRAPH.json"
BINDING = HERE / "DNa02_BINDING.json"
GENERATION = "1780396134870867"
OFFICIAL = ("https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
            "compiled_data/banc_888/banc_888_edgelist_simple_v2.feather?generation=" + GENERATION)


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm)


def source_ref(path):
    return dict(path=str(path.resolve()), sha256=digest(path).hexdigest())


def derive():
    headers = HERE / "edges_v2.download.headers"
    assert f"x-goog-generation: {GENERATION}" in headers.read_text().lower()
    assert SOURCE.stat().st_size == 305250378
    assert base64.b64encode(digest(SOURCE, "md5").digest()).decode() == "OUQG+Km98JPIlfla/09sSQ=="
    bound = json.loads(BINDING.read_text())
    assert [p["root_id"] for p in bound["ports"] if p["direction"] == "DN"] == list(ROOTS)
    metadata = feather.read_table(METADATA, columns=["banc_888_id", "root_id", "root_888",
                                      "cell_type", "region", "super_class", "side", "neuromere"])
    by_root = defaultdict(list)
    for row in metadata.to_pylist():
        if row["root_888"]:
            by_root[row["root_888"]].append(row)
    for side, rid in zip(("left", "right"), ROOTS, strict=True):
        row, = by_root[rid]
        assert row["side"] == side and row["cell_type"] == "DNa02"
        assert row["root_id"] == row["banc_888_id"] == rid

    outgoing = {rid: {} for rid in ROOTS}
    with pa.memory_map(str(SOURCE), "r") as data:
        reader = ipc.open_file(data)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            chosen = batch.filter(pc.is_in(batch.column("pre"), value_set=pa.array(ROOTS)))
            for row in chosen.select(["pre", "post", "count"]).to_pylist():
                assert row["count"] > 0 and row["post"] not in outgoing[row["pre"]]
                outgoing[row["pre"]][row["post"]] = row["count"]
    posts = set().union(*(edges.keys() for edges in outgoing.values()))
    joined = {rid: by_root[rid][0] for rid in posts if len(by_root[rid]) == 1}
    nodes = [joined[rid] for rid in sorted(joined) if joined[rid]["region"] == "ventral_nerve_cord"]
    connections = []
    for presynaptic in ROOTS:
        for node in nodes:
            post = node["root_888"]
            if post in outgoing[presynaptic]:
                connections.append(dict(pre=presynaptic, post=post, synapses=outgoing[presynaptic][post]))
    assert set(edge["pre"] for edge in connections) == set(ROOTS)
    with CONTRACT.open("x") as stream:
        json.dump(dict(
            scope="Publication v888 v2 DNa02 direct VNC targets with unique exact root_888 join.",
            materialization=888, synapse_release="v2", official_url=OFFICIAL, generation=GENERATION,
            sources=[source_ref(path) for path in (SOURCE, METADATA, headers, BINDING)],
            roots=ROOTS, targets=nodes, connections=connections,
            all_outgoing_edge_count={rid: len(outgoing[rid]) for rid in ROOTS},
            unjoined_post_root_ids=sorted(posts-joined.keys()),
            non_vnc_posts=sorted(set(joined)-{row["root_888"] for row in nodes}),
            engineering_projection="Nonnegative synapse counts divided by largest count at each target.",
            native_synaptic_gain_established=False,
        ), stream, indent=2)
        stream.write("\n")
    print("Official VNC graph", len(nodes), len(connections), "synapses", sum(edge["synapses"] for edge in connections))


if __name__ == "__main__":
    derive()
