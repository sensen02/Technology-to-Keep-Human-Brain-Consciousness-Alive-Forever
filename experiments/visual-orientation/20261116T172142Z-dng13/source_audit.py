"""Verify bilateral DNg13 provenance, cervical continuity, and official v888 VNC edges."""
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
NECK = PROJECT / "experiments/neck-real-structure"
GEOMETRY = NECK / "20261112T214803Z"
VISUAL = HERE.parent / "20261116T023521Z"
sys.path.insert(0, str(GEOMETRY / "geometry_sources"))
from validate_swc import parse_swc, sha, validate

SKELETONS = {
    "left": ("720575941445859754", "1778894315609221", 3186494, "NlOIavppPfQ7hkV62qIffA=="),
    "right": ("720575941535086424", "1778894354842115", 3166453, "CuaeqmpkJ+vH4THelmcnSA=="),
}
GCS = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
SWC_PREFIX = GCS + "compiled_data/banc_888/banc_banc_space_swc/"
DOWNLOADS = GEOMETRY / "geometry_downloads"
META = PROJECT / "data/banc888/banc_888_meta.feather"
EDGE = VISUAL / "banc_888_edgelist_simple_v2.feather"
EDGE_HEADERS = VISUAL / "edges_v2.download.headers"
PLANES = NECK / "20261112T225331Z/dnp09_geometry/720575941433155799_METADATA.json"
MARKERS = DOWNLOADS / "neck_connective_y92500.parquet"


def pin(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def check_download(path, headers, generation, size, md5):
    assert f"x-goog-generation: {generation}" in headers.read_text().lower()
    assert path.stat().st_size == size
    with path.open("rb") as stream:
        assert base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode() == md5


def crossing_path(swc, geometry, marker):
    nodes, _, _ = parse_swc(swc)
    front_y, back_y = map(str, geometry["plane_y_nm"])
    point = [a * b for a, b in zip(marker[0]["pt_position"], geometry["metadata"]["planes"]["voxel_size_nm"], strict=True)]
    back = {}
    for crossing in geometry["plane_intersections"][back_y]:
        child = crossing["child"]
        while child != -1:
            back.setdefault(child, []).append(crossing["child"])
            child = nodes[child][3]
    candidates = []
    for front in geometry["plane_intersections"][front_y]:
        downstream = back.get(front["child"], [])
        candidates.append(dict(front=front, distance_to_annotated_voxel_nm=math.dist(front["point_nm"], point),
                               posterior_child_on_same_path=downstream))
    continuous = [row for row in candidates if row["posterior_child_on_same_path"]]
    assert continuous
    qualified = min(continuous, key=lambda row: row["distance_to_annotated_voxel_nm"])
    assert qualified["distance_to_annotated_voxel_nm"] < 2000
    return dict(marker_voxels=marker[0]["pt_position"], marker_tag=marker[0]["tag"],
                nearest_continuous=qualified, all_front=candidates)


def main():
    check_download(EDGE, EDGE_HEADERS, "1780396134870867", 305250378, "OUQG+Km98JPIlfla/09sSQ==")
    metadata = feather.read_table(META, columns=["root_id","root_888","banc_888_id","cell_type","super_class","side",
                                                  "tract","proofread","region","neuromere"]).to_pylist()
    index = {}
    duplicate = set()
    for row in metadata:
        root = row["root_888"]
        if root:
            if root in index:
                duplicate.add(root)
            index[root] = row
    points = parquet.read_table(MARKERS)
    planes = json.loads(PLANES.read_text())["planes"]
    source_paths = [META, EDGE, EDGE_HEADERS, PLANES, MARKERS, DOWNLOADS / "neck_connective_y92500.md",
                    DOWNLOADS / "neck_connective_y121000.md", DOWNLOADS / "banc_swc_skeletons.md", Path(__file__)]
    provenance = [pin(path) for path in source_paths]
    ports = {}
    for side, (root, generation, size, md5) in SKELETONS.items():
        row = index[root]
        assert root not in duplicate and row["root_id"] == row["banc_888_id"] == root
        assert (row["side"], row["cell_type"], row["super_class"], row["tract"], row["proofread"]) == (
            side, "DNg13", "descending", "cervical_connective", "TRUE")
        marker = points.filter(pc.and_(pc.equal(points["pt_root_id"], int(root)), pc.equal(points["valid"], True))).to_pylist()
        assert len(marker) == 1 and marker[0]["tag"] == ("neck connective (right)" if side == "left" else "neck connective (left)")
        swc = HERE / f"{root}_skeleton.swc"
        headers = HERE / f"{root}.download.headers"
        check_download(swc, headers, generation, size, md5)
        contract = dict(schema="BANC_V888_SWC_GEOMETRY_CHECK_V1",dataset="BANC",materialization=888,
                        root_id=root,coordinate_space="BANC_NATIVE",coordinate_units="nm",swc_sha256=sha(swc),
                        source_evidence=provenance+[pin(headers)],version_identity_evidence=dict(root_888=root,verified=True,source_row=row),
                        planes=planes)
        geometry = validate(swc, contract)
        assert geometry["is_single_tree"] and geometry["each_plane_intersected"]
        name = f"{root}_GEOMETRY.json"
        geometry["continuous_branch"] = crossing_path(swc, geometry, marker)
        geometry["generation_url"] = SWC_PREFIX + swc.name + f"?generation={generation}"
        (HERE / name).write_text(json.dumps(geometry, indent=2)+"\n")
        ports[root] = dict(side=side, source_evidence=[pin(swc),pin(HERE/name)])
    outgoing = {root: {} for root in ports}
    with pa.memory_map(str(EDGE), "r") as data:
        reader = ipc.open_file(data)
        roots = pa.array(list(ports))
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            selected = batch.filter(pc.is_in(batch.column("pre"), value_set=roots))
            for row in selected.select(["pre","post","count"]).to_pylist():
                assert row["count"] > 0 and row["post"] not in outgoing[row["pre"]]
                outgoing[row["pre"]][row["post"]] = row["count"]
    all_targets = {}
    for root, edges in outgoing.items():
        vnc = {post: count for post, count in edges.items() if post in index and post not in duplicate
               and index[post]["region"] == "ventral_nerve_cord"}
        assert vnc
        by_neuromere = Counter()
        for post, count in vnc.items():
            by_neuromere[index[post]["neuromere"]] += count
            all_targets[post] = index[post]
        ports[root].update(vnc_targets=len(vnc),vnc_synapses=sum(vnc.values()),all_outgoing=len(edges),
                           vnc_synapses_by_target_neuromere=dict(by_neuromere),
                           vnc_connections=[dict(post=post,synapses=count) for post,count in sorted(vnc.items())],
                           unmatched_post_ids=sorted(post for post in edges if post not in index or post in duplicate))
    result = dict(status="PINNED_BILATERAL_DNG13_CERVICAL_AND_VNC_SOURCE",materialization=888,
                  synapse_release="v2",sources=provenance,ports=ports, targets=list(all_targets.values()),
                  physiology_calibrated=False, physical_steering_established=False)
    (HERE / "DNG13_SOURCE.json").write_text(json.dumps(result,indent=2)+"\n")
    for root, port in ports.items():
        print(root,port["side"],port["vnc_targets"],port["vnc_synapses"],port["vnc_synapses_by_target_neuromere"])


if __name__ == "__main__":
    main()
