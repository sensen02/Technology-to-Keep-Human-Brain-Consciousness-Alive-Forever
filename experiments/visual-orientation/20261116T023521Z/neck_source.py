"""Verify bilateral DNa02 source provenance and build the scoped transport binding."""
import base64
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.parquet as parquet

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
NECK = PROJECT / "experiments/neck-real-structure"
GEOMETRY = NECK / "20261112T214803Z"
OLD = NECK / "20261112T225331Z/dnp09_interface"
sys.path.insert(0, str(GEOMETRY / "geometry_sources"))
from validate_swc import sha, validate

PORTS = {
    "left": ("720575941510475536", "1778894342874524", 4154629, "4JP+lkGtSi79IQaFSTQXEA=="),
    "right": ("720575941456897005", "1778894319194186", 4174858, "c+w7i5i7dvz7YM5oUXqsPQ=="),
}
META = PROJECT / "data/banc888/banc_888_meta.feather"
EDGES = PROJECT / "data/flywire/banc_connections_princeton.csv.gz"
DOWNLOADS = GEOMETRY / "geometry_downloads"


def pin(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def save(path, data):
    with path.open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def main():
    metadata = feather.read_table(META, columns=["root_id", "root_888", "cell_type", "super_class", "side", "tract", "proofread"])
    points = parquet.read_table(DOWNLOADS / "neck_connective_y92500.parquet")
    valid = points.filter(pc.equal(points["valid"], True)).to_pylist()
    projections = {entry[0]: Counter() for entry in PORTS.values()}
    with gzip.open(EDGES, "rt", newline="") as stream:
        for edge in csv.DictReader(stream):
            if edge["pre_root_id"] in projections:
                projections[edge["pre_root_id"]][edge["neuropil"]] += int(edge["syn_count"])

    old_binding = json.loads((OLD / "SOURCE_BINDING.json").read_text())
    common = [pin(path) for path in (META, EDGES, DOWNLOADS / "neck_connective_y92500.parquet",
                                    DOWNLOADS / "neck_connective_y92500.md",
                                    DOWNLOADS / "neck_connective_y121000.md", DOWNLOADS / "banc_swc_skeletons.md")]
    ports = []
    for side, (root, generation, size, md5) in PORTS.items():
        skeleton = HERE / f"{root}_skeleton.swc"
        header = HERE / f"{root}.download.headers"
        assert f"x-goog-generation: {generation}" in header.read_text().lower()
        assert skeleton.stat().st_size == size
        assert base64.b64encode(hashlib.md5(skeleton.read_bytes()).digest()).decode() == md5
        rows = metadata.filter(pc.equal(metadata["root_888"], root)).to_pylist()
        assert len(rows) == 1
        row = rows[0]
        assert row["side"] == side and row["cell_type"] == "DNa02"
        assert row["super_class"] == "descending" and row["tract"] == "cervical_connective"
        annotations = [item["pt_position"] for item in valid if str(item["pt_root_id"]) == root]
        assert annotations
        previous = json.loads((NECK / "20261112T225331Z/dnp09_geometry/720575941433155799_METADATA.json").read_text())
        dependencies = common + [pin(header)]
        facts = dict(schema=previous["schema"], dataset="BANC", materialization=888, root_id=root,
                     coordinate_space="BANC_NATIVE", coordinate_units="nm", swc_sha256=sha(skeleton),
                     source_evidence=dependencies, version_identity_evidence=dict(root_888=root, verified=True, source_row=row),
                     planes=previous["planes"])
        geometry = validate(skeleton, facts)
        assert geometry["is_single_tree"] and geometry["each_plane_intersected"]
        assert any("ProNm" in region for region in projections[root]), "missing T1 outputs in raw edge table"
        geometry.update(annotated_front_plane_voxels=annotations, gcs_generation=generation, md5_base64=md5,
                        exact_raw_endpoint_output_synapses_by_region=dict(projections[root].most_common()),
                        raw_projection_version_scope="Counts from exact endpoint strings in local raw edge table.")
        report = HERE / f"{root}_GEOMETRY.json"
        save(report, geometry)
        ports.append(dict(root_id=root, direction="DN", side=side, source_reviewed=True,
                          structural_evidence_status="SOURCE_GEOMETRY_REVIEWED", source_cell_type="DNa02",
                          native_tuning_established=False, evidence=[pin(report), pin(skeleton)]))
    ports.extend(port for port in old_binding["ports"] if port["direction"] == "AN")
    transport = old_binding["transport_config"]
    support = common + [pin(HERE / "neck_source.py"), pin(OLD / "SOURCE_BINDING.json"),
                        pin(OLD / "source_transport.py")]
    assert len(ports) == 3
    save(HERE / "DNa02_BINDING.json", dict(schema=old_binding["schema"], scope=old_binding["scope"],
         status="SCOPED_DNA02_VISUAL_TASK_SOURCE_BINDING", ports=ports, source_files=support,
         transport_config=transport, an_runtime_scope="Archived source identity retained, inactive in this task."))
    for port in ports[:2]:
        report = json.loads(Path(port["evidence"][0]["path"]).read_text())
        print(port["side"], port["root_id"], report["nodes"], report["each_plane_intersected"],
              dict(list(report["exact_raw_endpoint_output_synapses_by_region"].items())[:5]))


if __name__ == "__main__":
    main()
