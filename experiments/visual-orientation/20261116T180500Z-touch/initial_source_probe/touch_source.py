"""Audit and bind the two T1 AN06B002 axons, their inputs and brain targets."""
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
sys.path.insert(0, str(HERE.parent / "20261116T170000Z"))
from forward_source import (META, POINTS, EDGE, PRIOR, GEOMETRY_REFERENCE,
                            pin, write, validate, parse_swc, sha)

DN = HERE.parent / "20261116T170000Z/DNg100_BINDING.json"
ROOTS = ("720575941633589179", "720575941626534090")


def path_to_brain(swc, geometry, marks):
    tree, _, _ = parse_swc(swc)
    anterior, posterior = (geometry["plane_intersections"][str(y)] for y in geometry["plane_y_nm"])
    posterior_index = {x["child"]: x for x in posterior}
    candidates = []
    for front in anterior:
        node = front["parent"]
        while node != -1 and node not in posterior_index:
            node = tree[node][3]
        if node != -1:
            distance = min(math.dist(front["point_nm"], point) for point in marks)
            candidates.append((distance, front, posterior_index[node]))
    distance, front, rear = min(candidates, key=lambda item: item[0])
    assert distance <= 1500, distance
    return dict(distance_to_front_seed_nm=distance, front=front, rear=rear)


def main():
    columns = ["root_888","banc_888_id","cell_type","region","side","super_class","tract",
               "neuromere","cell_class","cell_sub_class","body_part_sensory"]
    table = feather.read_table(META, columns=columns)
    meta = defaultdict(list)
    for row in table.to_pylist():
        meta[row["root_888"]].append(row)
    pts = parquet.read_table(POINTS)
    markers = pts.filter(pc.equal(pts["valid"], True)).to_pylist()
    planes = json.loads(GEOMETRY_REFERENCE.read_text())["planes"]
    prior = json.loads(PRIOR.read_text())
    edge_pin = next(s for s in prior["sources"] if s["path"] == str(EDGE))
    assert prior["synapse_release"] == "v2" and sha(EDGE) == edge_pin["sha256"]
    dependencies = [pin(p) for p in (META, POINTS, PRIOR, GEOMETRY_REFERENCE, DN, Path(__file__))]
    outputs = defaultdict(list)
    local_inputs = defaultdict(list)
    with pa.memory_map(str(EDGE), "r") as source:
        reader = ipc.open_file(source)
        roots = pa.array(ROOTS)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            for direction, field, partner in (("out","pre","post"),("in","post","pre")):
                hits = batch.filter(pc.is_in(batch.column(field), value_set=roots))
                for edge in hits.select(["pre","post","count"]).to_pylist():
                    id_ = edge[field]
                    partner_rows = meta[edge[partner]]
                    if len(partner_rows) != 1:
                        continue
                    row = partner_rows[0]
                    if direction == "out" and row["region"] in ("central_brain","optic_lobe"):
                        outputs[id_].append(edge)
                    if direction == "in" and row["super_class"] == "sensory" and row["neuromere"] == "T1":
                        local_inputs[id_].append(dict(edge=edge, metadata=row))
    ports = []
    for side, root in zip(("left","right"), ROOTS, strict=True):
        row, = meta[root]
        assert (row["side"], row["cell_type"], row["super_class"], row["neuromere"], row["banc_888_id"]) == (
            side, "AN06B002", "ascending", "T1", root)
        swc = HERE / f"{root}_skeleton.swc"
        headers = swc.with_suffix(".headers")
        generation, = re.findall(r"^x-goog-generation: (\d+)$", headers.read_text(), re.M)
        md5, = re.findall(r"^x-goog-hash: md5=(\S+)$", headers.read_text(), re.M)
        assert base64.b64encode(hashlib.md5(swc.read_bytes()).digest()).decode() == md5
        marks = [[float(v)*s for v,s in zip(m["pt_position"], planes["voxel_size_nm"], strict=True)]
                 for m in markers if str(m["pt_root_id"]) == root]
        facts = dict(schema="BANC_V888_SWC_GEOMETRY_CHECK_V1", dataset="BANC", materialization=888,
                     root_id=root, coordinate_space="BANC_NATIVE", coordinate_units="nm",
                     swc_sha256=sha(swc), planes=planes, source_evidence=[*dependencies,pin(headers)],
                     version_identity_evidence=dict(root_888=root, verified=True, source_row=row))
        geom = validate(swc,facts)
        assert geom["is_single_tree"] and marks and geom["each_plane_intersected"]
        geom.update(gcs_generation=generation, md5=md5, continuity=path_to_brain(swc,geom,marks))
        path = HERE / f"{root}_GEOMETRY.json"
        write(path,geom)
        assert outputs[root] and any(x["metadata"]["side"] == side for x in local_inputs[root])
        ports.append(dict(root_id=root, side=side, direction="AN", source_reviewed=True,
                          structural_evidence_status="SOURCE_GEOMETRY_REVIEWED", source_cell_type="AN06B002",
                          evidence=[pin(swc),pin(path)]))
        print(root,geom["continuity"]["distance_to_front_seed_nm"],len(local_inputs[root]),len(outputs[root]))
    graph = HERE / "AN_GRAPH.json"
    write(graph,dict(roots=ROOTS, physical_contact_encoding="engineering, no available AN waveform labels",
                     source=edge_pin, inputs=local_inputs, brain_outputs=outputs))
    base = json.loads(DN.read_text())
    base.update(ports=[p for p in base["ports"] if p["direction"]=="DN"] + ports,
                source_files=[*base["source_files"],*dependencies,pin(graph)], AN_runtime="bilateral T1 tactile candidate")
    write(HERE/"TOUCH_BINDING.json",base)


if __name__ == "__main__":
    main()
