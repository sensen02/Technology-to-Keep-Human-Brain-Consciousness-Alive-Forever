"""Admit the bilateral T1 AN19B004 pair alongside archived lateral touch."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from touch_source import (META, POINTS, EDGE, PRIOR, GEOMETRY_REFERENCE, pin, write,
                          validate, path_to_brain, sha)

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "20261116T180500Z-touch"
TOUCH = ("720575941633589179", "720575941626534090")
LOAD = ("720575941685691375", "720575941565099623")
BINDING = HERE / "SEPARATED_BINDING.json"
GRAPH = HERE / "SEPARATED_GRAPH.json"


def bind():
    candidates = json.loads((HERE / "CANDIDATE_GRAPH.json").read_text())
    old = json.loads((OLD/"TOUCH_BINDING.json").read_text())
    touch = json.loads((OLD/"AN_GRAPH.json").read_text())
    prior = json.loads(PRIOR.read_text())
    edge_pin = next(row for row in prior["sources"] if row["path"]==str(EDGE))
    assert prior["synapse_release"]=="v2" and sha(EDGE)==edge_pin["sha256"]
    deps = [pin(path) for path in (META, POINTS, PRIOR, GEOMETRY_REFERENCE, OLD/"TOUCH_BINDING.json",
            OLD/"AN_GRAPH.json", HERE/"CANDIDATE_GRAPH.json", HERE/"candidates.py",Path(__file__))]
    outputs = {**touch["brain_outputs"], **{root:candidates["targets"][root] for root in LOAD}}
    pts = pq.read_table(POINTS).to_pylist()
    frames = json.loads(GEOMETRY_REFERENCE.read_text())["planes"]
    ports, evidence = [], {}
    for side, root in zip(("left","right"), LOAD, strict=True):
        meta = candidates["meta"][root]
        assert (meta["side"],meta["cell_type"],meta["super_class"],meta["neuromere"]) == (
            side,"AN19B004","ascending","T1")
        geometry = HERE / f"{root}_skeleton.swc"
        headers = geometry.with_suffix(".headers")
        generation, = re.findall(r"^x-goog-generation: (\d+)$",headers.read_text(),re.M)
        digest, = re.findall(r"^x-goog-hash: md5=(\S+)$",headers.read_text(),re.M)
        assert base64.b64encode(hashlib.md5(geometry.read_bytes()).digest()).decode()==digest
        marks = [[float(v)*s for v,s in zip(r["pt_position"],frames["voxel_size_nm"], strict=True)]
                 for r in pts if r["valid"] and str(r["pt_root_id"])==root]
        contract = dict(schema="BANC_V888_SWC_GEOMETRY_CHECK_V1",dataset="BANC",materialization=888,
                     root_id=root,coordinate_space="BANC_NATIVE",coordinate_units="nm",swc_sha256=sha(geometry),
                     planes=frames,source_evidence=[*deps,pin(headers)],
                     version_identity_evidence=dict(root_888=root,verified=True,source_row=meta))
        actual = validate(geometry,contract)
        assert marks and actual["is_single_tree"] and actual["each_plane_intersected"]
        continuity = path_to_brain(geometry,actual,marks)
        actual.update(continuity=continuity,gcs_generation=generation,md5=digest)
        filename = HERE / f"{root}_GEOMETRY.json"
        write(filename,actual)
        local = candidates["inputs"][root]
        input_counts = Counter()
        for r in local:
            if r["metadata"]["side"] in (side,"bilateral"):
                input_counts[r["metadata"]["cell_sub_class"]] += r["edge"]["count"]
        load = sum(value for key,value in input_counts.items() if "campaniform" in key)
        assert load >= 10 and len(outputs[root])>0
        evidence[root] = dict(name="AN19B004",side=side,counts=dict(input_counts),
                         same_side_campaniform_synapses=load,brain_target_edges=len(outputs[root]),
                         anterior_distance_nm=continuity["distance_to_front_seed_nm"],afferent_edges=local)
        ports.append(dict(root_id=root,side=side,direction="AN",source_reviewed=True,
                          structural_evidence_status="SOURCE_GEOMETRY_REVIEWED",source_cell_type="AN19B004",
                          evidence=[pin(geometry),pin(filename)]))
        deps.extend((pin(geometry),pin(headers),pin(filename)))
    assert old["ports"][-2]["root_id"]==TOUCH[0] and old["ports"][-1]["root_id"]==TOUCH[1]
    write(GRAPH,dict(root_order=TOUCH+LOAD,source=edge_pin,load_source=evidence,brain_outputs=outputs,
                  channels=["left_lateral","right_lateral","left_load","right_load"],
                  dynamics="none measured; distinct morphology with a deliberately isolated engineering sensor coupling"))
    known = {row["path"]:row for row in [*old["source_files"],*deps,edge_pin,pin(GRAPH)]}
    old["ports"] += ports
    old["source_files"] = list(known.values())
    old["AN_runtime"] = "separated engineering components; AN06B002 lateral, AN19B004 forefoot normal load"
    write(BINDING,old)
    print({root:{k:v for k,v in row.items() if k!="afferent_edges"} for root,row in evidence.items()},flush=True)


if __name__=="__main__":
    bind()
