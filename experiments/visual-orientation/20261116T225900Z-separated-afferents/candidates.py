"""Find bilateral front-leg campaniform recipients with a documented neck marker."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

from touch_source import META, POINTS, EDGE

columns = ["root_888","banc_888_id","cell_type","region","side","super_class","tract",
           "neuromere","cell_class","cell_sub_class","body_part_sensory"]
rows = feather.read_table(META, columns=columns).to_pylist()
by_id = {r["root_888"]:r for r in rows if r["root_888"]}
marks = {str(r["pt_root_id"]) for r in parquet.read_table(POINTS).to_pylist() if r["valid"]}
ascending = {i for i,r in by_id.items() if r["super_class"]=="ascending" and i in marks and r["side"] in ("left","right")}
sources = {i:r for i,r in by_id.items() if r["super_class"]=="sensory" and
           (r["cell_sub_class"] or "").startswith("front_leg_")}
source_records = defaultdict(list)
brain_targets = defaultdict(list)
with pa.memory_map(str(EDGE), "r") as source:
    batches = ipc.open_file(source)
    input_ids, possible = pa.array(list(sources)), pa.array(list(ascending))
    for index in range(batches.num_record_batches):
        batch = batches.get_batch(index)
        incoming = batch.filter(pc.and_(pc.is_in(batch.column("pre"),value_set=input_ids),
                                       pc.is_in(batch.column("post"),value_set=possible)))
        for r in incoming.select(["pre","post","count"]).to_pylist():
            source_records[r["post"]].append(dict(edge=r,metadata=sources[r["pre"]]))
    load_roots = {i for i, edges in source_records.items() if any(
        "campaniform" in x["metadata"]["cell_sub_class"] for x in edges)}
    chosen = pa.array(list(load_roots))
    for index in range(batches.num_record_batches):
        batch = batches.get_batch(index)
        outgoing = batch.filter(pc.is_in(batch.column("pre"),value_set=chosen))
        for r in outgoing.select(["pre","post","count"]).to_pylist():
            if by_id.get(r["post"],{}).get("region")=="central_brain":
                brain_targets[r["pre"]].append(r)
for side in ("left","right"):
    print(side)
    summary=[]
    for i in load_roots:
        if by_id[i]["side"] != side or not brain_targets[i]:
            continue
        counts=Counter()
        for entry in source_records[i]:
            r=entry["metadata"]
            if r["side"] not in (side,"bilateral"):
                continue
            counts[r["cell_sub_class"]] += entry["edge"]["count"]
        load=sum(n for k,n in counts.items() if "campaniform" in k)
        if load:
            summary.append((load,load/sum(counts.values()),by_id[i]["cell_type"],i,
                            by_id[i]["neuromere"],len(brain_targets[i]),dict(counts)))
    for row in sorted(summary,reverse=True)[:22]:
        print(row)
with Path("CANDIDATE_GRAPH.json").open("x") as fh:
    json.dump(dict(inputs={i:source_records[i] for i in load_roots},targets=brain_targets,
                   meta={i:by_id[i] for i in load_roots}),fh,indent=2)
