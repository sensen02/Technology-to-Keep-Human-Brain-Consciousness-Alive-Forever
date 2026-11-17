"""Measure actual swing/stance fore-aft excursion around saved reload events."""
import numpy as np

from probe import HERE, LEGS, W, read, stan

result = {}
source = read(HERE/"STEERING_AUDIT.json")
for mode, audit in source["audit"].items():
    run = HERE/f"FLAT_{mode.upper()}"
    assert stan.sha(run/"TRACE.npz") == read(run/"RESULT.json")["trace_files"]["TRACE.npz"]
    with np.load(run/"TRACE.npz") as trace:
        foot, loads, z = [trace[key] for key in ("foot","loads","padz")]
    rows = {}
    for index,leg in enumerate(LEGS):
        rows[leg] = []
        touchdown = None
        for event in audit["actual_world_reload_steps"][leg]:
            end=event["ms"]-9  # qualification uses 10 consecutive loaded samples
            low=max(0,end-4000)
            lifted=np.flatnonzero(z[low:end,index]>.015)
            if not len(lifted):
                continue
            last_raised=low+int(lifted[-1])
            start_support=np.flatnonzero((loads[low:last_raised,index]>=.10*W)&(z[low:last_raised,index]<=.004))
            if not len(start_support):
                continue
            start=low+int(start_support[-1])
            swing=foot[start:end+1,index,0]
            stance=foot[touchdown:start+1,index,0] if touchdown is not None and touchdown<start else None
            rows[leg].append(dict(qualified_reload_ms=event["ms"],last_loaded_before_last_raised_ms=start,
                                  reload_start_ms=end,swing_signed_local_x=float(swing[-1]-swing[0]),
                                  swing_range_local_x=float(np.ptp(swing)),
                                  preceding_stance_signed_local_x=None if stance is None else float(stance[-1]-stance[0])))
            touchdown=end
    result[mode]=rows
stan.put(HERE/"EXCURSION_AUDIT.json",dict(
    scope="saved physics positions only; qualified reload events, last loaded through last lifted to sustained reload",
    physics_steps=0,source_sha256=stan.sha(HERE/"STEERING_AUDIT.json"),arms=result))
