"""Independently check saved navigation geometry, contact and run accounting."""
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import ApproachLink

HERE = Path(__file__).resolve().parent
RUNS = HERE/"RUNS"
scenes = json.loads((HERE/"SCENES.json").read_text())["scenes"]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()



def audit_transport(out, policy, result, decisions):
    steps = result["actual_ms"]
    journal = json.loads((out/"TRANSPORT.json").read_text())
    groups = {}
    stages = (("DN",5,2),("AN",1,4)) if policy=="joint" else (("DN",20,4),)
    for direction, interval, count in stages:
        sent = [e for e in journal if e["direction"]==direction and e["event"]=="enqueue"]
        released = [e for e in journal if e["direction"]==direction and e["event"]=="release"]
        assert [r["capture_step"] for r in sent]==list(range(0,steps,interval))
        expected = [r for r in sent if r["capture_step"]+2<=steps]
        assert len(released)==len(expected) and len(sent[0]["root_ids"])==count
        for a,b in zip(expected,released,strict=True):
            assert all(a[k]==b[k] for k in ("root_ids","values","capture_step","release_step"))
            assert b["consume_step"]==b["capture_step"]+2==b["release_step"]
            assert b["target_part"]==("brain" if direction=="AN" else "other")
            assert b["source_bindings_sha256"]==result["source"]["binding_sha256"]
        groups[direction]=released
    records = groups["DN"]
    when = np.asarray([r["release_step"] for r in records])
    values = np.asarray([r["values"] for r in records])
    delivered = np.searchsorted(when,decisions[:,0],side="right")-1
    pad = delivered>=0
    if policy=="joint":
        np.testing.assert_array_equal(values[:,0],values[:,1])
        prediction = np.where(pad,values[delivered,0]>.5,0.)
        np.testing.assert_array_equal(prediction,decisions[:,3])
    else:
        codec = ApproachLink().codec
        decoded = {tuple(row):codec.read(list(row))[0] for row in np.unique(values[:,:2],axis=0)}
        expected_yaw = [decoded[tuple(values[n,:2])] if n>=0 else 0. for n in delivered]
        np.testing.assert_array_equal(expected_yaw,decisions[:,4])
        expected_speed = np.where(pad,values[delivered,2]*.1,0.)
        np.testing.assert_allclose(expected_speed,decisions[:,3],rtol=0.,atol=1e-12)
    return {direction:len(events) for direction,events in groups.items()}

def load_case(scene, policy):
    name = f"{scene}__{policy}"
    out = RUNS/name
    assert (RUNS/f"{name}.exit").read_text().strip()=="0", name
    r = json.loads((out/"RESULT.json").read_text())
    identity = json.loads((out/"RUN_IDENTITY.json").read_text())
    assert r["scene"]==scene and r["policy"]==policy and identity["parameters"]==scenes[scene]
    assert all(digest(out/n)==v for n,v in r["artifacts"].items())
    assert all(digest(Path(p))==v for p,v in identity["inputs"].items())
    assert r["actual_ms"]*80==r["native_steps"] and r["actual_ms"]<=r["max_duration_ms"]
    assert r["source"]["actual_steps"]==r["actual_ms"] and r["source"]["source_files_unchanged"]
    assert not r["source"]["full_neck_admitted"] and not r["source"]["physiology_verified"]
    with np.load(out/"TRACE.npz",allow_pickle=False) as s:
        r["independent_release_counts"] = audit_transport(out,policy,r,s["decisions"])
        assert len(s["com"])==r["actual_ms"]+1
        assert s["stamps"][-1]==r["actual_ms"] and len(s["qpos"])==len(s["feet"])==len(s["stamps"])
        native_clock = np.full(r["native_steps"]+1,12.5e-6)
        native_clock[0] = s["time"][0]
        np.cumsum(native_clock,out=native_clock)
        np.testing.assert_array_equal(s["time"],native_clock[::80])
        assert abs(np.linalg.norm(s["head"][-1,:2]-s["goal"][:2])-r["final_range"])<1e-12
        raw = s["native_contact_max_by_ms"]
        assert raw.shape[0]==r["actual_ms"]
        for i, count in enumerate(r["contact_native_counts"]):
            hits = np.flatnonzero(raw[:,i]>0.)
            assert len(hits)<=count<=len(hits)*80
            assert (r["first_contact_step"][i]//80==hits[0]) if len(hits) else (r["first_contact_step"][i]==-1)
        bounds = scenes[scene]["target_range"]
        ranges = np.linalg.norm(s["head"][:,:2]-s["goal"][:2],axis=1)
        inside = np.flatnonzero((ranges>bounds[0]) & (ranges<bounds[1]))
        r["first_in_goal_zone_ms"] = int(inside[0]) if len(inside) else None
        r["new_evidence"] = {}
        boxes = r["geometry"]["boxes"]
        if boxes:
            far = max(x+hx for x,_,_,hx,_,_ in boxes)
            clear = np.flatnonzero(s["feet"][:,0]>=far+.03)
            if len(clear):
                at = int(s["stamps"][clear[0]])
                r["new_evidence"]["first_all_geometry_cleared_ms"] = at
                r["new_evidence"]["closing_after_clear"] = float(ranges[at]-ranges[-1])
        if policy=="joint":
            r["new_evidence"]["visual_move_score_max_after_target"] = float(
                s["decisions"][s["decisions"][:,0]>=520.,7].max())
        if scene=="flat_paired" and r["actual_ms"]>=30_000:
            m = (s["decisions"][:,0]>=20_000) & (s["decisions"][:,0]<30_000)
            r["new_evidence"]["20_to_30s_forward_x"] = float(s["com"][30_000,0]-s["com"][20_000,0])
            r["new_evidence"]["20_to_30s_delivered_propulsion_fraction"] = float(np.mean(s["decisions"][m,3]>0.))
        if r["pit_stats"]:
            edge = r["geometry"]["pit"]["far"]
            np.testing.assert_allclose(r["pit_stats"]["final_full_body_margin"],s["feet"][-1,0]-edge)
        for (x,_,_,hx,_,_), (_, gap) in zip(r["geometry"]["boxes"],r["box_margins"],strict=True):
            np.testing.assert_allclose(gap,s["feet"][-1,0]-x-hx)
        if policy=="joint":
            assert not s["decisions"][:,[2,4]].any()
        if policy=="neck_probe":
            assert len(s["eyes"])==0 and not s["decisions"][:,[2,4]].any()
        return r, {k:s[k] for k in ("qpos","stamps","com","decisions","head","goal","feet")}


def common_prefix(a, b, n):
    n=min(n,len(a),len(b))
    rows=np.flatnonzero(np.max(abs(a[:n]-b[:n]),axis=1)>1e-10)
    return int(rows[0]) if len(rows) else n


def main():
    all_cases={}
    stored={}
    for scene,definition in scenes.items():
        for policy in definition["policies"]:
            tag=f"{scene}__{policy}"
            all_cases[tag],stored[tag]=load_case(scene,policy)
    assert all(np.array_equal(row["qpos"][0],stored["near_left__joint"]["qpos"][0]) for row in stored.values())
    paired={}
    for name_a,name_b in (("flat_paired__joint","narrow_trench__joint"),
                          ("flat_paired__joint","wide_trench__joint"),
                          ("flat_paired__neck_probe","narrow_trench__neck_probe"),
                          ("flat_paired__neck_probe","wide_trench__neck_probe"),
                          ("one_offset_box__detour","staggered_boxes__detour")):
        a,b=(stored[k] for k in (name_a,name_b))
        paired[f"{name_a} / {name_b}"] = {
            "com_prefix_ms_at_1e-10":common_prefix(a["com"],b["com"],min(len(a["com"]),len(b["com"]))),
            "same_initial_qpos":True}
    legacy_path=HERE.parents[1]/"body-motor-learning/20261116T153927Z/gap/HYBRID_TRANSIT_ACCEPTANCE.json"
    legacy=json.loads(legacy_path.read_text())
    assert legacy["passed"] and legacy["independent_exact_bytes"]
    assert not legacy["transverse_gap_under_whole_footprint"]
    result=dict(runs=len(all_cases),scenes=len(scenes),native_steps=sum(r["native_steps"] for r in all_cases.values()),
                frozen_policies=True,matched_nominal_qpos=True,paired=paired,cases=all_cases,
                legacy_local_slot=dict(path=str(legacy_path),sha256=digest(legacy_path),points=legacy["gap_opening_xyz"],
                                       transverse=False,has_foothold_teacher=True))
    (HERE/"BENCHMARK.json").write_text(json.dumps(result,indent=2)+"\n")
    for tag,r in all_cases.items():
        failed=[key for key,val in r["checks"].items() if not val]
        print(f"{tag:<31} {r['actual_ms']/1000:5.2f}s  goal {r['initial_range']:.3f}->{r['final_range']:.3f} "
              f"heading {r['final_bearing_deg']:+.1f}  contact {r['contact_peak_W']}  {failed}")
    print("native steps",result["native_steps"],"cases",result["runs"])


if __name__=="__main__":
    main()
