"""Independently replay delivery and compare the intervention prefixes."""
import json
from pathlib import Path

import numpy as np
import torch

from unified import BINDING, CHECKPOINT, HERE, PARENT, Neck, sha

RUNS=HERE/"RUNS"
FIRST=HERE.parent/"20261117T005200Z-unified-neck/RUNS"
SCENES=json.loads((HERE/"SCENES.json").read_text())


def load(root,case):
    path=root/case
    assert (root/f"{case}.exit").read_text().strip()=="0"
    result=json.loads((path/"RESULT.json").read_text())
    assert all(sha(path/name)==digest for name,digest in result["hashes"].items())
    assert result["scene"]==SCENES[case]
    with np.load(path/"TRACE.npz",allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    times=arrays["time"]
    native=np.full(result["native_steps"]+1,12.5e-6)
    native[0]=times[0]
    np.cumsum(native,out=native)
    np.testing.assert_array_equal(times,native[::80])
    assert result["source"]["actual_steps"]==SCENES[case]["end"]
    assert result["source"]["source_files_unchanged"] and not result["source"]["physiology_verified"]
    assert result["source"]["binding_sha256"]==sha(root.parent/"UNIFIED_BINDING.json")
    assert np.isfinite(arrays["forecasts"]).all() and arrays["forecasts"].shape[1]==43
    return result,arrays


def audit_events(case,arr):
    events=json.loads((RUNS/case/"TRANSPORT.json").read_text())
    neck=Neck()
    end=SCENES[case]["end"]
    delivery={}
    counts={}
    for direction, ids in (("DN",neck.ids),("AN",neck.an_roots)):
        group=[r for r in events if r["direction"]==direction]
        sent=[r for r in group if r["event"]=="enqueue"]
        got=[r for r in group if r["event"]=="release"]
        case_kind=SCENES[case]
        if direction=="DN":
            begin,interval=(end if case_kind.get("dn_cut") else 0),5
        else:
            begin=(case_kind.get("an_restore",end) if case_kind.get("an_cut") else 0)
            interval=1
        assert [r["capture_step"] for r in sent]==list(range(begin,end,interval))
        expected=[r for r in sent if r["capture_step"]+2<=end]
        for p,q in zip(expected,got,strict=True):
            assert p["root_ids"]==q["root_ids"]==ids and q["source_part"]=="neck"
            assert q["target_part"]==("brain" if direction=="AN" else "other")
            assert p["values"]==q["values"]
            assert q["capture_step"]==p["capture_step"]
            assert p["release_step"]==q["release_step"]==q["consume_step"]==p["capture_step"]+2
        if got:
            digest=sha(BINDING)
            assert all(q["source_bindings_sha256"]==digest for q in got)
        delivery[direction]=got
        counts[direction]=len(got)
    d=arr["decisions"]
    times=d[:,0].astype(int)
    for direction in ("DN","AN"):
        got=delivery[direction]
        released=np.array([r["release_step"] for r in got],dtype=int)
        where=np.searchsorted(released,times,side="right")-1
        if len(got):
            steps=np.where(where<0,-1,released[where])
        else:
            steps=np.full(len(where),-1)
        if direction=="AN":
            np.testing.assert_array_equal(steps,d[:,16])
            np.testing.assert_array_equal(steps>=0,d[:,12].astype(bool))
            reconstructed=np.zeros((len(d),4))
            if len(got):
                act=np.array([r["values"] for r in got])[where[steps>=0]]
                projected=act@neck.matrix.T
                reconstructed[steps>=0]=np.column_stack([projected[:,p].mean(1) for p in neck.pool])
            np.testing.assert_allclose(reconstructed,d[:,7:11],rtol=0.,atol=1e-14)
        else:
            np.testing.assert_array_equal(steps,d[:,17])
            v=np.zeros((len(d),2))
            if len(got):
                values=np.asarray([r["values"] for r in got])
                yaw={tuple(row):neck.codec.read(list(row))[0] for row in np.unique(values[:,:2],axis=0)}
                for i in np.flatnonzero(steps>=0):
                    payload=values[where[i]]
                    v[i]=.1*payload[2],yaw[tuple(payload[:2])]
            np.testing.assert_allclose(v,d[:,3:5],rtol=0.,atol=1e-12)
    return counts


def common_physics(a,b,until):
    for key in ("time","com","head","yaw","loads","nonfoot","upright"):
        np.testing.assert_array_equal(a[key][:until+1],b[key][:until+1])
    frames=a["eye_stamps"]+20<=until
    np.testing.assert_array_equal(a["eye_stamps"][frames],b["eye_stamps"][:frames.sum()])
    np.testing.assert_array_equal(a["eyes"][frames],b["eyes"][:frames.sum()])


def divergence(a,b):
    ar,br=a["decisions"],b["decisions"]
    length=min(len(ar),len(br))
    moved=np.flatnonzero(np.any(ar[:length,3:5]!=br[:length,3:5],axis=1))
    return int(ar[moved[0],0]) if len(moved) else None


def main():
    torch.set_num_threads(1)
    old=torch.load(PARENT,weights_only=True,map_location="cpu")["brain"]
    new=torch.load(CHECKPOINT,weights_only=True,map_location="cpu")["brain"]
    assert all(torch.equal(t,new[k]) for k,t in old.items())
    rows={}
    arrays={}
    first={}
    for case in SCENES:
        rows[case],arrays[case]=load(RUNS,case)
        rows[case]["independent_releases"]=audit_events(case,arrays[case])
        first[case],old_trace=load(FIRST,case)
        diff=divergence(old_trace,arrays[case])
        rows[case]["first_delivered_difference_from_original_ms"]=diff
        upper=min(len(old_trace["com"]),len(arrays[case]["com"]))-1
        common_physics(old_trace,arrays[case],diff if diff is not None else upper)
        print(case,rows[case]["passed"],rows[case]["independent_releases"],"first difference",diff,flush=True)
    paired={}
    for case in ("an_cut","an_restore"):
        a,b=arrays["an_intact"],arrays[case]
        first_change=divergence(a,b)
        assert first_change is not None
        common_physics(a,b,first_change)
        np.testing.assert_array_equal(a["touch_max"][:first_change],b["touch_max"][:first_change])
        assert first_change>=rows["an_intact"]["contact_ms"][0]
        paired[case]=first_change
    d=divergence(arrays["left"],arrays["dn_cut"])
    assert d is not None
    common_physics(arrays["left"],arrays["dn_cut"],d)
    result=dict(passed=all(r["passed"] for r in rows.values()),total_native_steps=sum(r["native_steps"] for r in rows.values()),
                checkpoint_sha256=sha(CHECKPOINT),all_original_learned_weights_preserved=True,
                cases=rows,first_revision_pass={k:v["passed"] for k,v in first.items()},
                same_physics_through_first_an_difference_ms=paired,same_physics_through_dn_cut_difference_ms=d)
    (HERE/"ACCEPTANCE.json").write_text(json.dumps(result,indent=2)+"\n")
    print("total",result["total_native_steps"],"combined",result["passed"],"paired",paired,flush=True)


if __name__=="__main__":
    main()
