"""Audit the repaired live interface against its baseline and old failing control."""
from pathlib import Path
import json
import os
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"20261118T042507Z-observer-closed-loop"
ARCH=Path(os.environ["P1_ARCHIVE_ROOT"])/"experiments/open-brain"
FIXED=ARCH/"20261117T164207Z-fixed-neck/PARAMETERS.json"
BIND=ARCH/"20261117T235930Z-ascending-reference-fit/BINDING.json"
sys.path.insert(0,str(PRIOR))
from shadow_stress import file_sha, load, phase_encode, read, spike_arrivals_to_rate
from treatment import ARMS, DT, SEEDS, prescribed_drift

CORE=("dn_emitted","dn_arrived","an_emitted","an_arrived","brain_an_events",
      "rates","population","left_used_hz")
WINDOW=slice(10500,14000)


def batch(folder):
    r=read(folder/"RESULT.json")
    for path,hashval in r["source_sha256"].items():
        assert file_sha(Path(path))==hashval,path
    return r,load(folder/"NEURAL.npz",r["neural_sha256"])


def bodies(folder):
    result=[]
    for arm in ARMS:
        p=folder/arm
        m=read(p/"RESULT.json")
        data=load(p/"TRACE.npz",m["trace_sha256"])
        result.append((m,data))
    return result


def same(a,b):
    assert set(a)==set(b)
    return {key:bool(np.array_equal(a[key],b[key])) for key in a}


def rmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))


def audit_transport(n,r):
    params=read(FIXED)["ports"]
    an=[x["banc_root"] for x in read(BIND)["records"]]
    for key,roots in (("dn",r["dn_roots"]),("an",an)):
        for i,root in enumerate(roots):
            delay=params[root]["delay_ticks"]+int(key=="dn")
            a,b=n[key+"_emitted"][:,:,i],n[key+"_arrived"][:,:,i]
            assert not b[:delay].any() and np.array_equal(a[:-delay],b[delay:])
    for lane in range(4):
        for side in range(2):
            rate=spike_arrivals_to_rate(n["an_arrived"][:,lane,side])
            assert np.array_equal(rate,n["rates"][:,lane,side])
            assert np.array_equal(phase_encode(rate),n["physical_encoded_events"][:,lane,side])
        assert np.array_equal(phase_encode(n["left_used_hz"][:,lane]),n["brain_an_events"][:,lane,0])
    assert np.array_equal(n["brain_an_events"][:,[0,1,3]],n["physical_encoded_events"][:,[0,1,3]])
    assert np.array_equal(n["brain_an_events"][:,:,1],n["physical_encoded_events"][:,:,1])
    assert np.array_equal(n["brain_an_events"][:,2,0],n["legacy_candidate_events"][:,2])


def formal(seed):
    path=HERE/"FORMAL"/f"seed{seed}"
    previous=PRIOR/"FORMAL"/f"seed{seed}"
    meta,n=batch(path)
    old_meta,old=batch(previous)
    assert meta["dn_roots"]==old_meta["dn_roots"]
    assert np.array_equal(n["central_hits"],old["central_hits"])
    audit_transport(n,meta)
    now=bodies(path)
    then=bodies(previous)
    for lane in (0,2):
        assert all(np.array_equal(n[key][:,lane],old[key][:,lane]) for key in CORE)
        assert all(same(now[lane][1],then[lane][1]).values())
    core={}
    for lane in (1,3):
        parity={key:bool(np.array_equal(n[key][:,0],n[key][:,lane])) for key in CORE}
        physical=same(now[0][1],now[lane][1])
        assert all(parity.values()) and all(physical.values())
        core[ARMS[lane]]=dict(neural=parity,physical=physical)
    g=prescribed_drift(len(n["rates"]))
    ref=n["rates"][:,:,0]
    assert np.array_equal(ref*np.c_[np.ones((len(ref),2)),g,g],n["left_raw_hz"])
    marked=n["confirmed_readout_drift"]
    assert np.array_equal(n["advisory_left_hz"],np.where(marked,ref,n["left_raw_hz"]))
    assert not marked[:,0:2].any()
    assert not marked[16200:].any()
    expected=(ref[:,3]>=15.)
    assert marked[WINDOW,3][expected[WINDOW]].all()
    assert np.array_equal(n["advisory_left_hz"][WINDOW,3][expected[WINDOW]],ref[WINDOW,3][expected[WINDOW]])
    out=dict(seed=seed,exact_conditional_repair=core,legacy_archives_identical=True,
       first_drift_confirmation_ms=float(np.flatnonzero(marked[:,3])[0]*DT*1000),
       real_applied_readout_plateau_error_hz=rmse(n["left_used_hz"][WINDOW,3],ref[WINDOW,3]),
       raw_readout_plateau_error_hz=rmse(n["left_raw_hz"][WINDOW,3],ref[WINDOW,3]),
       advisory_plateau_error_hz=rmse(n["advisory_left_hz"][WINDOW,3],ref[WINDOW,3]),
       old_adapted_candidate_different_event_bits=int(np.count_nonzero(
           n["legacy_candidate_events"][:,1]!=n["physical_encoded_events"][:,1,0])),
       metrics={arm:{k:m[k] for k in ("safety_pass","advance_pass","final_stand_pass",
                                     "stand_rms_mm_s","yaw_degrees","forward_dx_mm")}
                      for arm,(m,_) in zip(ARMS,now,strict=True)},
       original_legacy_outcome_facts={
            arm:{k:m[k] for k in ("yaw_degrees","forward_dx_mm","final_stand_pass","stand_rms_mm_s")}
            for arm,(m,_) in zip(ARMS,then,strict=True)})
    print(seed,"exact body and neural, raw/advisory",out["raw_readout_plateau_error_hz"],
          out["advisory_plateau_error_hz"],out["metrics"]["drift_on"],flush=True)
    return out,meta


def main():
    rows=[formal(s) for s in SEEDS]
    folder=HERE/"FORMAL/seed11701"
    replay=HERE/"REPLAY/seed11701"
    _,a=batch(folder)
    repeat,b=batch(replay)
    parity={"neural":same(a,b)}
    for arm,x,y in zip(ARMS,bodies(folder),bodies(replay),strict=True):
        parity[arm]=same(x[1],y[1])
    assert all(all(x.values()) for x in parity.values())
    metas=[m for _,m in rows]+[repeat]
    value=dict(project="P1_open_source_brain",physical_event_route_repaired=True,
               analog_amplitude_repair_requires_independent_event_stream=True,
               rows=[row for row,_ in rows],anchor_fixture=read(HERE/"ANCHOR_CHECK.json"),
               independent_parity=parity,independent_exact_arrays=sum(map(len,parity.values())),
               brain_batch_ticks=sum(m["brain_batch_ticks"] for m in metas),
               body_steps=sum(m["native_body_steps"] for m in metas),optimizer_updates=0)
    with (HERE/"FINAL_EVENT_REPAIR.json").open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    print("replay arrays",value["independent_exact_arrays"],flush=True)


if __name__=="__main__":
    main()
