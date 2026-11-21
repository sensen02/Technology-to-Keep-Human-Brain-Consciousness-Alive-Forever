"""Reconstruct each live arrival and compare frozen-control behavior."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ARCHIVE=BASE/"20261118T031132Z-flat-neck-assay"
sys.path.insert(0,str(BASE/"20261118T002600Z-ascending-behavior-comparison"))
from neural_runtime import DT, FIXED, FIT, read, verify, write, sha
from implant_timing import CanonicalEvents
from profiles import MODES, SEEDS

SAMPLE=17999
PRE_INSERT=4500
RELEASE=10500


def arrays(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def trial(folder):
    result=read(folder/"RESULT.json")
    verify(result["source_sha256"])
    path=folder/"NEURAL.npz"
    assert sha(path)==result["neural_sha256"]
    return result,arrays(path)


def body(folder,mode):
    path=folder/mode/"TRACE.npz"
    result=read(path.parent/"RESULT.json")
    assert sha(path)==result["trace_sha256"]
    return result,arrays(path)


def equal(a,b):
    assert a.keys()==b.keys()
    return {k:bool(np.array_equal(a[k],b[k])) for k in a}


def first(a,b):
    idcs=np.flatnonzero((a!=b).reshape(len(a),-1).any(1))
    return float(idcs[0]*DT*1000) if len(idcs) else None


def verify_transport(meta, data, sections):
    baseline=read(FIXED/"PARAMETERS.json")["ports"]
    an_roots=[x["banc_root"] for x in read(FIT/"BINDING.json")["records"]]
    changed = {mode:[0]*len(sections) for mode in MODES}
    for group, roots in (("an",an_roots),("dn",meta["dn_roots"])):
        for col,root in enumerate(roots):
            default=baseline[root]["delay_ticks"]
            selected=sections.index(root) if root in sections else None
            for lane,mode in enumerate(MODES):
                emissions=np.flatnonzero(data[f"{group}_emitted"][:,lane,col])
                queue_ticks=emissions+int(group=="dn")
                inside=queue_ticks < len(data["dn_emitted"])
                queue_ticks=queue_ticks[inside]
                delays=np.full(len(queue_ticks),default)
                if selected is not None:
                    delays=data["departure_delay_ticks"][queue_ticks,lane,selected]
                    changed[mode][selected]=int(np.count_nonzero(delays!=default))
                due = queue_ticks+delays
                assert len(due)==len(np.unique(due))
                target=np.zeros(len(data["dn_emitted"]),dtype=bool)
                target[due[due<len(target)]]=True
                assert np.array_equal(target,data[f"{group}_arrived"][:,lane,col])
    encoder=CanonicalEvents((4,2))
    for tick,row in enumerate(data["an_arrived"]):
        assert np.array_equal(encoder.step(row),data["brain_an_events"][tick])
        assert np.array_equal(encoder.rate,data["rates"][tick])
    return changed


def main():
    physical=read(HERE/"MECHANICS.json")
    assert physical["output_sha"]==sha(HERE/"MECHANICS.npz")
    verify(physical["source_sha256"])
    cable=read(HERE/"CABLE.json")
    verify(cable["source_sha256"])
    sens=read(HERE/"SENSITIVITY.json")
    verify(sens["input_sha256"])
    expected=arrays(HERE/"TRANSPORT_PROFILES.npz")
    assert sha(HERE/"TRANSPORT_PROFILES.npz")==cable["data_sha256"]
    selected=expected["root_ids"].tolist()
    allrows=[]
    steps=0
    for seed in SEEDS:
        run=HERE/"FORMAL"/f"seed{seed}"
        old=ARCHIVE/"FORMAL"/f"seed{seed}"
        inputs=read(run/"INPUTS.json")
        meta, data=trial(run)
        oldmeta, old_data=trial(old)
        assert inputs["selected_roots"]==selected
        assert meta["dn_roots"]==oldmeta["dn_roots"]
        assert np.array_equal(data["departure_delay_ticks"],expected["delay_ticks"])
        changed = verify_transport(meta,data,selected)
        assert np.array_equal(data["central_hits"],old_data["central_hits"])
        for key in data:
            if key not in ("central_hits","departure_delay_ticks"):
                assert np.array_equal(data[key][:,0],old_data[key][:,0])
        control_status, control=body(run,MODES[0])
        _, backup=body(old,"intact")
        assert all(equal(control,backup).values())
        assert sha(run/"unloaded_control.mjb")==sha(old/"intact.mjb")
        assert all(control_status[key] for key in ("safety_pass","advance_pass","final_stand_pass"))
        yaw=np.rad2deg(np.unwrap(control["yaw"])-control["yaw"][0])
        metrics={}
        for lane,mode in enumerate(MODES):
            status,trace=body(run,mode)
            for key in trace:
                if key!="initial_position":
                    assert np.array_equal(trace[key][:PRE_INSERT],control[key][:PRE_INSERT])
            if lane==2:
                _, retained=body(run,MODES[1])
                for key in trace:
                    assert np.array_equal(trace[key][:RELEASE],retained[key][:RELEASE])
            angle=np.rad2deg(np.unwrap(trace["yaw"])-trace["yaw"][0])
            delta=float(angle[SAMPLE]-yaw[SAMPLE])
            xy=float(np.linalg.norm(trace["position"][SAMPLE,:2]-control["position"][SAMPLE,:2]))
            tags=dict(
                AN_arrival=first(data["an_arrived"][:,0],data["an_arrived"][:,lane]),
                AN_brain=first(data["brain_an_events"][:,0],data["brain_an_events"][:,lane]),
                DN_emitted=first(data["dn_emitted"][:,0],data["dn_emitted"][:,lane]),
                command=first(control["command"],trace["command"]),
                qpos=first(control["qpos"],trace["qpos"]))
            row=dict(yaw_1p8_deg=float(angle[SAMPLE]),yaw_difference_deg=delta,
                xy_difference_1p8_mm=xy,peak_abs_yaw_difference_deg=float(abs(angle-yaw).max()),
                arrival_count_by_modeled_port=changed[mode],
                first_differences_ms=tags, end_yaw_deg=status["yaw_degrees"],
                advance_pass=status["advance_pass"],final_stand_pass=status["final_stand_pass"],
                safety_pass=status["safety_pass"], stand_rms_mm_s=status["stand_rms_mm_s"],
                behavior_exact_to_unloaded=all(equal(control,trace).values()))
            row["engineering_pass"]=abs(delta)<=5. and xy<=.5 and all(
                status[k] for k in ("advance_pass","final_stand_pass","safety_pass"))
            metrics[mode]=row
        allrows.append(dict(seed=seed,modes=metrics,baseline_archive_exact=True,transport_verified=True))
        steps+=meta["native_body_steps"]
        print(seed,[(mode,round(r["yaw_difference_deg"],4),round(r["xy_difference_1p8_mm"],5),
                     r["engineering_pass"],r["arrival_count_by_modeled_port"]) for mode,r in metrics.items()],flush=True)
    formal=HERE/"FORMAL/seed11701"
    replay=HERE/"REPLAY/seed11701"
    _,f=trial(formal)
    m,r=trial(replay)
    steps+=m["native_body_steps"]
    parity={"neural":equal(f,r)}
    for mode in MODES:
        parity[mode]=equal(body(formal,mode)[1],body(replay,mode)[1])
    assert all(x for row in parity.values() for x in row.values())
    write(HERE/"FINAL_ASSAY.json",dict(project="P1_open_source_brain",rows=allrows,
        selected_roots=selected,phase_main_gate_1p8=1.8,
        physical_cases=physical["summary"], cable_sensitivity=sens["alternatives"],
        repeat_exact=parity, body_episodes=16,body_steps=steps,brain_batch_steps=80000,
        neck_batch_steps=80000,optimizer_updates=0, calibrated_biological_injury=False))
    print("all pulse deliveries audited; independent arrays",sum(map(len,parity.values())),flush=True)


if __name__=="__main__":
    main()
