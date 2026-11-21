"""Verify fresh replays and bind the next flight phase to its archive."""
from pathlib import Path

import numpy as np

from archive_io import BIDIR, FIXED, HERE, digest, read, write

FLIGHT_GROUPS=("takeoff","wing_amplitude","flight_steer","landing")


def check_pins(pins):
    for path, expected in pins.items():
        assert digest(Path(path))==expected,path


def compare(kind, pin_field):
    a=HERE/kind
    b=HERE/(kind+"_REPLAY")
    result=[read(p/"RESULT.json") for p in (a,b)]
    for p, row in zip((a,b), result, strict=True):
        assert digest(p/"TRACE.npz")==row["trace_sha256"]
        check_pins(row[pin_field])
    with np.load(a/"TRACE.npz") as x,np.load(b/"TRACE.npz") as y:
        assert set(x.files)==set(y.files)
        for key in x.files:
            assert np.array_equal(x[key],y[key],equal_nan=True),key
        count=len(x.files)
    return result[0],count


def flight_audit():
    catalog=read(BIDIR/"CATALOG.json")["ports"]
    old_eon=read(BIDIR/"EON/RESULT.json")["admitted"]
    parameters=read(FIXED/"PARAMETERS.json")["ports"]
    ports=[]
    pins={str(p):digest(p) for p in (BIDIR/"CATALOG.json",BIDIR/"EON/RESULT.json",
                                     FIXED/"PARAMETERS.json",BIDIR/"free_flight.py",
                                     BIDIR/"FLIGHT_ENV_LOCK.txt")}
    for root, p in catalog.items():
        if p["group"] not in FLIGHT_GROUPS:
            continue
        source=BIDIR/"geometry"/p["object"]["filename"]
        assert digest(source)==p["object"]["sha256"]
        pins[str(source)]=p["object"]["sha256"]
        ports.append(dict(root=root,flywire_root=old_eon.get(root),group=p["group"],
                          name=p["cell_type"],direction=p["direction"],side=p["side"],
                          delay_ticks=parameters[root]["delay_ticks"],targets=len(p["outgoing"])))
    assert len(ports)==8 and all(p["direction"]=="DN" and p["flywire_root"] for p in ports)
    assert all({p["side"] for p in ports if p["group"]==g}=={"left","right"} for g in FLIGHT_GROUPS)
    rows={}
    for name in ("intact","cut","cut_restore_short","cut_restore_long"):
        folder=BIDIR/"FREE_FLIGHT_QUAL"/name
        meta=read(folder/"RESULT.json")
        assert digest(folder/"TRACE.npy")==meta["trace_sha256"]
        assert digest(folder/"QPOS.npy")==meta["pose_sha256"]
        assert digest(BIDIR/"flight_reference/flight/saved_model.pb")==meta["network_file_sha256"]
        assert digest(BIDIR/"flight_reference/wing_pattern_fmech.npy")==meta["wing_pattern_sha256"]
        pins[str(folder/"RESULT.json")]=digest(folder/"RESULT.json")
        rows[name]={k:meta[k] for k in ("simulated_time","displacement","ground_contact_steps","min_height_cm","terminal_flag")}
    new={}
    for case, local in (("intact","FLIGHT_INTACT"),("cut_restore_short","FLIGHT_REOPEN")):
        folder=HERE/local
        original=BIDIR/"FREE_FLIGHT_QUAL"/case
        if not (folder/"RESULT.json").is_file():
            new[case]={"completed":False}
            continue
        meta=read(folder/"RESULT.json")
        old=read(original/"RESULT.json")
        assert meta["network_file_sha256"]==old["network_file_sha256"]
        assert meta["gate_sha256"]==old["gate_sha256"]
        assert digest(folder/"TRACE.npy")==meta["trace_sha256"]
        assert digest(folder/"QPOS.npy")==meta["pose_sha256"]
        a=np.load(folder/"QPOS.npy",allow_pickle=False)
        b=np.load(original/"QPOS.npy",allow_pickle=False)
        x=np.load(folder/"TRACE.npy",allow_pickle=False)
        y=np.load(original/"TRACE.npy",allow_pickle=False)
        new[case]=dict(completed=True,steps=meta["steps"],simulated_time=meta["simulated_time"],
            displacement=meta["displacement"],ground_contact_steps=meta["ground_contact_steps"],
            plant_policy_trace_exact=bool(np.array_equal(x,y)),qpos_exact=bool(np.array_equal(a,b)),
            max_abs_qpos_difference=float(np.max(np.abs(a-b))) if a.shape==b.shape else None,
            terminal_flag=meta["terminal_flag"])
    check_pins(pins)
    return dict(actual_port_rows=ports, archived=rows,new_fixed_policy_replay=new,source_sha256=pins,
                official_body_source="https://github.com/TuragaLab/flybody",
                mode_scope="Conditioned level flight and gate only; independently controlled yaw, takeoff and landing require new runs.")


def main():
    dynamic,dcount=compare("DYNAMIC","source_sha256")
    observer,ocount=compare("OBSERVER","inputs_sha256")
    assert (HERE/"OBSERVER/FIT.json").read_bytes()==(HERE/"OBSERVER_REPLAY/FIT.json").read_bytes()
    flight=flight_audit()
    write(HERE/"FLIGHT_AUDIT.json",flight)
    summary=dict(dynamic=dynamic,observer=observer,
                 replay_exact_arrays=dict(dynamic=dcount,observer=ocount),
                 observer_fit_equal=True,new_eon_steps=0,new_walking_steps=0,
                 new_flight_steps=sum(r.get("steps",0) for r in flight["new_fixed_policy_replay"].values()))
    write(HERE/"FINAL_ADAPTIVE_NECK.json",summary)
    print("Exact replay arrays",dcount,ocount,"flight",flight["new_fixed_policy_replay"],flush=True)


if __name__=="__main__":
    main()
