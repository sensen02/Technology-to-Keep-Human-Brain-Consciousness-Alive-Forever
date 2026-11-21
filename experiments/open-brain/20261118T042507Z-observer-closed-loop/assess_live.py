"""Report signal recovery and behavior recovery independently."""
from pathlib import Path
import json
import os

import numpy as np

from treatment import ARMS, SEEDS, FIT, DT, prescribed_drift
from shadow_stress import event_score, load, file_sha, phase_encode, read, rms, spike_arrivals_to_rate

HERE=Path(__file__).resolve().parent
ARCH=Path(os.environ["P1_ARCHIVE_ROOT"])/"experiments/open-brain"
OLD=ARCH/"20261118T031132Z-flat-neck-assay/FORMAL/seed11701"
BIND=ARCH/"20261117T235930Z-ascending-reference-fit/BINDING.json"
PARAM=ARCH/"20261117T164207Z-fixed-neck/PARAMETERS.json"
ACTIVE=slice(3500,14500)
PLATEAU=slice(10500,14000)
AFTER=slice(17500,None)


def batch(folder):
    r=read(folder/"RESULT.json")
    for p,sha in r["source_sha256"].items():
        assert file_sha(Path(p))==sha,p
    return r,load(folder/"NEURAL.npz",r["neural_sha256"])


def bodies(folder):
    result=[]
    for mode in ARMS:
        path=folder/mode
        meta=read(path/"RESULT.json")
        result.append((meta,load(path/"TRACE.npz",meta["trace_sha256"])))
    return result


def all_arrays(a,b):
    assert set(a)==set(b)
    return {k:bool(np.array_equal(a[k],b[k])) for k in a}


def first_ms(a,b):
    hit=np.flatnonzero((a!=b).reshape(len(a),-1).any(axis=1))
    return float(hit[0]*DT*1000) if hit.size else None


def audit(data, meta):
    ports=read(PARAM)["ports"]
    side=[r["banc_root"] for r in read(BIND)["records"]]
    for kind, roots in (("an",side),("dn",meta["dn_roots"])):
        for k,root in enumerate(roots):
            delay=int(ports[root]["delay_ticks"])+int(kind=="dn")
            raw=data[kind+"_emitted"][:,:,k]
            arrival=data[kind+"_arrived"][:,:,k]
            assert not arrival[:delay].any() and np.array_equal(raw[:-delay],arrival[delay:])
    gains=np.c_[np.ones((len(data["rates"]),2)),np.repeat(prescribed_drift(len(data["rates"]))[:,None],2,axis=1)]
    assert np.array_equal(data["rates"][:,:,0]*gains,data["left_raw_hz"])
    assert np.array_equal(data["left_used_hz"],data["left_raw_hz"]/np.c_[
          gains[:,0],data["estimated_gain"][:,1],gains[:,0],data["estimated_gain"][:,3]])
    assert np.array_equal(data["true_clean_events"][:,:,1],data["brain_an_events"][:,:,1])
    for lane in range(4):
        for side in range(2):
            real=data["an_arrived"][:,lane,side]
            rate=data["rates"][:,lane,side]
            assert np.array_equal(spike_arrivals_to_rate(real),rate)
            assert np.array_equal(phase_encode(rate),data["true_clean_events"][:,lane,side])
        assert np.array_equal(phase_encode(data["left_used_hz"][:,lane]),
                              data["brain_an_events"][:,lane,0])
    assert np.array_equal(data["brain_an_events"][:6100,0],data["brain_an_events"][:6100,2])
    assert np.array_equal(data["brain_an_events"][:6100,1],data["brain_an_events"][:6100,3])
    assert np.array_equal(data["dn_emitted"][:6100,0],data["dn_emitted"][:6100,2])
    assert np.array_equal(data["dn_emitted"][:6100,1],data["dn_emitted"][:6100,3])


def body_stats(base,entry):
    b,bd=base
    m,d=entry
    delta=np.rad2deg(np.unwrap(d["yaw"])-np.unwrap(bd["yaw"]))
    return dict(xy_end_difference_mm=float(np.linalg.norm(d["position"][-1,:2]-bd["position"][-1,:2])),
        yaw_end_difference_deg=float(delta[-1]), max_active_abs_yaw_deg=float(np.max(abs(delta[ACTIVE]))),
        active_forward_speed_difference_mm_s=float(m["active_forward_mm_s"]-b["active_forward_mm_s"]),
        forward_dx_mm=m["forward_dx_mm"],final_yaw_degrees=m["yaw_degrees"],
        stand_rms_mm_s=m["stand_rms_mm_s"],
        safety_pass=m["safety_pass"],advance_pass=m["advance_pass"],stand_pass=m["final_stand_pass"],
        first_body_difference_ms=first_ms(bd["qpos"],d["qpos"]))


def main():
    rows=[]
    for seed in SEEDS:
        folder=HERE/"FORMAL"/f"seed{seed}"
        meta, data=batch(folder)
        audit(data,meta)
        inp=read(folder/"INPUTS.json")
        assert file_sha(FIT)==inp["observer_fit_sha256"]
        body=bodies(folder)
        control=body[0]
        old_parity=None
        if seed==11701:
            _, original=batch(OLD)
            for key in ("dn_emitted","dn_arrived","an_emitted","an_arrived","brain_an_events","rates","population"):
                assert np.array_equal(data[key][:,0],original[key][:,0]),key
            assert np.array_equal(data["central_hits"],original["central_hits"])
            p=OLD/"intact"
            assert all(all_arrays(control[1],load(p/"TRACE.npz",read(p/"RESULT.json")["trace_sha256"])).values())
            old_parity=True
        result={}
        for i, arm in enumerate(ARMS):
            stats=body_stats(control,body[i])
            clean=data["rates"][:,i,0]
            actual=data["left_used_hz"][:,i]
            stats.update(rate_plateau_rmse_hz=rms(actual[PLATEAU],clean[PLATEAU]),
                 rate_after_rmse_hz=rms(actual[AFTER],clean[AFTER]),
                 rate_active_rmse_hz=rms(actual[ACTIVE],clean[ACTIVE]),
                 left_an_event_plateau=event_score(data["brain_an_events"][PLATEAU,i,0],
                                                  data["true_clean_events"][PLATEAU,i,0]),
                 active_events_changed_ticks=int(np.count_nonzero(data["true_clean_events"][ACTIVE,i,0]!=
                                                data["brain_an_events"][ACTIVE,i,0])),
                 first_an_difference_vs_clean_off_ms=first_ms(data["brain_an_events"][:,0],data["brain_an_events"][:,i]),
                 first_dn_difference_vs_clean_off_ms=first_ms(data["dn_emitted"][:,0],data["dn_emitted"][:,i]))
            result[arm]=stats
        normal=result["clean_on"]
        raw=result["drift_off"]
        fixed=result["drift_on"]
        ability_keys=("advance_pass","stand_pass","safety_pass")
        normal_checks=dict(both_body_gates=all(r[key] for r in (normal,result["clean_off"]) for key in ability_keys),
             endpoint=normal["xy_end_difference_mm"]<=.5,yaw=normal["max_active_abs_yaw_deg"]<=5.,
             forward_speed=abs(normal["active_forward_speed_difference_mm_s"])<=1.)
        recovery_checks=dict(same_body_rate=fixed["rate_plateau_rmse_hz"]<=.8*raw["rate_plateau_rmse_hz"],
             same_body_event_timing=fixed["left_an_event_plateau"]["f1"]>raw["left_an_event_plateau"]["f1"],
             final_xy=fixed["xy_end_difference_mm"]<raw["xy_end_difference_mm"],
             final_yaw=abs(fixed["yaw_end_difference_deg"])<abs(raw["yaw_end_difference_deg"]),
             body_gates=all(fixed[key] for key in ability_keys))
        rows.append(dict(seed=seed,archived_baseline_exact=old_parity,arms=result,
                         normal_checks=normal_checks,normal_pass=all(normal_checks.values()),
                         recovery_checks=recovery_checks,recovery_pass=all(recovery_checks.values()),
                         body_steps=meta["native_body_steps"],brain_ticks=meta["brain_batch_ticks"]))
        print(seed,normal_checks,recovery_checks,flush=True)
    official=HERE/"FORMAL/seed11903"
    replay=HERE/"REPLAY/seed11903"
    _, aa=batch(official)
    meta, bb=batch(replay)
    replay_parity={"neural":all_arrays(aa,bb)}
    for arm,a,b in zip(ARMS,bodies(official),bodies(replay),strict=True):
        replay_parity[arm]=all_arrays(a[1],b[1])
    assert all(all(x.values()) for x in replay_parity.values())
    output=dict(rows=rows,normal_every_case=all(r["normal_pass"] for r in rows),
          restored_every_case=all(r["recovery_pass"] for r in rows),
          replay_arrays=sum(len(g) for g in replay_parity.values()),
          physical_steps=sum(r["body_steps"] for r in rows)+meta["native_body_steps"],
          brain_batch_ticks=sum(r["brain_ticks"] for r in rows)+meta["brain_batch_ticks"],
          optimizer_updates=0)
    with (HERE/"FINAL_LIVE.json").open("x") as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write("\n")
    print("Normal in all",output["normal_every_case"],"Recovery in all",output["restored_every_case"],
          "Replay arrays",output["replay_arrays"],flush=True)


if __name__=="__main__":
    main()
