"""Verify the localized lesion on every closed-loop event and body trace."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
ARCHIVE = BASE / "20261118T031132Z-flat-neck-assay"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BASE/"20261118T002600Z-ascending-behavior-comparison"))
from neural_runtime import DT, FIXED, FIT, read, write, verify, sha
from fault_schedule import ARMS, FAULT_ON, FAULT_OFF, SEEDS, DROPOUT_PROBABILITY, RNG_DOMAIN
from fault_timing import CanonicalEvents


ENDPOINT = 17999
GATES = ("advance_pass","final_stand_pass","safety_pass")


def load(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def record(folder):
    meta = read(folder/"RESULT.json")
    assert sha(folder/"NEURAL.npz") == meta["neural_sha256"]
    verify(meta["source_sha256"])
    return meta, load(folder/"NEURAL.npz")


def body(folder, arm):
    path = folder/arm
    meta = read(path/"RESULT.json")
    assert sha(path/"TRACE.npz") == meta["trace_sha256"]
    return meta, load(path/"TRACE.npz")


def identical(a,b):
    assert a.keys()==b.keys()
    return {k:bool(np.array_equal(a[k],b[k])) for k in a}


def first(a,b):
    indices = np.flatnonzero((a!=b).reshape(len(a),-1).any(axis=1))
    return int(indices[0]) if len(indices) else None


def audit_transport(folder, meta, data, geometry):
    n = meta["steps"]
    inputs = read(folder/"INPUTS.json")
    rows = {r["root"]:r for r in geometry["trace_intersections"]}
    selected, = [r for r in rows.values() if r["cut_by_projected_slit"]]
    down, = [r for r in rows.values() if r["direction"]=="DN"]
    targets = [None, *[selected["root"]]*3, *[down["root"]]*2]
    offsets = [0] + [rows[root]["entry_to_fault_ticks"] for root in targets[1:]]
    assert inputs["target_roots"] == targets and inputs["entry_to_fault_ticks"] == offsets
    length = n+max(offsets)
    crossing_ticks = np.arange(n, dtype=int)[:,None] + np.asarray(offsets)
    on = crossing_ticks >= FAULT_ON
    in_window = on & (crossing_ticks < FAULT_OFF)
    generator = np.random.default_rng(np.random.SeedSequence([meta["seed"],RNG_DOMAIN]))
    choices = generator.random((2,length)) < DROPOUT_PROBABILITY
    conditions = np.zeros((n,len(ARMS)),dtype=bool)
    conditions[:,1] = on[:,1]
    conditions[:,2] = in_window[:,2] & choices[0,crossing_ticks[:,2]]
    conditions[:,3] = in_window[:,3]
    conditions[:,4] = in_window[:,4] & choices[1,crossing_ticks[:,4]]
    conditions[:,5] = in_window[:,5]
    assert np.array_equal(conditions,data["target_departure_loss"])
    params = read(FIXED/"PARAMETERS.json")["ports"]
    ans = [r["banc_root"] for r in read(FIT/"BINDING.json")["records"]]
    counts = {}
    for direction, roots in (("an",ans),("dn",meta["dn_roots"])):
        for col,root in enumerate(roots):
            for lane,mode in enumerate(ARMS):
                queue = np.flatnonzero(data[f"{direction}_emitted"][:,lane,col])+int(direction=="dn")
                queue = queue[queue+params[root]["delay_ticks"]<n]
                detected = targets[lane]==root
                lost = conditions[queue,lane] if detected else np.zeros(len(queue),dtype=bool)
                arrived = queue[~lost]+params[root]["delay_ticks"]
                assert np.array_equal(arrived,np.flatnonzero(data[f"{direction}_arrived"][:,lane,col]))
                if detected:
                    at = queue+offsets[lane]
                    within = (at>=FAULT_ON)&(at<FAULT_OFF)
                    later = at>=FAULT_OFF
                    detail = {}
                    for label,select in (("intervention",within),("after_release",later)):
                        detail[label] = dict(attempted=int(np.count_nonzero(select)),
                            lost=int(np.count_nonzero(select & lost)),
                            delivered=int(np.count_nonzero(select & ~lost)))
                    removed = queue[lost]+params[root]["delay_ticks"]
                    detail["first_removed_arrival_tick"] = int(removed[0]) if len(removed) else None
                    resumed = queue[later & ~lost]+params[root]["delay_ticks"]
                    detail["first_delivered_after_release_tick"] = int(resumed[0]) if len(resumed) else None
                    assert detail["intervention"]["attempted"]>0
                    if "dropout" in mode:
                        assert detail["intervention"]["lost"]>0 and detail["intervention"]["delivered"]>0
                    else:
                        assert detail["intervention"]["delivered"]==0
                    if mode=="tear_an":
                        assert detail["after_release"]["attempted"]>0 and detail["after_release"]["delivered"]==0
                    else:
                        assert detail["after_release"]["lost"]==0 and detail["after_release"]["delivered"]>0
                    counts[mode] = detail
    assert len(counts)==len(ARMS)-1
    encoder = CanonicalEvents((len(ARMS),2))
    for tick,sample in enumerate(data["an_arrived"]):
        assert np.array_equal(encoder.step(sample), data["brain_an_events"][tick])
        assert np.array_equal(encoder.rate,data["rates"][tick])
    return counts


def angle(trace):
    yaw = np.unwrap(trace["yaw"])
    return np.rad2deg(yaw-yaw[0])


def study(seed,geometry):
    folder = ROOT/"FORMAL"/f"seed{seed}"
    archive = ARCHIVE/"FORMAL"/f"seed{seed}"
    meta,data = record(folder)
    old_meta,old_data = record(archive)
    assert meta["dn_roots"] == old_meta["dn_roots"] and meta["steps"] == 20000
    assert meta["seed"] == seed and meta["optimizer_updates"] == 0
    assert np.array_equal(data["central_hits"],old_data["central_hits"])
    for key in ("dn_emitted","dn_arrived","an_emitted","an_arrived","brain_an_events","rates","population"):
        assert np.array_equal(data[key][:,0],old_data[key][:,0])
    _,previous = body(archive,"intact")
    control_meta,control = body(folder,"intact")
    assert all(identical(control,previous).values()) and all(control_meta[k] for k in GATES)
    model_sha = sha(archive/"intact.mjb")
    counts = audit_transport(folder,meta,data,geometry)
    tear_lane, block_lane = ARMS.index("tear_an"), ARMS.index("block_an")
    reunion = counts["block_an"]["first_delivered_after_release_tick"]
    assert first(data["an_arrived"][:,tear_lane],data["an_arrived"][:,block_lane])==reunion
    assert all(np.array_equal(data[name][:reunion,tear_lane],data[name][:reunion,block_lane])
               for name in ("an_emitted","brain_an_events","dn_emitted","dn_arrived","rates","population"))
    outcomes = {}
    for lane,arm in enumerate(ARMS):
        assert sha(folder/(arm+".mjb"))==model_sha
        status,trace = body(folder,arm)
        assert status["steps"] == meta["steps"] and status["teacher_motor_calls"]==0
        if arm=="block_an":
            _,tear_body = body(folder,"tear_an")
            assert np.array_equal(tear_body["initial_position"],trace["initial_position"])
            assert all(np.array_equal(tear_body[k][:reunion],trace[k][:reunion])
                       for k in trace if k!="initial_position")
        yaw = angle(trace)
        offset = yaw-angle(control)
        distance = np.linalg.norm(trace["position"][:,:2]-control["position"][:,:2],axis=1)
        changes = {name:first(data[name][:,0],data[name][:,lane]) for name in
            ("an_emitted","an_arrived","brain_an_events","dn_emitted","dn_arrived")}
        changes.update(command=first(control["command"],trace["command"]),
                       qpos=first(control["qpos"],trace["qpos"]))
        effect = counts.get(arm)
        if effect:
            target = "dn_arrived" if arm.endswith("_dn") else "an_arrived"
            earliest = effect["first_removed_arrival_tick"]
            assert changes[target]==earliest
            assert all(v is None or v>=earliest for v in changes.values())
        dn_tick = changes["dn_emitted"]
        differences = (data["dn_emitted"][dn_tick,0]!=data["dn_emitted"][dn_tick,lane]
                       if dn_tick is not None else np.zeros(len(meta["dn_roots"]),dtype=bool))
        first_changed_dn = [root for root,changed in zip(meta["dn_roots"],differences,strict=True) if changed]
        outcomes[arm] = dict(
            target_events=effect, first_difference_ms={k:(v*.1 if v is not None else None) for k,v in changes.items()},
            first_changed_emitting_dn_roots=first_changed_dn,
            peak_abs_command_delta=float(np.abs(trace["command"]-control["command"]).max()),
            yaw_1p8_deg=float(yaw[ENDPOINT]),yaw_delta_1p8_deg=float(offset[ENDPOINT]),
            xy_delta_1p8_mm=float(distance[ENDPOINT]),max_abs_yaw_delta_deg=float(abs(offset).max()),
            max_xy_delta_mm=float(distance.max()),forward_dx_mm=status["forward_dx_mm"],
            stand_rms_mm_s=status["stand_rms_mm_s"], **{k:status[k] for k in GATES},
            equivalent_flat=bool(abs(offset[ENDPOINT])<=5 and distance[ENDPOINT]<=.5 and all(status[k] for k in GATES)))
    return dict(seed=seed,intact_archive_exact=True,all_arrivals_recomputed=True,
                tear_block_identical_before_reopened_arrival=True,arms=outcomes),meta["native_body_steps"]


def main():
    geometry = read(ROOT/"GEOMETRY_FAULT.json")
    verify(geometry["source_sha256"])
    assert read(ROOT/"QUEUE_PROBE.json")["pre_on_transit_delivered"]
    runs, steps = [], 0
    for seed in SEEDS:
        trial, count = study(seed,geometry)
        print(seed, [(arm,round(m["yaw_delta_1p8_deg"],2),round(m["xy_delta_1p8_mm"],3))
                     for arm,m in trial["arms"].items()],flush=True)
        runs.append(trial)
        steps += count
    formal, repeat = ROOT/"FORMAL/seed11701", ROOT/"REPLAY/seed11701"
    _, older = record(formal)
    meta, newer = record(repeat)
    audit_transport(repeat,meta,newer,geometry)
    assert meta["seed"]==11701 and meta["optimizer_updates"]==0
    exact = {"neural": identical(older,newer)}
    for arm in ARMS:
        exact[arm] = identical(body(formal,arm)[1],body(repeat,arm)[1])
    assert all(v for component in exact.values() for v in component.values())
    steps += meta["native_body_steps"]
    write(ROOT/"FINAL_FAULT_ASSAY.json",dict(project="P1_open_source_brain",geometry=geometry,
        seed_runs=runs,independent_replay_exact=exact,body_steps=steps,
        body_episodes=4*len(ARMS),brain_batch_steps=80000,neck_batch_steps=80000,optimizer_updates=0,
        short_term_engineering_sensitivity=True,predicts_real_tear=False))
    print(f"transport, controls and {sum(map(len,exact.values()))} replay arrays verified",flush=True)


if __name__=="__main__":
    main()
