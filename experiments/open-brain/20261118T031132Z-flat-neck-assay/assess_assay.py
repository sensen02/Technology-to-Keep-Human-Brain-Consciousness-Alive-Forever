"""Verify the matched closed-loop experiment from its saved native ticks."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "20261118T020425Z-front-touch-timing-repair"
sys.path.insert(0, str(HERE.parent/"20261118T002600Z-ascending-behavior-comparison"))
from neural_runtime import DT, FIT, FIXED, read, sha, verify, write
from audit_queue import expected_arrivals
from assay_timing import CanonicalEvents
from schedule import ARMS, SEEDS, CUT_ROOT, CUT_ON, RESTORE, YAW_SAMPLE, YAW_EFFECT_THRESHOLD_DEG

META = {"target_blocked", "scheduled_event", "central_hits"}


def checked(path):
    meta = read(path/"RESULT.json")
    verify(meta["source_sha256"])
    neural_file = path/"NEURAL.npz"
    assert sha(neural_file) == meta["neural_sha256"]
    with np.load(neural_file, allow_pickle=False) as neural:
        data = {key: neural[key].copy() for key in neural.files}
    return meta, data


def body_record(path):
    meta = read(path/"RESULT.json")
    trace = path/"TRACE.npz"
    assert sha(trace) == meta["trace_sha256"]
    with np.load(trace, allow_pickle=False) as body:
        return meta, {key: body[key].copy() for key in body.files}


def first(a,b):
    diff = np.flatnonzero((a!=b).reshape(len(a),-1).any(axis=1))
    return int(diff[0]) if diff.size else None


def same_arrays(a, b, on=None):
    assert set(a)==set(b)
    return {k: bool(np.array_equal(a[k] if on is None else a[k][:on],
                                   b[k] if on is None else b[k][:on])) for k in a}


def timeline(data, body, reference, changed, start):
    earlier = data["an_arrived"][:, reference, 0]
    later = data["an_arrived"][:, changed, 0]
    r, c = body[reference], body[changed]
    times = dict(left_neck_arrival=first(earlier, later),
                 left_brain_AN=first(data["brain_an_events"][:,reference,0],
                                     data["brain_an_events"][:,changed,0]),
                 DN=first(data["dn_emitted"][:,reference],data["dn_emitted"][:,changed]),
                 neck_motor_command=first(r["command"], c["command"]),
                 qpos=first(r["qpos"],c["qpos"]),
                 own_foot_feedback=first(r["normal_load"],c["normal_load"]))
    known = [value for value in times.values() if value is not None]
    # Contact forces and state may be updated on the same native step.
    assert all(t>=start for t in known)
    order = [times[k] for k in ("left_neck_arrival","left_brain_AN","DN",
                                  "neck_motor_command","qpos")]
    present = [v for v in order if v is not None]
    ordered = all(x<=y for x,y in zip(present, present[1:])) and order[:len(present)]==present
    whole_chain = len(present)==len(order) and ordered
    return dict(tick=times, first_millisecond={key:(t*DT*1000 if t is not None else None)
                                              for key,t in times.items()},
                complete_observed_chain=whole_chain, all_observed_differences_ordered=ordered)


def audit_queue_and_codec(data, meta):
    blocked = data["target_blocked"]
    flags = data["scheduled_event"]
    assert not blocked[:,0:2].any()
    expected_mask = np.zeros_like(blocked)
    expected_mask[CUT_ON:,2] = True
    expected_mask[CUT_ON:RESTORE,3] = True
    assert np.array_equal(blocked, expected_mask)
    assert np.flatnonzero(flags[:,1]).tolist()==[CUT_ON,RESTORE]
    assert not flags[:,0].any()
    assert tuple(flags[CUT_ON,1:])==(1,1,1)
    assert tuple(flags[RESTORE,1:])==(-1,0,-1)
    assert np.count_nonzero(flags)==5

    bound = read(FIT/"BINDING.json")["records"]
    roots = [row["banc_root"] for row in bound]
    assert roots[0]==CUT_ROOT
    port_delays = read(FIXED/"PARAMETERS.json")["ports"]
    for kind, port_ids in (("an",roots),("dn",meta["dn_roots"])):
        for channel, root in enumerate(port_ids):
            delay = port_delays[root]["delay_ticks"] + int(kind=="dn")
            for lane in range(4):
                mask = blocked[:,lane] if root==CUT_ROOT else np.zeros(len(blocked),dtype=bool)
                expected = expected_arrivals(data[f"{kind}_emitted"][:,lane,channel], delay, mask)
                assert np.array_equal(data[f"{kind}_arrived"][:,lane,channel],expected)

    codec = CanonicalEvents((4, 2))
    for tick, arrival in enumerate(data["an_arrived"]):
        assert np.array_equal(codec.step(arrival),data["brain_an_events"][tick])
        assert np.array_equal(codec.rate,data["rates"][tick])
    return port_delays[CUT_ROOT]["delay_ticks"]


def main():
    queue = read(HERE/"QUEUE_PROBE.json")
    assert queue["pass_"] and sha(HERE/"QUEUE_PROBE.npz")==queue["trace_sha256"]
    rows = []
    body_steps = brain_steps = 0
    for seed in SEEDS:
        folder = HERE/"FORMAL"/f"seed{seed}"
        meta, data = checked(folder)
        delay = audit_queue_and_codec(data, meta)
        body_steps += meta["native_body_steps"]
        brain_steps += meta["brain_batch_ticks"]
        obtained = [body_record(folder/arm) for arm in ARMS]
        metrics, arrays = map(list,zip(*obtained,strict=True))
        neural = [dict((key,value[:,lane]) for key,value in data.items() if key not in META)
                  for lane in range(4)]
        assert all(same_arrays(arrays[0],arrays[1]).values())
        assert all(same_arrays(neural[0],neural[1]).values())
        for arm in (1,2,3):
            assert all(same_arrays(arrays[0],arrays[arm],CUT_ON).values())
            assert all(same_arrays(neural[0],neural[arm],CUT_ON).values())
        assert all(same_arrays(arrays[2],arrays[3],RESTORE).values())
        assert all(same_arrays(neural[2],neural[3],RESTORE).values())
        baseline_qualified = all(metrics[0][gate] for gate in ("advance_pass", "safety_pass", "final_stand_pass"))
        archived_baseline = None
        if seed in (10901,11311):
            archive = BASELINE/"FORMAL"/f"seed{seed}"
            old, old_data = checked(archive)
            _, old_body = body_record(archive/"repaired")
            assert sha(folder/"intact.mjb")==sha(archive/"repaired.mjb")
            assert all(same_arrays(arrays[0],old_body).values())
            assert np.array_equal(old_data["central_hits"],data["central_hits"])
            for k,value in neural[0].items():
                assert np.array_equal(value,old_data[k][:,0])
            archived_baseline = True

        continuous = timeline(data,arrays,1,2,CUT_ON)
        reopening = timeline(data,arrays,2,3,RESTORE)
        assert continuous["all_observed_differences_ordered"]
        assert reopening["all_observed_differences_ordered"]
        arrival = reopening["tick"]["left_neck_arrival"]
        assert not data["an_arrived"][RESTORE:,2,0].any()
        if arrival is not None:
            assert arrival >= RESTORE + delay
            assert data["an_emitted"][arrival-delay,3,0]
        target_count = {}
        for arm, lane in zip(ARMS, range(4), strict=True):
            e = data["an_emitted"][:,lane,0]
            a = data["an_arrived"][:,lane,0]
            b = data["brain_an_events"][:,lane,0]
            target_count[arm] = dict(source_spikes_during_shared_cut=int(e[CUT_ON:RESTORE].sum()),
                arrivals_during_shared_cut=int(a[CUT_ON:RESTORE].sum()),
                arrivals_after_reopen_until_stop=int(a[RESTORE:15000].sum()),
                brain_AN_events_after_reopen_until_stop=int(b[RESTORE:15000].sum()))
        yaw = np.array([np.rad2deg(np.unwrap(a["yaw"])[YAW_SAMPLE]-a["yaw"][0]) for a in arrays])
        effect, recovered = yaw[[2,3]]-yaw[1]
        positions = np.array([a["position"][YAW_SAMPLE,:2] for a in arrays])
        row = dict(seed=seed, control_parity=True, prior_baseline_exact=archived_baseline,
            untreated_baseline_qualified=baseline_qualified,
            restored_target_traffic_observed=arrival is not None,
            paired_prefixes_exact=True, physical_transport_verified=True,
            intervention=continuous, recovery=reopening, target_counts=target_count,
            yaw_1p8_degrees=dict(zip(ARMS,yaw.tolist(),strict=True)),
            sustained_minus_sham_yaw_deg=float(effect), restore_minus_sham_yaw_deg=float(recovered),
            recovery_reduces_abs_yaw_error=bool(abs(recovered)<abs(effect)),
            signed_effect_at_least_threshold=bool(abs(effect)>=YAW_EFFECT_THRESHOLD_DEG),
            xy_sustained_sham_1p8_mm=float(np.linalg.norm(positions[2]-positions[1])),
            xy_restore_sham_1p8_mm=float(np.linalg.norm(positions[3]-positions[1])),
            first_fresh_left_spike_at_brain_input_tick=reopening["tick"]["left_brain_AN"],
            metrics={arm:{key:m[key] for key in (
                "safety_pass","safety_gates","advance_pass","final_stand_pass","forward_dx_mm",
                "yaw_degrees","stand_rms_mm_s","min_height_mm")} for arm,m in zip(ARMS,metrics,strict=True)})
        rows.append(row)
        print(seed, "sustained effect",round(effect,3),"reopen error",round(recovered,3),
              "chain",continuous["first_millisecond"],flush=True)

    trial = HERE/"FORMAL"/"seed11701"
    replay = HERE/"REPLAY"/"seed11701"
    _, a = checked(trial)
    rep,b = checked(replay)
    body_steps += rep["native_body_steps"]
    brain_steps += rep["brain_batch_ticks"]
    parity = {"neural":same_arrays(a,b)}
    for arm in ARMS:
        parity[arm] = same_arrays(body_record(trial/arm)[1],body_record(replay/arm)[1])
    assert all(all(x.values()) for x in parity.values())
    effects = [row["sustained_minus_sham_yaw_deg"] for row in rows]
    same_sign = all(e>0 for e in effects) or all(e<0 for e in effects)
    repeated_yaw = same_sign and all(row["intervention"]["complete_observed_chain"]
                                  and row["untreated_baseline_qualified"]
                                  and row["signed_effect_at_least_threshold"] for row in rows)
    control_range = [row["yaw_1p8_degrees"]["sham"] for row in rows]
    write(HERE/"FINAL_ASSAY.json",dict(project="P1_open_source_brain", target=CUT_ROOT, rows=rows,
        intervention_cut_tick=CUT_ON, reopen_tick=RESTORE, yaw_sample_tick=YAW_SAMPLE,
        yaw_primary_threshold_deg=YAW_EFFECT_THRESHOLD_DEG,
        repeated_primary_behavior_effect=repeated_yaw,
        sham_yaw_range_deg=[min(control_range),max(control_range)], same_sign=same_sign,
        exact_replay=parity, body_episodes=(len(SEEDS)+1)*4, body_steps=body_steps,
        brain_batch_ticks=brain_steps, brain_lane_ticks=brain_steps*4,
        neck_batch_ticks=brain_steps+queue["probe_ticks"], neck_lane_ticks=brain_steps*4+queue["lane_ticks"],
        optimizer_updates=0))
    print("Primary repeated behavior",repeated_yaw,"independent arrays",
          sum(len(g) for g in parity.values()),flush=True)


if __name__=="__main__":
    main()
