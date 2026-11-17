"""Check paired physical source influence and independently replay native contacts."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from retarget import ARCHIVE, BASELINE, CONTROL_MS, HERE, PROFILE, WALK_MS, WALK_START, Policy, bench

NATIVE_SOURCE = ARCHIVE.parent / "20261116T004235Z"


def source_effect(run, control):
    result = bench.read(run / "RESULT.json")
    provenance = bench.read(run / "SOURCE_PROVENANCE.json")
    paired = bench.read(control / "RESULT.json")
    ref = bench.read(HERE / "SWING_REFERENCE.json")
    assert provenance["result_sha256"] == bench.stand.sha(run / "RESULT.json")
    assert provenance["source_ref_sha256"] == ref["ref_sha256"] == bench.stand.sha(PROFILE)
    assert result["start"] == paired["start"] and result["walk_ms"] == paired["walk_ms"]
    for path, meta in ((run, result), (control, paired)):
        assert bench.stand.sha(path / "TRACE.npz") == meta["trace_sha256"]
        assert bench.stand.sha(path / "DECISIONS.npz") == meta["decision_sha256"]
    p = Policy(provenance["source_gain"], ref)
    with np.load(run / "DECISIONS.npz", allow_pickle=False) as x:
        ms, intent, actual, feature = (x[k] for k in ("ms", "intent", "action", "x"))
    base_on_same_obs = np.stack([p.base(row, go) for row, go in zip(feature, intent)])
    action_delta = np.abs(actual - base_on_same_obs)
    with np.load(PROFILE, allow_pickle=False) as z:
        source = z["correction_native"]
    with np.load(run / "TRACE.npz", allow_pickle=False) as z:
        transferred_z = z["padz"]
        transferred_body = z["foot"][:, :, 2]
    with np.load(control / "TRACE.npz", allow_pickle=False) as z:
        control_z = z["padz"]
        control_body = z["foot"][:, :, 2]
    windows = []
    for row in ref["receiver_windows"]:
        lo, hi = row["from_ms"], row["to_ms"]
        if hi >= len(transferred_z):
            continue
        li = bench.LEGS.index(row["leg"])
        d = transferred_z[lo:hi + 1, li] - control_z[lo:hi + 1, li]
        d_body = transferred_body[lo:hi + 1, li] - control_body[lo:hi + 1, li]
        target = source[lo:hi + 1, li]
        weights = np.sin(np.linspace(0., np.pi, len(d))) ** 2
        windows.append(dict(**row,
            weight=float(weights.sum()),
            direction=float(np.dot(weights, d*target)),
            direction_thorax=float(np.dot(weights, d_body*target)),
            changed_sq=float(np.dot(weights, d*d)),
            source_error_sq_before=float(np.dot(weights, target*target)),
            source_error_sq_after=float(np.dot(weights, (target-d)**2)),
        ))
    if not windows:
        raise ValueError("No complete source effect window")
    by_leg = {leg: sum(w["direction"] for w in windows if w["leg"] == leg) for leg in ref["source_legs"]}
    norm = sum(w["weight"] for w in windows)
    ag = {key: sum(w[key] for w in windows) for key in ("direction", "direction_thorax", "changed_sq",
          "source_error_sq_before", "source_error_sq_after")}
    rms = float(np.sqrt(ag["changed_sq"]/norm))
    stats = dict(
        passed=bool(ag["direction"] > 0 and sum(d > 0 for d in by_leg.values()) >= 2 and rms >= .0002
                    and ag["source_error_sq_after"] < ag["source_error_sq_before"]),
        paired_control=str(control), gain_off_sha256=paired["trace_sha256"], candidate_trace_sha256=result["trace_sha256"],
        candidate_decision_sha256=result["decision_sha256"], effect_windows=windows,
        projection_sum=ag["direction"], projection_thorax_frame_sum=ag["direction_thorax"],
        pad_z_change_rms_native=rms, per_leg_projection=by_leg,
        pad_z_source_error_before_rms_native=float(np.sqrt(ag["source_error_sq_before"]/norm)),
        pad_z_source_error_after_rms_native=float(np.sqrt(ag["source_error_sq_after"]/norm)),
        walk_action_mae_against_base_on_same_observations=float(action_delta[intent == 1].mean()),
        max_action_delta_against_base_on_same_observations=float(action_delta.max()),
        all_nonwalking_actions_exact_base=bool(not action_delta[intent == 0].any()),
        changed_walk_decisions=int(np.count_nonzero(action_delta[intent == 1].max(1))),
    )
    bench.stand.put(run / "SOURCE_EFFECT.json", stats)
    print({k:v for k,v in stats.items() if k!="effect_windows"}, flush=True)
    return stats


class Replay:
    def __init__(self, decisions):
        with np.load(decisions, allow_pickle=False) as z:
            self.x, self.a, self.intent, self.ms = (z[k] for k in ("x", "action", "intent", "ms"))
        if not np.array_equal(self.ms, CONTROL_MS * np.arange(len(self.ms))):
            raise ValueError("Replay reference decision grid changed")
        self.i = 0

    def __call__(self, x, intent):
        np.testing.assert_array_equal(x, self.x[self.i])
        assert intent == self.intent[self.i]
        self.i += 1
        return self.a[self.i - 1]


def score_native(record, result, trace):
    warm, walk = WALK_START, result["walk_ms"]
    end, final = warm + walk, warm + walk + 1200
    if record.count != final * 80:
        raise ValueError("Incomplete raw integration")
    np.testing.assert_array_equal(record.endpoints, trace["qpos"])
    np.testing.assert_allclose(record.poses[:,6], trace["time"][0] + np.arange(record.count)*12.5e-6,
                               atol=3e-8, rtol=0.)
    if record.poses[:,7].any():
        raise ValueError("Extra force at native rate")
    report = {}
    for key, a, b in (("walking", warm, end), ("initial_stand", warm-200, warm), ("terminal_stand", final-200, final)):
        forces = record.forces[a*80:b*80]
        pose = record.poses[a*80:b*80]
        foot_load = forces[:, :6]
        norm = foot_load.sum(1) / bench.W
        out = dict(
            samples=len(forces), foot_support_mean_w=float(norm.mean()), foot_support_p10_w=float(np.quantile(norm, .1)),
            nonfoot_peak_w=float(forces[:, 12].max()/bench.W),
            com_min=float(min(pose[:, 2].min(), trace["com"][b, 2])),
            upright_dot_min=float(min(pose[:, 5].min(), trace["upright"][b])),
            three_foot_fraction=float(((foot_load >= .02*bench.W).sum(1) >= 3).mean()),
        )
        gates = dict(support=out["foot_support_mean_w"]>=.9, com=out["com_min"]>=1.2)
        if key == "walking":
            dt = .001 * (b-a)
            move = trace["com"][b, :2] - trace["com"][a, :2]
            longitudinal = float(move[0]/dt)
            speed = np.linalg.norm(move) / dt
            slip = forces[:, 6:12].sum(0) / foot_load.sum(0) / speed
            out.update(forward_signed_native_s=longitudinal, slip_max=float(slip.max()),
                       per_leg_slip=dict(zip(bench.LEGS, slip.tolist())), cycles=result["cycles"])
            gates.update(nonfoot=out["nonfoot_peak_w"] <= .20, forward=longitudinal >= .05,
                         slip=bool(np.isfinite(slip).all() and slip.max() <= .5), cycles=bool(all(result["cycles"].values())))
        else:
            # Match the originally recorded stand velocity estimator.
            speed = result[key]["speed_rms"]
            out["speed_rms_body_lengths_s"] = speed
            gates.update(speed=speed<=.01, nonfoot=out["nonfoot_peak_w"]<.01,
                         loaded_legs=out["three_foot_fraction"]>=.95, posture=bool(out["upright_dot_min"]>=np.cos(np.deg2rad(20.))))
        out.update(gates=gates, passed=bool(all(gates.values())))
        report[key] = out
    return dict(physical_pass=bool(all(v["passed"] for v in report.values())), windows=report,
                native_samples=record.count, compact_extractor_cross_checks=record.cross_checks,
                all_1ms_qpos_matches=True, zero_extra_force=True, full_D1_D8=False)


def native_replay(run, recover=False):
    sys.path.insert(0, str(NATIVE_SOURCE))
    from verify_native import Record, SO
    result = bench.read(run / "RESULT.json")
    provenance = bench.read(run / "SOURCE_PROVENANCE.json")
    assert result["joint_sampled_pass"] and result["actual_ms"] == WALK_START + WALK_MS + 1200
    effect = bench.read(run / "SOURCE_EFFECT.json")
    assert effect["passed"] and effect["all_nonwalking_actions_exact_base"]
    assert effect["candidate_decision_sha256"] == result["decision_sha256"]
    policy = Replay(run / "DECISIONS.npz")
    load_initial = bench.load_initial
    record = []

    def attach(standing_bench, *rest):
        load_initial(standing_bench, *rest)
        record.append(Record(standing_bench.body, result["actual_ms"]))
        standing_bench.body.recorder = record[-1]

    raw_path = run / "NATIVE.npz"
    if recover:
        # Enter only after replay and raw write succeeded and the bookkeeping write failed.
        prior_log = run.parent / f"{run.name}_verify.log"
        raw_error = prior_log.read_text()
        assert 'verifier_sha256=bench.stand.sha(__file__)' in raw_error
        assert "AttributeError: 'str' object has no attribute 'open'" in raw_error
        replay = bench.read(run / "NATIVE_REPLAY/RESULT.json")
        with np.load(raw_path, allow_pickle=False) as z:
            count = len(z["pose"])
            rec = SimpleNamespace(forces=z["foot_normal_and_slip_weighted"], poses=z["pose"],
                                  endpoints=z["qpos_1ms"], count=count, cross_checks=(count-1)//4031+1)
        executed_code = HERE / "verify_teacher_native_executed.py"
    else:
        bench.load_initial = attach
        replay = bench.rollout(run / "NATIVE_REPLAY", result["start"], result["walk_ms"], policy)
        assert policy.i == len(policy.ms)
        rec, = record
        with raw_path.open("xb") as raw:
            np.savez(raw, foot_normal_and_slip_weighted=rec.forces, pose=rec.poses, qpos_1ms=rec.endpoints)
        executed_code = Path(__file__)
    for key in ("trace_sha256", "decision_sha256"):
        assert replay[key] == result[key], f"Independent replay diverged at {key}"
    for name, key in (("TRACE.npz", "trace_sha256"), ("DECISIONS.npz", "decision_sha256")):
        assert bench.stand.sha(run / "NATIVE_REPLAY" / name) == result[key]
    with np.load(run / "TRACE.npz", allow_pickle=False) as trace:
        native = score_native(rec, result, trace)
    native.update(
        native_arrays_sha256=bench.stand.sha(raw_path),
        native_library_path=str(SO), native_library_sha256=bench.stand.sha(SO),
        replay_trace_sha256=replay["trace_sha256"], replay_decisions_sha256=replay["decision_sha256"],
        native_executed_verifier_sha256=bench.stand.sha(executed_code),
        verifier_sha256=bench.stand.sha(Path(__file__)),
        recovered_existing_replay_metadata=recover,
        native_force_columns=[*[f"normal_{leg}" for leg in bench.LEGS],
                              *[f"normal_times_xy_speed_{leg}" for leg in bench.LEGS], "nonfoot_normal"],
        native_pose_columns=["com_x", "com_y", "com_z", "heading_x", "heading_y", "upright_dot", "time_before_s", "max_extra_forces"],
    )
    bench.stand.put(run / "NATIVE_AUDIT.json", native)
    qualified = bool(native["physical_pass"] and effect["passed"])
    bench.stand.put(run / "TEACHER_ACCEPTANCE.json", dict(
        passed=qualified, result_sha256=bench.stand.sha(run / "RESULT.json"),
        trace_sha256=result["trace_sha256"], decision_sha256=result["decision_sha256"],
        native_audit_sha256=bench.stand.sha(run / "NATIVE_AUDIT.json"),
        source_effect_sha256=bench.stand.sha(run / "SOURCE_EFFECT.json"),
        source_provenance_sha256=bench.stand.sha(run / "SOURCE_PROVENANCE.json"),
        flygym_trace_sha256=provenance["source_trace_sha256"], source_ref_sha256=provenance["source_ref_sha256"],
        scope="same-body FlyGym-informed teacher, cold stand 500 ms, walking 9600 ms, stop 1200 ms",
    ))
    print(native, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--control", type=Path, default=BASELINE)
    p.add_argument("--native-replay", action="store_true")
    p.add_argument("--recover-native-metadata", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    args.run = args.run.resolve()
    if not (args.run / "SOURCE_EFFECT.json").exists():
        source_effect(args.run, args.control.resolve())
    if args.native_replay or args.recover_native_metadata:
        native_replay(args.run, recover=args.recover_native_metadata)
