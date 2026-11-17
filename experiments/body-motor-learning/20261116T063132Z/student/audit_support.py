"""Compare native load attribution to the same policy restricted to RF only."""
import argparse
from pathlib import Path

import numpy as np

from fit_gap import load_decisions, verified_files
from gap_policy import LEG_ORDER, checksum, read, write

RF = LEG_ORDER.index("RF")
PER_MS = 80
LOAD_W = .02
STEADY_MS = 20


def arm(path):
    result, _, _ = verified_files(path)
    native = read(path / "NATIVE_AUDIT.json")
    if (
        result["walk_ms"] != 9600 or result["actual_ms"] != 11300
        or native["native_integrations"] != 904000
        or not native["zero_extra_force"] or native["native_sha256"] != checksum(path / "NATIVE.npz")
    ):
        raise ValueError(f"Incomplete native integration evidence {path}")
    with np.load(path / "NATIVE.npz", allow_pickle=False) as stored:
        raw = stored["forces"][:, :6].copy()
    assert raw.shape == (904000, 6) and np.isfinite(raw).all()
    _, _, _, command, nominal, ms = load_decisions(path)
    return dict(path=path, result=result, native=native, forces=raw, delta=command - nominal, ms=ms)


def first_contiguous(samples, mark, threshold, duration_ms=STEADY_MS):
    on = samples[mark:] >= threshold
    starts = np.flatnonzero(np.diff(np.r_[False, on, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, on, False].astype(int)) == -1)
    accepted = starts[ends - starts >= duration_ms * PER_MS]
    return float((mark + accepted[0])/PER_MS) if len(accepted) else None


def force_summary(force, a, b, weight):
    selected = force[a * PER_MS:b * PER_MS] / weight
    return {leg: dict(mean_W=float(selected[:, i].mean()), minimum_W=float(selected[:, i].min()),
                      maximum_W=float(selected[:, i].max()), fraction_bearing=float((selected[:, i] >= LOAD_W).mean()))
            for i, leg in enumerate(LEG_ORDER)}


def audit(output, full, deploy_retract, control):
    # Import the normalization from the plain frozen platform, not from an arm.
    from walking_bench import W

    a, b, baseline = map(arm, (full, deploy_retract, control))
    for key in ("scenario", "initial_snapshot_sha256", "sensor_fields"):
        if not a["result"][key] == b["result"][key] == baseline["result"][key]:
            raise ValueError(f"Unmatched comparison {key}")
    h = read(full / "STUDENT_RUN.json")
    r = read(deploy_retract / "STUDENT_RUN.json")
    if h["checkpoint_sha256"] != r["checkpoint_sha256"] or h["retract_only"] or not r["retract_only"]:
        raise ValueError("Support attribution requires one checkpoint with and without alternate-leg outputs")
    leg_groups = np.unique(np.flatnonzero((np.abs(a["delta"]) > 1e-9).any(0)) // 7).tolist()
    helpers = [i for i in leg_groups if i != RF]
    if not helpers or RF not in leg_groups:
        raise ValueError("Full policy must deploy RF and at least one other leg group")
    if np.any(np.abs(b["delta"][:, np.arange(42) // 7 != RF]) > 1e-9):
        raise ValueError("Retract-only arm also corrected a helper")
    helper_cols = np.isin(np.arange(42) // 7, helpers)
    chose = (np.abs(a["delta"][:, helper_cols]) > 1e-9).any(axis=1)
    activity = a["ms"][chose]
    start = int(activity[0])
    raw_begin = start * PER_MS
    if not np.array_equal(a["forces"][:raw_begin], b["forces"][:raw_begin]):
        raise ValueError("Native records diverge before first action of the ablated helper")
    with np.load(full / "TRACE.npz", allow_pickle=False) as f, np.load(deploy_retract / "TRACE.npz", allow_pickle=False) as z:
        for key in ("qpos", "qvel", "com", "loads", "padz"):
            np.testing.assert_array_equal(f[key][:start+1], z[key][:start+1], err_msg=key)
    horizon = min(11300, start + 700) * PER_MS
    reports = {}
    for i in helpers:
        label = LEG_ORDER[i]
        start_commands = np.flatnonzero((np.abs(a["delta"][:, 7*i:7*(i+1)]) > 1e-9).any(1))
        t = int(a["ms"][start_commands[0]])
        j = t * PER_MS
        g = {}
        for k, record in (("full", a), ("RF_only", b), ("frozen", baseline)):
            supported = record["forces"][:, i] / W
            gap = supported[max(0, j - STEADY_MS * PER_MS):j]
            load_at_arm_start = float(gap.max())
            g[k] = dict(load_peak_preceding_20ms_W=load_at_arm_start,
                        mean_W_200ms=float(supported[j:j + 200 * PER_MS].mean()),
                        mean_W_500ms=float(supported[j:j + 500 * PER_MS].mean()),
                        first_sustained_after_ms=first_contiguous(supported[:horizon], j, LOAD_W))
        contact_if_unloaded = g["full"]["load_peak_preceding_20ms_W"] < LOAD_W
        new_contact = contact_if_unloaded and g["full"]["first_sustained_after_ms"] is not None
        if new_contact:
            target_first = g["full"]["first_sustained_after_ms"]
            ablated_first = g["RF_only"]["first_sustained_after_ms"]
            role = ("replacement_bearing_completed_earlier_than_RF_only" if ablated_first is None or target_first < ablated_first
                    else "new_contact_observed_in_both_arms_no_earlier_replacement")
        else:
            role = "already_bearing_at_intervention" if not contact_if_unloaded else "no_sustained_replacement_bearing"
        reports[label] = dict(first_helper_command_ms=t, role=role, raw_pair=g,
                              helper_mean_gain_W_200ms=g["full"]["mean_W_200ms"] - g["RF_only"]["mean_W_200ms"],
                              helper_mean_gain_W_500ms=g["full"]["mean_W_500ms"] - g["RF_only"]["mean_W_500ms"])
    windows = {}
    for span in (20, 100, 200, 500):
        windows[str(span)] = {k: force_summary(record["forces"], start, start+span, W)
                              for k, record in (("full", a), ("RF_only", b), ("frozen", baseline))}
    summary = dict(scope="Post-intervention native load comparison with one unchanged ablated checkpoint",
                   full=str(full), retract_only=str(deploy_retract), frozen=str(control),
                   full_native_joint=bool(a["native"]["native_joint_pass"]),
                   retract_only_native_joint=bool(b["native"]["native_joint_pass"]),
                   frozen_native_joint=bool(baseline["native"]["native_joint_pass"]),
                   first_any_helper_command_ms=start, exact_pre_helper=True,
                   native_step_seconds=.0000125, bearing_threshold_W=LOAD_W, steady_duration_ms=STEADY_MS,
                   helpers=reports, support_windows_from_first_helper_ms=windows)
    write(output, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--retract-only", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output.resolve(), args.full.resolve(), args.retract_only.resolve(), args.control.resolve())
