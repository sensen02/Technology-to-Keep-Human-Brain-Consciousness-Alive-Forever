"""Audit learned stumble trials against saved frozen controls and repeats."""
from pathlib import Path
import sys

import numpy as np

from fit_reflex import decisions, verified_files
from stumble_policy import HERE, checksum, read, write

TEACHER = HERE.parent / "teacher"
sys.path.insert(0, str(TEACHER))
from physical import W
from qualify import qualify


def read_replay(run, checkpoint):
    result, _, hashes = verified_files(run)
    saved = read(run / "STUDENT_RUN.json")
    if saved["student_run_sha256"] != hashes or saved["checkpoint_sha256"] != checkpoint:
        raise ValueError(f"Unverified student replay {run}")
    if not saved["reverse_decisions_bitwise"]:
        raise ValueError(f"State is not independently replayable {run}")
    audit = read(run / "NATIVE_AUDIT.json")
    if (checksum(run / "NATIVE.npz") != audit["native_sha256"]
            or checksum(run / "NATIVE_AUDIT.json") != result["native_audit_sha256"]):
        raise ValueError(f"Native source identity changed {run}")
    return result, saved, audit


def case(run, control, checkpoint):
    r, replay, audit = read_replay(run, checkpoint)
    cert = qualify(run, control, run / "STUDENT_ACCEPTANCE.json", HERE/"FLAT_FROZEN")
    _, local, _, _, _, ms = decisions(run)
    engagement = replay["active_first_ms"]
    trigger = None
    if engagement is not None:
        frame = np.flatnonzero(ms == engagement).item()
        trigger = local[frame, :9].reshape(3, 3).tolist()
    reload = cert["stable_ground_reload_ms"]
    dwell = None
    if reload is not None and reload+200 <= r["actual_ms"]:
        with np.load(run/"NATIVE.npz", allow_pickle=False) as b:
            load = b["forces"][reload*80:(reload+200)*80, 3]/W
        dwell = dict(duration_ms=200, minimum_load_W=float(load.min()), mean_load_W=float(load.mean()),
                     fraction_at_least_point_zero_two_W=float(np.mean(load >= .02)))
    info = dict(physical_scope=cert["scope"], evidence=str(run / "STUDENT_ACCEPTANCE.json"),
        passed=cert["passed"], checks=cert["checks"], cycles=r["cycles"],
        native_integrations=audit["native_integrations"], ground_slip=audit["max_ground_normalized_slip"],
        contact_begin_ms=r["event"]["obstacle_first_contact_ms"],
        contact_end_ms=r["event"]["obstacle_last_contact_ms"],
        native_resisting_impulse_Ws=audit["obstacle_resisting_impulse_Ws"],
        rf_pad_peak_native=r["event"]["target_pad_peak_during_recovery"],
        pad_clearance_gain_native=cert["pad_clearance_gain"],
        rf_reload_floor_ms=reload, rf_reload_following_native=dwell,
        joint_stand_pass=bool(r["initial_stand"]["passed"] and r["terminal_stand"]["passed"]),
        corrective_decisions=replay["active_decisions"], first_corrective_decision_ms=engagement,
        sensed_at_first_intervention_rf_segment_Fxyz_per_W=trigger)
    return info


def main():
    fit = read(HERE/"FIT_RF_A"/"RESULT.json")
    checkpoint = fit["sha256"]
    if checksum(Path(fit["checkpoint"])) != checkpoint:
        raise ValueError("Trained policy changed")
    a = HERE/"STUDENT_PEG_A"
    b = HERE/"STUDENT_PEG_B"
    flat = HERE/"STUDENT_FLAT"
    repeat = HERE/"STUDENT_A_REPEAT"
    one = case(a, TEACHER/"CONTROL_PEG_A", checkpoint)
    two = case(b, TEACHER/"CONTROL_PEG_B", checkpoint)
    flat_result, flat_replay, flat_audit = read_replay(flat, checkpoint)
    rpt, _, _ = read_replay(repeat, checkpoint)
    rerun_equal = {}
    for name in ("TRACE.npz", "DECISIONS.npz", "NATIVE.npz"):
        digest = checksum(a/name)
        if checksum(repeat/name) != digest:
            raise AssertionError(f"Independent replay diverged in {name}")
        rerun_equal[name] = digest
    flat_pass = flat_result["native_joint_pass"] and flat_replay["active_decisions"] == 0
    flat_match = len(flat_replay["matching_evidence"]) == 3
    report = dict(checkpoint=fit["checkpoint"], checkpoint_sha256=checkpoint,
                  main_qualified=one["passed"] and flat_pass and flat_match,
                  main_geometry=one, withheld_geometry=two,
                  flat=dict(native_joint_pass=flat_pass, exact_to_control=flat_match,
                            active=flat_replay["active_decisions"], native_steps=flat_audit["native_integrations"],
                            ground_slip=flat_audit["max_ground_normalized_slip"]),
                  independently_repeat_native_steps=rpt["physical_native_iterations"],
                  independently_repeated_hashes=rerun_equal,
                  full_D1_D8=False, matched_geometry_train_a_only=True)
    write(HERE/"STUDENT_SUITE.json",report)
    print(report,flush=True)


if __name__ == "__main__":
    main()
