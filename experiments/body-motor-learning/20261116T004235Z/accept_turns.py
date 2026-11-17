"""Verify the archived native contacts, continuation, and fixed turn controller."""
from pathlib import Path

import numpy as np

from turning import HERE, MODES, W, read, stand


def native_phases(path, result, native):
    arrays = path / "NATIVE.npz"
    assert stand.sha(arrays) == native["native_arrays_sha256"]
    assert native["native_samples"] == result["nsteps"]
    begin, end, after, finish = result["bounds"]
    with np.load(arrays, allow_pickle=False) as data:
        contacts = data["forces"]
        positions = data["pose"][:, :3].copy()
    with np.load(path / "TRACE.npz", allow_pickle=False) as trace:
        com = trace["com"].copy()
        yaw = np.unwrap(trace["yaw"])
    assert len(contacts) == len(positions) == finish * 80
    np.testing.assert_array_equal(positions[::80], com[:-1])

    def straight_window(first, last):
        delta = com[last, :2] - com[first, :2]
        heading = np.array([np.cos(yaw[first]), np.sin(yaw[first])])
        duration = (last - first) * .001
        forces = contacts[first * 80:last * 80]
        support = float(forces[:, :6].sum() / len(forces) / W)
        mean_slip = forces[:, 6:12].sum(0) / forces[:, :6].sum(0)
        return dict(
            forward=float(np.dot(delta, heading) / duration),
            yaw_change_deg=float(np.rad2deg(yaw[last] - yaw[first])),
            com_min=float(min(positions[first*80:last*80, 2].min(), com[last, 2])),
            support=support,
            nonfoot=float(forces[:, 12].max()),
            straight_slip_ratio=float(mean_slip.max() / (np.linalg.norm(delta) / duration)),
        )

    summary = dict(nonfoot_all_phases=float(contacts[:, 12].max()))
    if begin > 500:
        summary["entry"] = straight_window(500, begin)
        summary["continuation"] = continuation = straight_window(end, after)
        assert continuation["forward"] >= .05 and abs(continuation["yaw_change_deg"]) <= 10
        assert continuation["com_min"] >= 1.2 and continuation["support"] >= .9
        assert continuation["nonfoot"] == 0.
        displacement_speed = np.linalg.norm(com[end, :2] - com[begin, :2]) / ((end-begin)*.001)
        summary["straight_slip_formula_diagnostic_on_arc"] = float(
            max(native["turn"]["slip_absolute"]) / displacement_speed
        )
    return summary


if __name__ == "__main__":
    audit = read(HERE / "AUDIT_FINAL.json")
    assert audit["source_verified"] and audit["straight_and_standing_actions_bitwise"] == 2260
    expected = {f"{label}_{mode}" for label in ("NATIVE", "HELD") for mode in MODES}
    assert {Path(case["case"]).name for case in audit["cases"]} == expected
    bank = read(HERE / "BUILD.json")
    evidence = []
    for case in audit["cases"]:
        path = Path(case["case"])
        result = read(path / "RESULT.json")
        assert result["joint_pass"] and case["engineering_result"] and case["action_max_error"] == 0.
        assert case["tail_yaw_drift_deg"] <= .5
        assert all(result["cycles"])
        if result["mode"].startswith("arc"):
            assert case["resume"]["passed"]
        row = dict(case=path.name, start=result["start"], cycles=result["cycles"],
                   result_sha256=stand.sha(path / "RESULT.json"))
        if path.name.startswith("NATIVE"):
            native = read(path / "NATIVE.json")
            assert native["joint"] and native["all_endpoints_bitwise"] and native["extra_force_zero"]
            assert native["native_library_sha256"] == stand.sha(HERE / "native_contact.so")
            row["native"] = native_phases(path, result, native)
            row["native_summary_sha256"] = stand.sha(path / "NATIVE.json")
        evidence.append(row)
    visuals = {}
    for label, frame_count in (("VIDEO_PIVOT_VERIFIED", 566), ("VIDEO_ARC_VERIFIED", 1046)):
        manifest = read(HERE / label / "MANIFEST.json")
        assert manifest["decoded_frames"] == frame_count and manifest["physics_steps"] == 0
        assert stand.sha(Path(manifest["video"])) == manifest["file_sha256"]
        visuals[label] = manifest["file_sha256"]
    exits = {row["case"]: row["native"]["continuation"]["straight_slip_ratio"]
             for row in evidence if row["case"].startswith("NATIVE_arc_")}
    stand.put(HERE / "FINAL_ACCEPTANCE.json", dict(
        result="PASS",
        scope="flat-body discrete-command prototype, fixed turn window and arc entry",
        demonstration_samples=sum(entry["samples"] for entry in bank.values()),
        audit_sha256=stand.sha(HERE / "AUDIT_FINAL.json"),
        cases=evidence,
        additional_low_slip_exit=dict(
            result="PASS" if all(value <= .5 for value in exits.values()) else "FAIL",
            threshold=.5,
            straight_slip_ratio=exits,
        ),
        video_sha256=visuals,
    ))
