"""Check the live repair against the locked historic comparisons and replay."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BEHAVIOR = HERE.parent / "20261118T002600Z-ascending-behavior-comparison"
sys.path.insert(0, str(BEHAVIOR))
from assess import arrays, relative
from neural_runtime import FIT, FIXED, read, sha, verify, write
from timing import CanonicalEvents

MODES = ("repaired", "fixed_canonical", "fixed_raw", "matched_reference_boundary")


def checked_batch(folder):
    meta = read(folder / "RESULT.json")
    verify(meta["source_sha256"])
    assert sha(folder / "NEURAL.npz") == meta["neural_sha256"]
    return meta, arrays(folder / "NEURAL.npz")


def checked_body(folder, mode):
    row = read(folder / mode / "RESULT.json")
    assert sha(folder / mode / "TRACE.npz") == row["trace_sha256"]
    return row, arrays(folder / mode / "TRACE.npz")


def difference_count(a, b):
    return int(np.count_nonzero(a != b))


def audit_transport(neural, meta, inputs):
    model = read(FIXED / "PARAMETERS.json")["ports"]
    front = [r["banc_root"] for r in read(FIT / "BINDING.json")["records"]]
    offset = {"dn": 1, "an": 0}
    for label, roots in (("dn", meta["dn_roots"]), ("an", front)):
        for index, root in enumerate(roots):
            ticks = model[root]["delay_ticks"] + offset[label]
            emitted = neural[f"{label}_emitted"][:, :, index]
            arrived = neural[f"{label}_arrived"][:, :, index]
            assert not arrived[:ticks].any()
            assert np.array_equal(emitted[:-ticks], arrived[ticks:])
    encoder = CanonicalEvents((4, 2))
    events = []
    for tick, arrivals in enumerate(neural["an_arrived"]):
        encoded = encoder.step(arrivals)
        encoded[2] = arrivals[2]
        events.append(encoded)
        assert np.array_equal(encoder.rate, neural["rates"][tick])
    assert np.array_equal(events, neural["brain_an_events"])
    assert inputs["matched_feedback_lanes"] == [0]
    return True


def main():
    rows = []
    total_bodies = total_batch = 0
    for seed in (10901, 11311):
        old = BEHAVIOR / "FORMAL" / f"seed{seed}"
        live = HERE / "FORMAL" / f"seed{seed}"
        meta, data = checked_batch(live)
        histmeta, hist = checked_batch(old)
        inp = read(live / "INPUTS.json")
        historic_inputs = read(old / "INPUTS.json")
        assert np.array_equal(data["central_hits"], hist["central_hits"])
        assert meta["dn_roots"] == histmeta["dn_roots"]
        assert set(inp["body_model_hashes"]) == set(historic_inputs["body_model_hashes"])
        assert inp["source_indices"] == historic_inputs["source_indices"]
        total_bodies += meta["body_episodes"]
        total_batch += meta["brain_batch_ticks"]
        transport = audit_transport(data, meta, inp)

        original, _ = checked_body(old, "native_full")
        reference, reference_body = checked_body(old, "reference_boundary")
        previous, _ = checked_body(old, "translated")
        old_comparisons = {name: relative(row, previous)
                           for name, row in (("native", original), ("controlled", reference))}
        control_neural = {key: bool(np.array_equal(data[key][:, 3], hist[key][:, 1]))
                          for key in ("brain_an_events", "dn_emitted", "dn_arrived", "population")}
        records = {}
        for index, mode in enumerate(MODES):
            row, body = checked_body(live, mode)
            ports = [meta["dn_roots"].index(root) for root in ("720575941500851362", "720575941535086424")]
            records[mode] = dict(yaw_degrees=row["yaw_degrees"],
                stand_pass=row["final_stand_pass"], advance_pass=row["advance_pass"],
                safety_pass=row["safety_pass"], forward_mm=row["forward_dx_mm"],
                final_stand_rms_mm_s=row["stand_rms_mm_s"],
                native=relative(original, row), controlled=relative(reference, row),
                full_body_exact_to_controlled=all(np.array_equal(body[k], reference_body[k]) for k in body),
                AN_event_differences_to_controlled=difference_count(data["brain_an_events"][:, index],
                                                                    hist["brain_an_events"][:, 1]),
                DN_event_differences_to_controlled=difference_count(data["dn_emitted"][:, index],
                                                                    hist["dn_emitted"][:, 1]),
                actual_DN_arrived_on_AN_feedback_ports=[int(data["dn_arrived"][:, index, p].sum()) for p in ports])
        assert all(control_neural.values())
        assert records["matched_reference_boundary"]["full_body_exact_to_controlled"]
        rows.append(dict(seed=seed, old=old_comparisons, prior_yaw={
            "native": original["yaw_degrees"], "controlled": reference["yaw_degrees"],
            "translated": previous["yaw_degrees"]}, transport_audit=transport,
            matched_boundary_exact=control_neural, modes=records))
        print(seed, [(mode, round(row["yaw_degrees"], 3), row["controlled"]["pass_"],
                      row["native"]["pass_"]) for mode, row in records.items()], flush=True)

    live = HERE / "FORMAL/seed10901"
    replay = HERE / "REPLAY/seed10901"
    firstmeta, first = checked_batch(live)
    secondmeta, second = checked_batch(replay)
    total_bodies += secondmeta["body_episodes"]
    total_batch += secondmeta["brain_batch_ticks"]
    parity = {key: bool(np.array_equal(first[key], second[key])) for key in first}
    body_parity = {}
    for mode in MODES:
        _, a = checked_body(live, mode)
        _, b = checked_body(replay, mode)
        body_parity[mode] = {key: bool(np.array_equal(a[key], b[key])) for key in a}
    assert all(parity.values()) and all(v for p in body_parity.values() for v in p.values())
    diagnosis = read(HERE / "TIMING_DIAGNOSIS.json")
    verify(diagnosis["source_sha256"])
    assert diagnosis["cut_passed"] and diagnosis["feedback_matches_fixed_conductance"]
    repaired = [row["modes"]["repaired"] for row in rows]
    relative_pass = [bool(m["controlled"]["pass_"] and m["stand_pass"] and m["advance_pass"]) for m in repaired]
    native_pass = [bool(m["native"]["pass_"] and m["stand_pass"] and m["advance_pass"]) for m in repaired]
    write(HERE / "FINAL_REPAIR.json", dict(project="P1_open_source_brain",
          rows=rows, controlled_pass_per_seed=relative_pass, native_pass_per_seed=native_pass,
          exact_replay=dict(neural=parity, body=body_parity), optimizer_updates=0,
          brain_batch_ticks=total_batch, brain_lane_ticks=total_batch*4, body_episodes=total_bodies,
          body_steps=total_bodies*20000, fixed_neck_live_batch_ticks=total_batch,
          fixed_neck_live_lane_ticks=total_batch*4, diagnostic_neck_lane_ticks=2*20000+2*250,
          cut_passed=True, returned_DN_conductance_verified=True))
    print("independent replay exact", "controlled", relative_pass, "native", native_pass)


if __name__ == "__main__":
    main()
