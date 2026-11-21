"""Audit passed-through commands and score side-by-side arena comparisons."""
from pathlib import Path

import numpy as np

from arena_trial import TRIAL, ORIGINAL, TIMING
from neural_runtime import DT, FIT, FIXED, read, sha, verify, write
from assess import relative
from repair_binding import CanonicalEvents


MODES = ("native_full", "reference_boundary", "translated", "repaired")
TEACHERS = ("forward_probe", "native_command", "repair_command")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def same(a, b):
    assert set(a) == set(b)
    return {key: bool(np.array_equal(a[key], b[key])) for key in a}


def starting_at(mask):
    indices = np.flatnonzero(mask)
    return float((indices[0]+1)*DT) if indices.size else None


def first_difference(a, b):
    return starting_at((a != b).reshape(len(a), -1).any(axis=1))


def connection_audit(result, neural):
    bound = read(FIT/"BINDING.json")["records"]
    parameters = read(FIXED/"PARAMETERS.json")["ports"]
    for group, roots, extra in (("an", [r["banc_root"] for r in bound], 0),
                                 ("dn", result["dn_roots"], 1)):
        for i, root in enumerate(roots):
            delay = parameters[root]["delay_ticks"] + extra
            emitted = neural[f"{group}_emitted"][:, :, i]
            arrived = neural[f"{group}_arrived"][:, :, i]
            assert not arrived[:delay].any()
            assert np.array_equal(arrived[delay:], emitted[:-delay])
    codec = CanonicalEvents((4, 2))
    for tick, arrivals in enumerate(neural["an_arrived"]):
        events = codec.step(arrivals)
        for lane in (1, 3):
            assert np.array_equal(events[lane], neural["brain_an_events"][tick, lane])
    return True


def physical(path):
    result = read(path/"RESULT.json")
    trace = path/"TRACE.npz"
    assert result["trace_sha256"] == sha(trace)
    return result, load(trace)


def batch(path):
    result = read(path/"RESULT.json")
    verify(result["source_sha256"])
    assert result["neural_sha256"] == sha(path/"NEURAL.npz")
    return result, load(path/"NEURAL.npz")


def main():
    rows = []
    total_neural_ticks = total_neural_lanes = total_body_steps = total_bodies = 0
    for seed in (10901, 11311):
        suite = TRIAL/"FORMAL"/f"seed{seed}"
        archived = ORIGINAL/"FORMAL"/f"seed{seed}"
        flat_repaired = TIMING/"FORMAL"/f"seed{seed}"
        result, brain = batch(suite)
        flat_neural = load(archived/"NEURAL.npz")
        flat_fixed_neural = load(flat_repaired/"NEURAL.npz")
        assert np.array_equal(brain["central_hits"], flat_neural["central_hits"][:len(brain["central_hits"])])
        assert connection_audit(result, brain)
        total_body_steps += result["native_body_steps"]
        total_bodies += result["body_episodes"]
        total_neural_ticks += result["brain_batch_ticks"]
        total_neural_lanes += result["brain_lane_ticks"]
        stats = {}
        brain_traces = {}
        flat = {}
        flat_timing = {}
        for lane, mode in enumerate(MODES):
            row, trace = physical(suite/mode)
            brain_traces[mode] = trace
            flat_row, _ = physical((flat_repaired if mode == "repaired" else archived)/mode)
            flat[mode] = {k: flat_row[k] for k in ("yaw_degrees", "forward_dx_mm", "safety_pass", "final_stand_pass")}
            terrain = trace["terrain_normal"]
            row["first_next_block_contact_s"] = starting_at(terrain[:, 1:3, :].max(axis=(1,2)) > 1e-6)
            stats[mode] = row
            expected = flat_fixed_neural["dn_emitted"][:, 0] if mode == "repaired" else flat_neural["dn_emitted"][:, lane]
            actual = brain["dn_emitted"][:, lane]
            flat_timing[mode] = first_difference(expected[:len(actual)], actual)
        pair_timing = {}
        for mode in ("native_full", "reference_boundary", "translated"):
            lane = MODES.index(mode)
            repair = brain_traces["repaired"]
            control = brain_traces[mode]
            pair_timing[mode] = dict(
                first_dn_difference_s=first_difference(brain["dn_emitted"][:,lane],brain["dn_emitted"][:,3]),
                first_command_difference_s=first_difference(control["command"],repair["command"]),
                first_physical_difference_s=first_difference(control["qpos"],repair["qpos"]),
                first_load_difference_s=first_difference(control["normal_load"],repair["normal_load"]))

        teachers = {}
        for name in TEACHERS:
            root = TRIAL/"TEACHER"/f"seed{seed}"/name
            row, trace = physical(root)
            verify(row["source_sha256"])
            assert row["model_sha256"] == sha(root/"BODY.mjb") == sha(suite/"repaired.mjb")
            if name != "forward_probe":
                mode = "native_full" if name == "native_command" else "repaired"
                assert np.array_equal(trace["command"], brain_traces[mode]["command"][:len(trace["command"])])
            teachers[name] = row
            total_body_steps += row["steps"]
            total_bodies += 1
        reference = {mode: relative(stats[mode], stats["repaired"])
                     for mode in ("native_full", "reference_boundary", "translated")}
        rows.append(dict(seed=seed, live=stats, teachers=teachers, flat=flat, repair_vs=reference,
                         first_DN_difference_vs_flat_s=flat_timing, repair_beginning_of_difference=pair_timing))
        for mode, row in {**stats, **teachers}.items():
            print(seed, mode, round(row["max_x_mm"],3), row["course_pass"],
                  row["bilateral_exit_pass"], row["final_stand_pass"], row["safety_pass"], flush=True)
    formal = TRIAL/"FORMAL/seed10901"
    replay = TRIAL/"REPLAY/seed10901"
    _, a = batch(formal)
    rep, b = batch(replay)
    total_body_steps += rep["native_body_steps"]
    total_bodies += rep["body_episodes"]
    total_neural_ticks += rep["brain_batch_ticks"]
    total_neural_lanes += rep["brain_lane_ticks"]
    parity = {"neural": same(a, b)}
    for mode in MODES:
        parity[mode] = same(physical(formal/mode)[1], physical(replay/mode)[1])
    assert all(v for d in parity.values() for v in d.values())
    write(TRIAL/"FINAL_ARENA.json", dict(project="P1_open_source_brain", rows=rows, exact_replay=parity,
          body_episodes=total_bodies, body_steps=total_body_steps,
          brain_batch_ticks=total_neural_ticks, brain_lane_ticks=total_neural_lanes,
          neck_batch_ticks=total_neural_ticks, neck_lane_ticks=total_neural_lanes, optimizer_updates=0))
    print("independent replay arrays",sum(len(x) for x in parity.values()),flush=True)


if __name__ == "__main__":
    main()
