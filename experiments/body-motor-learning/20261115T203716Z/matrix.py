"""Evaluate unchanged standing controllers from independent physical snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch

from snapshots import CHECKPOINT, HERE, PREVIOUS, Neural, load, model, stand
from bench import Push, live_push, run
from collect import read
from learn import arrays
from native_contact import ContactLog, DT, W, evaluate

ARMS = ("restoring", "model")
FACTORS = (Push("quiet", (0., 0.), 0, onset_ms=0),
           Push("medium_x", (.50, 0.), 35, onset_ms=20),
           Push("medium_y", (0., .50), 35, onset_ms=20),
           Push("strong_x", (.80, 0.), 35, onset_ms=20),
           Push("strong_y", (0., .80), 35, onset_ms=20))
CONTROL_DATA = PREVIOUS / "RUN/TEST"


def cases(bank):
    result = [(initial, p) for initial in bank for p in FACTORS]
    recorded = read(CONTROL_DATA / "CONFIG.json")["tests"]
    result.extend(("nominal", Push(**row)) for row in recorded if row["name"] in ("n02", "n07"))
    return result


def trial(bench, push, policy, dest):
    bench.reset()
    steps = (push.end + 1200)*80
    recorder = ContactLog(bench.body, steps)
    ms_path = dest.with_suffix(".npz")
    bench.body.recorder = recorder
    try:
        old = run(bench, push, policy, ms_path)
    finally:
        bench.body.recorder = None
    native = recorder.arrays()
    ms = arrays(ms_path, old["trace_sha256"])
    row = evaluate(native, ms, push.end)
    p = dest.with_suffix(".native.npz")
    np.savez_compressed(p, **native)
    wrench_impulse = ms["wrench"].sum(0) * .0005 / W
    return dict(legacy=old, event=row, native_sha256=stand.sha(p),
                impulse_div_weight=wrench_impulse.tolist(), file=dest.name,
                steps=steps)


def run_worker(job):
    output, worker, bank_path, casespec, identifiers = job
    torch.set_num_threads(1)
    dest = output / f"worker_{worker:02}"
    dest.mkdir()
    bench = stand.Bench(dest)
    frozen = model()
    controller = dict(restoring=Neural(frozen.restoring), model=Neural(frozen))
    result = {}
    for initial, p in casespec:
        load(bench, bank_path / f"{initial}.npz", identifiers[initial]["sha256"])
        pair = {}
        prepared = f"{initial}__{p.name}"
        for arm, policy in controller.items():
            report = trial(bench, p, policy, dest / f"{prepared}_{arm}")
            report["file"] = f"{dest.name}/{report['file']}"
            pair[arm] = report
        result[prepared] = pair
        stand.put(dest / f"{prepared}.json", pair)
        print(prepared, " ".join(f"{a}:{r['event']['behavior']}/{r['event']['recovery_ms']}"
                                 for a,r in pair.items()), flush=True)
    stand.identity.verify_runtime(bench.manifest, bench.body)
    return result


def launch(root, workers):
    dest = root / "MATRIX"
    dest.mkdir()
    start = time.monotonic()
    bank_path = root / "INITIAL"
    bank = read(bank_path / "BANK.json")
    case_list = cases(bank)
    source_paths = [Path(__file__), HERE / "native_contact.py", HERE / "snapshots.py", HERE / "PREREG.zh-CN.md",
                    CHECKPOINT, PREVIOUS / "bounded.py", bank_path / "BANK.json",
                    Path(stand.__file__), Path(sys.modules["bench"].__file__),
                    CONTROL_DATA / "CONFIG.json", CONTROL_DATA / "RESULT.json"]
    identities = {str(p): stand.sha(p) for p in source_paths}
    stand.put(dest / "CONFIG.json", dict(cases=[dict(start=k, push=asdict(p)) for k,p in case_list],
                                       sha256=identities, control_sha256=stand.sha(CHECKPOINT),
                                       worker_count=workers, native_dt_s=DT))
    jobs = [(dest, i, bank_path, case_list[i::workers], bank) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = {k: v for group in pool.map(run_worker, jobs) for k, v in group.items()}
    if any(stand.sha(Path(p)) != sha for p,sha in identities.items()):
        raise RuntimeError("Frozen run input changed")
    stand.put(dest / "RESULT.json", dict(cases=result, native_steps=sum(r["steps"] for pair in result.values() for r in pair.values()),
                                       worker_settle_steps=workers*80000, wall_s=time.monotonic()-start))


def assert_sparse(native):
    events = native["nonfoot"]
    steps = events[:, 0].astype(int)
    body = events[:, 1].astype(int)
    z = np.maximum(events[:, 8], 0.)
    nonfoot = np.bincount(steps, weights=z, minlength=len(native["foot"]))
    trunk = native["trunk_body"][body]
    body_force = np.bincount(steps[trunk], weights=z[trunk], minlength=len(native["foot"]))
    np.testing.assert_array_equal(nonfoot, native["nonfoot_normal"])
    np.testing.assert_array_equal(body_force, native["trunk_normal"])


@torch.no_grad()
def replay(root):
    dest = root / "MATRIX"
    bank_path = root / "INITIAL"
    config = read(dest / "CONFIG.json")
    if not all(stand.sha(Path(p)) == v for p,v in config["sha256"].items()):
        raise RuntimeError("Frozen matrix inputs changed before replay")
    bank = read(bank_path / "BANK.json")
    assert stand.identity.verify(read(bank_path / "IDENTITY.json"))
    for path in sorted(dest.glob("worker_*/IDENTITY.json")):
        assert stand.identity.verify(read(path))
    old = read(CONTROL_DATA / "RESULT.json")["cases"]
    weights = model()
    arms = dict(restoring=Neural(weights.restoring), model=Neural(weights))
    results = read(dest / "RESULT.json")["cases"]
    spec = {f"{r['start']}__{r['push']['name']}": (r["start"], Push(**r["push"])) for r in config["cases"]}
    greatest = 0.
    for initial, item in bank.items():
        if item["mother_sha256"] is not None:
            assert stand.sha(bank_path / f"{initial}_mother.npz") == item["mother_sha256"]
    for key, pair in results.items():
        initial, p = spec[key]
        z = arrays(bank_path / f"{initial}.npz", bank[initial]["sha256"])
        matched = np.r_[z["observation"], z["qpos"], z["qvel"], z["act_target_rad"]]
        reference = None
        for arm, record in pair.items():
            path = dest / record["file"]
            ms = arrays(path.with_suffix(".npz"), record["legacy"]["trace_sha256"])
            native = arrays(path.with_suffix(".native.npz"), record["native_sha256"])
            assert_sparse(native)
            expected = arms[arm](ms["x"][::-1].copy())[::-1]
            error = float(np.abs(expected-ms["action"]).max())
            greatest = max(greatest,error)
            np.testing.assert_allclose(expected, ms["action"], atol=1e-12, rtol=0.)
            np.testing.assert_array_equal(ms["initial_state"], matched)
            np.testing.assert_array_equal(native["qpos_ms"][0], z["qpos"])
            np.testing.assert_array_equal(native["qvel_ms"][0], z["qvel"])
            np.testing.assert_array_equal(ms["applied"], ms["action"][np.arange(len(ms["applied"]))//5])
            if reference is None:
                reference = ms["wrench"]
            else:
                np.testing.assert_array_equal(reference, ms["wrench"])
            if p.name in ("n02", "n07"):
                prior = old[p.name][arm]
                historical = arrays(CONTROL_DATA / prior["path"], prior["trace_sha256"])
                for name, values in historical.items():
                    np.testing.assert_array_equal(values, ms[name])
            evaluated = evaluate(native, ms, p.end)
            if json.loads(json.dumps(evaluated)) != record["event"]:
                raise AssertionError("Native event replay changed")
            old_metrics = live_push.measure(ms, p.end, ms["ref_target"], ms["ref_joints"])
            assert all(old_metrics[k] == record["legacy"][k] for k in old_metrics)
    replay_folder = dest / "PHYSICAL_REPLAY"
    replay_folder.mkdir()
    bench = stand.Bench(replay_folder)
    replay_steps = 80000
    selection = ("nominal__n07", "lean_right__medium_y")
    for key in selection:
        initial, p = spec[key]
        path = replay_folder / key
        load(bench, bank_path / f"{initial}.npz", bank[initial]["sha256"])
        r = trial(bench, p, arms["model"], path)
        replay_steps += r["steps"]
        recorded = results[key]["model"]
        for suffix, record_sha, replay_sha in ((".npz", recorded["legacy"]["trace_sha256"], r["legacy"]["trace_sha256"]),
                (".native.npz", recorded["native_sha256"], r["native_sha256"])):
            baseline = arrays((dest / recorded["file"]).with_suffix(suffix), record_sha)
            actual = arrays(path.with_suffix(suffix), replay_sha)
            for field, expected in baseline.items():
                np.testing.assert_array_equal(expected, actual[field])
    stand.identity.verify_runtime(bench.manifest, bench.body)
    stand.put(dest / "REPLAY.json", dict(pairs=len(results), trajectories=len(results)*len(ARMS),
        historical_exact_pairs=["n02", "n07"], maximum_action_error=greatest,
        fresh_exact_cases=selection, native_replay_steps=replay_steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "replay"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "run":
        launch(HERE / "RUN", args.workers)
    else:
        replay(HERE / "RUN")


if __name__ == "__main__":
    main()
