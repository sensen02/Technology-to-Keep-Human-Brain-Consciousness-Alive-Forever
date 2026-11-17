"""Screen and collect physically verified standing recovery demonstrations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import torch

from bench import GAINS, HERE, Neural, Teacher, cases, eligible, improvement, live_push, run, stand


def read(path):
    return json.loads(path.read_text())


def work(job):
    output, worker, pushes, names, gain, policy_path = job
    torch.set_num_threads(1)
    group = output / f"worker_{worker:02}"
    group.mkdir(exist_ok=False)
    bench = stand.Bench(group)
    neural = Neural(live_push.load())
    use_student = policy_path is not None
    if use_student:
        from learn import load
        neural = Neural(load(policy_path))
    controls = {"old": neural, "zero": lambda x: bench.zero}
    for name in names:
        if name in GAINS or name == "teacher":
            controls[name] = Teacher(bench, gain if name == "teacher" else name)
    gathered = {}
    for push in pushes:
        row = {}
        switch_ms = 15 + 5 * (int(push.name.split("_")[-1]) % 8) * 2 if use_student and push.name.startswith("on_") else 0
        for name in names:
            prefix = neural if switch_ms and name == "teacher" else None
            dest = group / f"{push.name}_{name}.npz"
            report = run(bench, push, controls[name], dest, prefix=prefix,
                         switch_ms=switch_ms if prefix is not None else 0)
            report["file"] = str(dest.relative_to(output))
            row[name] = report
        gathered[push.name] = row
        print(push.name, " ".join(f"{k}:{v['recovery_ms']}/{v['end_drift_body_lengths']:.3f}"
                                  for k, v in row.items()), flush=True)
        stand.put(group / f"{push.name}.json", row)
    stand.identity.verify_runtime(bench.manifest, bench.body)
    return gathered


def collect(output, pushes, names, gain, workers, policy_path=None):
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    sources = [HERE / name for name in ("PREREG.zh-CN.md", "bench.py", "collect.py")]
    sources += [Path(m.__file__) for m in (live_push, stand)] + [live_push.CHECKPOINT]
    if policy_path is not None:
        sources.extend([policy_path, HERE / "learn.py"])
    identity = {str(p): stand.sha(p) for p in sources}
    stand.put(output / "CONFIG.json", dict(sources=identity, cases=[asdict(p) for p in pushes], arms=names,
                                          teacher=gain, workers=workers))
    batches = [pushes[i::workers] for i in range(workers)]
    jobs = [(output, i, group, names, gain, policy_path) for i, group in enumerate(batches) if group]
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        row = {name: data for group in pool.map(work, jobs) for name, data in group.items()}
    if any(stand.sha(Path(p)) != sha for p, sha in identity.items()):
        raise RuntimeError("Collection inputs changed")
    stand.put(output / "RESULT.json", dict(evaluation=row, settle_native_steps=len(jobs)*80000,
        total_trial_native_steps=sum(v["native_steps"] for pair in row.values() for v in pair.values()),
        wall_s=time.monotonic()-start))
    return row


def qualify(screen):
    ranked = {}
    for name in GAINS:
        pairs = [(r[name], r["old"]) for r in screen.values()]
        timings = [a["recovery_ms"] - b["recovery_ms"] for a, b in pairs if a["recovered"] and b["recovered"]]
        drift = [a["end_drift_body_lengths"] / max(.001, b["end_drift_body_lengths"])
                 for a, b in pairs if a["recovered"] and b["recovered"]]
        ranked[name] = (
            sum(a["recovered"] for a, _ in pairs), sum(improvement(a, b) for a, b in pairs),
            -sum(drift) / max(1, len(drift)), -sum(timings) / max(1, len(timings))
        )
    return max(ranked, key=ranked.get), ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "train", "on", "test"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run", type=Path, default=HERE / "RUN")
    args = parser.parse_args()
    root = args.run.resolve()
    output = root / args.stage
    root.mkdir(exist_ok=True)
    pushes = cases()[args.stage]
    if args.stage == "dev":
        data = collect(output, pushes, ("old", *GAINS), None, args.workers)
        gain, ranked = qualify(data)
        stand.put(output / "TEACHER.json", dict(gain=gain, ranked=ranked))
        print("teacher", gain, ranked, flush=True)
        return
    selection = read(root / "dev/TEACHER.json")
    gain = selection["gain"]
    if selection["ranked"][gain][1] < 1:
        raise RuntimeError("Teacher lacks demonstrated physical recovery improvement")
    if args.stage in ("train", "on"):
        model = root / "learn/step_0600.pt" if args.stage == "on" else None
        data = collect(output, pushes, ("old", "teacher"), gain, args.workers, model)
        seats = {name: dict(eligible=eligible(row["teacher"], row["old"]),
                           improves=improvement(row["teacher"], row["old"])) for name, row in data.items()}
        stand.put(output / "DATASET.json", seats)
        print("admitted", sum(v["eligible"] for v in seats.values()), "of", len(seats), flush=True)
        return
    raise NotImplementedError("Use learn.py to verify saved final checkpoint on the test group")


if __name__ == "__main__":
    main()
