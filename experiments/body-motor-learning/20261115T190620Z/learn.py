"""Fit and independently check a stateless residual from verified trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from bench import HERE, Neural, Push, SEED, cases, improvement, live_push, run, stand
from collect import read

FORMAT = "STAND_CURRENT_OBSERVATION_RESIDUAL_V1"
STAGE_ONE = 600
FINAL = 1000


class Residual(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.base = live_push.load().requires_grad_(False)
        self.correction = stand.StandPolicy(center, scale).double()

    def forward(self, x):
        return (self.base(x) + .48 * self.correction(x)).clamp(-.48, .48)


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != FORMAT or saved["base_sha256"] != live_push.REFERENCE_SHA:
        raise ValueError("Incompatible standing checkpoint")
    params = saved["correction"]
    model = Residual(params["center"], params["scale"])
    model.correction.load_state_dict(params)
    return model.eval()


def dataset(root, cohort):
    if cohort == "legacy":
        source = stand.HERE / "RUN"
        reports = read(source / "TEACHERS.json")
        rows = [(source / f"teacher_{int(k):02}.npz", v["saved_sha256"], True) for k, v in reports.items() if v["passed"]]
    else:
        source = root / cohort
        admission = read(source / "DATASET.json")
        reports = read(source / "RESULT.json")["evaluation"]
        rows = [(source / v["teacher"]["file"], v["teacher"]["trace_sha256"], False)
                for k, v in reports.items() if admission[k]["eligible"]]
    obs, actions, active = [], [], []
    manifest = []
    for path, digest, legacy in rows:
        if stand.sha(path) != digest:
            raise RuntimeError(f"Training trace changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            first = 0 if legacy else int(data["teacher_start_row"])
            x, y = data["x"][first:], data["action"][first:]
            early = np.arange(len(x)) < 50
            if not legacy and cohort == "train":
                # Loss sampling begins once the exogenous force can affect a reading.
                push = next(p for p in cases()[cohort] if path.name.startswith(p.name + "_"))
                early &= np.arange(len(x)) * 5 > push.onset_ms
            obs.append(x)
            actions.append(y)
            active.append(early)
            manifest.append(dict(path=str(path), sha256=digest, teacher_start=first, rows=len(x)))
    if not rows:
        return None, manifest
    x, y = (torch.from_numpy(np.concatenate(values)).double() for values in (obs, actions))
    active = torch.from_numpy(np.flatnonzero(np.concatenate(active)))
    return (x, y, active), manifest


def train(root, finish):
    start = time.monotonic()
    dest = root / "learn"
    dest.mkdir(exist_ok=finish)
    names = ["legacy", "train"]
    if finish:
        names.append("on")
    data = {}
    manifests = {}
    for name in names:
        data[name], manifests[name] = dataset(root, name)
    if data["train"] is None or len(manifests["train"]) < 4:
        raise RuntimeError("Insufficient verified standing demonstrations")
    lower, upper = (STAGE_ONE, FINAL) if finish else (0, STAGE_ONE)
    if finish and data["on"] is None:
        raise RuntimeError("No verified demonstrations on states visited by the first policy")
    saved_data = dest / f"DATA_{upper:04}.json"
    stand.put(saved_data, manifests)
    state = data["train"][0]
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    if finish:
        path = dest / f"step_{lower:04}.pt"
        old = torch.load(path, map_location="cpu", weights_only=True)
        model = load(path)
        rng.set_state(old["rng"])
    else:
        center = live_push.load().center
        scale = torch.cat((state, data["legacy"][0])).std(0).clamp_min(.004)
        model = Residual(center, scale)
    model.train()
    opt = torch.optim.Adam(model.correction.parameters(), lr=.0007)
    if finish:
        opt.load_state_dict(old["optimizer"])
        if any(int(state["step"]) != lower for state in opt.state.values()):
            raise ValueError("Optimizer count does not match the rollout checkpoint")

    @torch.no_grad()
    def mse():
        err = {}
        for name, (x, y, idx) in data.items():
            error = (model(x) - y).square().mean(1)
            err[name] = dict(full=float(error.mean()), early=float(error[idx].mean()), rows=len(x))
        return err

    sources = [Path(__file__), HERE / "bench.py", HERE / "collect.py", saved_data, live_push.CHECKPOINT,
               HERE / "PREREG.zh-CN.md"]
    if finish:
        sources.append(path)
    identity = {str(p): stand.sha(p) for p in sources}
    before = mse()
    effective = False
    journal_path = dest / f"JOURNAL_{upper:04}.jsonl"
    for step in range(lower, upper + 1):
        loss_value = None
        if step > lower:
            batch_x, batch_y = [], []
            counts = (64, 64, 64) if finish else (64, 128)
            for (x, y, prefer), size in zip(data.values(), counts, strict=True):
                any_idx = torch.randint(len(x), (size // 2,), generator=rng)
                active_idx = prefer[torch.randint(len(prefer), (size // 2,), generator=rng)]
                ids = torch.cat((any_idx, active_idx))
                batch_x.append(x[ids])
                batch_y.append(y[ids])
            loss = (model(torch.cat(batch_x)) - torch.cat(batch_y)).square().mean()
            loss_value = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1., error_if_nonfinite=True)
            effective |= float(grad) > 0.
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.correction.parameters()):
                raise RuntimeError("Optimizer produced nonfinite weights")
        if step == lower and finish:
            continue
        checkpoint = dest / f"step_{step:04}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save(dict(format=FORMAT, correction=model.correction.state_dict(), optimizer=opt.state_dict(),
                        rng=rng.get_state(), step=step, base_sha256=live_push.REFERENCE_SHA), temporary)
        temporary.rename(checkpoint)
        with journal_path.open("a") as journal:
            journal.write(json.dumps(dict(update=step, loss=loss_value, sha256=stand.sha(checkpoint)),
                                     allow_nan=False) + "\n")
    if not effective or any(int(state["step"]) != upper for state in opt.state.values()):
        raise RuntimeError("Optimizer did not complete all expected steps")
    for group, entries in manifests.items():
        for row in entries:
            if stand.sha(Path(row["path"])) != row["sha256"]:
                raise RuntimeError(f"{group} dataset changed")
    if any(stand.sha(Path(p)) != v for p, v in identity.items()):
        raise RuntimeError("Training inputs changed")
    stand.put(dest / f"TRAIN_{upper:04}.json", dict(source_sha256=identity, checkpoint_sha256=stand.sha(checkpoint),
              optimizer_updates=upper-lower, before=before, after=mse(), wall_s=time.monotonic()-start))
    print("trained", upper, stand.sha(checkpoint), mse(), flush=True)


def evaluate_worker(job):
    dest, worker, pushes, checkpoints = job
    torch.set_num_threads(1)
    output = dest / f"worker_{worker:02}"
    output.mkdir()
    bench = stand.Bench(output)
    controllers = dict(old=Neural(live_push.load()), zero=lambda x: bench.zero)
    controllers.update({arm: Neural(load(path)) for arm, path in checkpoints.items()})
    results = {}
    for push in pushes:
        pair = {}
        for arm, controller in controllers.items():
            path = output / f"{push.name}_{arm}.npz"
            metrics = run(bench, push, controller, path)
            metrics["file"] = str(path.relative_to(dest))
            pair[arm] = metrics
        results[push.name] = pair
        stand.put(output / f"{push.name}.json", pair)
        print(push.name, " ".join(f"{k}:{v['recovery_ms']}/{v['end_drift_body_lengths']:.4f}"
                                 for k, v in pair.items()), flush=True)
    stand.identity.verify_runtime(bench.manifest, bench.body)
    return results


def evaluate(root, workers):
    pushes = cases()["test"]
    dest = root / "test"
    dest.mkdir()
    checkpoints = {k: root / "learn" / f"step_{step:04}.pt" for k, step in
                   (("stage_one", STAGE_ONE), ("final", FINAL))}
    sources = [Path(__file__), HERE / "bench.py", HERE / "PREREG.zh-CN.md", live_push.CHECKPOINT,
               Path(live_push.__file__), Path(stand.__file__), *checkpoints.values()]
    digests = {str(p): stand.sha(p) for p in sources}
    stand.put(dest / "CONFIG.json", dict(case_names=[p.name for p in pushes], hashes=digests,
                                       checkpoints={name: str(path) for name, path in checkpoints.items()}))
    jobs = [(dest, i, pushes[i::workers], checkpoints) for i in range(min(workers, len(pushes)))]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        result = {key: pair for rows in pool.map(evaluate_worker, jobs) for key, pair in rows.items()}
    if any(stand.sha(Path(p)) != sha for p, sha in digests.items()):
        raise RuntimeError("Evaluation inputs changed")
    stand.put(dest / "RESULT.json", dict(evaluation=result, wall_s=time.monotonic() - start,
          settle_native_steps=80000 * len(jobs),
          trial_native_steps=sum(v["native_steps"] for pair in result.values() for v in pair.values())))


def arrays(path, sha):
    if stand.sha(path) != sha:
        raise RuntimeError(f"Trace changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def replay(root):
    dest = root / "test"
    config = read(dest / "CONFIG.json")
    for path, digest in config["hashes"].items():
        if stand.sha(Path(path)) != digest:
            raise RuntimeError("Fresh-process input identity mismatch")
    for end, start in ((STAGE_ONE, 0), (FINAL, STAGE_ONE+1)):
        with (root / "learn" / f"JOURNAL_{end:04}.jsonl").open() as journal:
            rows = [json.loads(line) for line in journal]
        assert [r["update"] for r in rows] == list(range(start, end + 1))
        for row in rows:
            assert stand.sha(root / "learn" / f"step_{row['update']:04}.pt") == row["sha256"]
    models = {"old": live_push.load(), **{k: load(Path(v)) for k, v in config["checkpoints"].items()}}
    cases_by_name = {p.name: p for p in cases()["test"]}
    metrics = read(dest / "RESULT.json")["evaluation"]
    errors = {}
    for name, pair in metrics.items():
        errors[name] = {}
        push = cases_by_name[name]
        baseline = arrays(dest / pair["old"]["file"], pair["old"]["trace_sha256"])
        for arm, summary in pair.items():
            z = arrays(dest / summary["file"], summary["trace_sha256"])
            with torch.no_grad():
                if arm == "zero":
                    out = np.zeros_like(z["action"])
                else:
                    out = models[arm](torch.from_numpy(z["x"][::-1].copy())).numpy()[::-1]
            errors[name][arm] = float(np.max(np.abs(out - z["action"])))
            np.testing.assert_allclose(out, z["action"], rtol=0., atol=1e-12)
            for key in ("initial_state", "wrench", "motor_injection"):
                np.testing.assert_array_equal(baseline[key], z[key])
            applied = z["action"][np.arange(len(z["applied"])) // 5].copy()
            applied[:30] += z["motor_injection"]
            np.testing.assert_array_equal(applied, z["applied"])
            row = live_push.measure(z, push.end, z["ref_target"], z["ref_joints"])
            if any(row[k] != summary[k] for k in row):
                raise RuntimeError("Trace metric mismatch")
    replay_path = dest / "physical_replay"
    replay_path.mkdir()
    bench = stand.Bench(replay_path)
    for name in ("test_03", "leg_pair"):
        push = cases_by_name[name]
        saved = metrics[name]["final"]
        output = replay_path / f"{name}.npz"
        current = run(bench, push, Neural(models["final"]), output)
        result = arrays(output, current["trace_sha256"])
        baseline = arrays(dest / saved["file"], saved["trace_sha256"])
        for key in baseline:
            np.testing.assert_array_equal(result[key], baseline[key])
    stand.identity.verify_runtime(bench.manifest, bench.body)
    stand.put(dest / "REPLAY.json", dict(all_verified=True, errors=errors,
              checked_checkpoints=FINAL+1, physical_exact=["test_03", "leg_pair"],
              physical_native_steps=80000 + sum(metrics[n]["final"]["native_steps"] for n in ("test_03", "leg_pair"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initial", "finish", "test", "replay"))
    parser.add_argument("--run", type=Path, default=HERE / "RUN")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.run.resolve()
    if args.stage == "test":
        evaluate(root, args.workers)
    elif args.stage == "replay":
        replay(root)
    else:
        train(root, args.stage == "finish")


if __name__ == "__main__":
    main()
