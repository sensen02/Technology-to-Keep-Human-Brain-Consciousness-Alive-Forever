"""Standing body control with current-target restoration and learned braking."""
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
from torch import nn

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "20261115T190620Z"
sys.path.insert(0, str(PREVIOUS))

from bench import Neural, Push, cases, live_push, run, stand
from collect import read
from learn import arrays, dataset

SEED = 2026111531
FINAL = 800


def tests():
    rng = np.random.default_rng(SEED)
    directions = np.arange(12) * np.pi / 6 + rng.uniform(-.27, .27, 12)
    pushes = []
    for i, angle in enumerate(directions):
        value = rng.uniform(.72, .89) if i % 3 == 2 else rng.uniform(.31, .69)
        force = tuple(value * np.array([np.cos(angle), np.sin(angle)]))
        pushes.append(Push(f"n{i:02}", force, int(rng.integers(17, 45)), float(rng.uniform(-.28, .28)),
                           int(rng.integers(5, 31))))
    return pushes + cases()["test"][-3:]


class Body(nn.Module):
    def __init__(self, center, scale, half):
        super().__init__()
        self.correction = stand.StandPolicy(center, scale).double()
        self.register_buffer("joint_half_range", half)

    def restoring(self, x):
        err = (self.correction.center[126:168] - x[..., 126:168]) * self.joint_half_range
        return (err * (12. / stand.physics.MAX_RATE)).clamp(-.48, .48)

    def forward(self, x):
        motion = x[..., 177:183] - self.correction.center[177:183]
        energy = motion[..., :3].square().sum(-1) + .36 * motion[..., 3:].square().sum(-1)
        gate = -torch.expm1(-energy / .25)
        return (self.restoring(x) + .42 * gate[..., None] * self.correction(x)).clamp(-.48, .48)


def load(path):
    chk = torch.load(path, map_location="cpu", weights_only=True)
    state = chk["body"]
    model = Body(state["correction.center"], state["correction.scale"], state["joint_half_range"])
    model.load_state_dict(state)
    return model.eval()


def train(root):
    root.mkdir()
    train_dir = root / "TRAIN"
    train_dir.mkdir()
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    data, identities = zip(*(dataset(PREVIOUS / "RUN", name) for name in ("legacy", "train", "on")))
    stand.put(root / "DATA.json", dict(zip(("legacy", "train", "on"), identities)))
    core = stand.physics.Core(adhesion=False)
    identity = stand.identity.capture(core, dict(scope="stand current target restoration ranges"),
                                      extra_paths=[Path(__file__), HERE / "PREREG.zh-CN.md",
                                                   PREVIOUS / "bench.py", PREVIOUS / "learn.py"])
    stand.put(root / "IDENTITY.json", identity)
    half = torch.from_numpy(np.ptp(core.plant.jnt_range[core.actuator.joint_ids], axis=1) / 2.).clamp_min(.1)
    center = live_push.load().center
    model = Body(center, torch.cat([g[0] for g in data]).std(0).clamp_min(.004), half)
    opt = torch.optim.Adam(model.correction.parameters(), lr=.0007)

    @torch.no_grad()
    def losses():
        return [float((model(x) - y).square().mean()) for x, y, _ in data]

    before = losses()
    with (train_dir / "JOURNAL.jsonl").open("x", buffering=1) as journal:
        for step in range(FINAL+1):
            value = None
            if step:
                observations, actions = [], []
                for x, y, early in data:
                    ids = torch.cat((torch.randint(len(x), (32,), generator=gen),
                                     early[torch.randint(len(early), (32,), generator=gen)]))
                    observations.append(x[ids])
                    actions.append(y[ids])
                loss = (model(torch.cat(observations)) - torch.cat(actions)).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                value = float(loss.detach())
            path = train_dir / f"step_{step:04}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(body=model.state_dict(), optimizer=opt.state_dict(), rng=gen.get_state(), step=step), temporary)
            temporary.rename(path)
            journal.write(json.dumps(dict(step=step, loss=value, sha256=stand.sha(path)), allow_nan=False) + "\n")
    for group in identities:
        for row in group:
            if stand.sha(Path(row["path"])) != row["sha256"]:
                raise RuntimeError("Training data changed")
    assert all(int(state["step"]) == FINAL for state in opt.state.values())
    assert losses()[1] < before[1]
    stand.identity.verify_runtime(identity, core)
    stand.put(root / "TRAIN.json", dict(before=before, after=losses(), updates=FINAL, sha256=stand.sha(path)))
    print("trained", FINAL, before, losses(), flush=True)


def controls(checkpoint):
    candidate = load(checkpoint)
    return dict(old=Neural(live_push.load()), restoring=Neural(candidate.restoring), model=Neural(candidate))


def evaluate_job(job):
    output, pushes, checkpoint = job
    torch.set_num_threads(1)
    output.mkdir()
    bench = stand.Bench(output)
    actions = controls(checkpoint)
    rows = {}
    for push in pushes:
        results = {}
        for name, ctrl in actions.items():
            path = output / f"{push.name}_{name}.npz"
            m = run(bench, push, ctrl, path)
            results[name] = dict(**m, path=str(path.relative_to(output.parent)))
        rows[push.name] = results
        print(push.name, " ".join(f"{k}:{v['recovered']}/{v['recovery_ms']}/{v['end_drift_body_lengths']:.3f}"
                                 for k, v in results.items()), flush=True)
    stand.identity.verify_runtime(bench.manifest, bench.body)
    return rows


def evaluate(root, workers):
    dest = root / "TEST"
    dest.mkdir()
    checkpoint = root / "TRAIN" / f"step_{FINAL:04}.pt"
    peek = [checkpoint, Path(__file__), HERE / "PREREG.zh-CN.md", PREVIOUS / "bench.py",
            Path(stand.__file__), Path(live_push.__file__), live_push.CHECKPOINT]
    ids = {str(p): stand.sha(p) for p in peek}
    pushes = tests()
    stand.put(dest / "CONFIG.json", dict(identity=ids, tests=[asdict(p) for p in pushes]))
    jobs = [(dest / f"worker_{i:02}", pushes[i::workers], checkpoint) for i in range(workers)]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        m = {k: v for rows in pool.map(evaluate_job, jobs) for k, v in rows.items()}
    assert all(stand.sha(Path(p)) == sha for p, sha in ids.items())
    stand.put(dest / "RESULT.json", dict(cases=m, workers=workers, wall_s=time.monotonic()-start,
            native_steps=workers * 80000 + sum(v["native_steps"] for row in m.values() for v in row.values())))


@torch.no_grad()
def replay(root):
    dest = root / "TEST"
    cfg = read(dest / "CONFIG.json")
    assert all(stand.sha(Path(p)) == sha for p, sha in cfg["identity"].items())
    with (root / "TRAIN/JOURNAL.jsonl").open() as f:
        j = [json.loads(l) for l in f]
    assert [r["step"] for r in j] == list(range(FINAL+1))
    for r in j:
        assert stand.sha(root / "TRAIN" / f"step_{r['step']:04}.pt") == r["sha256"]
    m = read(dest / "RESULT.json")["cases"]
    ctrl = controls(root / "TRAIN" / f"step_{FINAL:04}.pt")
    pushes = {p.name: p for p in tests()}
    maximum = 0.
    for key, row in m.items():
        ref = arrays(dest / row["old"]["path"], row["old"]["trace_sha256"])
        for arm, entry in row.items():
            z = arrays(dest / entry["path"], entry["trace_sha256"])
            out = ctrl[arm](z["x"][::-1].copy())[::-1]
            maximum = max(maximum, float(np.abs(out-z["action"]).max()))
            np.testing.assert_allclose(out, z["action"], atol=1e-12, rtol=0.)
            for field in ("initial_state", "wrench", "motor_injection"):
                np.testing.assert_array_equal(ref[field], z[field])
            applied = z["action"][np.arange(len(z["applied"])) // 5].copy()
            applied[:30] += z["motor_injection"]
            np.testing.assert_array_equal(applied, z["applied"])
            metrics = live_push.measure(z, pushes[key].end, z["ref_target"], z["ref_joints"])
            assert all(metrics[k] == entry[k] for k in metrics)
    folder = dest / "physical_replay"
    folder.mkdir()
    bench = stand.Bench(folder)
    physical_steps = 80000
    for key in ("n01", "leg_pair"):
        output = folder / f"{key}.npz"
        rerun = run(bench, pushes[key], ctrl["model"], output)
        new = arrays(output, rerun["trace_sha256"])
        old = m[key]["model"]
        physical_steps += rerun["native_steps"]
        for field, values in arrays(dest / old["path"], old["trace_sha256"]).items():
            np.testing.assert_array_equal(values, new[field])
    stand.identity.verify_runtime(bench.manifest, bench.body)
    stand.put(dest / "REPLAY.json", dict(verified=True, maximum_action_error=maximum,
              checkpoints_checked=len(j), native_steps=physical_steps, exact_cases=["n01", "leg_pair"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "eval", "replay"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(1)
    root = HERE / "RUN"
    if args.stage == "train":
        train(root)
    elif args.stage == "eval":
        evaluate(root, args.workers)
    else:
        replay(root)


if __name__ == "__main__":
    main()
