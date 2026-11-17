"""Recurrent behavior cloning from verified, state-aligned body demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from policy import BodyPolicy, FORMAT

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WALKING = PROJECT / "experiments/banc-walking-recovery/20261110T152754Z"
OLD_MOTOR = WALKING.parent / "20261110T152714Z"
MOTION = PROJECT / "experiments/locomotion-enhancement/20261113T161918Z"
CHECKPOINT_SHA = "246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809"
SEED = 20261115
WINDOW = 192
BURN = 64
UPDATES = 480


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def put(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bind(path, ledger, expected=None):
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen identity mismatch for {path}")
    ledger[str(path)] = digest
    return digest


def data():
    ledger = {}
    checkpoint = OLD_MOTOR / "BC_GRU_ROLE_3000/checkpoint_002500.pt"
    bind(checkpoint, ledger, CHECKPOINT_SHA)
    acceptance = WALKING / "FINAL_ACCEPTANCE.json"
    bind(acceptance, ledger)
    if read(acceptance)["status"] != "PASS":
        raise ValueError("Missing walk qualification")
    refs = [
        ("walk", WALKING / "NATIVE_ROLE_2500", "neural_decisions.npz", "neural", "actually_applied_action", "control_source"),
        ("walk_seed11", WALKING / "PERTURB_ROLE_2500_SEED_20261111", "decisions_5ms.npz", "decisions", "action", "source"),
        ("walk_seed12_heldout", WALKING / "PERTURB_ROLE_2500_SEED_20261112", "decisions_5ms.npz", "decisions", "action", "source"),
    ]
    refs.extend((task, directory / "trials" / task, "teacher_samples_5ms.npz", "teacher_samples", "Y42", "control_source")
                for task, directory in (("stop", MOTION / "stop_shadow_control"),
                                        ("left_pivot", MOTION / "pivot_fixed_shadow_v2"),
                                        ("right_pivot", MOTION / "pivot_fixed_shadow_v2")))
    episodes = {}
    for name, folder, filename, metadata, action_key, source_key in refs:
        source = folder / filename
        result_file = folder / "RESULT.json"
        result = read(result_file)
        result_sha = bind(result_file, ledger)
        bind(source, ledger, result[metadata]["sha256"])
        if metadata == "teacher_samples":
            proof = folder / "VERIFY.json"
            bind(proof, ledger)
            audit = read(proof)
            if not (audit["integrity_all_pass"] and audit["result_sha256"] == result_sha and result["full_duration"]):
                raise ValueError(f"Invalid physical demonstration {name}")
            if not all(result["candidate_metrics"]["gates"].values()):
                raise ValueError(f"Failed engineering teacher {name}")
        with np.load(source, allow_pickle=False) as z:
            clock, x, y, kinds = (z[key] for key in ("time", "X238", action_key, source_key))
            if not (x.shape == (2880, 238) and y.shape == (2880, 42)
                    and np.allclose(np.diff(clock), .005, rtol=0, atol=1e-8)
                    and np.isfinite(x).all() and np.isfinite(y).all()
                    and np.abs(y).max() <= 1.000000000001):
                raise ValueError(f"Invalid arrays for {name}")
            if not np.array_equal(x[1:, 195:237], y[:-1]):
                raise ValueError(f"Previous action disagrees with the actual label for {name}")
            command = np.repeat([[1., 0.]], len(clock), axis=0)
            if metadata == "teacher_samples":
                candidate = z["candidate_phase"]
                if not np.array_equal(candidate, np.arange(len(clock)) >= 960):
                    raise ValueError(f"Wrong teacher transition for {name}")
                command = np.stack((z["requested_speed"] / .075, z["requested_yaw_rate"] / .15), -1)
            elif not (np.all(kinds[:960] == 2) and np.all(kinds[960:] == 3)):
                raise ValueError(f"Wrong authenticated walk control sources for {name}")
        episodes[name] = tuple(torch.from_numpy(a).double() for a in (x, command, y))
    return episodes, ledger


@torch.no_grad()
def metrics(model, episodes):
    report = {}
    for name, (x, command, action) in episodes.items():
        output, _ = model.sequence(x[:, None], command[:, None])
        error = output[:, 0] - action
        report[name] = dict(prefix_mse=float(error[:960].square().mean()),
                            task_mse=float(error[960:].square().mean()))
    return report


def batch(episodes, generator):
    windows = []
    for name, (x, u, action) in episodes.items():
        if name == "walk_seed12_heldout":
            continue
        # One window covers the switch, another covers subsequent physical behavior.
        for start in (int(generator.integers(760, 920)), int(generator.integers(960, len(x) - WINDOW + 1))):
            windows.append((x[start:start + WINDOW], u[start:start + WINDOW], action[start:start + WINDOW]))
    return [torch.stack(items, dim=1) for items in zip(*windows)]


def snapshot(path, model, optimizer, step, generator, source_hashes):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, numpy_rng=generator.bit_generator.state, torch_rng=torch.get_rng_state(),
                    source_hashes=source_hashes), temporary)
    temporary.rename(path)
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    args.output.mkdir(parents=True, exist_ok=False)
    episodes, ledger = data()
    for name in ("policy.py", "train.py", "evaluate.py", "PREREG.zh-CN.md"):
        bind(HERE / name, ledger)
    put(args.output / "INPUTS.json", dict(source_hashes=ledger, seed=SEED, budget_updates=UPDATES,
                                          python=sys.executable, torch=torch.__version__, numpy=np.__version__))
    samples = torch.cat([episode[0] for name, episode in episodes.items() if name != "walk_seed12_heldout"])
    model = BodyPolicy(samples.mean(0), samples.std(0).clamp_min(.015)).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    started = time.monotonic()
    print("initial", json.dumps(metrics(model, episodes)), flush=True)

    with (args.output / "TRAINING.jsonl").open("x", buffering=1) as journal:
        for step in range(1, UPDATES + 1):
            x, command, y = batch(episodes, rng)
            with torch.no_grad():
                _, state = model.sequence(x[:BURN], command[:BURN])
            prediction, _ = model.sequence(x[BURN:], command[BURN:], state)
            loss = (prediction - y[BURN:]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if step == 1 and (not torch.isfinite(loss) or not grad > 0
                              or any(p.grad is None for p in model.parameters())):
                raise RuntimeError("BC gradient smoke check failed")
            optimizer.step()
            path = args.output / f"update_{step:04d}.pt"
            digest = snapshot(path, model, optimizer, step, rng, ledger)
            row = dict(step=step, loss=float(loss.detach()), grad=float(grad),
                       checkpoint=path.name, sha256=digest, wall_s=time.monotonic() - started)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            if step in (1, 30, 120, 240, 360, UPDATES):
                print(json.dumps(row), flush=True)
    if not all(sha(Path(p)) == h for p, h in ledger.items()):
        raise RuntimeError("Training input drift")
    put(args.output / "FIT.json", dict(updates=UPDATES, checkpoint=path.name, sha256=digest,
                                      metrics=metrics(model, episodes), wall_s=time.monotonic() - started))


if __name__ == "__main__":
    main()
