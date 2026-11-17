"""Supervise a local gap correction from verified same-body physical teachers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gap_policy import (
    Correction, FORMAT, HERE, LEG_ORDER, WALK_INPUTS, base_sources, body_indices, checksum,
    features, load, prototype, read, write,
)

DOFS = 42


def verified_files(path):
    files = {
        "result": path / "RESULT.json",
        "trace": path / "TRACE.npz",
        "decision": path / "DECISIONS.npz",
    }
    result = read(files["result"])
    hashes = {key: checksum(file) for key, file in files.items()}
    for key in ("trace", "decision"):
        if result[f"{key}_sha256"] != hashes[key]:
            raise ValueError(f"Trace/decision hash differs for {path}")
    return result, files, hashes


def load_decisions(path):
    with np.load(path / "DECISIONS.npz", allow_pickle=False) as z:
        local = "local" if "local" in z else "sensors"
        x, y, intent, action, base, ms = (z[k] for k in ("x", local, "intent", "action", "base", "ms"))
    n = len(intent)
    if (
        x.shape != (n, WALK_INPUTS) or y.shape[0] != n or y.ndim != 2
        or intent.shape != (n,) or action.shape != base.shape or base.shape != (n, DOFS)
        or any(not np.isfinite(arr).all() for arr in (x, y, action, base))
        or np.max(np.abs(action)) > 1. or np.max(np.abs(base)) > 1.
    ):
        raise ValueError(f"Unexpected decisions in {path}")
    np.testing.assert_array_equal(ms, np.arange(len(ms)) * 5)
    np.testing.assert_array_equal(intent.astype(bool), (ms >= 500) & (ms < 10100))
    return x, y, intent, action, base, ms


def admit(path, certification):
    result, files, hashes = verified_files(path)
    native_path = path / "NATIVE_AUDIT.json"
    raw_path = path / "NATIVE.npz"
    native = read(native_path)
    if (
        int(result["walk_ms"]) != 9600 or int(result["actual_ms"]) != 11300
        or int(native["native_integrations"]) != 904000
        or native["native_sha256"] != checksum(raw_path)
        or not native["zero_extra_force"] or not native["native_joint_pass"]
        or not result["joint_sampled_pass"]
    ):
        raise ValueError(f"Native gates or evidence failed in {path}")
    if certification:
        cert_path = path / "TEACHER_ACCEPTANCE.json"
        cert = read(cert_path)
        if cert["passed"] is not True or cert["role"] != "teacher":
            raise ValueError(f"No physically qualified teacher at {path}")
        for key, fingerprint in hashes.items():
            if cert[f"{key}_sha256"] != fingerprint:
                raise ValueError(f"Teacher certificate does not bind {key} in {path}")
        if cert["native_audit_sha256"] != checksum(native_path):
            raise ValueError(f"Teacher certificate does not bind native data in {path}")
        files["acceptance"] = cert_path
    elif result["mode"] != "control" or (result["scenario"] is not None and not result["scenario"]["filled"]):
        raise ValueError("Negative anchor must come from the original floor or matched filled scene")
    files.update(native=native_path, raw=raw_path)
    return result, {str(f): checksum(f) for f in files.values()}


def verify_base(records):
    policy = prototype.policy()
    x, _, intent, action, base, _ = records
    expected = np.asarray([policy(state, int(move)) for state, move in zip(x, intent, strict=True)])
    np.testing.assert_array_equal(base, expected)
    np.testing.assert_array_equal(action[~intent.astype(bool)], base[~intent.astype(bool)])


def fit(teacher, anchors, output, updates, seed, detector_floor):
    if HERE.parent not in teacher.parents or any(HERE.parent not in path.parents for path in anchors):
        raise ValueError("Fit uses current matched physical experiments")
    info, provenance = admit(teacher, True)
    sensor_fields = info["sensor_fields"]
    anchor_runs = []
    for floor in anchors:
        flat_info, flat_provenance = admit(floor, False)
        for key in ("initial_snapshot_sha256", "sensor_fields"):
            if info[key] != flat_info[key]:
                raise ValueError(f"The intact anchor differs in {key}")
        if flat_info["scenario"] is not None:
            if flat_info["scenario"] != dict(info["scenario"], filled=True):
                raise ValueError("The filled scene does not match the teacher opening")
        provenance.update(flat_provenance)
        anchor = load_decisions(floor)
        verify_base(anchor)
        np.testing.assert_array_equal(anchor[3], anchor[4])
        anchor_runs.append(anchor)
    for entry in (Path(__file__), HERE / "gap_policy.py", HERE.parent / "teacher" / "physical.py"):
        provenance[str(entry.resolve())] = checksum(entry)
    demonstration = load_decisions(teacher)
    verify_base(demonstration)
    x, sensed, moving, action, base, _ = demonstration
    if len(sensor_fields) != sensed.shape[1]:
        raise ValueError("Sensor descriptions differ from actual samples")
    move = moving.astype(bool)
    delta = action[move] - base[move]
    active = (np.abs(delta) > 1e-9).any(axis=1)
    if not active.any() or active.all():
        raise ValueError("Teacher must demonstrate a bounded selective intervention")
    channels = np.flatnonzero((np.abs(delta) > 1e-9).any(axis=0))
    named_legs = [LEG_ORDER[k] for k in np.unique(channels // 7)]
    if "RF" not in named_legs:
        raise ValueError("Study requires actual affected RF correction")
    indices = body_indices(named_legs)
    inputs = features(x[move], sensed[move], indices)
    intact = np.concatenate([features(cx[cmoving.astype(bool)], cy[cmoving.astype(bool)], indices)
                             for cx, cy, cmoving, *_ in anchor_runs])
    combined = np.concatenate((inputs, intact))
    bounds = np.clip(np.max(np.abs(delta[:, channels]), axis=0) * 1.035, .035, 2)
    torch.manual_seed(seed)
    model = Correction(combined.mean(0), np.maximum(combined.std(0), .006), bounds)
    optim = torch.optim.Adam(model.parameters(), lr=.002)
    rng = np.random.default_rng(seed)
    tx, ax, truth = (torch.as_tensor(a, dtype=torch.float32) for a in (inputs, intact, delta[:, channels]))
    joined = torch.cat((tx, ax))
    positive = np.flatnonzero(active)
    negative = np.flatnonzero(~active)
    edge = negative[(negative >= positive[0]-30) & (negative <= positive[-1]+30)]
    local_times = np.arange(max(0, positive[0]-30), min(len(tx), positive[-1]+31))
    local_anchor = (np.arange(len(anchor_runs))[:, None] * len(tx) + local_times).ravel()
    output.mkdir()
    source = base_sources()
    args = dict(teacher=str(teacher), intact_anchors=[str(path) for path in anchors], provenance=provenance, base_sources=source,
                seed=seed, updates=updates, sensor_fields=sensor_fields, legs=named_legs,
                channels=channels.tolist(), body_indices=indices.tolist(), feature_count=combined.shape[1],
                active_teacher_frames=len(positive), teacher_walking_frames=len(inputs),
                intact_walking_frames=len(intact), max_residual_per_channel=bounds.tolist())
    write(output / "INPUTS.json", args)
    with (output / "JOURNAL.jsonl").open("x") as log:
        for step in range(updates + 1):
            status = {}
            if step:
                p = rng.choice(positive, 48)
                near = rng.choice(edge if len(edge) else negative, 24)
                n = np.r_[near, rng.choice(negative, 24), len(tx) + rng.choice(local_anchor, 28),
                          len(tx) + rng.integers(len(ax), size=24)]
                plogit, estimate = model(tx[p])
                nlogit, passive = model(joined[n])
                detection = F.softplus(-plogit).mean() + F.softplus(nlogit).mean()
                action_error = F.smooth_l1_loss(estimate, truth[p], beta=.045)
                rest = passive.square().mean()
                optim.zero_grad(set_to_none=True)
                (action_error + .23*detection + .025*rest).backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise RuntimeError("Nonfinite supervised gradients")
                optim.step()
                status = dict(detection=float(detection.detach()), regression=float(action_error.detach()),
                              inactive_l2=float(rest.detach()))
            with torch.no_grad():
                flat_logit, _ = model(ax)
                threshold = max(detector_floor, float(flat_logit.max()) + .001)
            saved = dict(format=FORMAT, model=model.state_dict(), optimizer=optim.state_dict(),
                         channels=channels.tolist(), body_indices=indices.tolist(), base_sources=source,
                         sensor_fields=sensor_fields, threshold=threshold, step=step)
            path = output / f"step_{step:04d}.pt"
            with path.open("xb") as stream:
                torch.save(saved, stream)
            log.write(json.dumps(dict(step=step, sha256=checksum(path), threshold=threshold, **status)) + "\n")
            log.flush()
    for name, fingerprint in {**source, **provenance}.items():
        if checksum(name) != fingerprint:
            raise ValueError(f"Provenance changed during fit {name}")
    candidate = load(path)
    predictions = np.asarray([candidate(state, int(intent), f)
                             for state, f, intent in zip(x, sensed, moving, strict=True)])
    for cx, cy, ci, _, base_control, _ in anchor_runs:
        reverse = np.asarray([candidate(state, int(intent), f)
                              for state, f, intent in zip(cx[::-1], cy[::-1], ci[::-1], strict=True)])[::-1]
        np.testing.assert_array_equal(reverse, base_control)
    detected = (np.abs(predictions - base) > 1e-9).any(axis=1)
    active_all = (np.abs(action - base) > 1e-9).any(axis=1)
    summary = dict(checkpoint=str(path), sha256=checksum(path), training=args,
                   active_true=int(active_all.sum()), active_detected=int((detected & active_all).sum()),
                   false_positive_on_teacher=int((detected & ~active_all).sum()),
                   baseline_action_mae=float(np.abs((base - action)[active_all][:, channels]).mean()),
                   model_action_mae=float(np.abs((predictions - action)[active_all][:, channels]).mean()),
                   intact_reverse_decisions_exact=True, threshold=threshold)
    write(output / "RESULT.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--flat", type=Path, nargs="+", required=True)
    parser.add_argument("--updates", type=int, default=280)
    parser.add_argument("--seed", type=int, default=64887)
    parser.add_argument("--detector-floor", type=float, default=0.)
    options = parser.parse_args()
    if not 1 <= options.updates <= 1200:
        parser.error("Expected 1–1200 supervised updates")
    torch.set_num_threads(1)
    fit(options.teacher.resolve(), [path.resolve() for path in options.flat], options.output.resolve(),
        options.updates, options.seed, options.detector_floor)
