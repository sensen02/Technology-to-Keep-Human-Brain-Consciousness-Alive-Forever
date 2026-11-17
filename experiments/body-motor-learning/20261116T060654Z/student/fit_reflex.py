"""Fit supervised motor residuals using physically admitted stumble recovery labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stumble_policy import (
    Correction, FORMAT, HERE, LEG_ORDER, WALK_INPUTS, base_sources, checksum, load, local_features,
    prototype, read, write,
)


def verified_files(path):
    result_path = path / "RESULT.json"
    result = read(result_path)
    items = {
        "result": result_path,
        "trace": path / "TRACE.npz",
        "decision": path / "DECISIONS.npz",
    }
    fingerprints = {key: checksum(file) for key, file in items.items()}
    for key in ("trace", "decision"):
        if result[f"{key}_sha256"] != fingerprints[key]:
            raise ValueError(f"Unverified file {items[key]}")
    return result, items, fingerprints


def decisions(path):
    with np.load(path / "DECISIONS.npz", allow_pickle=False) as z:
        x, local, intent, action, base, ms = (
            z[k] for k in ("x", "local", "intent", "action", "base", "ms")
        )
    n = len(intent)
    assert x.shape == (n, WALK_INPUTS) and local.shape[0] == n and local.ndim == 2
    assert action.shape == base.shape == (n, 42)
    assert all(np.isfinite(a).all() for a in (x, local, action, base))
    assert np.max(np.abs(action)) <= 1. and np.max(np.abs(base)) <= 1.
    assert np.array_equal(ms, np.arange(len(ms))*5)
    return x, local, intent, action, base, ms


def admit(path, absent):
    result, items, hashes = verified_files(path)
    acceptance = path / "TEACHER_ACCEPTANCE.json"
    native = path / "NATIVE_AUDIT.json"
    cert = read(acceptance)
    for key, fingerprint in hashes.items():
        if cert[f"{key}_sha256"] != fingerprint:
            raise ValueError(f"Unbound teacher {key} {path}")
    if (cert["passed"] is not True or cert["role"] != "teacher"
            or cert["native_audit_sha256"] != checksum(native)):
        raise ValueError(f"Missing same-body qualified native teacher {path}")
    audit = read(native)
    if not audit["native_joint_pass"] or not audit["zero_extra_force"]:
        raise ValueError(f"Failed native physical evidence {native}")
    raw = path / "NATIVE.npz"
    if audit["native_sha256"] != checksum(raw):
        raise ValueError(f"Native samples changed after admission {raw}")
    if (int(result["walk_ms"]) != 9600 or int(result["actual_ms"]) != 11300
            or int(cert["native_samples"]) != 904000):
        raise ValueError(f"Unexpected teacher horizon {path}")
    items.update(acceptance=acceptance, native=native, raw=raw)
    clean, c_items, _ = verified_files(absent)
    if clean["scenario"] is not None or clean["mode"] != "control" or not clean["native_joint_pass"]:
        raise ValueError("Expected passing native absent control")
    if clean["initial_snapshot_sha256"] != result["initial_snapshot_sha256"]:
        raise ValueError("Absent-obstacle anchor starts from another physical snapshot")
    if clean["sensor_fields"] != result["sensor_fields"]:
        raise ValueError("Absent anchor uses another sensor contract")
    items.update({f"flat_{key}": file for key, file in c_items.items()})
    return {str(file): checksum(file) for file in items.values()}


def fit(teacher, absent, output, leg, updates, seed, threshold, force_index, force_floor, force_sign,
        support_isolation):
    if HERE.parent / "teacher" not in teacher.parents:
        raise ValueError("Training paths must come from the current teacher study")
    if absent != HERE / "FLAT_FROZEN" and HERE.parent / "teacher" not in absent.parents:
        raise ValueError("Use the current matched absent-obstacle control as the flat anchor")
    provenance = admit(teacher, absent)
    for p in (Path(__file__), HERE / "stumble_policy.py", HERE.parent / "teacher" / "physical.py"):
        provenance[str(p)] = checksum(p)
    x, local, intent, action, base, ms = decisions(teacher)
    fx, flocal, fintent, faction, fbase, fms = decisions(absent)
    np.testing.assert_array_equal(intent, fintent)
    np.testing.assert_array_equal(ms, fms)
    np.testing.assert_array_equal(faction, fbase)
    np.testing.assert_array_equal(intent.astype(bool), (ms >= 500) & (ms < 10100))

    pi = prototype.policy()
    for rows, intents, commands in ((x, intent, base), (fx, fintent, fbase)):
        expected = np.array([pi(row, int(move)) for row, move in zip(rows, intents, strict=True)])
        np.testing.assert_array_equal(expected, commands)

    j = LEG_ORDER.index(leg) * 7
    if np.any(action[~intent.astype(bool)] != base[~intent.astype(bool)]):
        raise ValueError("Teacher altered the frozen standing policy")
    other = np.r_[0:j, j+7:42]
    support_change = np.any(action[:, other] != base[:, other], axis=1)
    if support_change.any():
        if not support_isolation or local.shape[1] != 19:
            raise ValueError("Other-leg responses require the declared current-state frozen-support rule")
        with np.load(prototype.BANK, allow_pickle=False) as z:
            halfspan = z["halfspan"][j:j+7]
        obs = x[support_change].copy()
        obs[:, 126+j:133+j] -= local[support_change, 12:19] / halfspan
        expected = np.array([pi(row, 1) for row in obs])
        np.testing.assert_array_equal(expected[:, other], action[support_change][:, other])
    pick, fpick = intent.astype(bool), fintent.astype(bool)
    raw_delta = action[pick, j:j+7] - base[pick, j:j+7]
    active = abs(raw_delta).max(1) > 1e-8
    if not active.any() or active.all():
        raise ValueError("Expected a finite actual corrective window and inactive teacher frames")
    if force_floor is not None:
        if force_index is None or not 0 <= force_index < local.shape[-1]:
            raise ValueError("A measured force channel is required when using a force floor")
        if np.min(force_sign * local[pick][active, force_index]) < force_floor:
            raise ValueError("Teacher requires correction after the immediate force window")
    tx = local_features(x[pick], local[pick], leg)
    flat = local_features(fx[fpick], flocal[fpick], leg)
    bound = float(min(2., max(.05, abs(raw_delta).max() * 1.02)))
    concat = np.concatenate((tx, flat))
    torch.manual_seed(seed)
    model = Correction(concat.mean(0), np.maximum(concat.std(0), .01), bound)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    rng = np.random.default_rng(seed)
    tx, ax, truth = (torch.as_tensor(a, dtype=torch.float32) for a in (tx, flat, raw_delta))
    joined = torch.cat((tx, ax))
    idx_positive = np.flatnonzero(active)
    idx_negative = np.flatnonzero(~active)
    adjacent = idx_negative[(idx_negative >= idx_positive[0]-24) & (idx_negative <= idx_positive[-1]+24)]
    if not len(adjacent):
        adjacent = idx_negative
    anchor_band = np.arange(max(0, idx_positive[0]-24), min(len(flat), idx_positive[-1]+25))

    output.mkdir()
    source = base_sources()
    manifest = dict(teacher=str(teacher), absent=str(absent), provenance=provenance, base_sources=source,
                    leg=leg, seed=seed, updates=updates, teacher_active_samples=len(idx_positive),
                    teacher_walking_samples=len(tx), flat_anchor_samples=len(flat), bound=bound,
                    runtime="Current joints/target/state, local leg forces and foot mechanics",
                    support_isolation=support_isolation, force_floor=force_floor,
                    force_channel=force_index, force_sign=force_sign, gate_floor=threshold)
    write(output / "INPUTS.json", manifest)
    with (output / "JOURNAL.jsonl").open("x") as log:
        prev, after = None, None
        for step in range(updates + 1):
            status = {}
            if step:
                p = rng.choice(idx_positive, 45)
                n = np.r_[rng.choice(adjacent, 20), rng.choice(idx_negative, 16),
                          len(tx) + rng.choice(anchor_band, 20), len(tx) + rng.integers(len(ax), size=20)]
                logits, correction = model(tx[p])
                negative_logits, inactive_correction = model(joined[n])
                detection = F.softplus(-logits).mean() + F.softplus(negative_logits).mean()
                action_error = F.smooth_l1_loss(correction, truth[p], beta=.055)
                norm = inactive_correction.square().mean()
                loss = action_error + .25*detection + .03*norm
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in model.parameters()]
                assert all(g is not None and torch.isfinite(g).all() for g in gradients)
                optimizer.step()
                status = dict(detection=float(detection.detach()), regression=float(action_error.detach()),
                              inactive_l2=float(norm.detach()))
            with torch.no_grad():
                flat_logits, _ = model(ax)
                threshold_at_update = float(max(threshold, float(flat_logits.max()) + 1e-4))
            state = dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         base_sources=source, leg=leg, threshold=threshold_at_update,
                         support_isolation=support_isolation, force_floor=force_floor,
                         force_index=force_index, force_sign=force_sign, step=step)
            path = output / f"step_{step:04d}.pt"
            with path.open("xb") as stream:
                torch.save(state, stream)
            digest = checksum(path)
            log.write(json.dumps(dict(step=step, sha256=digest, threshold=threshold_at_update, **status)) + "\n")
            log.flush()
            prev, after = after, digest
    assert prev != after
    for path_, digest in {**source, **provenance}.items():
        assert checksum(path_) == digest
    candidate = load(path)
    obtained = np.array([candidate(row, int(move), sensor)
                         for row, move, sensor in zip(x, intent, local, strict=True)])
    anchor = np.array([candidate(row, int(move), sensor)
                       for row, move, sensor in zip(fx, fintent, flocal, strict=True)])
    np.testing.assert_array_equal(anchor, fbase)
    assigned = abs(obtained - base).max(1) > 1e-8
    active_all = abs(action - base).max(1) > 1e-8
    report = dict(checkpoint=str(path), sha256=checksum(path), training=manifest,
                  flat_decisions_bitwise=True, teacher_triggered=int(active_all.sum()),
                  student_triggered=int(assigned.sum()), true_positive=int((assigned & active_all).sum()),
                  false_positive_teacher=int((assigned & ~active_all).sum()),
                  active_baseline_mae=float(abs(base[active_all,j:j+7] - action[active_all,j:j+7]).mean()),
                  active_student_mae=float(abs(obtained[active_all,j:j+7] - action[active_all,j:j+7]).mean()),
                  gate_logit_cutoff=threshold_at_update)
    write(output / "RESULT.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--absent", type=Path, required=True)
    p.add_argument("--leg", choices=LEG_ORDER, required=True)
    p.add_argument("--updates", type=int, default=240)
    p.add_argument("--seed", type=int, default=63575)
    p.add_argument("--threshold", type=float, default=0.)
    p.add_argument("--force-index", type=int)
    p.add_argument("--force-floor", type=float)
    p.add_argument("--force-sign", type=float, default=1.)
    p.add_argument("--support-isolation", action="store_true")
    a = p.parse_args()
    if not 1 <= a.updates <= 1200:
        p.error("Use 1–1200 bounded training updates for this study")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    fit(a.teacher.resolve(), a.absent.resolve(), a.output.resolve(), a.leg, a.updates, a.seed, a.threshold,
        a.force_index, a.force_floor, a.force_sign, a.support_isolation)
