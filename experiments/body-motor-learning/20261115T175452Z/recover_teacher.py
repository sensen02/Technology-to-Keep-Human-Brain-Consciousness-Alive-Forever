"""Qualify an old walking teacher on states reached by the new policy."""
import argparse
import sys
from pathlib import Path
import time

import numpy as np
import torch

from evaluate import load
from train import WALKING, OLD_MOTOR, put, read, sha, CHECKPOINT_SHA


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy, candidate_sha = load(Path("TRAIN/update_0480.pt"))
    reference = OLD_MOTOR / "BC_GRU_ROLE_3000/checkpoint_002500.pt"
    if sha(reference) != CHECKPOINT_SHA:
        raise ValueError("Old walking checkpoint identity mismatch")
    sys.path.insert(0, str(OLD_MOTOR))
    import physics
    import identity
    import student_wave as sw

    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    body = physics.Core(adhesion=False)
    old, _ = sw.load_student(reference, body)
    manifest = identity.capture(body, dict(scope="teacher recovery diagnostic only",
                                           candidate_sha=candidate_sha, old_sha=CHECKPOINT_SHA),
                                extra_paths=[Path("PREREG_RECOVERY.zh-CN.md"), reference])
    put(args.output / "IDENTITY.json", manifest)
    trace = sw.Trace(full=True)
    sw.settle(body, trace)
    warmup = sw.StaticTeacher(body, sw.teacher_config(3))
    previous = np.zeros(42)
    new_state = None
    old_state = old.actor.initial_state(1)
    command = torch.tensor([1., 0.], dtype=torch.float64)
    decisions = []
    unsafe = 0
    n_candidate = 5
    n_recovery = 480
    for k in range(960 + n_candidate + n_recovery):
        x = body.obs238(previous, target_speed=.075)
        xt = torch.from_numpy(x)
        old_mean, old_state = old.actor.step(xt[None], old_state)
        old_action = old_mean.tanh()[0].numpy()
        if k < 960 + n_candidate:
            new_action, new_state = policy.step(xt, command, new_state)
            if k < 960:
                previous = sw.bounded_action(warmup.action(body, k * .005)[0])
                source = 2
            else:
                previous = sw.bounded_action(new_action.numpy())
                source = 3
        else:
            previous = sw.bounded_action(old_action)
            source = 5
        decisions.append((x.copy(), old_action.copy(), previous.copy(), source))
        for _ in range(5):
            m = trace.tick(body, previous, source)
            unsafe = unsafe + 1 if m["com"] < .8 or m["nonfoot"] > .5 * physics.W else 0
            if unsafe >= 100:
                break
        if k in (959, 964):
            print("boundary", k, float(body.d.time), flush=True)
        if unsafe >= 100:
            break
    identity.verify_runtime(manifest, body)
    trace_proof = trace.save(args.output / "trace_1ms.npz")
    xs, ys, actions, sources = (np.asarray(a) for a in zip(*decisions))
    with np.load("EVAL_walk/decisions.npz", allow_pickle=False) as original:
        prefix_matches = (len(xs) >= 965 and np.array_equal(xs[:965], original["x"][:965])
                          and np.array_equal(actions[:965], original["applied"][:965]))
    evidence = trace.summary(body, 5)
    complete = trace.sources.count(5) == n_recovery * 5
    # This short recovery window defines a safety/progress probe, not gait certification.
    qualified = (prefix_matches and complete and evidence["gates"]["support"]
                 and evidence["gates"]["com"] and evidence["gates"]["forward"]
                 and evidence["nonfoot_force_max_native"] == 0.)
    data_path = args.output / "decisions.npz"
    np.savez_compressed(data_path, x=xs, old_label=ys, actually_applied=actions, source=sources)
    put(args.output / "RESULT.json", dict(recovery_qualification=qualified,
        old_checkpoint_sha256=CHECKPOINT_SHA, candidate_sha256=candidate_sha,
        identical_candidate_prefix=prefix_matches, candidate_ms=trace.sources.count(3),
        teacher_recovery_ms=trace.sources.count(5), teacher_recovery=evidence,
        trace=trace_proof, decision_sha256=sha(data_path), actual_native_steps=len(trace.rows)*80,
        wall_s=time.monotonic() - started, optimizer_steps=0))
    print("recovery qualification", qualified, complete, evidence.get("gates"), flush=True)


if __name__ == "__main__":
    main()
