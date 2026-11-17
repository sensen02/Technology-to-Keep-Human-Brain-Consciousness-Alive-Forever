"""Independently audit saved decisions, applied commands and matched physical prefixes."""
import argparse
from pathlib import Path

import numpy as np
import torch

from residual import baseline_policy, load, read, stand
from physical import KEYS, LIGHT_ONSET, PULSE_MS, PUSH_ONSET, PUSH_W, W, WARM, legacy


def check(root, control=None):
    report = read(root / "RESULT.json")
    for name, field in (("TRACE.npz", "trace_sha256"), ("DECISIONS.npz", "decision_sha256")):
        assert stand.sha(root / name) == report[field]
    checkpoint = Path(report["checkpoint"]) if report["checkpoint"] else None
    if checkpoint:
        assert stand.sha(checkpoint) == report["checkpoint_sha256"]
    policy = load(checkpoint) if checkpoint else baseline_policy()
    actual = report["actual_ms"]
    with np.load(root / "DECISIONS.npz", allow_pickle=False) as z:
        observations, intents, actions, ms = (z[name] for name in ("x", "intent", "action", "ms"))
    np.testing.assert_array_equal(ms, np.arange(0, actual, 5))
    np.testing.assert_array_equal(intents, (WARM <= ms) & (ms < WARM + report["walk_ms"]))
    replay = np.array([legacy.bounded_action(policy(x, a)) for x, a in zip(observations[::-1], intents[::-1], strict=True)])[::-1]
    assert replay.tobytes() == actions.tobytes()

    expected = np.repeat(actions, 5, axis=0)[:actual]
    wrench = np.zeros((2 * actual, 6))
    pulse = report["pulse"]
    onset = 0
    if pulse == "light":
        onset = LIGHT_ONSET
        noise = np.random.default_rng(3011).normal(0., .035, 42)
        expected[onset:onset+PULSE_MS] = np.clip(expected[onset:onset+PULSE_MS] + noise, -1., 1.)
    if pulse == "push":
        onset = PUSH_ONSET
        wrench[2*onset:2*(onset+PULSE_MS), 1] = PUSH_W * W
    with np.load(root / "TRACE.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["applied"][0], np.zeros(42))
        np.testing.assert_array_equal(z["applied"][1:], expected)
        np.testing.assert_array_equal(z["actual_wrench"], wrench)
        np.testing.assert_allclose(z["time"], z["time"][0] + np.arange(actual+1)*.001, atol=5e-8, rtol=0.)
        if control:
            baseline = read(control / "RESULT.json")
            assert stand.sha(control / "TRACE.npz") == baseline["trace_sha256"]
            assert report["start"] == baseline["start"] and report["walk_ms"] == baseline["walk_ms"]
            if pulse != "none":
                assert baseline.get("pulse") == "none" and report["checkpoint_sha256"] == baseline.get("checkpoint_sha256")
            else:
                assert actual == baseline["actual_ms"]
                assert report["control"] == "baseline" or report["walk_ms"] == 0
                onset = actual
            with np.load(control / "TRACE.npz", allow_pickle=False) as frozen:
                for key in (*KEYS, "applied"):
                    assert frozen[key][:onset+1].tobytes() == z[key][:onset+1].tobytes(), key
    native_audit = None
    if report["raw_native_contacts"]:
        native_audit = read(root / "NATIVE.json")
        assert native_audit["native_arrays_sha256"] == stand.sha(root / "NATIVE.npz")
        assert native_audit["all_endpoints_bitwise"] and native_audit["expected_wrench_exact"]
        assert native_audit["native_samples"] == actual * 80
    result = dict(reverse_replay_bitwise=True, ticks=len(ms), all_applied_commands_exact=True,
                  applied_wrench_exact=True, arity=42, control=str(control) if control else None,
                  paired_prefix_ms=onset if control else None, paired_prefix_bitwise=True if control else None,
                  native_check=native_audit)
    stand.put(root / "SAVED_AUDIT.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--control", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    check(args.run.resolve(), args.control.resolve() if args.control else None)
