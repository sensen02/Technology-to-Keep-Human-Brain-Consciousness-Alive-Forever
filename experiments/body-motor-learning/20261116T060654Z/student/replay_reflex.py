"""Independently load and integrate a saved current-state stumble policy."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from fit_reflex import decisions, verified_files
from stumble_policy import HERE, checksum, load, read, write

BENCH_ROOT = HERE.parent / "teacher"


def run(checkpoint, scenario, output, match, native):
    summary = read(checkpoint.parent / "RESULT.json")
    fingerprint = checksum(checkpoint)
    if fingerprint != summary["sha256"]:
        raise ValueError("Checkpoint has no frozen training identity")
    sys.path.insert(0, str(BENCH_ROOT))
    from physical import run as physical_run

    body = load(checkpoint)
    kwargs = dict(policy=body, native=native, policy_source=checkpoint)
    physical_run(output, "student", scenario, **kwargs)
    result, items, hashes = verified_files(output)
    assert checksum(checkpoint) == fingerprint
    x, local, intent, commanded, base, ms = decisions(output)
    expected = np.array([body(state, int(move), sensed)
                         for state, move, sensed in zip(x[::-1], intent[::-1], local[::-1], strict=True)])[::-1]
    np.testing.assert_array_equal(expected, commanded)
    corrective = np.flatnonzero(np.any(commanded != base, axis=1))
    if scenario is None and len(corrective):
        raise AssertionError("The absent companion unexpectedly activates the reflex")
    matched = {}
    if match:
        comparison, sample_items, sample_hashes = verified_files(match)
        for key in ("trace", "decision"):
            if scenario is None:
                with np.load(items[key], allow_pickle=False) as new, np.load(sample_items[key], allow_pickle=False) as old:
                    for name in set(new.files) & set(old.files):
                        np.testing.assert_array_equal(new[name], old[name], err_msg=name)
            else:
                assert hashes[key] == sample_hashes[key], items[key]
            matched[key] = sample_hashes[key]
        if native and (match / "NATIVE.npz").exists() and (output / "NATIVE.npz").exists():
            a = checksum(output / "NATIVE.npz")
            assert a == checksum(match / "NATIVE.npz")
            matched["native"] = a

    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint,
                  scenario=str(scenario) if scenario else None, student_run_sha256=hashes,
                  reverse_decisions_bitwise=True, decision_ticks=len(ms),
                  comparison=str(match) if match else None, matching_evidence=matched,
                  active_decisions=len(corrective),
                  active_first_ms=int(ms[corrective[0]]) if len(corrective) else None,
                  active_last_ms=int(ms[corrective[-1]]) if len(corrective) else None)
    write(output / "STUDENT_RUN.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario", type=Path)
    p.add_argument("--match", type=Path)
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a.checkpoint.resolve(), a.scenario.resolve() if a.scenario else None, a.output.resolve(),
        a.match.resolve() if a.match else None, a.native)
