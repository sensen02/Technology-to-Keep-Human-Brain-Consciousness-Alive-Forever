"""Load one saved gap checkpoint, integrate, then verify all stored decisions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from fit_gap import load_decisions, verified_files
from gap_policy import HERE, LEG_ORDER, checksum, load, read, write

BENCH = HERE.parent / "teacher"


def replay(checkpoint, scenario, output, match, retract_only=False):
    fingerprint = checksum(checkpoint)
    if fingerprint != read(checkpoint.parent / "RESULT.json")["sha256"]:
        raise ValueError("No frozen training result for checkpoint")
    sys.path.insert(0, str(BENCH))
    from physical import run

    candidate = load(checkpoint)
    rf = LEG_ORDER.index("RF")
    cut = candidate.channels[candidate.channels // 7 != rf]

    def policy(body, intent, local):
        action = candidate(body, intent, local)
        if retract_only:
            action = action.copy()
            action[cut] = candidate.nominal(body, intent)[cut]
        return action

    run(output, "student", scenario, policy=policy, native=True, policy_source=checkpoint)
    result, file_items, hashes = verified_files(output)
    if checksum(checkpoint) != fingerprint or tuple(result["sensor_fields"]) != candidate.sensor_fields:
        raise ValueError("Runtime sensor or parameter identity changed")
    present_body, local, intent, commanded, nominal, ms = load_decisions(output)
    generated = np.asarray([policy(x, int(i), y)
                            for x, y, i in zip(present_body[::-1], local[::-1], intent[::-1], strict=True)])[::-1]
    np.testing.assert_array_equal(generated, commanded)
    changed = np.abs(commanded - nominal) > 1e-9
    channels = candidate.channels if not retract_only else candidate.channels[candidate.channels // 7 == rf]
    unexpected = set(np.flatnonzero(changed.any(axis=0))) - set(channels.tolist())
    if unexpected:
        raise AssertionError(f"Unsupervised motor channels were changed {unexpected}")
    active = np.flatnonzero(changed.any(axis=1))
    matched = {}
    if match is not None:
        comparison, items, other = verified_files(match)
        same_scene = result["scenario"] == comparison["scenario"]
        if not same_scene:
            raise ValueError("Replay matching requires a shared physical scenario")
        for key in ("trace", "decision"):
            with np.load(file_items[key], allow_pickle=False) as new, np.load(items[key], allow_pickle=False) as old:
                if set(new.files) != set(old.files):
                    raise ValueError("Incompatible matching trial archives")
                for name in new.files:
                    np.testing.assert_array_equal(new[name], old[name], err_msg=f"{key}:{name}")
            matched[key] = bool(hashes[key] == other[key])
        matched["native"] = checksum(output / "NATIVE.npz") == checksum(match / "NATIVE.npz")
        if not all(matched.values()):
            raise ValueError("Physically repeated arrays did not match byte-for-byte")
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint, source_result_sha256=hashes["result"],
                  reverse_decisions_exact=True, native_joint_pass=result["native_joint_pass"],
                  controlled_channels=channels.tolist(), retract_only=retract_only,
                  corrective_decisions=len(active), first_active_ms=int(ms[active[0]]) if len(active) else None,
                  last_active_ms=int(ms[active[-1]]) if len(active) else None,
                  comparison=str(match) if match else None, exact_matched=matched)
    write(output / "STUDENT_RUN.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--match", type=Path)
    parser.add_argument("--retract-only", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    replay(options.checkpoint.resolve(), options.scenario.resolve() if options.scenario else None,
           options.output.resolve(), options.match.resolve() if options.match else None, options.retract_only)
