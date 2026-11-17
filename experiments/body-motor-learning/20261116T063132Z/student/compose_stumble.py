"""Run the previously accepted stumble episode with both local responses available."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from gap_policy import HERE, STUMBLE, checksum, load, read, write
from stumble_policy import load as load_stumble

GAP_BENCH = HERE.parent / "teacher"
STUMBLE_BENCH = STUMBLE.parent / "teacher"
STUMBLE_CHECKPOINT = STUMBLE / "FIT_RF_A" / "step_0240.pt"


def source_bench():
    sys.path.insert(0, str(GAP_BENCH))
    import physical as gap

    sys.path.remove(str(GAP_BENCH))
    for name in ("physical", "geometry"):
        sys.modules.pop(name)
    sys.path.insert(0, str(STUMBLE_BENCH))
    import physical as stumble
    if Path(gap.__file__).parent != GAP_BENCH or Path(stumble.__file__).parent != STUMBLE_BENCH:
        raise ImportError("Could not isolate the two experiment benches")
    return gap, stumble


def episode(checkpoint, output, scenario, expected):
    old_fp = checksum(STUMBLE_CHECKPOINT)
    to_verify = [(STUMBLE_CHECKPOINT, old_fp)]
    new_fp = None
    if checkpoint is not None:
        new_fp = checksum(checkpoint)
        to_verify.append((checkpoint, new_fp))
    for p, fingerprint in to_verify:
        if read(p.parent / "RESULT.json")["sha256"] != fingerprint:
            raise ValueError(f"Checkpoint is not bound to a frozen train result {p}")

    gap_bench, stumble_bench = source_bench()
    gap = load(checkpoint) if checkpoint is not None else None
    old = load_stumble(STUMBLE_CHECKPOINT)
    old_fields = tuple(stumble_bench.LOCAL_FIELDS)
    gap_fields = tuple(gap_bench.LOCAL_FIELDS)
    if gap is not None and gap_fields != gap.sensor_fields:
        raise ValueError("Real current sensor features do not align")
    split = len(old_fields)
    native_old_local = stumble_bench.Local

    class JoinedLocal(native_old_local):
        def __init__(self, body, walk_ms):
            super().__init__(body, walk_ms)
            self.gap = gap_bench.Local(body, walk_ms)

        def sense(self, m):
            return np.r_[super().sense(m), self.gap.sense(m)]

    gap_activations = []

    def policy(x, intent, local):
        stumble = old(x, intent, local[:split])
        sensed_gap = local[split:]
        if gap is None or gap.correction(x, sensed_gap, intent) is None:
            gap_activations.append(False)
            return stumble
        gap_activations.append(True)
        return gap(x, intent, sensed_gap)

    stumble_bench.Local = JoinedLocal
    stumble_bench.LOCAL_FIELDS = old_fields + gap_fields
    stumble_bench.run(output, "student", scenario, policy=policy, native=True,
                      policy_source=checkpoint or STUMBLE_CHECKPOINT)
    actual, default = (read(p / "RESULT.json") for p in (output, expected))
    report = dict(gap_checkpoint=str(checkpoint) if checkpoint is not None else None, gap_sha256=new_fp,
                  scope="complete composition" if gap is not None else "actual old stumble policy, new gap observation passive only",
                  stumble_checkpoint=str(STUMBLE_CHECKPOINT),
                  stumble_sha256=old_fp, reference=str(expected), complete=actual["actual_ms"] == 11300,
                  gap_activations=int(sum(gap_activations)), regress_native_joint=actual["native_joint_pass"])
    exact = {}
    for key in ("TRACE.npz", "NATIVE.npz"):
        exact[key] = checksum(output / key) == checksum(expected / key)
    with np.load(output / "DECISIONS.npz", allow_pickle=False) as got, np.load(expected / "DECISIONS.npz", allow_pickle=False) as ref:
        exact["commands"] = np.array_equal(got["action"], ref["action"])
        exact["previous_sensors"] = np.array_equal(got["local"][:, :split], ref["local"])
    report.update(matches=exact, certain_old_case_pass=bool(default["native_joint_pass"]),
                  regression_passed=bool(actual["native_joint_pass"] and all(exact.values()) and not any(gap_activations)))
    write(output / "COMPOSITION_AUDIT.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--sense-only", action="store_true")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=STUMBLE / "STUDENT_PEG_A")
    options = parser.parse_args()
    torch.set_num_threads(1)
    episode(options.checkpoint.resolve() if options.checkpoint else None, options.output.resolve(),
            options.case.resolve(), options.reference.resolve())
