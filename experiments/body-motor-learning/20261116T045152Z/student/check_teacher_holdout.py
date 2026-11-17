"""Score a frozen candidate on independently qualified whole teacher runs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fit import teacher_data
from residual import baseline_policy, load, read, stand


def check(fit_dir, runs, output):
    training = read(fit_dir / "INPUTS.json")
    checkpoint_info = read(fit_dir / "RESULT.json")
    checkpoint = Path(checkpoint_info["checkpoint"])
    assert stand.sha(checkpoint) == checkpoint_info["checkpoint_sha256"]
    assert not training["archive_preflight"]
    train_starts = {read(Path(p) / "RESULT.json")["start"] for p in training["teachers"]}
    starts = [read(p / "RESULT.json")["start"] for p in runs]
    assert not train_starts.intersection(starts) and len(set(starts)) == len(starts)
    frozen = baseline_policy()
    candidate = load(checkpoint)
    teacher_inputs = {str(Path(__file__)): stand.sha(Path(__file__))}
    scores = {}
    for root, start in zip(runs, starts, strict=True):
        observations, actions = teacher_data(root, teacher_inputs, False)
        base = np.asarray([frozen(x, 1) for x in observations])
        found = np.asarray([candidate(x, 1) for x in observations])
        reference_error = abs(actions - base)
        candidate_error = abs(actions - found)
        active = reference_error > .001
        scores[start] = dict(source=str(root), frame_count=len(actions), baseline_mae=float(reference_error.mean()),
                             candidate_mae=float(candidate_error.mean()), active_action_element_fraction=float(active.mean()),
                             active_original_mae=float(reference_error[active].mean()) if active.any() else None,
                             active_corrected_mae=float(candidate_error[active].mean()) if active.any() else None)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=stand.sha(checkpoint),
                  used_for_fitting=False, teacher_inputs=teacher_inputs, scores=scores)
    stand.put(output, report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    check(a.fit.resolve(), [p.resolve() for p in a.runs], a.output.resolve())
