"""Check the realized physical effect of a source-derived swing transfer."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261116T000536Z"))
from prototype import policy
from walking_bench import LEGS, W, read, stand


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main(reference, candidate, output):
    torch.set_num_threads(1)
    meta = read(reference)
    src = Path(meta["source_dir"])
    baseline = Path(meta["receiver_baseline_dir"])
    ref = reference.with_suffix(".npz")
    result = read(candidate / "RESULT.json")
    provenance = read(candidate / "SOURCE_PROVENANCE.json")
    original = read(baseline / "RESULT.json")
    assert stand.sha(src / "TRACE.npz") == meta["source_trace_sha256"]
    assert stand.sha(ref) == meta["ref_sha256"]
    assert stand.sha(baseline / "TRACE.npz") == original["trace_sha256"]
    assert stand.sha(candidate / "TRACE.npz") == result["trace_sha256"]
    assert stand.sha(candidate / "DECISIONS.npz") == result["decision_sha256"]
    assert provenance["result_sha256"] == stand.sha(candidate / "RESULT.json")
    assert provenance["source_ref_sha256"] == meta["ref_sha256"]
    assert provenance["source_trace_sha256"] == meta["source_trace_sha256"]
    if result["start"] != original["start"]:
        raise ValueError("Physical arc comparisons require matching initial snapshots")
    p = policy()
    with np.load(candidate / "DECISIONS.npz", allow_pickle=False) as z:
        active = z["intent"].astype(bool)
        state, labels = z["x"][active], z["action"][active]
    correction = labels - np.asarray([p(x, 1) for x in state])
    with np.load(ref, allow_pickle=False) as profile, np.load(baseline / "TRACE.npz", allow_pickle=False) as old, \
            np.load(candidate / "TRACE.npz", allow_pickle=False) as new:
        offset = profile["correction_native"]
        old_z, new_z = old["padz"], new["padz"]
        before_force, after_force = old["loads"], new["loads"]
        old_com, new_com = old["com"], new["com"]
    events = []
    for e in meta["receiver_windows"]:
        leg, a, b = e["leg"], e["from_ms"], e["to_ms"]
        if b >= len(new_z):
            continue
        k = LEGS.index(leg)
        span = slice(a, b+1)
        d_ref = offset[span, k]
        old_height = old_z[span, k]
        new_height = new_z[span, k]
        displacement = new_height - old_height
        vertical_shape_delta = displacement - np.linspace(displacement[0], displacement[-1], len(displacement))
        intended_offset = provenance["source_gain"] * d_ref
        power = float(d_ref @ d_ref)
        events.append(dict(leg=leg, from_ms=a, to_ms=b, realized_shape_rms_body_lengths=rms(vertical_shape_delta)/stand.LENGTH,
                           realized_height_rms_body_lengths=rms(displacement)/stand.LENGTH,
                           baseline_full_reference_error_body_lengths=rms(d_ref)/stand.LENGTH,
                           candidate_full_reference_error_body_lengths=rms(d_ref-vertical_shape_delta)/stand.LENGTH,
                           requested_gain_error_body_lengths=rms(intended_offset-vertical_shape_delta)/stand.LENGTH,
                           realized_projection_of_full_gain=float(vertical_shape_delta @ d_ref / power) if power else None,
                           old_peak_load_W=float(before_force[span,k].max()/W),
                           new_peak_load_W=float(after_force[span,k].max()/W)))
    count = min(len(old_z),len(new_z))
    stand.put(output, dict(candidate=str(candidate), reference=str(reference), original=str(baseline),
          source_legs=meta["source_legs"], source_gain=provenance["source_gain"], valid_decisions=len(state),
          same_observation_action_correction_mae=float(abs(correction).mean()),
          same_observation_action_correction_max=float(abs(correction).max()),
          walking_decisions_with_nonzero_source_correction=int(np.count_nonzero(np.any(correction != 0.,axis=1))),
          across_trajectory_body_com_max_delta_body_lengths=float(np.linalg.norm(new_com[:count]-old_com[:count],axis=1).max()/stand.LENGTH),
          whole_trial_arc_observations=events, gates=result["joint_sampled_pass"],
          note="Fixed-time comparison in nominated nominal swing windows with endpoint chord removed. The reference reproduces only the tapered shape contribution."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.reference.resolve(), args.candidate.resolve(), args.output.resolve())
