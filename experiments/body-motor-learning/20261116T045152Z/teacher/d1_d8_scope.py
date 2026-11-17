"""Assess original D1–D8 wherever the compact native replay is sufficient."""
import argparse
from pathlib import Path

import numpy as np

from retarget import WALK_START, bench


def evaluate(run):
    result = bench.read(run / "RESULT.json")
    audit = bench.read(run / "NATIVE_AUDIT.json")
    raw = run / "NATIVE.npz"
    assert bench.stand.sha(raw) == audit["native_arrays_sha256"]
    end = WALK_START + result["walk_ms"]
    with np.load(raw, allow_pickle=False) as z:
        sample = z["foot_normal_and_slip_weighted"][WALK_START*80:end*80]
    foot = sample[:, :6] / bench.W
    nonfoot = sample[:, 12] / bench.W
    force = foot.sum(1) + nonfoot
    p10 = float(np.quantile(foot.sum(1), .1))
    minimum_support = float(force.min())
    median_legs = float(np.median((foot >= .1).sum(1)))
    nonfoot_fraction = float(np.divide(nonfoot, force, out=np.zeros_like(nonfoot), where=force > 0).mean())
    com_min = audit["windows"]["walking"]["com_min"]
    d8_threshold = .8 * 1.2072528224791905
    report = dict(
        native_audit_sha256=bench.stand.sha(run / "NATIVE_AUDIT.json"),
        native_raw_sha256=audit["native_arrays_sha256"],
        D1=dict(passed=bool(not nonfoot.any()), peak_nonfoot_w=float(nonfoot.max())),
        D2=dict(passed=True, min_total_plane_support_w=minimum_support,
                reason="No native time interval satisfies the <.2W support condition")
           if minimum_support >= .2 else dict(status="UNKNOWN", reason="Compact records lack native COM velocity"),
        D3=dict(status="UNKNOWN", reason="No same-trial auxiliary walking phase"),
        D4=dict(passed=p10>=.5, foot_support_p10_w=p10),
        D4b=dict(passed=nonfoot_fraction<=.2, nonfoot_fraction=nonfoot_fraction),
        D5=dict(status="UNKNOWN", reason="No same-trial auxiliary walking phase"),
        D6=dict(status="UNKNOWN", reason="Compact records lack original native COM velocity difference for force balance"),
        D7=dict(passed=True, median_legs_at_point_one_w=median_legs,
                reason="Conservative ≥0.1W threshold supplies sufficient evidence")
           if median_legs >= 3 else dict(status="UNKNOWN", median_legs_at_point_one_w=median_legs),
        D8=dict(passed=com_min>=d8_threshold, com_min=com_min, threshold=d8_threshold),
        full_D1_D8=False,
    )
    bench.stand.put(run / "D1_D8_SCOPE.json", report)
    print(run, report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    for directory in p.parse_args().runs:
        evaluate(directory.resolve())
