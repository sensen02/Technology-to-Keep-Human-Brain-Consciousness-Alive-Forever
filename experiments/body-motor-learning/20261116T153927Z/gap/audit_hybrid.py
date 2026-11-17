"""Current physical slot counterfactuals under the corrected floor partition."""
import argparse
import json
from pathlib import Path

from audit_crossing import paired_prefix, view
from crossing import RELEASE, read, stan


def main(candidate, off, filled, output):
    case,bare,floor=map(view,(candidate,off,filled))
    result=read(candidate/"RESULT.json")
    shape=result["scenario"]
    assert shape["partition"]=="left_continuous" and not shape["filled"]
    assert read(off/"RESULT.json")["scenario"]==shape
    assert read(filled/"RESULT.json")["scenario"]==dict(shape,filled=True)
    onset=case["controller_trigger_ms"]
    events=[case[k] for k in ("prior_near_support","first_slot_miss","first_lift_after_loss",
            "first_airborne_far_rim_clearance","first_in_lane_far_support")]
    open_far=bare["first_in_lane_far_support"]
    active_far=events[-1]
    paired=bool(onset and paired_prefix(candidate,off,onset))
    checks=dict(paired_local_physics=paired,
                named_chronology=bool(all(events) and events[1]["ms"]<=onset<events[2]["ms"] and
                    all(x["ms"]<y["ms"] for x,y in zip(events[:-1],events[1:],strict=True)) and active_far["ms"]<RELEASE),
                remaining_in_slot_y_during_crossing=case["all_mean_pad_y_inside_slot_until_airborne_crossing"],
                early_far_wins=bool(active_far and (not open_far or active_far["ms"]<open_far["ms"])),
                real_loss_in_unmodified_case=bool(bare["first_slot_miss"]),
                fill_has_upper_support=bool(floor["filled_support_at_same_locus"] and not floor["first_slot_miss"]),
                first_200ms_five_support=bool(case["helpers"] and case["helpers"]["min_total_W"]>=.8 and
                                              case["helpers"]["min_feet_above_02W"]>=3),
                stable_before_bench_renewal=bool(case["before_renewal_stand"] and case["before_renewal_stand"]["passed"]),
                native_zero_external_force=case["posture"]["zero_extra_force"])
    out=dict(scope="RF slot clear on left-continuous geometry",passed=all(checks.values()),checks=checks,
            candidate=case,off=bare,filled=floor,matched_geometry_prefix_ms=onset,
            actual_physical_opening=shape,one_released_bench_renewal_at_ms=RELEASE)
    stan.put(output,out)
    print(json.dumps(dict(passed=out["passed"],checks=checks,far_ms=active_far["ms"] if active_far else None)))
    return out["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate",type=Path)
    p.add_argument("off",type=Path)
    p.add_argument("filled",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    raise SystemExit(0 if main(a.candidate,a.off,a.filled,a.output) else 1)
