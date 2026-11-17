"""Audit six-foot stationary window while RF stays across the slot."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_crossing import paired_prefix, view
from crossing import PER_MS, RELEASE, RF, W, read, stan


def main(hold, direct, output):
    data=view(hold)
    source=read(hold/"RESULT.json")
    start=next(t for t,kind in source["edges"] if kind=="holding_far_ground_anchor")
    assert source["actual_ms"]==RELEASE and paired_prefix(hold,direct,start)
    far_x=source["scenario"]["opening"][1]
    tail=200*PER_MS
    with np.load(hold/"NATIVE.npz",allow_pickle=False) as n, np.load(hold/"GEOMETRY_NATIVE.npz",allow_pickle=False) as g:
        foot=n["forces"][-tail:,:6]/W
        stats=dict(far_min_RF_pad_x=float(g["rf_spatial"][-tail:,3].min()),
                   far_min_top_support_W=float(n["rf_floor"][-tail:,1].min()/W),
                   min_total_ground_support_W=float(foot.sum(1).min()),
                   peak_nonfoot_W=float(n["forces"][-tail:,12].max()/W),
                   minimum_COM_native=float(n["pose"][-tail:,2].min()),
                   fraction_three_loaded_feet=float(np.mean(((foot>=.02).sum(1)>=3))),
                   final_RF_center=g["rf_spatial"][-1,:3].tolist())
    checks=dict(same_direct_crossing_prefix=True,zero_external_force=data["posture"]["zero_extra_force"],
                prior_real_loss_and_far_support=bool(data["first_slot_miss"] and data["first_in_lane_far_support"]),
                far_pad_maintained=stats["far_min_RF_pad_x"]>far_x and stats["far_min_top_support_W"]>=.02,
                native_support=stats["min_total_ground_support_W"]>=.90 and stats["peak_nonfoot_W"]<.01
                    and stats["fraction_three_loaded_feet"]>=.95 and stats["minimum_COM_native"]>=1.2,
                sampled_rest=bool(source["before_renewal_stand"]["passed"]))
    result=dict(scope="far-side foothold station after a real RF slot miss",passed=all(checks.values()),
                hold_from_ms=start,final_window_ms=[RELEASE-200,RELEASE],
                final_center=data["center_end_x"],far_rim_x=far_x,stats=stats,checks=checks,
                first_in_lane_far_support=data["first_in_lane_far_support"],source=str(hold.resolve()),
                direct_repeated_prefix_source=str(direct.resolve()),native_steps=data["steps"],
                sampled_renewal_after_releasing_far_anchor_succeeds=False)
    stan.put(output,result)
    print(json.dumps(result,ensure_ascii=False))
    return result["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("hold",type=Path)
    p.add_argument("direct",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    raise SystemExit(0 if main(a.hold,a.direct,a.output) else 1)
