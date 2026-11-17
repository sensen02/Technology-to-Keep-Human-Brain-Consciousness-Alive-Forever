"""Independently score both new physical series and inspect causal prefixes."""
import argparse
from pathlib import Path

import numpy as np

from touch_runtime import sha, turn

CASES=("intact","right","sham","cut","restore","empty","dn_cut")


def load(folder):
    info=turn.read(folder/"RESULT.json")
    assert all(sha(folder/name)==digest for name,digest in info["hashes"].items())
    with np.load(folder/"RUN.npz",allow_pickle=False) as z:
        arrays={k:z[k] for k in ("qpos","qvel","com","loads","raw","commands","eye_stamps","eyes")}
    return info, arrays


def verify(root):
    conditions={}
    for version in ("joint","behavior"):
        stored={name:load(root/version/name) for name in CASES}
        cases={}
        for name,(res,data) in stored.items():
            moves=np.flatnonzero(data["commands"][:,4])
            interruptions=np.diff(moves)-1
            cases[name]=dict(passed=res["passed"],checks=res["checks"],start=res["first"],stop=res["stop"],
                       physical_touch=res["side_onsets_ms"],stop_delay_ms=res["stop_delay_ms"],
                       stand=res["stand"]["passed"],settled_ms=res["settle_ms"],
                       maximum_internal_pause_ms=int(max([0,*interruptions])*5),
                       advance_native=float(data["com"][-1,0]-data["com"][0,0]),
                       native_steps=res["native_steps"])
        matching={}
        regular=stored["intact"][1]
        for name in ("cut","restore","sham","dn_cut"):
            partner=stored[name][1]
            first_difference=np.flatnonzero(regular["commands"][:,4]!=partner["commands"][:,4])
            if not len(first_difference):
                continue
            tick=int(regular["commands"][first_difference[0],0])
            first_touch=stored["intact"][0]["side_onsets_ms"][0]
            frames=regular["eye_stamps"]<=tick
            matched={}
            for key in ("qpos","qvel","com","loads"):
                matched[key]=bool(np.array_equal(regular[key][:tick+1],partner[key][:tick+1]))
            matched["native"]=bool(np.array_equal(regular["raw"][:tick*80],partner["raw"][:tick*80]))
            matched["eyes"]=bool(np.array_equal(regular["eyes"][frames],partner["eyes"][frames]))
            matching[name]=dict(first_decoded_difference_ms=tick,through_contact=first_touch is not None and tick>=first_touch,
                                exact=matched,passed=all(matched.values()))
        conditions[version]=dict(complete_behavior_checks=all(case["passed"] for case in cases.values()),
                                  passed_cases=sum(case["passed"] for case in cases.values()),
                                  matched_prefixes=matching,cases=cases)
    return conditions


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    evidence=verify(args.root)
    turn.stand.put(args.out,evidence)
    print({version:row["passed_cases"] for version,row in evidence.items()},flush=True)
