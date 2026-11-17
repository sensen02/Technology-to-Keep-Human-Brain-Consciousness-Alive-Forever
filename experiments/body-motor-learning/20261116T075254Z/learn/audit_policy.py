"""Verify saved actions against the current-only fitted policy in reverse sequence."""
import argparse
from pathlib import Path

import numpy as np
import torch

from physical import bench,load_provider,sha,write


def verify(path, policy):
    result=bench.read(path/"RESULT.json")
    raw=path/"DECISIONS.npz"
    saved=result.get("decision_sha256",result.get("decisions_sha256"))
    if sha(raw)!=saved:
        raise ValueError(f"Decision archive changed at {path}")
    with np.load(raw,allow_pickle=False) as data:
        x,cmd,expected=(data[k] for k in ("x","command","action"))
    order=np.arange(len(x)-1,-1,-1)
    actual=np.asarray([policy(x[i],cmd[i]) for i in order])
    diff=np.abs(actual-expected[order])
    exact=np.array_equal(actual,expected[order])
    return dict(path=str(path),rows=len(x),decision_sha256=saved,
                action_mae=float(diff.mean()),maximum_abs=float(diff.max()),bitwise=bool(exact))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("provider",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("tested",type=Path,nargs="+")
    parser.add_argument("--right-fallback-match",action="store_true")
    a=parser.parse_args()
    torch.set_num_threads(1)
    policy=load_provider(a.provider,a.checkpoint)
    observed=[verify(path.resolve(),policy) for path in a.tested]
    extra={}
    if a.right_fallback_match:
        base=a.output.resolve().parent
        pair=[base/"RIGHT_SIGNED",base/"RIGHT_GUARDED"]
        extra=dict(compare=[str(path) for path in pair],
            exact_files={f:sha(pair[0]/f)==sha(pair[1]/f) for f in ("TRACE.npz","DECISIONS.npz","NATIVE.npz")},
            note="Independent physical run with an optional previous arc fallback produced the same arrays. The signed-only run has no old arc fallback.")
    out=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
             provider=str(a.provider.resolve()),provider_sha256=sha(a.provider),
             reverse_order=observed,right_fallback_ablation=extra,
             matches_all=all(r["bitwise"] for r in observed))
    write(a.output.resolve(),out)


if __name__=="__main__":
    main()
