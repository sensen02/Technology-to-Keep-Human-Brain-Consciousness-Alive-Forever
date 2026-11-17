"""Verify serialized gait command sequences and their joint behavior decisions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fit_walk import load_policy
from prototype import policy as prototype_policy
from walking_bench import HERE, metrics, read, stand


def replay(path,policy):
    r=read(path/"RESULT.json")
    for name,digest in (("TRACE.npz",r["trace_sha256"]),("DECISIONS.npz",r["decision_sha256"])):
        assert stand.sha(path/name)==digest
    manifest=read(path/"RUN_IDENTITY.json")
    assert stand.identity.verify(manifest)
    with np.load(path/"TRACE.npz",allow_pickle=False) as data:
        samples={k:data[k] for k in data.files}
    with np.load(path/"DECISIONS.npz",allow_pickle=False) as f:
        x,c,a,t=(f[key] for key in ("x","intent","action","ms"))
    assert np.array_equal(t,5*np.arange(len(t)))
    expected=np.asarray([policy(xx,int(cc)) for xx,cc in zip(x[::-1],c[::-1],strict=True)])[::-1]
    np.testing.assert_allclose(expected,a,atol=1e-12,rtol=0)
    for i in range(len(samples["time"])-1):
        add=0
        if r["light"] and 2500<=i<2520:
            add=np.random.default_rng(3011).normal(0.,.035,42)
        np.testing.assert_array_equal(samples["applied"][i+1],np.clip(a[i//5]+add,-1,1))
    m=metrics(samples,500,r["walk_ms"],1700+r["walk_ms"])
    for k in m:
        assert m[k]==r[k]
    return dict(run=str(path),error=float(np.abs(expected-a).max()),decisions=len(t),
                trace_sha256=r["trace_sha256"],joint_sampled_pass=r["joint_sampled_pass"])


def check_fit(path):
    log=[]
    for line in (path/"JOURNAL.jsonl").read_text().splitlines():
        row=json.loads(line)
        assert stand.sha(path/f"step_{row['step']:04d}.pt")==row["sha256"]
        log.append(row["step"])
    assert log==list(range(1201))
    r=read(path/"RESULT.json")
    assert all(stand.sha(Path(k))==v for k,v in r["inputs"].items())
    return len(log)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--prototype",nargs="*",type=Path,default=[])
    parser.add_argument("--initial",nargs="*",type=Path,default=[])
    parser.add_argument("--aggregated",nargs="*",type=Path,default=[])
    args=parser.parse_args()
    torch.set_num_threads(1)
    checks = {}
    rows = []
    for name, group, folder in (("initial",args.initial,"FIT_INITIAL"),("aggregated",args.aggregated,"FIT_AGGREGATED"),
                                ("prototype",args.prototype,None)):
        if group:
            if folder is not None:
                policy=load_policy(HERE/folder/"step_1200.pt")
                checks[name]=check_fit(HERE/folder)
            else:
                policy=prototype_policy()
            rows.extend(replay(path.resolve(),policy) for path in group)
    stand.put(args.output,dict(replay=rows,verified_checkpoints=checks))


if __name__=="__main__":
    main()
