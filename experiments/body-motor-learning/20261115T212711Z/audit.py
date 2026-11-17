"""Independently verify righting traces and the saved checkpoint journal."""
import argparse
from pathlib import Path

import numpy as np
import torch

from learned import Policy, UPDATES
from search import HERE, INITIALS, STARTS, read, stand
from verify import qualifies


def load_result(name):
    directory=HERE/name
    summary=read(directory/"RESULT.json")
    trace=directory/"TRACE.npz"
    if stand.sha(trace)!=summary["trace_sha256"]:
        raise ValueError("Altered trace")
    with np.load(trace,allow_pickle=False) as z:
        return summary,{k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",default="FIT")
    p.add_argument("--tested",default="FIT")
    p.add_argument("--repeated")
    p.add_argument("--output",default="AUDIT.json")
    args=p.parse_args()
    torch.set_num_threads(1)
    teacher={}
    for seed, original, new in (("side_fallen","NATIVE_SIDE_D4","REPLAY_SIDE"),
                                ("near_supine","NATIVE_SUPINE_D6","REPLAY_SUPINE")):
        result,a=load_result(original)
        repeat,b=load_result(new)
        for key,value in a.items():
            np.testing.assert_array_equal(value,b[key])
        assert result["passed"] and repeat["passed"]
        assert qualifies(a,a,result["handoff_ms"])["passed"]
        teacher[seed]=dict(array_count=len(a),identical=True,passed=True)
    stage=HERE/args.fit
    import json
    journal=[json.loads(line) for line in (stage/"TRAIN.jsonl").read_text().splitlines()]
    assert [row["update"] for row in journal]==list(range(UPDATES+1))
    for row in journal:
        assert stand.sha(stage/f"step_{row['update']:04}.pt")==row["sha256"]
    controller=Policy(stage/f"step_{UPDATES:04}.pt")
    audit={}
    for seed in STARTS:
        result, z=load_result(f"{args.tested}_{seed}")
        with np.load(INITIALS/f"{seed}.npz",allow_pickle=False) as start:
            np.testing.assert_array_equal(start["qpos"],z["qpos_ms"][0])
            np.testing.assert_array_equal(start["qvel"],z["qvel_ms"][0])
        error=0.
        for i in reversed(range(len(z["observation"]))):
            rate,branch=controller.command(z["observation"][i], z["target"][5*i])
            assert branch==z["branches"][i]
            error=max(error,float(np.max(np.abs(rate/2.-z["action"][i]))))
        assert error <= 1e-12
        dist=qualifies(z,z,0)
        assert all(dist[k]==result[k] for k in dist)
        replayed=False
        if args.repeated:
            again,w=load_result(f"{args.repeated}_{seed}")
            for key,value in z.items():
                np.testing.assert_array_equal(value,w[key])
            assert again["full_pass"]==result["full_pass"]
            replayed=True
        audit[seed]=dict(maximum_action_error=error,passed=result["full_pass"],physics_repeated=replayed)
    stand.put(HERE/args.output, dict(teacher=teacher, learned=audit, checkpoints_checked=len(journal)))


if __name__=="__main__":
    main()
