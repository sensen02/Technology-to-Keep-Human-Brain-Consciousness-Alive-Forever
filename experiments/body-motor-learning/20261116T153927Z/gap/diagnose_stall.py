"""Reconstruct the frozen nearest-prototype phase during the confirmed long stall."""
import argparse
from pathlib import Path

import numpy as np

import crossing as q
from prototype import BANK


def main(run, output):
    info=q.read(run/"RESULT.json")
    log=run/"DECISIONS.npz"
    trace=run/"TRACE.npz"
    assert q.stan.sha(log)==info["decisions_sha256"] and q.stan.sha(trace)==info["trace_sha256"]
    with np.load(BANK,allow_pickle=False) as z:
        targets, joints, nominal, half=z["x"][:,:42],z["x"][:,42:],z["a"],z["halfspan"]
    with np.load(log,allow_pickle=False) as z:
        ms, obs, command, actions=map(z.__getitem__,("ms","x","command","action"))
    with np.load(trace,allow_pickle=False) as z:
        com=z["com"]
    active=(ms>=q.RELEASE)&(command[:,0]>0)
    selected=ms[active]
    indices=[]
    exact=[]
    for x,a in zip(obs[active],actions[active],strict=True):
        err=x[126:168]-targets
        qerr=x[:42]-joints
        idx=int((np.square(err).sum(1)+.001*np.square(qerr).sum(1)).argmin())
        indices.append(idx)
        calc=np.clip(nominal[idx]-45.*half*err[idx],-1.,1.)
        exact.append(np.array_equal(a,calc))
    indices=np.asarray(indices)
    changes=np.flatnonzero(np.diff(indices))
    last=int(changes[-1]+1)
    stable=selected>=25000
    speed=(com[52_050,0]-com[25_000,0])/27.05
    result=dict(scope="archive lookup fixation under nonzero forward intention",
                archive=str(BANK.resolve()), archive_sha256=q.stan.sha(BANK),prototype_count=len(nominal),
                replay_exact_all_active=bool(all(exact)), first_ms_terminal_prototype=int(selected[last]),
                last_prototype_index=int(indices[-1]), last_bank_index=len(nominal)-1,
                stable_25s_to_rollout_end_prototype_indices=np.unique(indices[stable]).tolist(),
                mean_forward_native_s_after_25s=float(speed),
                forward_command_unique_native_s=np.unique(command[active,0]).tolist(),
                max_abs_action_after_25s=float(np.max(np.abs(actions[active][stable]))),
                terminal_whole_COM=float(com[-1,0]))
    q.stan.put(output,result)
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.run,a.output)
