"""Fork the original frozen Eon state to isolate an archived pulse retiming."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"20261118T042507Z-observer-closed-loop"
ARCH=Path(os.environ["P1_ARCHIVE_ROOT"])/"experiments/open-brain"
RUNTIME=ARCH/"20261118T002600Z-ascending-behavior-comparison"
sys.path.insert(0,str(RUNTIME))
from neural_runtime import (BIDIR, ContextBrain, FrozenCPUBrain, context_groups,
                            eon, load_connectome, read, sha, verify, verify_upstream, write)

ARMS=("original","recorded_adapted","first_retimed_only","first_pair_put_back")


def earliest(a,b,start):
    v=np.flatnonzero((a!=b).any(axis=1))
    return int(v[0]+start) if v.size else None


@torch.inference_mode()
def run(seed, path):
    torch.set_num_threads(1)
    path.mkdir()
    verify_upstream()
    check=HERE/"PHASE_AUDIT.json"
    phase=next(r for r in read(check)["rows"] if r["seed"]==seed)
    first,second=phase["first_an_tick"],phase["corresponding_clean_an_tick"]
    end=phase["first_dn_tick"]+650
    receipt=PRIOR/"FORMAL"/f"seed{seed}"
    inputs=read(receipt/"INPUTS.json")
    trace=receipt/"NEURAL.npz"
    expected=read(receipt/"RESULT.json")
    assert sha(trace)==expected["neural_sha256"]
    with np.load(trace,allow_pickle=False) as raw:
        events=raw["brain_an_events"][:end,:2].copy()
        hits=raw["central_hits"][:end].copy()
        dn=raw["dn_emitted"][:end,:2].copy()
        population=raw["population"][:end,:2].copy()
    assert np.array_equal(events[:first,0],events[:first,1])
    lanes=events[:,[0,1,0,1],:].copy()
    lanes[[first,second],2,0]=events[[first,second],1,0]
    lanes[[first,second],3,0]=events[[first,second],0,0]

    index,_=eon.get_hash_tables(eon.path_comp)
    weights=load_connectome(index)
    ports=read(BIDIR/"EON/RESULT.json")["admitted"]
    columns=[index[int(ports[root])] for root in inputs["dn_roots"]]
    context=np.asarray(context_groups(index)[inputs["context"]])
    precursor=ContextBrain(FrozenCPUBrain(1,weights),inputs["source_indices"],[0])
    rng=torch.random.get_rng_state()
    for tick in range(first):
        active=context[hits[tick]].tolist()
        precursor.step([active],events[tick,:1])
        assert np.array_equal(precursor.spikes[0,columns].numpy().astype(bool),dn[tick,0])
        assert int(precursor.spikes.sum(1)[0])==population[tick,0]
    brain=ContextBrain(FrozenCPUBrain(4,weights),inputs["source_indices"],range(4))
    for field in ("g","queue","spikes","v","refrac"):
        old=getattr(precursor,field)
        setattr(brain,field,old.repeat((4,)+(1,)*(old.ndim-1)))
    brain.slot=precursor.slot

    fired=np.empty((end-first,4,len(columns)),dtype=bool)
    sums=np.empty((end-first,4),dtype=np.int32)
    for tick in range(first,end):
        active=context[hits[tick]].tolist()
        brain.step([active]*4,lanes[tick])
        fired[tick-first]=brain.spikes[:,columns].numpy().astype(bool)
        sums[tick-first]=brain.spikes.sum(1).numpy()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert np.array_equal(fired[:,:2],dn[first:end])
    assert np.array_equal(sums[:,:2],population[first:end])
    changes={}
    for lane,name in enumerate(ARMS[1:],1):
        start=earliest(fired[:,0],fired[:,lane],first)
        roots=[] if start is None else [inputs["dn_roots"][i] for i in
                                  np.flatnonzero(fired[start-first,0]!=fired[start-first,lane])]
        changes[name]=dict(first_dn_difference_tick=start,different_ports=roots,
            total_dn_event_differences_vs_original=int(np.count_nonzero(fired[:,0]!=fired[:,lane])))
    np.savez_compressed(path/"TRACE.npz",emitted=fired,population=sums,inputs=lanes[first:end])
    sources=(Path(__file__),check,trace,receipt/"INPUTS.json",receipt/"RESULT.json",
        RUNTIME/"neural_runtime.py",Path(eon.__file__),Path(eon.path_con),Path(eon.path_comp))
    pins={str(p):sha(p) for p in sources}
    verify(pins)
    write(path/"RESULT.json",dict(seed=seed,ports=inputs["dn_roots"],branches=changes,
        matches_both_archived_neural_lanes=True,first_retimed_ticks=[first,second],
        prefix_ticks=first,suffix_batch_ticks=end-first,total_lane_ticks=first+4*(end-first),
        window_end_exclusive=end,trace_sha256=sha(path/"TRACE.npz"),source_sha256=pins))
    print(seed,changes,flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("seed",type=int);arg.add_argument("output",type=Path)
    a=arg.parse_args();run(a.seed,a.output)
