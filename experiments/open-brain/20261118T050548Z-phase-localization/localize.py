"""Split the first AN threshold decision into prior phase and current drive."""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"20261118T042507Z-observer-closed-loop"
sys.path.insert(0,str(PRIOR))
from treatment import CausalCue, AdaptiveObserver, DT
from shadow_stress import file_sha, load, read


def difference(a,b):
    match=np.flatnonzero((a!=b).reshape(len(a),-1).any(1))
    return int(match[0]) if match.size else None


def write(path,value):
    import json
    with path.open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False)
        f.write("\n")


def threshold(rates):
    phase=np.zeros(rates.shape[1])
    before=np.empty_like(rates)
    events=np.zeros_like(rates,dtype=bool)
    for t,row in enumerate(rates):
        before[t]=phase
        phase+=row*DT
        events[t]=phase>=1.
        phase-=events[t]
    return before,events


def localize(seed):
    source=PRIOR/"FORMAL"/f"seed{seed}"
    meta=read(source/"RESULT.json")
    npath=source/"NEURAL.npz"
    neural=load(npath,meta["neural_sha256"])
    rates=neural["left_used_hz"][:,:2]
    clean_rate=neural["rates"][:,:2,0]
    p,e=threshold(rates)
    assert np.array_equal(e,neural["brain_an_events"][:,:2,0])
    first=difference(e[:,0],e[:,1])
    first_rate=difference(rates[:,0],rates[:,1])
    assert np.array_equal(clean_rate[:first+1,0],clean_rate[:first+1,1])
    assert np.array_equal(neural["an_arrived"][:first+1,0],neural["an_arrived"][:first+1,1])
    crossings=p[first,:,None]+rates[first,None,:]*DT
    crossing_bits=crossings>=1.
    assert np.array_equal(np.diag(crossing_bits),e[first])
    assert np.array_equal(crossing_bits,np.array([[False,False],[True,True]]))
    shift_end=int(np.flatnonzero(e[first:,0])[0])+first
    assert not e[first+1:shift_end+1,1].any()
    assert np.count_nonzero(e[:first,0])==np.count_nonzero(e[:first,1])

    filter_cue=CausalCue()
    observer=AdaptiveObserver()
    monitor=[]
    for t in range(first+1):
        predictions=filter_cue.estimate()
        m,o,f=[float(array[1]) for array in predictions]
        out=observer.step(float(clean_rate[t,1]),m,o,f,available=True)
        assert out.gain==neural["estimated_gain"][t,1]
        monitor.append([m,o,f,observer.observed,observer.expected,observer.observed/max(observer.expected,1.),
                       observer.persistent,out.gain,int(out.cue_agreement)])
        filter_cue.update(neural["dn_emitted"][t],neural["an_arrived"][t,:,1])
    local=monitor[first_rate]
    assert local[6]==1000
    assert np.isclose(p[first,1]-p[first,0],np.sum((rates[:first,1]-rates[:first,0])*DT),atol=1.e-12)
    pins={str(npath):file_sha(npath)}
    body=[]
    for name in ("clean_off","clean_on"):
        path=source/name
        m=read(path/"RESULT.json")
        trace=path/"TRACE.npz"
        body.append(load(trace,m["trace_sha256"]))
        pins[str(trace)]=file_sha(trace)
    steps={key:difference(body[0][key],body[1][key])
           for key in ("command","qpos","normal_load")}
    dn=difference(neural["dn_emitted"][:,0],neural["dn_emitted"][:,1])
    assert first<dn<=steps["command"]<=steps["qpos"]
    print(seed,first_rate,first,shift_end,dn,steps,local,crossings.tolist(),flush=True)
    return dict(seed=seed,first_rate_tick=first_rate,first_an_tick=first,corresponding_clean_an_tick=shift_end,
        first_dn_tick=dn,downstream_ticks=steps,prior_phase=p[first].tolist(),instant_rate_hz=rates[first].tolist(),
        instantaneous_phase_delta=float((rates[first,1]-rates[first,0])*DT),
        accumulated_phase_delta=float(p[first,1]-p[first,0]),
        thresholds_phase_rows_rate_columns=crossings.tolist(),threshold_decisions=crossing_bits.tolist(),
        observer_first_update=dict(zip(("motor","other","fused","filtered_observed","filtered_expected",
                               "ratio","persistent_ticks","gain","cues_agree"),local,strict=True)),
        clean_raw_and_actual_arrivals_equal_until_first_difference=True,source_sha256=pins)


if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("output",type=Path);p=arg.parse_args().output
    write(p,{"rows":[localize(seed) for seed in (11701,11903)],"analysis_only":True})
