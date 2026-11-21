"""Characterize frozen observer on archived timing and actual terrain contacts."""
import hashlib
import json
from pathlib import Path
import os
import sys

import numpy as np
from treatment import AdaptiveObserver, DT, load_models, predict

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"20261118T035424Z-adaptive-neck"
sys.path.insert(0,str(PRIOR))
from adaptive_observer import cues

ARCH=Path(os.environ["P1_ARCHIVE_ROOT"])/"experiments/open-brain"
HIST=ARCH/"20261118T031132Z-flat-neck-assay/FORMAL/seed11701"
PREV=ARCH/"20261118T022531Z-official-mixed-terrain/FORMAL"
PINS={}


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def load(path, expected=None):
    got=file_sha(path)
    assert expected is None or got==expected
    PINS[str(path)]=got
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def rms(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))


def spike_arrivals_to_rate(a):
    value=0.
    alpha=-np.expm1(-DT/.020)
    rate=np.empty(len(a))
    for tick,bit in enumerate(a):
        value+=alpha*(bit/DT-value)
        rate[tick]=value
    return rate


def phase_encode(row):
    phase=0.
    events=np.zeros(len(row),dtype=bool)
    for i,x in enumerate(row):
        phase+=x*DT
        events[i]=phase>=1.
        phase-=events[i]
    return events


def event_score(a,b,tolerance=10):
    a,b=np.flatnonzero(a),np.flatnonzero(b)
    i=j=matches=0
    while i<len(a) and j<len(b):
        difference=a[i]-b[j]
        if abs(difference)<=tolerance:
            i+=1;j+=1;matches+=1
        elif difference<0:
            i+=1
        else:
            j+=1
    return dict(a_count=len(a),b_count=len(b),matches=matches,
                f1=float(2*matches/max(1,len(a)+len(b))))


def correct(data,lane,raw):
    model=load_models()
    f=cues(data,lane)
    results=[predict(model[name],f[name]) for name in ("motor","other","fused")]
    observer=AdaptiveObserver()
    values=np.empty(len(raw))
    for i,(y,m,o,a) in enumerate(zip(raw,*results,strict=True)):
        values[i]=observer.step(float(y),float(m),float(o),float(a),available=True).estimate
    return values


def main():
    preserved=read(HIST/"RESULT.json")
    rec=load(HIST/"NEURAL.npz",preserved["neural_sha256"])
    base=rec["rates"][:,0,0]
    ref_pulses=rec["brain_an_events"][:,0,0]
    assert np.array_equal(phase_encode(base),ref_pulses)
    case=PRIOR/"DYNAMIC"
    shifted=load(case/"TRACE.npz",read(case/"RESULT.json")["trace_sha256"])
    assert np.array_equal(shifted["arrivals"][:,0,:2],rec["an_arrived"][:,0])
    w=slice(5500,13500)
    plain=spike_arrivals_to_rate(shifted["arrivals"][:,1,0])
    calibrated=correct(rec,0,plain)
    delayed=dict(rate_raw_rmse_hz=rms(plain[w],base[w]),rate_corrected_rmse_hz=rms(calibrated[w],base[w]),
                 exact_timing_raw=event_score(phase_encode(plain)[w],ref_pulses[w],0),
                 exact_timing_corrected=event_score(phase_encode(calibrated)[w],ref_pulses[w],0),
                 one_ms_raw=event_score(phase_encode(plain)[w],ref_pulses[w]),
                 one_ms_corrected=event_score(phase_encode(calibrated)[w],ref_pulses[w]))
    gain=np.interp(np.arange(len(base)),[0,6100,9100,14000,16200,len(base)-1],[1.,1.,1.34,1.34,1.,1.])
    shifted_amplitude=base*gain
    adjusted=correct(rec,0,shifted_amplitude)
    plateau=slice(10500,14000)
    expanded=dict(raw_rmse_hz=rms(shifted_amplitude[plateau],base[plateau]),
                  corrected_rmse_hz=rms(adjusted[plateau],base[plateau]),
                  raw_events=event_score(phase_encode(shifted_amplitude)[plateau],ref_pulses[plateau]),
                  corrected_events=event_score(phase_encode(adjusted)[plateau],ref_pulses[plateau]))
    terrain=[]
    for seed in (10901,11311):
        run=PREV/f"seed{seed}"
        r=read(run/"RESULT.json")
        n=load(run/"NEURAL.npz",r["neural_sha256"])
        repaired=run/"repaired"
        m=read(repaired/"RESULT.json")
        physical=load(repaired/"TRACE.npz",m["trace_sha256"])
        contact=np.flatnonzero(physical["terrain_normal"][:,1,:].max(1)>1.e-6)[0]
        a=n["rates"][:,3,0]
        b=correct(n,3,a)
        assert np.array_equal(phase_encode(a),n["brain_an_events"][:,3,0])
        w=slice(contact,contact+1000)
        active=slice(3500,14500)
        terrain.append(dict(seed=seed,first_contact_ms=float(contact*DT*1000),contact_window_added_rmse_hz=rms(a[w],b[w]),
                            active_window_added_rmse_hz=rms(a[active],b[active]),
                            active_events_changed=int(np.count_nonzero(phase_encode(a)[active] != phase_encode(b)[active])),
                            active_events_f1=event_score(phase_encode(a)[active],phase_encode(b)[active])))
    output=dict(simulation_only=True,timing_challenge=delayed,gain_expansion=expanded,
                clean_terrain_contact=terrain,data_sha256=PINS)
    with (HERE/"SHADOW_STRESS.json").open("x") as f:
        json.dump(output,f,indent=2);f.write("\n")
    print({k:v for k,v in output.items() if k!="data_sha256"},flush=True)


if __name__=="__main__":
    main()
