"""Use archived actual neck outputs to test signal anchoring and advisory amplitude."""
from pathlib import Path
import json
import os
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"20261118T042507Z-observer-closed-loop"
ADAPTIVE=HERE.parent/"20261118T035424Z-adaptive-neck"
ARCH=Path(os.environ["P1_ARCHIVE_ROOT"])/"experiments/open-brain"
sys.path.insert(0,str(PRIOR))
sys.path.insert(0,str(ARCH/"20261118T031132Z-flat-neck-assay"))
from assay_timing import CanonicalEvents
from anchored import AnchoredReadout
from shadow_stress import file_sha, phase_encode, read, spike_arrivals_to_rate

DT=.0001


def main():
    dyn=ADAPTIVE/"DYNAMIC"
    path=dyn/"TRACE.npz"
    assert file_sha(path)==read(dyn/"RESULT.json")["trace_sha256"]
    with np.load(path,allow_pickle=False) as d:
        arrival=d["arrivals"][:,:,:2].copy()
    codec=AnchoredReadout(CanonicalEvents((3,2)),3,DT)
    produced=np.array([codec.encode(row) for row in arrival])
    for lane in range(3):
        for side in range(2):
            independent=phase_encode(spike_arrivals_to_rate(arrival[:,lane,side]))
            assert np.array_equal(produced[:,lane,side],independent)
    assert np.array_equal(produced[:,0,1],produced[:,1,1])
    shifted=int(np.count_nonzero(produced[:,0,0]!=produced[:,1,0]))
    assert shifted>0
    assert not arrival[4500:9000,2,0].any()
    assert not produced[6000:9000,2,0].any()
    assert produced[9000:,2,0].any()

    four=np.repeat(arrival[:,0:1],4,axis=1)
    engine=AnchoredReadout(CanonicalEvents((4,2)),4,DT)
    verified=np.array([True,True,True,False])
    matched=[]
    rates=[]
    for tick, row in enumerate(four):
        exact=engine.encode(row)
        ref=engine.rates[:,0].copy()
        injected=np.array([1.,.66,1.34,.66]) if 6100<=tick<14000 else np.ones(4)
        raw=ref*injected
        corrected,flags=engine.reconcile_left_amplitude(raw,verified)
        assert np.array_equal(exact,produced[tick,:1].repeat(4,axis=0))
        matched.append(flags)
        rates.append(np.stack((ref,raw,corrected)))
    flags=np.asarray(matched)
    x=np.asarray(rates)
    assert not flags[:,[0,3]].any()
    assert np.flatnonzero(flags[:,1])[0]>=7100-1
    assert flags[10500:14000,1:3].all()
    assert not flags[14000:].any()
    assert np.array_equal(x[10500:14000,0,1:3],x[10500:14000,2,1:3])
    assert np.array_equal(x[:,1,3],x[:,2,3])
    result=dict(changed_real_delay_event_bits=shifted,cut_has_no_recreated_events=True,
               restored_after_reopen=True,advisory_deficit_detection_tick=int(np.flatnonzero(flags[:,1])[0]),
               advisory_expansion_detection_tick=int(np.flatnonzero(flags[:,2])[0]),
               amplitude_restoration_exact_with_independent_timestamps=True,
               healthy_and_unverified_readings_untouched=True,no_stale_correction_after_release=True,
               actual_archive=str(path),actual_archive_sha256=file_sha(path),observational_samples=len(arrival),
               new_brain_or_neck_or_body_ticks=0)
    with (HERE/"ANCHOR_CHECK.json").open("x") as out:
        json.dump(result,out,indent=2);out.write("\n")
    print(result,flush=True)


if __name__=="__main__":
    main()
