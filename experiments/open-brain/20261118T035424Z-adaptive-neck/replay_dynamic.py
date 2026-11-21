"""Use an archived bodily input and DN recording to test live axon delay changes."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from archive_io import ARCHIVE, ASSAY, BIDIR, FIXED, BINDING, HERE, digest, load_case, read, write
os.environ["P1_ARCHIVE_ROOT"] = str(ARCHIVE)
from dynamic_timing import FrontTouchNeck

TARGET = "720575941633589179"
CONTEXT_DN = ("720575941500851362", "720575941535086424")


def actual_arrivals(source, delay, cut):
    expected = np.zeros_like(source)
    for start in np.flatnonzero(source):
        arrival = start+int(delay[start])
        if arrival < len(cut) and not cut[start:arrival+1].any():
            assert not expected[arrival]
            expected[arrival] = True
    return expected


@torch.inference_mode()
def run(out):
    torch.set_num_threads(1)
    out.mkdir()
    case, meta, trace = load_case(11701)
    body_path = case / "intact/TRACE.npz"
    assert digest(body_path) == read(body_path.parent/"RESULT.json")["trace_sha256"]
    with np.load(body_path,allow_pickle=False) as source:
        load = source["normal_load"][:, [0,3]]
    ports = read(BIDIR/"CATALOG.json")["ports"]
    roots = [row["banc_root"] for row in read(BINDING)["records"]]
    assert roots[0] == TARGET
    context = [meta["dn_roots"].index(root) for root in CONTEXT_DN]
    chosen = {root:ports[root] for root in [*roots,*CONTEXT_DN]}
    model = FrontTouchNeck({"ports":chosen}, roots, range(3), range(3), batch=3, max_extra_delay=6)
    assert model.neck_delay[0] == read(FIXED/"PARAMETERS.json")["ports"][TARGET]["delay_ticks"]
    receptors = [model.validate_inputs([r for r,_ in ports[root]["incoming_receptors"]],"body") for root in roots]
    size = len(load)
    pauses = np.zeros((size,3,len(chosen)),dtype=bool)
    pauses[4500:9000,2,0] = True
    delays = np.zeros(pauses.shape,dtype=np.int16)
    for tick in range(5500,13500):
        delays[tick,1,0] = min(6, (tick-5500)//900+1, (13500-tick-1)//900+1)
    drive = np.zeros((3,len(model.roots)),dtype=np.float32)
    stim = np.zeros((3,len(chosen)),dtype=bool)
    em = np.zeros_like(pauses)
    ar = np.zeros_like(pauses)
    hold = np.zeros(2)
    for tick in range(size):
        if tick>0 and tick%5==0:
            hold = load[tick-1]
        drive.fill(0)
        stim.fill(False)
        for index, including in enumerate(receptors):
            if tick%23==0 and hold[index]>.04:
                drive[:, including] = 68.75
        if tick:
            stim[:,2:] = trace["dn_emitted"][tick-1,0,context]
        em[tick], ar[tick], _, _ = model.step(drive,pauses[tick],eon_output=stim,extra_delay=delays[tick])
    assert np.array_equal(em[:,0,:2],trace["an_emitted"][:,0])
    assert np.array_equal(ar[:,0,:2],trace["an_arrived"][:,0])
    assert np.array_equal(em[:,0],em[:,1]) and np.array_equal(em[:,0],em[:,2])
    for lane in range(3):
        for port in range(len(chosen)):
            found = actual_arrivals(em[:,lane,port],model.neck_delay[port]+delays[:,lane,port],pauses[:,lane,port])
            assert np.array_equal(ar[:,lane,port],found),(lane,port)
    assert np.array_equal(ar[:,0,1:],ar[:,1,1:]) and np.array_equal(ar[:,0,1:],ar[:,2,1:])
    assert not ar[4500:9000,2,0].any()
    fresh = int(np.flatnonzero(ar[9000:,2,0])[0])+9000
    assert fresh >= 9000+model.neck_delay[0]
    effect = np.flatnonzero(em[:,1,0] & (delays[:,1,0]>0))
    assert len(effect)>0
    np.savez_compressed(out/"TRACE.npz",emissions=em,arrivals=ar,blocked=pauses,extra_delay=delays)
    paths = [Path(__file__),HERE/"dynamic_timing.py",HERE/"dynamic_transport.py",HERE/"PROTOCOL.zh-CN.md",
        case/"NEURAL.npz",body_path,BIDIR/"CATALOG.json",FIXED/"PARAMETERS.json",BINDING,
        ASSAY/"assay_timing.py",ASSAY/"assay_transport.py",
        ARCHIVE/"data/banc888/banc_888_meta.feather"]
    write(out/"RESULT.json",dict(passes=True, baseline_archived_AN_bitexact=True, others_bitexact=True,
        every_arrival_matches_source_delay_and_gate=True, delayed_left_pulses=len(effect),
        extra_delay_max_ticks=int(delays.max()), baseline_left_delay_ticks=int(model.neck_delay[0]),
        first_fresh_open_tick=fresh, returned_DN_roots=CONTEXT_DN, batch_ticks=size,
        lane_ticks=size*3, trace_sha256=digest(out/"TRACE.npz"),
        source_sha256={str(p):digest(p) for p in paths}))
    print("Actual BANC replay; shifted",len(effect),"left arrivals; post-open",fresh,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);run(p.parse_args().output)
