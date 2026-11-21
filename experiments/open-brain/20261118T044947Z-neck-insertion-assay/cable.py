"""Measure geometry sensitivity in an explicitly proxy active NEURON cable."""
import json
from pathlib import Path

import neuron
from neuron import h
import numpy as np

from mechanics import FIXED, ROOTS, sha

HERE = Path(__file__).resolve().parent
DIAMETER = .60  # hypothetical, not the BANC skeleton radius
NEURAL_STEP = .0025
RA = 100.
MEASUREMENT_THRESHOLD = -5.


def crossing(time, signal):
    hits=np.flatnonzero((signal[:-1]<MEASUREMENT_THRESHOLD)&(signal[1:]>=MEASUREMENT_THRESHOLD))
    if len(hits)!=1:
        raise ValueError(f"expected one propagated crossing, observed {len(hits)}, peak {signal.max():.3f} mV")
    i=hits[0]
    share=(MEASUREMENT_THRESHOLD-signal[i])/(signal[i+1]-signal[i])
    return float(time[i]+share*(time[i+1]-time[i]))


def propagate(length, stretch, conserve_area, step=NEURAL_STEP, reverse=False):
    # Buffer both ends so the time difference is measured across the segment
    # rather than at the stimulated boundary or at a sealed end.
    parts=[h.Section(name=x) for x in ("upstream_buffer","cervical","downstream_buffer")]
    pro,neck,dist=parts
    neck.connect(pro(1),0)
    dist.connect(neck(1),0)
    for cable in parts:
        cable.insert("hh")
        cable.Ra=RA
        cable.cm=1.
    pro.L=dist.L=300.
    neck.L=float(length*stretch)
    for cable in parts:
        cable.diam=DIAMETER/(np.sqrt(stretch) if cable==neck and conserve_area else 1.)
        cable.nseg=int(np.ceil(cable.L/2))*2+1
    source=dist(.8) if reverse else pro(.2)
    pulse=h.IClamp(source)
    pulse.delay=1.
    pulse.dur=.22
    pulse.amp=.25
    times=h.Vector().record(h._ref_t)
    front=h.Vector().record(neck(0)._ref_v)
    back=h.Vector().record(neck(1)._ref_v)
    h.celsius=6.3
    h.dt=step
    h.finitialize(-65.)
    while h.t < 12.:
        h.fadvance()
    t=np.array(times)
    ts=[crossing(t,np.array(v)) for v in (front,back)]
    return abs(ts[1]-ts[0])


def ticks(array):
    return np.ceil(array/.1-1e-12).astype(np.int16)


def main():
    pinned=HERE/"MECHANICS.npz"
    m=json.loads((HERE/"MECHANICS.json").read_text())
    assert sha(pinned)==m["output_sha"]
    params=json.loads(FIXED.read_text())["ports"]
    with np.load(pinned) as data:
        responses={key:data[key] for key in ("release","retained","stick_release","refined")}
    bases=np.array([params[r]["propagation_ms"] for r in ROOTS])
    baseline=np.array([params[r]["delay_ticks"] for r in ROOTS])
    assert np.array_equal(ticks(bases),baseline)
    clean=responses["release"][0,5::2]
    print("SWC / selected param delay",clean.tolist(),bases.tolist(),flush=True)
    times=responses["release"][:,0]
    selected_outputs={}
    receipt={}
    for index,(root,length) in enumerate(zip(ROOTS,clean,strict=True)):
        all_stretch=np.concatenate([v[:,5+index*2]/length for v in responses.values()])
        grid=np.unique(np.r_[np.linspace(all_stretch.min(),all_stretch.max(),13),1.])
        reverse=index<2 # BANC front-to-rear morphology; AN travels rear-to-front.
        curve={str(c):np.array([propagate(length,s,c,reverse=reverse) for s in grid]) for c in (False,True)}
        control={str(c):propagate(length,1.,c,step=NEURAL_STEP/2,reverse=reverse) for c in (False,True)}
        reference={k:float(np.interp(1.,grid,v)) for k,v in curve.items()}
        assert all(abs(control[k]-reference[k])<.02 for k in curve)
        ratio=np.interp(responses["release"][:,5+index*2]/length,grid,curve["True"])/reference["True"]
        selected_outputs[root]=bases[index]*ratio
        receipt[root]=dict(stretch_grid=grid.tolist(),time_ms={k:v.tolist() for k,v in curve.items()},
                           fine_dt_control_ms=control, reference_proxy_ms=reference,
                           geometry_to_proxy_max_ms=float(np.max(np.abs(bases[index]*(ratio-1.)))))
        print(root,"proxy pristine",reference,"strain ratio",all_stretch.min(),all_stretch.max(),
              "max latency delta",receipt[root]["geometry_to_proxy_max_ms"],flush=True)

    live=np.arange(20000)*.0001
    transport=np.zeros((len(live),4,len(ROOTS)),dtype=np.int16)
    continuous=np.zeros_like(transport,dtype=float)
    for index,root in enumerate(ROOTS):
        pure_release=bases[index]*responses["release"][:,5+index*2]/clean[index]
        pure_retained=bases[index]*responses["retained"][:,5+index*2]/clean[index]
        keys=(np.ones_like(times)*bases[index], pure_retained,pure_release,selected_outputs[root])
        for group,values in enumerate(keys):
            continuous[:,group,index]=np.interp(live,times,values)
            transport[:,group,index]=ticks(continuous[:,group,index])
    assert np.array_equal(transport[:,0],np.tile(baseline,(len(live),1)))
    assert np.all(transport[:4501]==baseline)
    assert np.array_equal(transport[:10501,1],transport[:10501,2])
    assert np.max(np.abs(np.diff(transport.astype(int),axis=0)))<=1
    path=HERE/"TRANSPORT_PROFILES.npz"
    np.savez_compressed(path, time=live,root_ids=ROOTS, delay_ticks=transport,
                        continuous_delay_ms=continuous,baseline_ticks=baseline)
    result=dict(neuron_version=neuron.__version__, proxy="Hodgkin-Huxley demonstration, not fly-calibrated",
                diameter_um=DIAMETER, axial_resistivity_ohm_cm=RA, dt_ms=NEURAL_STEP,
                sensitivity_assumptions=["constant diameter","volume-conserving diameter"],
                live_use="relative conduction ratio of volume-conserving geometry, normalized to unchanged fixed baseline; no field or damage",
                measured_cables=receipt, data_sha256=sha(path), source_sha256={str(p):sha(p) for p in
                  (Path(__file__),Path(__file__).parent/"mechanics.py",pinned,FIXED)},
                changed_departure_ticks=np.count_nonzero(transport[:,1:]!=baseline,axis=0).tolist(),
                min_delay_ticks=np.min(transport,axis=0).tolist(), max_delay_ticks=np.max(transport,axis=0).tolist())
    (HERE/"CABLE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("changed opportunities",result["changed_departure_ticks"],flush=True)


if __name__=="__main__":
    main()
