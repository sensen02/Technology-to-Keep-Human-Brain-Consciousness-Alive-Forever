"""Actual fixed-neck interruption/recovery, without telling the encoder the cut."""
import argparse
import time
from shared import *
from encoder import load
from fixed_neck import FixedCircuit


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(1);start=time.monotonic();a.output.mkdir(parents=True)
    fit=read(HERE/'FIT_RESULT.json');verify(fit['source_sha256']);model,_=load(fit['candidate'])
    base,_=load_data('held_left')
    with np.load(PREP/'DATA/held_left/INPUTS.npz',allow_pickle=False) as z:stim=z['receptor_stimuli'].copy()
    catalog=read(BIDIR/'CATALOG.json');ports={r:p for r,p in catalog['ports'].items() if p['group']=='front_touch'}
    circuit=FixedCircuit(dict(catalog,ports=ports))
    inputs=[circuit.validate_inputs([r for r,_ in p['incoming_receptors']],'body') for p in ports.values()]
    drive=np.zeros((1,len(circuit.roots)),dtype=np.float32)
    f=np.zeros((2,3));alpha=-np.expm1(-DT/np.array([.005,.020,.050]))
    rows={k:[] for k in ('emitted','arrivals','features_hz','prediction')}
    pins=sources({HERE/'FIT_RESULT.json',Path(fit['candidate']),FIXED/'PARAMETERS.json',BIDIR/'CATALOG.json',
                  PREP/'DATA/held_left/INPUTS.npz',HERE/'DATA/held_left/PAIRED.npz',HERE/'CUT_PROTOCOL.zh-CN.md'})
    first,last=4500,9000
    for tick in range(len(stim)):
        drive.fill(0)
        for side in range(2):
            if stim[tick,side]:drive[0,inputs[side]]=68.75
        cut=np.array([first<=tick<last,False],dtype=bool)
        emitted,arrivals,_,_=circuit.step(drive,cut)
        f+=alpha*(arrivals[0,:,None]/DT-f)
        # No privileged cut flag and no reset of post-neck filter history.
        prediction=model(torch.from_numpy(f).double()).numpy()
        for k,v in dict(emitted=emitted[0],arrivals=arrivals[0],features_hz=f,prediction=prediction).items():rows[k].append(v.copy())
        if time.monotonic()-start>90:raise TimeoutError('cut/restore verification budget')
    arrays={k:np.asarray(v) for k,v in rows.items()}
    normal=model(torch.from_numpy(base['features_hz']).double()).numpy()
    d=int(circuit.neck_delay[0])
    restored=np.flatnonzero(arrays['arrivals'][last:,0])+last
    late=slice(first+3000,last)
    gates=dict(cut_arrivals_zero=not bool(arrays['arrivals'][first:last,0].any()),
        no_replay_before_new_transit=not bool(arrays['arrivals'][last:last+d,0].any()),
        restoration_received=bool(len(restored)>0),
        uncut_arrivals_exact=bool(np.array_equal(arrays['arrivals'][:,1],base['student_arrivals'][:,1])),
        uncut_encoding_exact=bool(np.array_equal(arrays['prediction'][:,1],normal[:,1])),
        late_cut_encoding_hz=bool(arrays['prediction'][late,0].max()<=2.),
        late_recovery_hz=bool(np.max(abs(arrays['prediction'][14000:,0]-normal[14000:,0]))<=1.))
    np.savez_compressed(a.output/'TRACE.npz',**arrays,normal_prediction=normal)
    verify(pins)
    write(a.output/'RESULT.json',dict(pass_=all(gates.values()),gates=gates,
        native_neck_ticks=len(stim),new_physical_steps=0,brain_steps=0,
        cut_first_tick=first,restore_tick=last,first_restored_arrival_tick=int(restored[0]) if len(restored) else None,
        max_late_cut_rate_hz=float(arrays['prediction'][late,0].max()),
        cut_tail_policy='causal filter tails retained; encoder never receives cut mask',
        source_sha256=pins,trace_sha256=sha(a.output/'TRACE.npz'),wall_seconds=time.monotonic()-start))
    print('cut/restore',gates,'late cut rate',float(arrays['prediction'][late,0].max()),flush=True)


if __name__=='__main__':main()
