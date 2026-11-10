"""Supplementary exact encoder port verification, no physics calls."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
from module import LocalCircuit, HERE


def main():
    if (HERE/'ENCODER_PROBE.json').exists():
        raise RuntimeError('preserve completed evidence')
    encoder_path=HERE.parent/'encoder/encoder.py'
    spec=importlib.util.spec_from_file_location('qualified_encoder',encoder_path)
    encmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(encmod)
    e=encmod.SensoryEncoder(); c=LocalCircuit()
    assert np.array_equal(e.root_ids,c.root_ids)
    assert np.isin(e.local_indices,c.allowed_sensory_local).all()
    assert len(e.local_indices)==556
    q=e.joint_ranges_rad.mean(axis=1); velocity=np.zeros(42); load=np.full(6,e.body_weight_native/6)
    encoded=e.encode(q,velocity,load)
    pulse=np.zeros(20000); pulse[e.local_indices]=.05
    zero=np.zeros(20000)
    np.savez_compressed(HERE/'ENCODER_FIXTURE_INPUTS.npz',local_indices=e.local_indices,q42_rad=q,qvel42_rad_s=velocity,load6_native=load,constant_current_nA=encoded,pulse_current_nA=pulse,descending_current_nA=zero)
    results={}; steps=0
    for name,current in [('encoder_exact_pulse',pulse),('encoder_constant_fixture',encoded)]:
        pair=[]
        for dt in (.5,.25):
            c=LocalCircuit(dt_ms=dt)
            rates=[]; counts=[]; volts=[]
            for k in range(round(200/dt)):
                drive=current if name.endswith('fixture') or 25<=k*dt<125 else zero
                rates.append(c.advance(drive,zero,dt))
                counts.append(c.net.spike_count[c.mn_local].copy())
                volts.append([c.net.v.min(),c.net.v.max()])
            steps+=c.net.steps
            count=c.net.spike_count[c.mn_local]
            perleg={leg:{'recruited':int(np.count_nonzero(count[c.mn_leg==leg])),'spikes':int(count[c.mn_leg==leg].sum())} for leg in np.unique(c.mn_leg)}
            summary={'dt_ms':dt,'neural_steps':c.net.steps,'MN_spikes':int(count.sum()),'MN_recruited':int(np.count_nonzero(count)),'MN_mean_Hz':float(count.mean()/ .2),'MN_max_cell_mean_Hz':float(count.max()/ .2),'voltage_min_mV':float(np.min(volts)),'voltage_max_mV':float(np.max(volts)),'per_leg':perleg,'all_six_legs_propagated':all(r['spikes']>0 for r in perleg.values())}
            np.savez_compressed(HERE/(name+'_dt'+str(dt)+'.npz'),MN_filtered_Hz=np.array(rates),MN_spike_count=np.array(counts),voltage_range_mV=np.array(volts),final_spike_count=c.net.spike_count,final_voltage_mV=c.net.v)
            pair.append(summary)
            print(name,summary,flush=True)
        results[name]={'runs':pair,'aggregate_rate_relative_difference':abs(pair[0]['MN_mean_Hz']-pair[1]['MN_mean_Hz'])/max(pair[0]['MN_mean_Hz'],1e-12)}
    report={'status':'EXECUTED_NO_CALIBRATION_CHANGE','results':results,'actual_neural_steps':steps,'physical_steps':0,'training_updates':0,'encoder_sha256':hashlib.sha256(encoder_path.read_bytes()).hexdigest(),'preregistration_sha256':hashlib.sha256((HERE/'ENCODER_PROBE_PREREGISTRATION.json').read_bytes()).hexdigest(),'qualified_cells':len(e.local_indices),'current_bounds_nA':[float(encoded.min()),float(encoded.max())],'calibration_status':'Original broad calibration remains FAILED; .02uS is original predeclared fallback','fixture_status':'engineering, not physical standing or brain output'}
    (HERE/'ENCODER_PROBE.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
