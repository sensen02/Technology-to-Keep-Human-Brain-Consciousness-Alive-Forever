"""Reconstruct motor filter from targeted emitted events, not saved filter rates."""
from pathlib import Path
import json
import numpy as np
from engine.embodied.closure_bundle import validate_bundle
from engine.embodied.motor_output import MotorOutput,MotorOutputConfig
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/neural_muscle_closure'

def reconstruct(manifest,episodes,arrays,config):
    checks={}
    for arm,ep in episodes.items():
        decoder=MotorOutput(manifest,MotorOutputConfig(**config)); ids=ep['probe_local_indices']; dt=ep['probe_spike_dt_s'];pending=np.zeros(manifest['n_actuators']);errors=[]
        motor={i for row in manifest['rows'] for i in row['local_indices']}
        if not motor.issubset(ids):raise ValueError('probe missing motor neurons')
        for k,r in enumerate(ep['records']):
            steps=round((r['time_end_s']-r['time_start_s'])/dt)
            spikes=np.zeros((steps,manifest['n_neurons']),dtype=bool)
            events=r['probe_emitted_spike_events_step_column']
            if len(events)!=len({tuple(e) for e in events}):raise ValueError('duplicate events')
            for step,col in events:
                if not (0<=step<steps and 0<=col<len(ids)):raise ValueError('invalid event index')
                spikes[step,ids[col]]=True
            actual=arrays[arm+'_controls'][k]
            expected=pending.copy()
            if arm=='motor_cut' and r['time_start_s']>=.015:expected[:]=0
            if arm=='replay':expected=arrays['normal_controls'][k]
            np.testing.assert_array_equal(actual,expected)
            pending,trace=decoder.consume(spikes,spike_dt_s=dt,start_s=r['time_start_s'],end_s=r['decoder']['source_end_s'],connected=r['decoder']['connected'])
            np.testing.assert_array_equal(trace['pool_spike_counts'],r['decoder']['pool_spike_counts'])
            np.testing.assert_array_equal(trace['rates_hz'],r['decoder']['rates_hz'])
            errors.append(float(np.max(np.abs(np.array(trace['rates_hz'])-r['decoder']['rates_hz']))))
        checks[arm]=dict(intervals=len(errors),max_rate_error_hz=max(errors),commands_exact=True)
    return checks

def main():
    bundle=validate_bundle(OUT);p=Path(bundle['path'])
    d=json.loads((p/'closure_results.json').read_text());m=json.loads((p/'motor_manifest.json').read_text())['measured']
    with np.load(p/'closure_traces.npz',allow_pickle=False) as a:
        checks=reconstruct(m,d['measured']['episodes'],a,d['provenance']['motor_translation'])
    result=dict(status='EXACT_EVENT_FILTER_RECONSTRUCTION',source_bundle=bundle,checks=checks,
                scope='Independently reconstruct saved filter from0.5ms emitted motor events; same implemented filter algorithm, not biological validation; no dynamics changed')
    path=OUT/'spike_filter_reconstruction.json';path.write_text(json.dumps(result,indent=2)+'\n');print(path);print(checks)
if __name__=='__main__':main()
