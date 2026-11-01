"""Independent saved-raster/parameter audit; no new network or neural steps."""
from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s4-snn-tuning-contract/20261101T214247Z')
A=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
def main():
    z=np.load(O/'PROBES.npz');r=json.loads((O/'RESULT.json').read_text());p=json.loads((O/'PREREGISTRATION.json').read_text());m=z['motor_indices'];n=len(z['roots'])
    baseline=np.unpackbits(z['neutral_spikes_packed'],axis=1)[:,:n]
    archived=np.unpackbits(np.load(A/'BASELINE_REPLAY.npz')['spikes_packed'][:40],axis=1)[:,:n]
    assert np.array_equal(baseline,archived)
    for row in r['probes']:
        i=row['probe'];sp=np.unpackbits(z[f'probe{i}_spikes_packed'],axis=1)[:,:n]
        assert int(sp.sum())==row['total_spikes'] and int(sp[:,m].sum())==row['motor202_spikes']
        assert int((sp!=baseline).sum())==row['spike_bits_differ_from_neutral']
        assert float(abs(z[f'probe{i}_final_v']-z['neutral_final_v']).max())==row['final_voltage_max_difference']
        values=row['parameters'];assert .25<=values['exc_gain']<=4 and .25<=values['inh_gain']<=4 and -5<=values['threshold_shift_mV']<=5
    assert len(r['rejected'])==9 and len(set(r['rejected']))==9 and r['counts']['neural_steps']==320
    for f,h in p['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    report=dict(passed=True,baseline_total_spikes=int(baseline.sum()),baseline_motor202_spikes=int(baseline[:,m].sum()),recomputed_rasters=7,baseline_archived_exact=True,invalid_cases_recorded=9,counts=dict(network_instances=0,neural_steps=0,body_steps=0,training=0),limits='Array-only verification of stored experiment results; sparse-mask/state parity assertions executed by primary script. No autograd or optimization claim.')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
