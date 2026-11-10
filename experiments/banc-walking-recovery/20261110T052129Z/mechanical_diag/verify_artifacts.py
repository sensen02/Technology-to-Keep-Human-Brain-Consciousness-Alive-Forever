"""Independent process checks saved diagnostics only; no physics simulation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
import identity

def read(name):
    return json.loads((P/name).read_text())

checks=[]
manifest=read('output_sha256.json')
for path,digest in manifest.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
checks.append(f'{len(manifest)} saved output hashes match')
assert read('exit_status.json')['exit_code']==0
assert (P/'mechanical_probe.py').read_bytes()==(P.parent/'mechanical_probe.py').read_bytes()
checks.append('exit status zero, script copy exact')
results={}
summary=read('summary.json')
initial=np.load(P/'initial_state.npz')
for label,report in summary['results'].items():
    z=np.load(P/(label+'_1ms.npz'))
    assert z['com'].shape==(2001,3) and z['loads'].shape==(2001,6)
    assert all(np.isfinite(z[key]).all() for key in z.files)
    assert np.allclose(z['time_s'],np.arange(2001)*.001,rtol=0,atol=2e-11)
    for key in ('qpos','qvel','target_rad'):
        assert np.array_equal(z[key][0],initial[key]),(label,key)
    if label.startswith('zero'):
        assert np.all(z['action']==0)
        assert np.all(z['target_rad']==initial['target_rad'])
    for window in report['windows']:
        a,b=window['window_s']
        mask=(z['time_s']>max(a,0)+1e-10)&(z['time_s']<=b+1e-10)
        loads=z['loads'][mask];speed=z['contact_speed'][mask]
        computed=float((loads*speed).sum()/loads.sum())
        assert abs(computed-window['load_weighted_contact_speed_native_per_s'])<1e-14
        assert mask.sum()==window['tick_end_samples']
    for n in (20,60):
        # Independently evaluate moving average directly, avoiding cumsum implementation.
        from numpy.lib.stride_tricks import sliding_window_view
        lp=sliding_window_view(z['com'],n,axis=0).mean(axis=-1)
        assert np.allclose(lp,z[f'lp{n}_com'],rtol=0,atol=1e-12)
    assert not any(report['warning_count'])
    identity.verify(read(label+'_identity.json'))
    results[label]=dict(windows_weighted_speed=[r['load_weighted_contact_speed_native_per_s'] for r in report['windows']],
        tail_blocks_weighted_speed=[r['load_weighted_contact_speed_native_per_s'] for r in report['tail_blocks']],
        late_com_xy_raw_mean=report['windows'][2]['com_xy_velocity_stats']['mean'],
        late_com_xy_20ms_mean=report['windows'][2]['com_lowpass']['20ms_com_xy_speed']['mean'],
        late_com_xy_60ms_mean=report['windows'][2]['com_lowpass']['60ms_com_xy_speed']['mean'],
        late_com_z_range=report['windows'][2]['com_z_range_native'],
        late_nonfoot_peak=report['windows'][2]['nonfoot_peak_native'],
        late_perleg_weighted_speed={leg:r['load_weighted_contact_speed_native_per_s'] for leg,r in report['windows'][2]['per_leg'].items()},
        legacy_engineering={key:report['legacy_engineering'][key] for key in ['forward_signed_native_per_s','foot_support_mean','com_height_min','nonfoot_force_max_native','slip_ratio_max']})
checks.append('all 4 cases: arrays finite, timestamps, initial states, zero-action targets, windows, filters, warnings and input hashes verified')
value=dict(exit_code=0,simulation_rerun=False,checks=checks,results=results)
identity.atomic_json(P/'artifact_verification.json',value)
print(json.dumps(value,ensure_ascii=False,indent=2))
