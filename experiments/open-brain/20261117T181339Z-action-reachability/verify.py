"""Independently recompute diagnostic gates from native stored arrays."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np

HERE=Path(__file__).resolve().parent
digest=lambda p:sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((HERE/'SUMMARY.json').read_text())
checks=[]
for p in sorted((HERE/'RUNS').glob('*/RESULT.json')):
    result=json.loads(p.read_text());path=p.parent/'TRACE.npz'
    assert digest(path)==result['trace_sha256']
    with np.load(path,allow_pickle=False) as a:
        assert len(a['position'])==17000 and a['control'].shape==(17000,42)
        assert all(np.isfinite(a[k]).all() for k in a.files)
        ranges=np.array(result['model']['joint_ranges_rad']);limited=np.array(result['model']['joint_limited']).astype(bool)
        q=a['joint_position'];excess=np.maximum(np.maximum(ranges[:,0]-q,q-ranges[:,1]),0)[:,limited].max(initial=0.)
        stop=np.linalg.norm(np.diff(a['position'][-2000:,:2],axis=0),axis=1)/.0001
        rms=float(np.sqrt(np.mean(stop*stop)))
        nf=float(a['nonfoot_force'].max()/result['model']['weight_native'])
        gates=dict(completed=True,finite=True,upright=bool(a['upright'].min()>=.8),
            height=bool(a['position'][:,2].min()>=.7*a['initial_position'][2]),nonfoot=nf<=.2,
            joint_limit=bool(excess<=.02),stand=rms<=.30)
        if result['scene']=='flat':gates['advance']=bool(a['position'][-1,0]-a['initial_position'][0]>=2.)
        else:
            gates['block_contact']=bool(a['block_force'].max()>1e-6)
            gates['cross']=bool(a['position'][-1,0]>=3.15)
        assert gates==result['gates'] and all(gates.values())==result['pass_']
        assert abs(rms-result['final_stand_rms_native_s'])<1e-10
        active=np.max(np.abs(a['command']),axis=1)>.3
        needed=np.quantile(np.abs(a['teacher'][active]-a['rhythmic'][active]),[.5,.95,.99,1])
        assert np.array_equal(needed,np.array(result['required_residual_abs_quantiles_rad']))
        if result['mode'].startswith('oracle'):
            bound=.015 if result['mode']=='oracle015' else .15
            # Applied control is held 10 ticks: compare on actual control update rows.
            gain=np.clip(np.max(np.abs(a['command'][::10]),axis=1)/.3,0,1)
            assert np.max(np.abs(a['control'][::10]-a['rhythmic'][::10])-bound*gain[:,None])<1e-12
        checks.append(dict(name=p.parent.name,metrics_match=True,pass_=result['pass_'],
                          configured_joint_limits=int(limited.sum()),anatomical_joint_compliance='UNKNOWN_NOT_CONFIGURED'))
assert len(checks)==24 and summary['physical_steps']==408000
inputs=json.loads((HERE/'INPUTS.json').read_text())
assert all(digest(p)==sha for p,sha in inputs['source_sha256'].items())
replays=[]
for name in ('flat_teacher_119','block015_oracle015_119'):
    path=HERE/'REPLAY'/name/'TRACE.npz'
    with np.load(path,allow_pickle=False) as replay,np.load(HERE/'RUNS'/name/'TRACE.npz',allow_pickle=False) as original:
        equal={k:bool(np.array_equal(replay[k],original[k])) for k in original.files}
    assert all(equal.values())
    replays.append(dict(name=name,all_arrays_exact=True,arrays=len(equal)))
out=dict(status='DIAGNOSTIC_METRICS_AND_REPLAY_VERIFIED',checks=checks,replays=replays,
         main_physical_steps=408000,replay_physical_steps=34000,training_updates=0,
         source_hashes_unchanged=True)
(HERE/'VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('24 native metrics and 2 independent physics replays verified')
