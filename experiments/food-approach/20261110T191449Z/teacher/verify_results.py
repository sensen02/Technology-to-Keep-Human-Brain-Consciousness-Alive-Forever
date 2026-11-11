"""Independent saved-evidence verification and actual trajectory plot."""
from pathlib import Path
import hashlib
import json
import numpy as np
from turning_teacher import pin_baseline
pin_baseline()
import physics
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    identity=json.loads((HERE/'IDENTITY.json').read_text())
    checks={}
    checks['all_frozen_inputs_unchanged']=all(sha(p)==v for p,v in identity['source_hashes'].items())
    conditions=json.loads((HERE/'PREREG.json').read_text())['conditions']
    prefixes=[]
    fig, axes=plt.subplots(1,3,figsize=(15,4.5))
    for cfg in conditions:
        name=cfg['name']; folder=HERE/name
        report=json.loads((folder/'RESULT.json').read_text())
        with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        with np.load(folder/'SAMPLES_5MS.npz',allow_pickle=False) as z:
            samples={k:z[k] for k in z.files}
        with np.load(folder/'YAW_1MS.npz',allow_pickle=False) as z:
            yt=z['time']; y=np.unwrap(z['yaw_world'])
        checks[name+'_complete15p4s']=bool(len(a['time'])==15400 and samples['X238'].shape==(2880,238) and samples['Y42'].shape==(2880,42) and abs(a['time'][-1]-15.4)<1e-8)
        checks[name+'_finite']=all(np.isfinite(v).all() for v in [a['qpos'],a['qvel'],a['action'],samples['X238'],samples['Y42'],samples['command2']])
        checks[name+'_external_zero']=bool(np.all(a['external_body_force_max']==0) and np.all(a['external_generalized_force_max']==0))
        checks[name+'_obs_target_speed_unchanged']=bool(np.all(samples['X238'][:,-1]==.075))
        checks[name+'_neutral_prime']=bool(np.all(samples['command2'][:960]==0))
        checks[name+'_action_alignment']=bool(np.array_equal(a['action'][1000::5],samples['Y42']))
        checks[name+'_action_bounds']=bool(np.max(np.abs(a['action']))<=1+1e-12)
        post=a['control_source']==5
        rec=list(zip(a['time'][post],a['com'][post],a['loads'][post],a['nonfoot'][post],a['speed'][post]))
        summary=physics.summarise(rec,int(post.sum())*80)
        checks[name+'_all_original_gates_recomputed']=summary['gates']==report['gates']
        checks[name+'_slip_exact_recomputed']=summary['slip_ratio_max']==report['slip_ratio_max']
        checks[name+'_trace_hash']=sha(folder/'TRACE_1MS.npz')==report['trace']['sha256']
        checks[name+'_sample_hash']=sha(folder/'SAMPLES_5MS.npz')==report['samples']['sha256']
        prefixes.append((name,a['qpos'][:5800].copy(),a['qvel'][:5800].copy(),samples['X238'][:960].copy(),samples['Y42'][:960].copy()))
        xy=a['com'][post,:2]; xy=xy-xy[0]
        axes[0].plot(xy[:,0],xy[:,1],label=name)
        axes[1].plot(yt[4800:]-5.8,(y[4800:]-y[4799])*180/np.pi,label=name)
        # One-second trailing average actual COM speed along initial heading.
        com=a['com'][post,:2]; dt=1000
        speed=(com[dt:,0]-com[:-dt,0])/(a['time'][post][dt:]-a['time'][post][:-dt])
        axes[2].plot(a['time'][post][dt:]-5.8,speed,label=name)
    checks['all4_neutral_prefixes_bitexact']=all(all(np.array_equal(x,y) for x,y in zip(prefixes[0][1:],p[1:])) for p in prefixes[1:])
    axes[0].set(xlabel='Actual COM delta world X (native)',ylabel='Actual COM delta world Y (native)',title='Actual physical paths')
    axes[1].set(xlabel='Seconds after prefix',ylabel='Actual signed thorax yaw (degrees)',title='Native physics, 1ms samples')
    axes[2].set(xlabel='Seconds after prefix',ylabel='Actual X speed (native/s)',title='Trailing 1s displacement / 1s')
    axes[2].axhline(.05,color='gray',ls=':',label='original forward gate .05')
    for ax in axes:
        ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.suptitle('Algorithm teacher diagnostic only — no neural autonomy / no food field')
    fig.tight_layout();fig.savefig(HERE/'ACTUAL_DIAGNOSTIC.png',dpi=150);plt.close(fig)
    result=dict(all_passed=all(checks.values()),checks=checks,scope='Evidence integrity, not assertion all physical gates pass')
    (HERE/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['all_passed']

if __name__=='__main__':main()
