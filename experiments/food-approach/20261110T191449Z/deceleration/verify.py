"""Fresh-process independent saved-evidence replay; never integrates physics."""
from pathlib import Path
import json
import hashlib
import numpy as np
from api import HERE, replay_cycles, previous
import physics
import identity

LEGS=('LF','LH','LM','RF','RH','RM')
def sha(p): return identity.sha256(p)
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text())
    checks={'all_frozen_sources_and_original_evidence_unchanged':all(sha(p)==h for p,h in frozen['source_hashes'].items())}
    prereg=json.loads((HERE/'PREREG.json').read_text())
    prefixes=[];recomputed={};artifacts={}
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for cfg in prereg['conditions']:
        name=cfg['name'];folder=HERE/name
        r=json.loads((folder/'RESULT.json').read_text())
        a=load(folder/'TRACE_1MS.npz');s=load(folder/'SAMPLES_5MS.npz');y=load(folder/'YAW_1MS.npz')
        def ck(key,value): checks[name+'_'+key]=bool(value)
        ck('runtime_identity',identity.verify(json.loads((folder/'IDENTITY.json').read_text())))
        ck('complete_trace',len(a['time'])==25000 and abs(a['time'][-1]-25)<1e-8 and r['early_stop'] is None)
        ck('trace_continuity',np.allclose(a['time'],np.arange(1,25001)*.001,atol=1e-8,rtol=0))
        ck('samples_shape',s['X238'].shape==(4800,238) and s['Y42'].shape==(4800,42) and s['command2'].shape==(4800,2))
        ck('sample_actual_times',np.allclose(s['actual_time'],1+np.arange(4800)*.005,atol=1e-8,rtol=0))
        ck('finite',all(np.isfinite(v).all() for v in list(a.values())+list(s.values())+list(y.values())))
        ck('sources',np.array_equal(a['control_source'],np.r_[np.zeros(1000,int),np.full(4800,2),np.full(19200,5)]))
        ck('zero_settle',np.all(a['action'][:1000]==0))
        ck('zero_extra_forces',np.all(a['external_body_force_max']==0) and np.all(a['external_generalized_force_max']==0))
        ck('neutral960_and3840post',np.all(s['command2'][:960]==0) and np.array_equal(s['post_prime'],np.arange(4800)>=960))
        ck('original_observation_speed',np.all(s['X238'][:,-1]==.075))
        ck('actual_actions_and_bounds',np.array_equal(a['action'][1000::5],s['Y42']) and np.max(np.abs(a['action']))<=1+1e-12 and np.array_equal(a['action'],a['actual_requested_action']))
        # Independently reconstruct analytic schedule without querying teacher.command.
        elapsed=np.arange(4800)*.005;dt=np.maximum(0,elapsed-4.8);u=np.minimum(dt,1)
        blend=u*u*(3-2*u);scale=1-.5*blend
        curvature=cfg['yaw_rate']/(.075*.5)*blend
        physical=np.column_stack([.075*scale,.075*scale*curvature])
        normalized=np.column_stack([scale-1,physical[:,1]/.008])
        phase=elapsed-.5*(u**3-.5*u**4+np.maximum(0,dt-1))
        ck('commands_reconstructed',np.allclose(s['command2'],normalized,atol=1e-15,rtol=0) and np.allclose(s['command_speed_yaw'],physical,atol=1e-15,rtol=0) and np.allclose(s['source_phase'],phase,atol=1e-14,rtol=0))
        post=a['control_source']==5;t=a['time'][post];com=a['com'][post];loads=a['loads'][post]
        duration=t[-1]-t[0];delta=com[-1,:2]-com[0,:2];forward=float(delta[0]/duration)
        progress=float(np.linalg.norm(delta)/duration);den=loads.sum(0)
        slip=(loads*a['speed'][post]).sum(0)/np.maximum(den,1e-12)/max(progress,1e-9)
        support=float((loads.sum(1)/physics.W).mean());nf=float(a['nonfoot'][post].max());height=float(com[:,2].min())
        original=dict(support=support>=.90,nonfoot=nf<=.20*physics.W,com=height>=1.20,forward=forward>=.05,slip=float(slip.max())<=.5)
        cycles=replay_cycles(a)
        half=dict(forward=forward>=.025,slip=float(slip.max())<=.5,support=support>=.90,com=height>=1.20,
            strict_nonfoot_zero=nf==0,valid_slip=bool(np.all(den>1e-12) and progress>1e-9),all6cycles=cycles['all_legs_completed'])
        ck('original_five_recomputed',original==r['gates'] and bool(all(original.values()))==r['original_all5_gates'])
        ck('halfspeed_gates_recomputed',half==r['halfspeed_gates'] and bool(all(half.values()))==r['all_halfspeed_gates'])
        ck('exact_slip_and_denominator',float(slip.max())==r['slip_ratio_max'] and np.array_equal(den,r['slip_load_denominator_native']))
        ck('sixleg_cycle_counts_and_events_replayed',cycles['complete_per_leg']==r['cycles']['complete_per_leg'] and cycles['events']==r['cycles']['events'])
        ck('all_metrics_recomputed',forward==r['forward_signed_native_per_s'] and support==r['foot_support_mean'] and nf==r['nonfoot_force_max_native'] and height==r['com_height_min'])
        ck('original_forward_failure_preserved',not original['forward'] and not r['original_all5_gates'])
        ck('artifact_hashes',all(sha(p)==h for p,h in r['artifacts'].items()))
        ck('yaw_time_alignment',np.array_equal(y['time'],a['time'][1000:]))
        yy=np.unwrap(y['yaw_world']);yawdelta=float(yy[-1]-yy[4799])
        ck('actual_yaw_recomputed',abs(yawdelta-r['actual_signed_yaw_rad'])<1e-12)
        recomputed[name]=dict(original_gates=original,original_all5=bool(all(original.values())),halfspeed_gates=half,
            all_halfspeed=bool(all(half.values())),forward=forward,slip_max=float(slip.max()),per_leg_slip=dict(zip(LEGS,map(float,slip))),
            load_denominator_native=den.tolist(),support_mean=support,com_min=height,nonfoot_max_native=nf,
            cycles=cycles['complete_per_leg'],actual_signed_yaw_rad=yawdelta)
        prefixes.append({k:a[k][:5800].copy() for k in ('qpos','qvel','action','com')})
        xy=com[:,:2]-a['com'][5799,:2]
        axes[0].plot(xy[:,0],xy[:,1],label=name)
        axes[1].plot(t-5.8,(yy[4800:]-yy[4799])*180/np.pi,label=name)
        axes[2].plot(t[1000:]-5.8,(com[1000:,0]-com[:-1000,0])/(t[1000:]-t[:-1000]),label=name)
        for p in folder.glob('*'):
            if p.is_file():artifacts[str(p)]=sha(p)
    checks['both_actual_baseline_prefixes_bitexact']=all(np.array_equal(prefixes[0][k],prefixes[1][k]) for k in prefixes[0])
    axes[0].set(xlabel='Actual COM delta world X (native)',ylabel='Actual COM delta world Y (native)',title='Complete modified physical window')
    axes[1].set(xlabel='Seconds after physical prefix',ylabel='Actual thorax yaw delta (degrees)',title='Continuous actual yaw')
    axes[2].set(xlabel='Seconds after physical prefix',ylabel='Trailing1s signed X speed (native/s)',title='Original forward FAIL retained')
    axes[2].axhline(.05,color='red',ls=':',label='original .05')
    axes[2].axhline(.025,color='black',ls='--',label='halfspeed .025')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.suptitle('Algorithm teacher only | 1s zero +4.8s baseline +19.2s deceleration | no training / neural autonomy')
    fig.tight_layout();fig.savefig(HERE/'ACTUAL_DIAGNOSTIC.png',dpi=150);plt.close(fig)
    for p in (HERE/'INPUT_IDENTITY.json',HERE/'SWEEP_RESULT.json',HERE/'ACTUAL_DIAGNOSTIC.png',HERE/'PREREG.json',HERE/'api.py',HERE/'run.py',HERE/'verify.py'):
        artifacts[str(p)]=sha(p)
    result=dict(all_integrity_checks_passed=all(checks.values()),checks=checks,recomputed=recomputed,artifact_sha256=artifacts,
        scope='Independent fresh-process saved-evidence logical replay; no new simulation, native peak or D1-D8 claim')
    identity.atomic_json(HERE/'VERIFICATION.json',result)
    print(json.dumps(result,indent=2))
    assert result['all_integrity_checks_passed']
if __name__=='__main__':main()
