"""Independent fresh-process evidence replay, no physics integration."""
from pathlib import Path
import json
import numpy as np
from api import HERE,CONDITIONS,load_verified_samples
import physics,identity
from cycle_reward import CycleTracker,CycleConfig
LEGS=('LF','LH','LM','RF','RH','RM')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return identity.sha256(p)

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text())
    checks={'all_frozen_sources_previous_teacher_and_deceleration_unchanged':all(sha(p)==h for p,h in frozen['source_hashes'].items())}
    original=load(HERE.parent/'teacher'/'left_full'/'TRACE_1MS.npz')
    osamples=load(HERE.parent/'teacher'/'left_full'/'SAMPLES_5MS.npz')
    recomputed={};artifacts={};fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for cfg in CONDITIONS:
        name=cfg['name'];folder=HERE/name;r=json.loads((folder/'RESULT.json').read_text())
        a=load(folder/'TRACE_1MS.npz');s=load(folder/'SAMPLES_5MS.npz');y=load(folder/'YAW_1MS.npz')
        def ck(k,v):checks[name+'_'+k]=bool(v)
        ck('runtime_inputs',identity.verify(json.loads((folder/'IDENTITY.json').read_text())))
        ck('complete_no_exception',len(a['time'])==15400 and r['early_stop'] is None and r['exception'] is None)
        ck('trace_times_continuous',np.allclose(a['time'],np.arange(1,15401)*.001,atol=1e-8,rtol=0))
        ck('fullbody_shapes',a['qpos'].shape==(15400,49) and a['qvel'].shape==(15400,48))
        ck('sample_shapes',s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42) and s['command2'].shape==(2880,2) and all(len(v)==2880 for v in s.values()))
        ck('sample_actual_time',np.allclose(s['actual_time'],1+np.arange(2880)*.005,atol=1e-8,rtol=0))
        ck('finite',all(np.isfinite(v).all() for v in list(a.values())+list(s.values())+list(y.values())))
        ck('zero_settle',np.all(a['action'][:1000]==0))
        ck('zero_extra_forces',np.all(a['external_body_force_max']==0) and np.all(a['external_generalized_force_max']==0))
        ck('exact_control_sources',np.array_equal(a['control_source'],np.r_[np.zeros(1000,int),np.full(4800,2),np.full(9600,5)]))
        ck('all_held_action_ticks',np.array_equal(a['action'][1000:],np.repeat(s['Y42'],5,axis=0)) and np.array_equal(a['action'],a['actual_requested_action']) and np.max(np.abs(a['action']))<=1+1e-12)
        ck('previous_actual_action_X238',np.array_equal(s['X238'][:,195:237],np.vstack([np.zeros((1,42)),s['Y42'][:-1]])))
        ck('observation_targetspeed_unchanged',np.all(s['X238'][:,-1]==.075))
        ck('neutral960and1920post',np.all(s['command2'][:960]==0) and np.array_equal(s['post_prime'],np.arange(2880)>=960))
        elapsed=np.arange(2880)*.005;u=np.clip(elapsed-4.8,0,1);scale=1+(cfg['stride_scale']-1)*u*u*(3-2*u)
        ck('independent_commands_and_original_phase',np.allclose(s['command2'],np.column_stack([scale-1,np.zeros(2880)]),atol=1e-15,rtol=0) and np.allclose(s['command_speed_yaw'],np.column_stack([.075*scale,np.zeros(2880)]),atol=1e-15,rtol=0) and np.array_equal(s['source_phase'],elapsed))
        ck('original5800_physical_prefix_bitexact',all(np.array_equal(a[k][:5800],v[:5800]) for k,v in original.items()))
        ck('original960_samples_bitexact',all(np.array_equal(s[k][:960],osamples[k][:960]) for k in ('X238','Y42','command2','actual_time','source_phase','post_prime')))
        post=a['control_source']==5;t=a['time'][post];com=a['com'][post];loads=a['loads'][post];duration=t[-1]-t[0];delta=com[-1,:2]-com[0,:2]
        forward=float(delta[0]/duration);progress=float(np.linalg.norm(delta)/duration);den=loads.sum(0)
        slip=(loads*a['speed'][post]).sum(0)/np.maximum(den,1e-12)/max(progress,1e-9)
        support=float((loads.sum(1)/physics.W).mean());nf=float(a['nonfoot'][post].max());height=float(com[:,2].min())
        originalg=dict(support=support>=.90,nonfoot=nf<=.20*physics.W,com=height>=1.20,forward=forward>=.05,slip=float(slip.max())<=.5)
        observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);events=[]
        for i in np.flatnonzero(post):events.extend(observer.update(dict(time=float(a['time'][i]),loads=a['loads'][i],padz=a['padz'][i],relativefoot=a['relativefoot'][i]))['events'])
        events.extend(observer.finalize()['events']);counts={leg:sum(e['type']=='complete' and e['leg']==leg for e in events) for leg in LEGS}
        all6=all(v>=1 for v in counts.values())
        gates=dict(forward=forward>=cfg['forward_threshold'],slip=float(slip.max())<=.5,support=support>=.90,com=height>=1.20,strict_nonfoot_zero=nf==0,valid_slip=bool(np.all(den>1e-12) and progress>1e-9),all6cycles=all6)
        ck('original_gates_recomputed',originalg==r['gates'] and bool(all(originalg.values()))==r['original_all5_gates'])
        ck('stride_gates_recomputed',gates==r['stride_gates'] and bool(all(gates.values()))==r['all_stride_gates'])
        ck('exact_metrics',forward==r['forward_signed_native_per_s'] and float(slip.max())==r['slip_ratio_max'] and support==r['foot_support_mean'] and nf==r['nonfoot_force_max_native'] and height==r['com_height_min'])
        ck('slip_original_denominator',np.array_equal(den,r['slip_load_denominator_native']))
        ck('sixleg_counts_and_events_replayed',counts==r['cycles']['complete_per_leg'] and events==r['cycles']['events'])
        ck('saved_artifact_hashes',all(sha(p)==h for p,h in r['artifacts'].items()))
        yy=np.unwrap(y['yaw_world']);dyaw=float(yy[-1]-yy[4799])
        ck('yaw_alignment',np.array_equal(y['time'],a['time'][1000:]) and abs(dyaw-r['actual_signed_yaw_rad'])<1e-12)
        recomputed[name]=dict(original_gates=originalg,original_all5=bool(all(originalg.values())),stride_gates=gates,all_stride_gates=bool(all(gates.values())),
            forward_threshold=cfg['forward_threshold'],forward=forward,slip_max=float(slip.max()),per_leg_slip=dict(zip(LEGS,map(float,slip))),
            load_denominator_native=den.tolist(),support_mean=support,com_min=height,nonfoot_max_native=nf,cycles=counts,actual_yaw_rad=dyaw)
        axes[0].plot(com[:,0]-a['com'][5799,0],com[:,1]-a['com'][5799,1],label=name)
        axes[1].plot(t-5.8,(yy[4800:]-yy[4799])*180/np.pi,label=name)
        axes[2].plot(t[1000:]-5.8,(com[1000:,0]-com[:-1000,0])/(t[1000:]-t[:-1000]),label=name)
        for p in folder.glob('*'):
            if p.is_file():artifacts[str(p)]=sha(p)
    axes[0].set(xlabel='Actual delta worldX (native)',ylabel='Actual delta worldY (native)',title='Complete modified physical window')
    axes[1].set(xlabel='Seconds after prefix',ylabel='Actual thorax yaw delta (degrees)',title='Actual unwrapped yaw')
    axes[2].set(xlabel='Seconds after prefix',ylabel='Trailing1s signedX speed native/s',title='Fixed cadence, shortened sagittal stride')
    for threshold,color in ((.05,'red'),(.04,'gray'),(.025,'black')):axes[2].axhline(threshold,color=color,ls=':',label=str(threshold))
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.suptitle('Algorithm teacher only | original .8s cadence +fixed4s cycle timeout | no training / neural autonomy')
    fig.tight_layout();fig.savefig(HERE/'ACTUAL_DIAGNOSTIC.png',dpi=150);plt.close(fig)
    for p in (HERE/'PREREG.json',HERE/'api.py',HERE/'run.py',HERE/'verify.py',HERE/'check_reference.py',HERE/'REFERENCE_CHECKS.json',HERE/'INPUT_IDENTITY.json',HERE/'SWEEP_RESULT.json',HERE/'ACTUAL_DIAGNOSTIC.png'):artifacts[str(p)]=sha(p)
    result=dict(all_integrity_checks_passed=all(checks.values()),checks=checks,recomputed=recomputed,artifact_sha256=artifacts,
        scope='Freshprocess saved evidence replay, no third physics trajectory/fullnativeD1-D8')
    identity.atomic_json(HERE/'VERIFICATION.json',result)
    # Default curriculum admission must match independent physical gates.
    admission={}
    for cfg in CONDITIONS:
        admitted=True
        try: samples=load_verified_samples(cfg['name'])
        except ValueError:admitted=False
        admission[cfg['name']]=dict(admitted=admitted,matches_recomputed=admitted==recomputed[cfg['name']]['all_stride_gates'])
    identity.atomic_json(HERE/'CURRICULUM_ADMISSION.json',admission)
    print(json.dumps(result,indent=2));assert result['all_integrity_checks_passed'];assert all(v['matches_recomputed'] for v in admission.values())
if __name__=='__main__':main()
