"""Fresh-process saved evidence replay of amended control+two fixed candidates."""
import json
import numpy as np
from api import HERE,CONDITIONS,load_verified_samples
import physics,identity
from cycle_reward import CycleTracker,CycleConfig
LEGS=physics.LEGS

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return identity.sha256(p)

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text());checks={};recomputed={};artifacts={};prefixes=[]
    checks['all_frozen_sources_and_previous_experiments_unchanged']=all(sha(p)==h for p,h in frozen['source_hashes'].items())
    ref=json.loads((HERE/'REFERENCE_CHECKS.json').read_text())
    checks['preflight_all_checks_and_frozen_source_binding']=ref['all_passed'] and all(ref['checks'].values()) and all(sha(p)==h for p,h in ref['source_hashes'].items())
    original=load(HERE.parent/'teacher'/'left_full'/'TRACE_1MS.npz');originals=load(HERE.parent/'teacher'/'left_full'/'SAMPLES_5MS.npz')
    fig,axes=plt.subplots(1,3,figsize=(16,4.7));control_forward=None;control_passed=None
    for cfg in CONDITIONS:
        name=cfg['name'];folder=HERE/name;r=json.loads((folder/'RESULT.json').read_text());a=load(folder/'TRACE_1MS.npz');s=load(folder/'SAMPLES_5MS.npz');y=load(folder/'YAW_1MS.npz')
        def ck(k,v):checks[name+'_'+k]=bool(v)
        ck('runtime_inputs',identity.verify(json.loads((folder/'IDENTITY.json').read_text())))
        ck('complete_no_earlystop_or_exception',len(a['time'])==15400 and r['early_stop'] is None and r['exception'] is None)
        ck('trace_time_and_body',np.allclose(a['time'],np.arange(1,15401)*.001,atol=1e-8,rtol=0) and a['qpos'].shape==(15400,49) and a['qvel'].shape==(15400,48))
        ck('sample_shapes_and_actualtimes',s['X238'].shape==(2880,238) and s['Y42'].shape==(2880,42) and s['command2'].shape==(2880,2) and all(len(v)==2880 for v in s.values()) and np.allclose(s['actual_time'],1+np.arange(2880)*.005,atol=1e-8,rtol=0))
        ck('finite',all(np.isfinite(v).all() for v in list(a.values())+list(s.values())+list(y.values())))
        ck('zero_settle_and_external',np.all(a['action'][:1000]==0) and np.all(a['external_body_force_max']==0) and np.all(a['external_generalized_force_max']==0))
        ck('source_and_phaseboundaries',np.array_equal(a['control_source'],np.r_[np.zeros(1000,int),np.full(4800,2),np.full(9600,5)]))
        ck('everyactual_action_tick',np.array_equal(a['action'][1000:],np.repeat(s['Y42'],5,axis=0)) and np.array_equal(a['action'],a['actual_requested_action']) and np.max(np.abs(a['action']))<=1+1e-12)
        ck('previousaction_and_originaltargets',np.array_equal(s['X238'][:,195:237],np.vstack([np.zeros((1,42)),s['Y42'][:-1]])) and np.all(s['X238'][:,-1]==.075))
        ck('neutral960_and1920post',np.all(s['command2'][:960]==0) and np.array_equal(s['post_prime'],np.arange(2880)>=960))
        elapsed=np.arange(2880)*.005;dt=np.maximum(0,elapsed-4.8);u=np.minimum(dt,1);scale=1+(cfg['speed_scale']-1)*u*u*(3-2*u)
        phase=elapsed+(cfg['speed_scale']-1)*(u**3-.5*u**4+np.maximum(0,dt-1))
        ck('independent_wholephase_commands',np.allclose(s['command2'],np.column_stack([scale-1,np.zeros(2880)]),atol=1e-15,rtol=0) and np.allclose(s['command_speed_yaw'],np.column_stack([.075*scale,np.zeros(2880)]),atol=1e-15,rtol=0) and np.allclose(s['source_phase'],phase,atol=1e-14,rtol=0))
        ck('original5800physicalprefix_bitexact',all(np.array_equal(a[k][:5800],v[:5800]) for k,v in original.items()))
        ck('original960sampleprefix_bitexact',all(np.array_equal(s[k][:960],originals[k][:960]) for k in ('X238','Y42','command2','actual_time','source_phase','post_prime')))
        prefixes.append({k:a[k][:5800].copy() for k in ('qpos','qvel','action')})
        post=a['control_source']==5;t=a['time'][post];com=a['com'][post];loads=a['loads'][post];duration=t[-1]-t[0];delta=com[-1,:2]-com[0,:2]
        forward=float(delta[0]/duration);progress=float(np.linalg.norm(delta)/duration);den=loads.sum(0);vt=(loads*a['speed'][post]).sum(0)/np.maximum(den,1e-12);slip=vt/max(progress,1e-9)
        support=float((loads.sum(1)/physics.W).mean());nf=float(a['nonfoot'][post].max());height=float(com[:,2].min())
        originalg=dict(support=support>=.90,nonfoot=nf<=.20*physics.W,com=height>=1.20,forward=forward>=.05,slip=float(slip.max())<=.5)
        observer=CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8);events=[]
        for i in np.flatnonzero(post):events.extend(observer.update(dict(time=float(a['time'][i]),loads=a['loads'][i],padz=a['padz'][i],relativefoot=a['relativefoot'][i]))['events'])
        events.extend(observer.finalize()['events']);counts={leg:sum(e['type']=='complete' and e['leg']==leg for e in events) for leg in LEGS};all6=all(v>=1 for v in counts.values())
        fg=dict(**originalg,strict_nonfoot_zero=nf==0,valid_slip=bool(np.all(den>1e-12) and progress>1e-9),all6cycles=all6)
        iscontrol=name=='straight_control_100'
        if iscontrol:control_forward=forward;control_passed=bool(all(fg.values()))
        else:fg.update(matched_control_passed=control_passed,deceleration=forward<=.95*control_forward)
        ck('original_five_recomputed',originalg==r['gates'] and bool(all(originalg.values()))==r['original_all5_gates'])
        ck('functional_gates_recomputed',fg==r['functional_gates'] and bool(all(fg.values()))==r['all_functional_gates'])
        ck('all_metrics_exact',forward==r['forward_signed_native_per_s'] and float(slip.max())==r['slip_ratio_max'] and support==r['foot_support_mean'] and nf==r['nonfoot_force_max_native'] and height==r['com_height_min'] and np.array_equal(den,r['slip_load_denominator_native']))
        ck('all6events_replayed',counts==r['cycles']['complete_per_leg'] and events==r['cycles']['events'])
        ck('matched_control_rule',r['is_matched_control']==iscontrol and (r['matched_control_forward'] is None if iscontrol else r['matched_control_forward']==control_forward and r['deceleration_maximum_forward']==.95*control_forward and r['achieved_forward_fraction_of_control']==forward/control_forward))
        ck('artifact_hashes',all(sha(p)==h for p,h in r['artifacts'].items()))
        yy=np.unwrap(y['yaw_world']);dy=float(yy[-1]-yy[4799]);ck('actualyaw',np.array_equal(y['time'],a['time'][1000:]) and abs(dy-r['actual_signed_yaw_rad'])<1e-12)
        recomputed[name]=dict(is_control=iscontrol,original_gates=originalg,original_all5=bool(all(originalg.values())),functional_gates=fg,all_functional_gates=bool(all(fg.values())),forward=forward,slip_max=float(slip.max()),per_leg_slip=dict(zip(LEGS,map(float,slip))),per_leg_vt=dict(zip(LEGS,map(float,vt))),
            support_mean=support,com_min=height,nonfoot_max_native=nf,cycles=counts,actual_yaw_rad=dy,control_forward=control_forward,deceleration_max_forward=.95*control_forward,forward_fraction_of_control=forward/control_forward)
        axes[0].plot(com[:,0]-a['com'][5799,0],com[:,1]-a['com'][5799,1],label=name)
        axes[1].plot(t-5.8,(yy[4800:]-yy[4799])*180/np.pi,label=name)
        axes[2].plot(t[1000:]-5.8,(com[1000:,0]-com[:-1000,0])/(t[1000:]-t[:-1000]),label=name)
        for p in folder.glob('*'):
            if p.is_file():artifacts[str(p)]=sha(p)
    checks['all3_actualphysicalprefixes_bitexact']=all(all(np.array_equal(prefixes[0][k],p[k]) for k in prefixes[0]) for p in prefixes[1:])
    axes[0].set(xlabel='Actualdelta worldX native',ylabel='Actualdelta worldY native',title='Same modified9.6s physical window')
    axes[1].set(xlabel='Seconds after prefix',ylabel='Actualsignedyaw degrees',title='Actualthoraxyaw')
    axes[2].set(xlabel='Seconds after prefix',ylabel='Trailing1s signedX native/s',title='Matchedcontrol and fixedmoderatespeeds')
    axes[2].axhline(.05,color='red',ls=':',label='original>=.05');axes[2].axhline(.95*control_forward,color='black',ls='--',label='95%control mean threshold')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.suptitle('Algorithmteacher only | matchedstraightcontrol+2fixedmoderatespeeds | no training/neuralautonomy')
    fig.tight_layout();fig.savefig(HERE/'ACTUAL_DIAGNOSTIC.png',dpi=150);plt.close(fig)
    for name in ('PREREG.json','AMENDMENT_BEFORE_EXECUTION.json','api.py','run.py','verify.py','check_reference.py','REFERENCE_CHECKS.json','INPUT_IDENTITY.json','SWEEP_RESULT.json','ACTUAL_DIAGNOSTIC.png'):artifacts[str(HERE/name)]=sha(HERE/name)
    result=dict(all_integrity_checks_passed=all(checks.values()),checks=checks,recomputed=recomputed,artifact_sha256=artifacts,matched_control_forward=control_forward,deceleration_maximum_forward=.95*control_forward)
    identity.atomic_json(HERE/'VERIFICATION.json',result)
    admission={}
    for cfg in CONDITIONS:
        admitted=True
        try:s=load_verified_samples(cfg['name'])
        except ValueError:admitted=False
        admission[cfg['name']]=dict(admitted=admitted,matches_recomputed=admitted==recomputed[cfg['name']]['all_functional_gates'])
    identity.atomic_json(HERE/'CURRICULUM_ADMISSION.json',admission)
    print(json.dumps(result,indent=2));assert result['all_integrity_checks_passed'];assert all(v['matches_recomputed'] for v in admission.values())
if __name__=='__main__':main()
