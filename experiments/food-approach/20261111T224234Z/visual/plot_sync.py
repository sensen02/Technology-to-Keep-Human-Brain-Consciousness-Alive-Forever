"""Six actual clamp conditions; default CODE_ONLY; no network/physics execution."""
import argparse,os
from contracts import HERE,ROOT,B_IDS,sha,read,require,save_new,events,audited_arrays,bind_functional,verify_unchanged

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
    if not a.execute_after_parent_release:
        print('CODE_ONLY: no arrays/plot/model/physics executed');return
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(k)=='1',k+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={condition:bind_functional(audited_arrays(condition)) for condition in B_IDS}
    baseline=data['B_noodor'];hashes={str(HERE/'plot_sync.py'):sha(HERE/'plot_sync.py')}
    out=HERE/'SYNC_SIX_CONDITIONS';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(6,3,figsize=(18,19),sharex=True)
    records=[]
    for row,condition in enumerate(B_IDS):
        case=data[condition];t=case['trace'];d=case['decisions'];r=case['result'];hashes.update(case['source_hashes'])
        require(np.array_equal(t['time'],baseline['trace']['time']),'unmatched baseline time axis')
        tx=t['time']-5.8;dx=d['time']-5.8;y=np.degrees(case['yaw']);by=np.degrees(baseline['yaw']);difference=y-by
        a0,a1,a2=axes[row]
        a0.plot(dx,d['receptors'][:,0],label='z left (causal)',color='#2563eb',lw=1.2);a0.plot(dx,d['receptors'][:,1],label='z right (causal)',color='#dc2626',lw=1.2)
        # Held concentrations influence next response; do not label as simultaneous neural input.
        a0.plot(dx,d['concentration'][:,0],label='held C left -> next z',color='#2563eb',ls=':',alpha=.6,lw=.7);a0.plot(dx,d['concentration'][:,1],label='held C right -> next z',color='#dc2626',ls=':',alpha=.6,lw=.7)
        a1.plot(dx,d['raw_turn'],label='behavior raw turn',color='#9333ea',lw=1.4);a1.plot(dx,d['command'][:,1],label='applied command turn',color='#16a34a',ls='--',lw=.9)
        a1.set_ylim(-1.05,1.05)
        a2.plot(tx,y,label='body delta yaw from handover',color='#2563eb',lw=1.2);a2.plot(tx,by,label='no-odor delta yaw',color='#737373',ls='--',lw=1.)
        a2.plot(tx,difference,label='body minus no-odor',color='#ea580c',ls=':',lw=1.3)
        for ax in (a0,a1,a2):
            ax.axvspan(-5.8,0,color='#64748b',alpha=.06);ax.axvline(0,color='black',ls='--',lw=.8);ax.grid(alpha=.2)
            for label,x in events(condition):ax.axvline(x,color='#a16207',ls=':',lw=.6,alpha=.65)
            if condition!='B_noodor':
                ax.axvspan(.6,.8,color='#eab308',alpha=.12)
                ax.axvspan(4.8,5.,color='#eab308',alpha=.12) if condition in B_IDS[-2:] else ax.axvspan(3.6,3.8,color='#eab308',alpha=.12)
        original5=bool(r['autonomous']['all_gates']);functional=case['visual_functional_pass']
        a0.set_ylabel(condition+'\nreceptor z / held C',fontsize=9)
        a1.set_ylabel('Neural turn')
        a2.set_ylabel('Yaw change (deg)')
        a2.set_title('Original5 '+('PASS' if original5 else 'FAIL')+' | functional '+('PENDING matched summary' if functional is None else 'PASS' if functional else 'FAIL'),fontsize=9)
        if row==0:
            a0.legend(loc='upper left',fontsize=7);a1.legend(loc='upper left',fontsize=7);a2.legend(loc='upper left',fontsize=7)
        records.append(dict(condition=condition,run=str(case['run']),result_sha256=sha(case['run']/'RESULT.json'),audit_sha256=sha(case['run']/'VERIFY.json'),engineering_pass=r['engineering_pass'],original_five_pass=original5,functional_pass=functional,functional_source=case['visual_functional_source'],functional_flags=case['visual_functional_flags'],functional_metrics=case['visual_functional_metrics'],slip=r['autonomous']['slip_ratio_max'],body_yaw_end_deg=float(y[-1]),noodor_yaw_end_deg=float(by[-1]),difference_end_deg=float(difference[-1]),events_ANN_seconds=events(condition)))
    for ax in axes[-1]:ax.set_xlabel('ANN-relative time (s), handover = 0; PRE-ANN shown at negative times')
    fig.suptitle('CONTROLLED BILATERAL ODOR / CONCENTRATION CLAMP — NOT NAVIGATION\nFull saved 1s settle + 4.8s teacher + 9.6s ANN; held C updates next causal z; no body receptor-cut in B6',fontsize=13)
    fig.text(.5,.005,'Event boundaries: ON .60/.80; OFF 3.60/3.80; reversal 4.80/5.00 (ramp start/end, ANN seconds). Samples are correlated time points, not independent replicates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.022,1,.962));path=out/'SIX_CONDITIONS_SYNC.png';fig.savefig(path,dpi=150);plt.close(fig)
    verify_unchanged(hashes)
    save_new(out/'PLOT.json',dict(status='SAVED_OUTPUT_PLOT_COMPLETE_REVIEW_PENDING',plot=str(path),plot_sha256=sha(path),source_hashes=hashes,rows=records,baseline='B_noodor',time_axis='actual_time_s-5.8; no interpolation or time shift',concentration_clamp_not_navigation=True,body_receptor_cut_performed=False,offline_cut_not_body_control=True,model_forward_calls=0,Core_constructions=0,physics_steps=0,visual_review='PENDING_PARENT_OPENING'))
    print(str(out/'PLOT.json'))
if __name__=='__main__':main()
