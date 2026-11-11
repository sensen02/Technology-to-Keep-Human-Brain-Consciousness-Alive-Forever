"""Saved arrays once; no MuJoCo import/model construction/integration."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'20261110T191449Z'
W=24.446354755878005
LEGS=['LF','LH','LM','RF','RH','RM']
CASES=[('Teacher left [algorithm PASS]',OLD/'teacher/left_full','#1b9e77','--'),('Teacher right [algorithm PASS]',OLD/'teacher/right_full','#66a61e','--'),('Residual500 left [neural FAIL]',OLD/'residual_candidate/EVAL_RESIDUAL500_left','#7570b3',':'),('Residual500 right [neural FAIL]',OLD/'residual_candidate/EVAL_RESIDUAL500_right','#e7298a',':'),('Fast1000 neutral [neural PASS]',ROOT/'EVAL_FAST_neutral','#333333','-'),('Fast1000 left [neural FAIL]',ROOT/'EVAL_FAST_left','#d95f02','-'),('Fast1000 right [neural FAIL]',ROOT/'EVAL_FAST_right','#e31a1c','-')]

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    hashes={};allmetrics={};data=[]
    fig,axes=plt.subplots(3,3,figsize=(22,16),constrained_layout=True)
    for label,folder,color,style in CASES:
        rp=folder/'RESULT.json';r=json.loads(rp.read_text());a=r.get('autonomous',r)
        hashes[str(rp)]=sha(rp)
        tp=folder/('TRACE_1MS.npz' if 'Teacher' in label else 'trace_1ms.npz')
        hashes[str(tp)]=sha(tp)
        assert hashes[str(tp)]==r['trace']['sha256'],('trace hash',folder)
        # Exactly one decompression per saved array, not per time row.
        with np.load(tp,allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
        assert len(t['time'])==15400 and t['com'].shape==(15400,3)
        assert np.isfinite(t['com']).all() and np.isfinite(t['speed']).all()
        assert np.max(t['external_body_force_max'])==0 and np.max(t['external_generalized_force_max'])==0
        s=slice(5800,None);time=t['time'][s]-t['time'][5799];com=t['com'][s];loads=t['loads'][s];speed=t['speed'][s]
        delta=com[-1,:2]-com[0,:2];duration=time[-1]-time[0];xy=float(np.linalg.norm(delta)/duration);forward=float(delta[0]/duration)
        vt=(loads*speed).sum(0)/loads.sum(0);ratios=vt/max(xy,1e-9)
        assert np.allclose(ratios,[a['per_leg_slip_ratio'][leg] for leg in LEGS],rtol=1e-12,atol=1e-12)
        assert np.isclose(forward,a['forward_signed_native_per_s'],rtol=1e-12,atol=1e-12)
        assert np.isclose(xy,a['forward_speed_native_per_s'],rtol=1e-12,atol=1e-12)
        # Free-root quaternion is saved wxyz, yaw from exact stored orientation.
        q=t['qpos'][:,3:7];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
        dyaw=float(yaw[-1]-yaw[5799]);expectedyaw=r.get('yaw_change_rad',r.get('actual_signed_yaw_rad'))
        assert abs(dyaw-expectedyaw)<1e-10,(label,dyaw,expectedyaw)
        support=loads.sum(1)/W
        assert np.isclose(support.mean(),a['foot_support_mean'],rtol=1e-12)
        assert np.isclose(com[:,2].min(),a['com_height_min'],rtol=1e-12)
        # Causal trailing1s velocity, warmup history included; no endpoint trimming.
        vel=(t['com'][1000:,0]-t['com'][:-1000,0])/(t['time'][1000:]-t['time'][:-1000])
        x=com[:,0]-t['com'][5799,0];y=com[:,1]-t['com'][5799,1]
        axes[0,0].plot(x,y,color=color,ls=style,label=label);axes[0,0].scatter(x[-1],y[-1],color=color,s=20)
        axes[0,1].plot(time,vel[4800:],color=color,ls=style)
        axes[0,2].plot(time,np.rad2deg(yaw[s]-yaw[5799]),color=color,ls=style)
        axes[1,0].plot(np.arange(6),ratios,color=color,ls=style,marker='o')
        axes[1,1].plot(np.arange(6),vt,color=color,ls=style,marker='o')
        axes[1,2].scatter(forward,xy,color=color,marker='o' if 'Fast' in label else 'x',s=65)
        axes[2,0].plot(time,support,color=color,ls=style,alpha=.8,lw=.8)
        axes[2,1].plot(time,com[:,2],color=color,ls=style,lw=.9)
        # Absolute load-weighted tangent speed over trailing1s; no ratio normalization.
        numerator=(loads*speed).sum(1);denominator=loads.sum(1)
        numsum=np.r_[0,np.cumsum(numerator)];densum=np.r_[0,np.cumsum(denominator)]
        end=np.arange(1,len(time)+1);start=np.maximum(0,end-1000)
        meanvt=(numsum[end]-numsum[start])/(densum[end]-densum[start])
        axes[2,2].plot(time,meanvt,color=color,ls=style)
        leg=np.argmax(ratios)
        metrics=dict(kind='algorithm teacher' if 'Teacher' in label else 'neural',reported_status=r.get('status','TEACHER_ENGINEERING_ONLY'),all_five=bool(a['all_gates']),gates=a['gates'],all_six=bool(r.get('autonomous_cycles',r.get('cycles',{})).get('all_legs_completed',False)),signed_forward=forward,slip_denominator_net_xy_speed=xy,net_com_xy_delta=delta.tolist(),integrated_com_path_length=float(np.linalg.norm(np.diff(com[:,:2],axis=0),axis=1).sum()),yaw_change_rad=dyaw,foot_load_mean_W=float(support.mean()),foot_load_min_W=float(support.min()),COM_min=float(com[:,2].min()),max_slip=float(ratios.max()),max_slip_leg=LEGS[leg],per_leg_load_weighted_tangent_speed=dict(zip(LEGS,map(float,vt))),per_leg_slip=dict(zip(LEGS,map(float,ratios))),per_leg_loaded_numerator_share=dict(zip(LEGS,map(float,((loads*speed)*(loads>=.1*W)).sum(0)/np.maximum((loads*speed).sum(0),1e-30)))),nonfoot_peak=float(t['nonfoot'][s].max()),trace_sha256=hashes[str(tp)],window='all9600 posthandover endpoints; ratio duration9.599s',extra_forces_zero=True)
        allmetrics[label]=metrics;data.append((label,metrics))
    ax=axes[0,0];ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='COM delta world X [model units]',ylabel='COM delta world Y [model units]',title='Actual saved full-window COM paths: same fixed axes');ax.legend(fontsize=8)
    axes[0,1].set(title='Actual signed world-X velocity, trailing1s',xlabel='Seconds after handover',ylabel='Model units / s');axes[0,1].axhline(.05,color='red',ls='--',label='original forward .05')
    axes[0,2].set(title='Actual root/thorax yaw change',xlabel='Seconds after handover',ylabel='Degrees')
    for ax in (axes[1,0],axes[1,1]):ax.set_xticks(range(6),LEGS)
    axes[1,0].set_yscale('log');axes[1,0].axhline(.5,color='red',lw=2,ls='--');axes[1,0].set(title='Original per-leg slip ratio: true limit .5 (log)',ylabel='Load-weighted tangent speed / net XY speed')
    axes[1,1].set_yscale('log');axes[1,1].set(title='Slip NUMERATOR: absolute load-weighted foot speed',ylabel='Model units / s (log)')
    axes[1,2].set(title='Signed forward is NOT slip denominator',xlabel='Signed world-X mean speed',ylabel='Net XY displacement norm / duration');axes[1,2].axvline(.05,color='red',ls='--');axes[1,2].axvline(0,color='grey',lw=.8)
    axes[2,0].set(title='Actual foot load at every1ms endpoint',xlabel='Seconds after handover',ylabel='Sum foot load / W');axes[2,0].axhline(.9,color='red',ls='--')
    axes[2,1].set(title='Actual center-of-mass height',xlabel='Seconds after handover',ylabel='Native model units');axes[2,1].axhline(1.2,color='red',ls='--')
    axes[2,2].set(title='Absolute all-foot loaded tangent speed, trailing1s',xlabel='Seconds after handover',ylabel='Model units / s')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('REAL SAVED PHYSICS — Fast1000 turns FAIL; neutral PASS; no food behavior\nAlgorithm teacher, prior residual500 and fast1000 are distinctly labeled. No integration / no trajectory correction.',fontsize=17)
    plot=HERE/'ACTUAL_NEURAL_TURN_COMPARISON.png';fig.savefig(plot,dpi=150);plt.close(fig)
    neutral=allmetrics['Fast1000 neutral [neural PASS]'];comparison={}
    for side in ('left','right'):
        fast=allmetrics[f'Fast1000 {side} [neural FAIL]'];prev=allmetrics[f'Residual500 {side} [neural FAIL]'];teacher=allmetrics[f'Teacher {side} [algorithm PASS]']
        comparison[side]=dict(fast_signed_forward_negative=fast['signed_forward']<0,denominator_ratio_fast_to_teacher=fast['slip_denominator_net_xy_speed']/teacher['slip_denominator_net_xy_speed'],per_leg_numerator_ratio_fast_to_teacher={leg:fast['per_leg_load_weighted_tangent_speed'][leg]/teacher['per_leg_load_weighted_tangent_speed'][leg] for leg in LEGS},per_leg_numerator_ratio_fast_to_prior={leg:fast['per_leg_load_weighted_tangent_speed'][leg]/prev['per_leg_load_weighted_tangent_speed'][leg] for leg in LEGS})
    assert all(sha(p)==h for p,h in hashes.items()),'read-only source drift'
    report=dict(status='SAVED_ARRAY_COMPARISON_VERIFIED_NOT_FOOD_SUCCESS',physics_steps=0,imports_physics_backend=False,source_hashes=hashes,script_sha256=sha(__file__),all_traces_hashes_and_saved_summary_recomputed=True,cases=allmetrics,comparisons=comparison,explanation='Original slip denominator is nonnegative norm(net XY displacement)/duration, not signed world-X progress. Negative forward causes forward gate failure, not a negative slip denominator. Numerator and denominator are quantified independently.',plot=str(plot),plot_sha256=sha(plot),scope='1ms endpoint engineering evidence, not full native D1-D8 or physiological validity')
    (HERE/'METRICS.json').write_text(json.dumps(report,indent=2)+'\n')
    desktop=Path('/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/续轮1-神经转向物理对照-未达标.png');shutil.copyfile(plot,desktop);assert sha(desktop)==sha(plot)
    (HERE/'DELIVERY.json').write_text(json.dumps(dict(source=str(plot),desktop=str(desktop),sha256=sha(plot)),indent=2)+'\n')
    print(json.dumps(dict(plot=str(plot),metrics=str(HERE/'METRICS.json'),comparisons=comparison,desktop=str(desktop)),indent=2))
if __name__=='__main__':main()
