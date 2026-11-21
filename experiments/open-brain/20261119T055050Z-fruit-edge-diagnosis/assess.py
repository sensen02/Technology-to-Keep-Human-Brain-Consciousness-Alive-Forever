"""Summarize the event order and response to selective body-input clipping."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261119T052620Z-hunger-foraging'/'VERIFIED'
FAR='food_ground_far'


def inspect(case):
    p=HERE/'RESULTS'/case
    d=json.loads((p/'DIAGNOSTIC.json').read_text())
    q=np.load(p/'TRACE.npy',allow_pickle=False)
    body=np.load(p/'TORSO_ANGLES.npy',allow_pickle=False)
    walk=np.load(p/'APPLIED_WALK.npy',allow_pickle=False)
    with np.load(p/'CONTACTS.npz',allow_pickle=False) as s:
        stamps,pairs,physical=[s[name].copy() for name in ('time','pair','physical')]
    selected=pairs[:,1]==FAR
    positions={}
    for side in ('left','right'):
        relevant=selected & (pairs[:,0]=='walker/claw_T1_'+side)
        first={}
        for name,key in (('top',physical[:,3]>.8),('side',physical[:,3]<.35)):
            ix=np.flatnonzero(relevant & key)
            first[name]=float(stamps[ix[0]]) if len(ix) else None
        positions[side]=first
    impact=np.flatnonzero((pairs[:,1]=='orchard_terrain') & (pairs[:,0]=='walker/tibia_T2_right'))
    interval=(4.42,5.26) if case.startswith('FADE') else (2.30,3.24)
    sample=q[q[:,0]>=interval[0]]
    update=walk[(walk[:,0]>=4.66)&(walk[:,0]<4.74),4]
    if len(update):
        close_yaw=float(np.mean(update))
    else:
        close_yaw=None
    after_zero=np.flatnonzero((walk[:,0]>=4.66)&(walk[:,0]<=4.90)&(walk[:,4]<=0.))
    from_prior=BASE/d['relative_to_original_drive']
    prior=np.load(from_prior/'TRACE.npy',allow_pickle=False)
    equal_before=None
    if d['intervention_starts_s'] is not None:
        n=round(d['intervention_starts_s']/.00005)
        equal_before=bool(np.array_equal(q[:n],prior[:n]))
    mouth=sample[sample[:,7]==1]
    d.update(first_valid_oral_contact_s=float(mouth[0,0]) if len(mouth) else None,
        valid_oral_contact_ms=float(len(mouth)*.05),
        first_foreclaw_top_side_s=positions,max_post_start_abs_roll_deg=float(
        np.max(abs(np.rad2deg(body[body[:,0]>=interval[0],2])))),
        applied_yaw_mean_4_66_4_74_rad_s=close_yaw,
        first_inward_yaw_after_crossing_s=float(walk[after_zero[0],0]) if len(after_zero) else None,
        first_right_mid_tibia_terrain_s=float(stamps[impact[0]]) if len(impact) else None,
        terminal_planar_mm=(10*sample[-1,1:3]).tolist(),
        equal_before_intervention=equal_before)
    return d,q,body,walk


def main():
    labels=dict(FADE='原下降组',HUNGRY='原低饱足对照',FADE_YAW='限偏航',FADE_BRAKE='限推进',
                FADE_LATE_YAW='果缘取消偏航',FADE_LATE_BRAKE='已倾斜才限推进')
    colors=dict(FADE='#aa412c',HUNGRY='#147d69',FADE_YAW='#3767a2',FADE_BRAKE='#8c639d',
                FADE_LATE_YAW='#267f92',FADE_LATE_BRAKE='#ae8740')
    finished={name:inspect(name) for name in labels
              if (HERE/'RESULTS'/name/'DIAGNOSTIC.json').is_file()}
    for case in ('FADE','HUNGRY'):
        facts=finished[case][0]
        assert facts['reference_trace_exact'] and facts['reference_qpos_exact']
    control={name:values[0] for name,values in finished.items()}
    assert all(v['equal_before_intervention'] in (None,True) for v in control.values())
    (HERE/'FINDINGS.json').write_text(json.dumps(control,indent=2)+'\n')
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig,ax=plt.subplots(3,1,layout='constrained',sharex=True,figsize=(10,7.4))
    fig.suptitle('P1 · 第二颗地果边缘失稳',x=.02,ha='left',fontsize=17)
    for name,(d,trace,body,walk) in finished.items():
        if name=='HUNGRY':
            continue
        color=colors[name]
        style='--' if name=='FADE_BRAKE' else '-'
        ax[0].plot(walk[:,0],walk[:,4],color=color,ls=style,lw=1.7,label=labels[name])
        ax[1].plot(body[:,0],np.abs(np.rad2deg(body[:,2])),color=color,ls=style,lw=1.6)
        ax[2].plot(trace[::12,0],trace[::12,3]*10,color=color,ls=style,lw=1.6)
    left=control['FADE']['first_foreclaw_top_side_s']['left']['top']
    right=control['FADE']['first_foreclaw_top_side_s']['right']['side']
    impact=control['FADE']['first_nonfoot'][0]
    for axis in ax:
        for t in (left,right,impact):
            axis.axvline(t,color='#657171',lw=.84,ls=':',alpha=.7)
        axis.grid(alpha=.16)
        axis.spines[['top','right']].set_visible(False)
    ax[0].set(ylabel='偏航（rad/s）',xlim=(4.42,5.16),ylim=(-2.7,2.7))
    ax[0].legend(loc='lower right',frameon=False,ncol=2,fontsize=9)
    ax[1].set(ylabel='横滚绝对值（°）')
    ax[2].set(ylabel='身体质心高度（mm）',xlabel='模拟时间（秒）')
    for t,label in ((left,'左前爪踩上果面'),(right,'右前爪碰侧壁'),(impact,'首个非足碰撞')):
        ax[2].text(t+.006,.045,label,rotation=90,transform=ax[2].get_xaxis_transform(),
                   va='bottom',color='#3e4b4b',fontsize=9)
    fig.savefig(HERE/'FRUIT_EDGE_DIAGNOSIS.png',dpi=160,facecolor='white')


if __name__=='__main__':
    main()
