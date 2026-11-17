"""Score a deliberately stopped physical recovery separately from walking."""
import argparse
from pathlib import Path

import numpy as np

from evidence import contrast
from physical import RF,RM,W,bench,read,stan
from qualify import first_persistent


def evaluate(path):
    r=read(path/'RESULT.json')
    a=read(path/'NATIVE_AUDIT.json')
    if a['native_sha256']!=stan.sha(path/'NATIVE.npz'):
        raise ValueError('Raw native identity differs')
    if r['trace_sha256']!=stan.sha(path/'TRACE.npz'):
        raise ValueError('Trajectory identity differs')
    with np.load(path/'TRACE.npz',allow_pickle=False) as file:
        z={key:file[key].copy() for key in file.files}
    with np.load(path/'NATIVE.npz',allow_pickle=False) as file:
        n={key:file[key].copy() for key in ('forces','pose','rf_floor')}
    start=r['teacher_engagement_ms']
    end=len(z['time'])-1
    stance=bench.standing(z,start if start is not None else 500,end)
    upper=n['rf_floor'][:,1]/W
    xmin,xmax,ymin,ymax=r['scenario']['opening']
    xyz=z['rf_pad_xyz'][:-1]
    outside=(xyz[:,0]<xmin)|(xyz[:,0]>xmax)|(xyz[:,1]<ymin)|(xyz[:,1]>ymax)
    relocated=first_persistent(np.repeat(outside,80)&(upper>.02),200*80,
                              start=start*80 if start is not None else 0)
    f=n['forces'][-200*80:]
    pose=n['pose'][-200*80:]
    forces=n['forces'][:,:6]/W
    t0=start*80 if start is not None else 500*80
    finite_support=np.mean(forces[t0:t0+200*80].sum(1)-forces[t0:t0+200*80,RF])
    terminal=dict(
        native_mean_foot_support=float(f[:,:6].sum(1).mean()/W),
        native_nonfoot_peak_W=float(f[:,12].max()/W),
        native_three_more_fraction=float(np.mean((f[:,:6]>=.02*W).sum(1)>=3)),
        native_min_COM=float(pose[:,2].min()),
        native_min_upright_dot=float(pose[:,5].min()))
    stance_ok=bool(stance['passed'] and terminal['native_mean_foot_support']>=.9 and
                  terminal['native_nonfoot_peak_W']<.01 and terminal['native_three_more_fraction']>=.95 and
                  terminal['native_min_COM']>=1.2 and terminal['native_min_upright_dot']>=np.cos(np.deg2rad(20)))
    event=r['event'].get('first_pad_drop_ms')
    return dict(audited_native_steps=a['native_integrations'],safe_stance_pass=stance_ok,physical_zero_external_force=a['zero_extra_force'],
                path=str(path.resolve()),act_after_loss=bool(start is not None and event is not None and start>=event),
                actuating_ms=start,first_pad_drop_ms=event,native_200ms_upper_support_reloaded=int(relocated)/80. if relocated is not None else None,
                retracted_height_difference=float(z['padz'][start:,RF].max()-z['padz'][event:,RF].min()) if start is not None and event is not None else None,
                other_five_support_first_200ms_mean_W=float(finite_support),terminal_stance=stance,terminal_native=terminal,
                standard_full_continued_walk_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('full',type=Path)
    p.add_argument('lift_only',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    a,b=evaluate(args.full),evaluate(args.lift_only)
    delta=contrast(args.full,args.lift_only,200)
    work=bool(a['safe_stance_pass'] and a['physical_zero_external_force'] and a['act_after_loss']
          and a['native_200ms_upper_support_reloaded'] is not None and a['other_five_support_first_200ms_mean_W']>.90
          and delta.get('mean_right_mid_increase_W',0.)>.008)
    stan.put(args.output,dict(scope='sensor triggered deliberate stationary safe stop; bounded 6.45 s protocol, separate from walking',
        localized_safe_stop_and_support_gate=work,full=a,rm_ablation=b,paired_support=delta))
