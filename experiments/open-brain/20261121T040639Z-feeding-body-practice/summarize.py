"""Summarize requested experiments from saved outputs and verify receipt consistency."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from practice import (OUT,PRIOR,HELD,demo,read_json,save_json,load_floor,T,local_axes,digest)


@torch.inference_mode()
def explain(path):
    info=read_json(path/'result.json')
    cfg=read_json(demo(info['seed'])/'RESULT.json')
    variant=path.name.split('-',3)[3]
    root=PRIOR/'adapted' if variant=='previous' else OUT/variant
    with np.load(path/'replay.npz') as z:
        body,mouth=z['markers'],z['mouth']
        intake=z['intake'];contact=z['mouth_contact']
        assert float(intake[-1])==info['intake']
        assert not np.any(np.diff(intake,prepend=0)[~contact]>0)
        assert abs(info['intake']-info['stock_loss'])<1e-9
        assert np.array_equal(z['brain_end'].sum(0),info['arrivals'])
        assert np.array_equal(z['brain_input'].sum(0),info['encoded'])
    floor=load_floor(root/'floor.pt')
    loads=floor(T(body[:-2]),T(body[1:-1])).numpy()
    goal=10*np.array([cfg['geometry']['goal_x'],cfg['geometry']['goal_y']])
    near=np.linalg.norm(mouth[:,:2]-goal,axis=1)<=4.55
    low=(mouth[:,2]>=.02)&(mouth[:,2]<=1.44)
    supported=(loads>=.5).sum(1)>=3
    upside=local_axes(T(body[1:-1]))[:,2,2].numpy()
    upright=upside>=.88
    tipping=np.flatnonzero(upside<.55)
    tail_support=(loads[-28:]>=.5).sum(axis=1).min()
    tail_v=np.linalg.norm(np.diff(body[-29:,0,:2],axis=0),axis=1)/.005
    held_stance=bool(tail_support>=4 and upside[-28:].min()>.94 and tail_v.max()<=.8)
    # Float conversion at the radius only affects samples exactly on the boundary.
    assert np.count_nonzero(contact!=(near&low&supported&upright))<=1
    return dict(trial=path.name,split='heldout' if info['seed'] in HELD else 'development',
                near_frames=int(near.sum()),near_low_frames=int((near&low).sum()),
                near_supported_frames=int((near&supported).sum()),teacher_intake=cfg['intake'],
                teacher_duration_s=cfg['duration_s'],meal_amount_pass=info['intake']>=.18,
                first_tip_s=float(tipping[0]*.005) if len(tipping) else None,
                tail_max_speed_mm_s=float(tail_v.max()),end_stance_pass=held_stance,
                full_meal_pass=bool(info['intake']>=.18 and held_stance and not len(tipping)),**info),mouth,goal


def main():
    p=argparse.ArgumentParser()
    p.add_argument('variant')
    p.add_argument('seeds',nargs='+',type=int)
    args=p.parse_args()
    current=OUT/args.variant
    model=read_json(current/'training.json')
    assert model['weights_sha256']==digest(current/'weights.pt')
    assert not set(model['train_seeds'])&set(HELD)
    fig,axs=plt.subplots(1,len(args.seeds),figsize=(4.3*len(args.seeds),4.6),layout='constrained',squeeze=False)
    rows=[]
    for ax,seed in zip(axs[0],args.seeds):
        cfg=read_json(demo(seed)/'RESULT.json')
        with np.load(demo(seed)/'external.npz') as z: oral=z['mouth_mm'][:,:2]
        ax.plot(*oral.T,label='Physical teacher',lw=1.5,color='#66748a')
        for arm,label,col in [('previous','Before practice','#be7738'),(args.variant,'After practice','#22846c')]:
            d=OUT/f'live-{seed}-on-{arm}'
            if not (d/'result.json').exists():continue
            info,mouth,goal=explain(d)
            rows.append(info)
            ax.plot(*mouth[:,:2].T,label=label,lw=1.5,color=col)
        ax.add_patch(Circle((10*cfg['geometry']['goal_x'],10*cfg['geometry']['goal_y']),4.55,color='#94be78',alpha=.22))
        ax.autoscale_view()
        ax.set_aspect('equal',adjustable='datalim')
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'{seed} · '+('held out' if seed in HELD else 'development'))
        ax.legend(loc='best',fontsize=8)
    if rows:
        fig.savefig(current/'foraging_paths.png',dpi=155)
        save_json(current/'closed-loop.json',rows)
        print([(r['trial'],round(r['closest_oral_center_mm'],3),r['near_frames'],r['near_supported_frames'],round(r['intake'],5)) for r in rows])
if __name__=='__main__':main()
