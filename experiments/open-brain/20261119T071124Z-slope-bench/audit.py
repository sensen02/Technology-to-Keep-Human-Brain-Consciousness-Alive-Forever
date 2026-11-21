"""Verify physical replay, command provenance, and incline diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from slope_solved import (HERE, SOURCE, WARMUP, WALK_END, STOP_END, END, make_environment)


def actual_neck_provenance(name):
    with np.load(SOURCE/name/'GROUND_DN.npz',allow_pickle=False) as data:
        emitted, arrived, delays = (data[k] for k in ('emitted','arrived','delay_ticks'))
        for i,d in enumerate(delays):
            assert not arrived[:d,i].any()
            assert np.array_equal(emitted[:-d,i],arrived[d:,i])
    return {'path':str(SOURCE/name/'GROUND_DN.npz'),'channels':len(delays),
            'delays_ms':(delays*.1).tolist(),
            'emitted_pulses':int(emitted.sum()),'arrived_pulses':int(arrived.sum())}


def normalized(m):
    name=m.split('/')[-1]
    return name.startswith(('head','thorax','abdomen','wing','rostrum','antenna','labrum','haustellum'))


def check_trial(folder, initial, bank):
    r=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'TRAJECTORY.npz',allow_pickle=False) as d:
        t, feet, drive, ids, start = (d[k] for k in ('trace','feet','drive','source_index','initial'))
        name=folder.name
        known=initial.get(name)
        if known is None:
            initial[name]=start
        else:
            assert np.array_equal(start,known),name
        around=(t[:,0]>=WARMUP)&(t[:,0]<WALK_END)
        expected=np.vstack([bank['FULL'][ids[t0:t1]] if quiet else bank['HUNGRY'][ids[t0:t1]]
            for t0,t1,quiet in ((0,1000,True),(1000,4700,False),(4700,7700,True),(7700,10000,False))])
        assert np.array_equal(drive,expected),folder
        assert np.isfinite(t).all() and np.isfinite(feet).all()
        moving=t[around]
        loss=np.flatnonzero((moving[:,9]<=2)&(moving[:,7]<.55)&(moving[:,10]>0))
    r['first_dynamic_fall_s']=float(moving[loss[0],0]) if len(loss) else None
    r['trunk_head_wing_contacts']=sum(v for name,v in r['nonfoot_by_body'].items() if normalized(name))
    r['native_nonfoot_ms']=r['native_nonfoot_samples']*r['physics_dt_s']*1000.
    return r


def plot(root):
    fig, axes = plt.subplots(3, 1, figsize=(9,8), sharex=True,layout='constrained')
    variants={'SOLVED':('Original body feedback','#952f3e'),
              'CONTACT_FRAME':('Change reference only','#806316'),
              'TANGENT':('Foot-estimated support coordinates','#175965'),
              'LOW_ADHESION':('Support coordinates, 35% adhesion','#584093')}
    folder='+20deg_yaw+00'
    for name,(label,color) in variants.items():
        p=root/name/folder/'TRAJECTORY.npz'
        if not p.is_file():
            continue
        with np.load(p) as d:
            t=d['trace']
        normal=np.deg2rad(20)
        axis=np.array([np.cos(normal),0.,np.sin(normal)])
        progress=10*(t[:,1:4]-t[np.searchsorted(t[:,0],WARMUP),1:4])@axis
        for panel,y in zip(axes,(progress,t[:,7],t[:,9]),strict=True):
            panel.plot(t[:,0],y,lw=1.35,label=label,color=color)
    axes[0].set_ylabel('Uphill displacement (mm)')
    axes[1].set_ylabel('Body up · support normal')
    axes[2].set_ylabel('Load-bearing feet')
    axes[2].set_xlabel('Physical time (s)')
    axes[0].legend(loc='lower left',frameon=False)
    for ax in axes:
        ax.axvspan(WALK_END,STOP_END,color='#999999',alpha=.11)
        ax.axvline(WARMUP,ls=':',c='#999999')
        ax.grid(axis='y',alpha=.20)
    fig.suptitle('P1 FlyBody 20° uphill · same recorded post-neck commands')
    fig.savefig(root/'INCLINE_TRACE.png',dpi=145,facecolor='white')
    plt.close(fig)


def render(root):
    from PIL import Image,ImageDraw
    from dm_control.mujoco import engine,wrapper
    variants=('SOLVED','TANGENT')
    times=(.20,.51,.88,1.40,1.94)
    w,h=320,218
    image=Image.new('RGB',(w*len(times),(h+27)*len(variants)),'#17282c')
    draw=ImageDraw.Draw(image)
    env,*_=make_environment(20,0)
    phy=env.physics
    opt=wrapper.MjvOption()
    opt.sitegroup[:]=0
    labels={'SOLVED':'Original','TANGENT':'Foot-estimated plane'}
    for row,var in enumerate(variants):
        p=root/var/'+20deg_yaw+00'/'TRAJECTORY.npz'
        with np.load(p) as d:
            poses=d['pose']
        for col,t in enumerate(times):
            index=round(t/.005)-1
            phy.data.qpos[:]=poses[index]
            phy.forward()
            center=phy.named.data.subtree_com['walker/'].copy()
            camera=engine.MovableCamera(phy,height=h,width=w)
            camera.set_pose(center,1.12,90.,-13.)
            f=Image.fromarray(camera.render(scene_option=opt))
            x,y=w*col,(h+27)*row
            image.paste(f,(x,y+26))
            draw.text((x+9,y+5),f'{labels[var]}  {t:.2f} s',fill='white')
    image.save(root/'BODY_FRAMES.png')
    env.close()


def main(args):
    root=args.root
    provenance={k:actual_neck_provenance(k) for k in ('HUNGRY','FULL')}
    bank={}
    for k in provenance:
        with np.load(SOURCE/k/'NEURAL.npz',allow_pickle=False) as d:
            bank[k]=d['ground']
    record={}
    start={}
    for set_name in args.groups:
        result={}
        for folder in sorted((root/set_name).iterdir()):
            if (folder/'RESULT.json').is_file():
                result[folder.name]=check_trial(folder,start,bank)
                same_base=root/'BASELINE'/folder.name/'TRAJECTORY.npz'
                if set_name=='SOLVED' and same_base.is_file():
                    with np.load(same_base) as old,np.load(folder/'TRAJECTORY.npz') as new:
                        assert np.array_equal(old['pose'],new['pose'])
                        assert np.array_equal(old['actions'],new['actions'])
                if set_name=='TANGENT_REPEAT' and folder.name in record['TANGENT']:
                    with np.load(root/'TANGENT'/folder.name/'TRAJECTORY.npz') as old, np.load(
                                  folder/'TRAJECTORY.npz') as new:
                        assert np.array_equal(old['trace'],new['trace'])
                        assert np.array_equal(old['pose'],new['pose'])
                        assert np.array_equal(old['actions'],new['actions'])
        record[set_name]=result
    (root/'AUDIT.json').write_text(json.dumps({'neural_sources':provenance,'cases':record},
                                               indent=2,allow_nan=False)+'\n')
    plot(root)
    if args.render:
        render(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',nargs='?',type=Path,default=HERE)
    parser.add_argument('--groups',nargs='+',required=True)
    parser.add_argument('--render',action='store_true')
    main(parser.parse_args())
