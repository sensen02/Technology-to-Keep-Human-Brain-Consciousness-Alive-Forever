"""Check odor ablation, transport and stock ledgers, then plot real paths."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from odor_env import HERE
from proximity import AssistedProxy
from train import CAL, DEV, HOLDOUT, Motor, actor_mean, assess, dump, summarize
from reachability import run as reach


def read(path):
    return json.loads(path.read_text())


def inspect(folder):
    r=read(folder/'RESULT.json')
    arm=read(folder/'ARM.json')
    with (np.load(folder/'BODY.npz',allow_pickle=False) as body,
          np.load(folder/'NECK.npz',allow_pickle=False) as neck,
          np.load(folder/'SENSE.npz',allow_pickle=False) as senses):
        trace, obs, meal = body['trace'], body['observation'],body['meal']
        sniff = senses['samples']
        emitted, arrived, delays = [neck[k] for k in ('emitted','arrived','delays')]
        assert len(delays)==17 and len(emitted)==r['native_steps']//2
        assert len(obs)==len(sniff) and np.max(np.abs(obs[:,0]-sniff[:,0]))<1e-8
        for i,d in enumerate(delays):
            assert not arrived[:d,i].any()
            assert np.array_equal(emitted[:-d,i],arrived[d:,i])
        assert r['native_extra_force']==0 and abs(r['intake']-r['stock_loss'])<1e-10
        assert abs(meal[-1,1]-r['intake'])<1e-12
        assert (body['initial'][:3]==np.array([0.,0.,.1378])).all()
        if arm['sense'] in ('odor','blank'):
            assert not obs[:,1].any()
        if arm['sense'] in ('vision','blank'):
            assert not sniff[:,9:11].any()
        if arm['sense']=='blank':
            assert not emitted.any() and not obs[:,-2:].any()
        goal=np.array([r['geometry']['goal_x'],r['geometry']['goal_y']])
        full_path=np.r_[body['initial'][:2][None],trace[:,1:3]]
        common=trace[trace[:,0] <= 3.2,1:3]
        contact_s=sum(end-t for (t,ids),end in zip(arm['wall_contact_changes'],
                       [x[0] for x in arm['wall_contact_changes'][1:]]+[r['duration_s']]) if ids)
        row=dict(r,**arm,first_odor=sniff[0,9:11].tolist(),
            first_command=obs[0,-2:].tolist(),first_head_yaw=float(sniff[0,14]),
            diagnostics_camera_hits=int(sniff[:,13].sum()),actual_dn_emitted=int(emitted.sum()),
            actual_dn_arrived=int(arrived.sum()),actual_an_arrived=int(neck['ascending'].sum()),
            initial=body['initial'].tolist(),proxy_material=False,
            max_displacement_mm=float(10*np.linalg.norm(full_path,axis=1).max()),
            max_displacement_first_3_2s_mm=float(10*np.linalg.norm(common,axis=1).max()),
            wall_contact_seconds=contact_s, wall_exit_xy_mm=None,
            closest_chest_to_fruit_center_mm=float(10*np.linalg.norm(full_path-goal,axis=1).min()),
            max_obstacle=sniff[:,11:13].max(axis=0).tolist(),
            mouth_contact_fraction=float(meal[:,3].mean()))
        if r['geometry']['stage']!='open':
            g=r['geometry']
            exit_rows=trace[(trace[:,1]>=g['far'])&
                            ((trace[:,2]<g['low'])|(trace[:,2]>g['high']))]
            if len(exit_rows):
                row['wall_exit_xy_mm']=(exit_rows[0,1:3]*10).tolist()
    return row,trace,sniff


def run(root):
    motor=Motor(read(CAL))
    reachability={s:reach(motor,AssistedProxy,s,DEV) for s in ('wall','detour')}
    dump(root/'REACHABILITY.json',dict(scores={s:summarize(v) for s,v in reachability.items()},rows=reachability))
    ref={}
    with np.load(root/'TRAIN_DETOUR/POLICY.npz',allow_pickle=False) as policy:
        import train
        import torch
        train.Proxy=AssistedProxy
        final=read(root/'TRAIN_DETOUR/FINAL.json')
        saved=torch.load(root/'TRAIN_DETOUR'/final['selected']['checkpoint'],
                         map_location='cpu',weights_only=True)
        actor=train.SAC(77511)
        actor.actor.load_state_dict(saved['actor'])
        r=np.random.default_rng(91571).normal(0.,.4,(1024,14)).astype(np.float32)
        parity=float(np.max(np.abs(actor.action(r,False)-actor_mean(policy,r))))
        assert parity<1.5e-5
        assert len(saved['actor_optimizer']['state'])==6
        assert len(saved['critic_optimizer']['state'])==12
        held=read(root/'TRAIN_DETOUR/HELDOUT.json')['SAC']
        for stage in ('open','detour'):
            replay=assess(lambda x:actor_mean(policy,x),motor,HOLDOUT,stage=stage)
            assert json.loads(json.dumps(replay))==held[stage]
        dump(root/'RELOAD.json',dict(actor_parity=parity,cases=48,replay_exact=True,
                                     updates=saved['updates'],step=saved['step']))

        def equalized(x):
            x=x.copy()
            x[1:3]=x[1:3].mean()
            x[5]=0.
            return actor_mean(policy,x)

        ref['equalized']={s:assess(equalized,motor,HOLDOUT,stage=s) for s in ('open','detour')}
    dump(root/'EQUALIZED_PROXY.json',dict(rows=ref['equalized'],
                          counts={s:summarize(v) for s,v in ref['equalized'].items()}))
    cases={}
    for folder in sorted(root.glob('LIVE_*')):
        if (folder/'ARM.json').exists():
            cases[folder.name]=inspect(folder)
    comparison={k:v[0] for k,v in cases.items()}
    paired={}
    for treated,control in (('LIVE_OPEN_ODOR','LIVE_OPEN_BLANK'),
                            ('LIVE_DETOUR_ODOR_L','LIVE_DETOUR_BLANK_L')):
        if treated in comparison and control in comparison:
            a,b=comparison[treated],comparison[control]
            assert a['initial']==b['initial'] and a['geometry']==b['geometry']
            live,blank=cases[treated][1],cases[control][1]
            live_prefix,blank_prefix=live[live[:,0]<.16],blank[blank[:,0]<.16]
            assert np.array_equal(live_prefix,blank_prefix)
            with (np.load(root/treated/'NECK.npz',allow_pickle=False) as on,
                  np.load(root/control/'NECK.npz',allow_pickle=False) as off):
                for k in ('emitted','arrived','ascending'):
                    assert np.array_equal(on[k][:1600],off[k][:1600])
            paired[treated]=dict(control=control,trace_rows=len(live_prefix),neural_ticks=1600,
                                 body_and_neck_prefix_identical=True)
    if not cases:
        return
    dump(root/'PHYSICAL_AUDIT.json',comparison)
    dump(root/'PAIRING.json',paired)
    ncols=min(len(cases),3)
    nrows=(len(cases)+ncols-1)//ncols
    fig,axs=plt.subplots(nrows,ncols,figsize=(4.2*ncols,3.6*nrows),layout='constrained',squeeze=False)
    for ax,(tag,(row,trace,sniff)) in zip(axs.flat,cases.items()):
        geo=row['geometry']
        if geo['stage']!='open':
            ax.add_patch(Rectangle((10*geo['near'],10*geo['low']),10*(geo['far']-geo['near']),
                         10*(geo['high']-geo['low']),color='gray',alpha=.65))
        ax.add_patch(Circle((10*geo['goal_x'],10*geo['goal_y']),4.9,facecolor='#ed7f54',
                           edgecolor='#9a4d3a',alpha=.4))
        ax.plot(trace[:,1]*10,trace[:,2]*10,color='#154d71',linewidth=1.8)
        ax.scatter([0,trace[-1,1]*10],[0,trace[-1,2]*10],c=['#05794f','#154d71'],s=22,zorder=5)
        ax.set(xlim=(-3,24),ylim=(-20,13),xlabel='Forward (mm)',ylabel='Lateral (mm)',
               title=tag.removeprefix('LIVE_').replace('_',' ').title()+f'\nIntake {row["intake"]:.3f}')
        ax.set_aspect('equal')
        ax.grid(alpha=.14)
    for ax in list(axs.flat)[len(cases):]:
        ax.set_visible(False)
    fig.savefig(root/'PHYSICAL_PATHS.png',dpi=155)
    plt.close(fig)
    left='LIVE_DETOUR_ODOR_L'
    if left in cases:
        _,trace,sniff=cases[left]
        fig,axes=plt.subplots(3,1,figsize=(8.5,7),layout='constrained')
        graph=axes[0]
        for idx,name,color in ((9,'Left antenna','#165b88'),(10,'Right antenna','#a44728')):
            graph.plot(sniff[:,0],sniff[:,idx],label=name,color=color)
        graph.set_ylabel('Normalized odor')
        graph.legend(frameon=False)
        with np.load(root/left/'BODY.npz',allow_pickle=False) as body:
            meal=body['meal']
            graph=axes[1]
            graph.plot(trace[:,0],10*trace[:,1],label='Forward')
            graph.plot(trace[:,0],10*trace[:,2],label='Lateral')
            graph.set_ylabel('Chest position (mm)')
            graph.legend(frameon=False)
            graph=axes[2]
            graph.plot(meal[:,0],meal[:,1],color='#166549')
            graph.axhline(.18,linestyle='--',color='gray',linewidth=1)
            graph.set_ylabel('Intake')
            graph.set_xlabel('Simulation time (s)')
        for graph in axes:
            graph.grid(alpha=.17)
        fig.savefig(root/'LEFT_TIMING.png',dpi=145)
        plt.close(fig)


if __name__=='__main__':
    run(HERE)
