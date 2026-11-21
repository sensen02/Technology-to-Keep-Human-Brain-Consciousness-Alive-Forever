"""Verify physical, neural and feeding records and plot their source geometry."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def audit(case):
    result=read(case/'RESULT.json')
    transfer=read(case/'TRANSFER.json')
    with (np.load(case/'BODY.npz',allow_pickle=False) as body,
          np.load(case/'NECK.npz',allow_pickle=False) as neck,
          np.load(case/'SENSE.npz',allow_pickle=False) as sense):
        trace=body['trace']
        obs=body['observation']
        samples=sense['samples']
        assert len(trace)>0 and len(obs)==len(samples)
        assert not obs[:,1].any(), 'Fruit visual features reached the head policy'
        assert np.max(abs(obs[:,0]-samples[:,0]))<1e-8
        assert result['native_steps']==2*len(neck['emitted'])
        assert len(neck['delays'])==17
        for i,d in enumerate(neck['delays']):
            assert not neck['arrived'][:d,i].any()
            assert np.array_equal(neck['emitted'][:-d,i],neck['arrived'][d:,i])
        if not transfer['odor_enabled']:
            assert not samples[:,9:11].any() and not neck['arrived'].any()
        assert result['native_extra_force']==0.
        assert abs(result['stock_loss']-result['intake'])<1e-10
        if result['intake']>0.:
            assert transfer['mouth_gate_samples']>0
            assert transfer['min_supported_skin_gap_mm']<=transfer['skin_reach_mm']
        contact=body['meal'][:,3]
        geom=result['geometry']
        first_trace=trace[trace[:,0]<.16]
        common=trace[trace[:,0]<3.2]
        orientation=trace[-1,6]
        near=10.*np.min(np.sqrt(np.sum((trace[:,1:3]-[geom['goal_x'],geom['goal_y']])**2,axis=1)))
        record=result | transfer | dict(neural_samples=len(neck['emitted']),
            arrived_dn=int(neck['arrived'].sum()),arrived_an=int(neck['ascending'].sum()),
            first_sample_odor=samples[0,9:11].tolist(),peak_obstacle=samples[:,11:13].max(axis=0).tolist(),
            displacement_mm=float(10*np.max(np.linalg.norm(trace[:,1:3],axis=1))),
            displacement_before_3_2_mm=float(10*np.max(np.linalg.norm(common[:,1:3],axis=1))),
            min_chest_to_patch_xy_mm=float(near),mouth_contact_steps=int(contact.sum()),
            final_upright=float(orientation), RGB_target_detected_frames=int(samples[:,13].sum()),
            native_prefix=len(first_trace))
        limit=[(-1.599,3.199),(-1.999,1.999)]
        record['left_crop']=bool(any(np.any((trace[:,c+1]<ab[0])|(trace[:,c+1]>ab[1]))
                                      for c,ab in enumerate(limit)))
        counts={}
        changes=transfer['contact_changes']
        for (start,ids),end in zip(changes,[t for t,_ in changes[1:]]+[result['duration_s']]):
            for label in {name.split('_')[1] for name in ids}:
                counts[label]=counts.get(label,0.)+end-start
        record['object_contact_seconds']=counts
    return record,trace


def paired(site,first,second,results):
    assert first['geometry']==second['geometry']
    a,b=(results/f'LIVE_{site}_{mode}' for mode in ('ON','OFF'))
    with np.load(a/'BODY.npz') as f,np.load(b/'BODY.npz') as g:
        assert np.array_equal(f['initial'],g['initial'])
        n=np.count_nonzero(f['trace'][:,0]<.16)
        assert np.array_equal(f['trace'][:n],g['trace'][:n])
    with np.load(a/'NECK.npz') as f,np.load(b/'NECK.npz') as g:
        for k in ('emitted','arrived','ascending'):
            assert np.array_equal(f[k][:1600],g[k][:1600])
    return dict(first_body_rows=int(n),first_neural_ticks=1600,exact=True)


def plot(cases,results):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for (site,mode),ax in zip(((s,m) for s in ('CLEAR','DEBRIS') for m in ('ON','OFF')),
                             axes.flat,strict=True):
        label=f'LIVE_{site}_{mode}'
        with np.load(HERE/site/'CROP.npz',allow_pickle=False) as crop:
            height=ax.contourf(crop['x']*10,crop['y']*10,crop['terrain']*10,
                               levels=np.linspace(-2.6,1.7,15),cmap='cividis',alpha=.40)
            ax.add_collection(PolyCollection(crop['skin'][:,:,:2]*10,facecolor='#b85d37',
                                             edgecolor='none',alpha=.26))
            if len(crop['debris']):
                ax.add_collection(PolyCollection(crop['debris'][:,:,:2]*10,facecolor='#526926',
                                                 edgecolor='#26351a',linewidth=.45,alpha=.53))
            goal=crop['goal'][:2]*10
            ax.add_patch(Circle(goal,4.9,fill=False,edgecolor='#a32b1d',linestyle=':',linewidth=1.4))
            ax.scatter(*goal,c='#a32b1d',s=22,zorder=3)
        if label in cases:
            row,trace=cases[label]
            ax.plot(trace[:,1]*10,trace[:,2]*10,color='#075e8c',linewidth=1.9,zorder=5)
            ax.scatter(*trace[-1,1:3]*10,color='#075e8c',s=22,zorder=6)
            note=f'Intake {row["intake"]:.3f}'
        else:
            note='Pending'
        ax.scatter(0,0,color='#1b6243',s=25,zorder=6)
        if label=='LIVE_CLEAR_ON' and 'LIVE_CLEAR_HEIGHTFIELD_ON' in cases:
            _,alt=cases['LIVE_CLEAR_HEIGHTFIELD_ON']
            ax.plot(alt[:,1]*10,alt[:,2]*10,linestyle='--',color='#4e387e',
                    linewidth=1.6,zorder=5,label='Heightfield check')
            ax.legend(loc='lower left',frameon=False)
        ax.set(title=f'{site.title()} · odor {mode.lower()} · {note}',xlabel='Forward (mm)',
               ylabel='Lateral (mm)',xlim=(-16,32),ylim=(-20,20),aspect='equal')
    fig.colorbar(height,ax=axes.ravel().tolist(),label='Soil elevation vs spawn (mm)',shrink=.68)
    fig.savefig(results/'ORCHARD_TRANSFER.png',dpi=160)
    plt.close(fig)


def main(results):
    cases={d.name:audit(d) for d in sorted(results.glob('LIVE_*')) if (d/'TRANSFER.json').exists()}
    comparisons={}
    for site in ('CLEAR','DEBRIS'):
        a,b=f'LIVE_{site}_ON',f'LIVE_{site}_OFF'
        if a in cases and b in cases:
            comparisons[site]=paired(site,cases[a][0],cases[b][0],results)
    report=dict(cases={k:v[0] for k,v in cases.items()},matched_prefix=comparisons)
    (results/'ORCHARD_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(cases,results)
    for name,(r,_) in cases.items():
        print(name,{k:r[k] for k in ('intake','meal_pass','min_oral_to_patch_mm','arrived_dn',
                                     'displacement_mm','object_contact_seconds','left_crop')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path,nargs='?',default=HERE)
    main(parser.parse_args().results)
