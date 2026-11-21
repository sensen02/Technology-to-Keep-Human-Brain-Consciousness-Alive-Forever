"""Verify paired input identity, neck chronology, actual intake and functional outcomes."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261119T205558Z-orchard-curriculum'
sys.path.insert(0,str(PREV))
from rewards import analyze
def verify(d):
    res=json.loads((d/'RESULT.json').read_text())
    with np.load(d/'NECK.npz') as n:
        assert len(n['arrived'])*2==res['native_steps'] and n['arrived'].shape[1]==17
        for i,lag in enumerate(n['delays']):
            assert np.array_equal(n['arrived'][lag:,i],n['emitted'][:-lag,i]) and not n['arrived'][:lag,i].any()
    with np.load(d/'BODY.npz') as b:
        m=b['meal'];assert abs(m[:,4].sum()-res['intake'])<1e-8
        assert res['native_extra_force']==0 and abs(res['intake']-res['stock_loss'])<1e-8
        if np.any(m[:,4]>0):
            fed=m[m[:,4]>0]
            assert (fed[:,3]>0).all()
            with np.load(d/'NECK.npz') as n:
                ticks=np.rint(fed[:,0]*1e4).astype(int)
                assert n['feeding'][ticks].all()
    _,row=analyze(d,3)
    row.update(stock_loss=res['stock_loss'],first_intake=res['first_intake_s'])
    return row
def main():
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    rows={}
    for col,site in enumerate(('CLEAR','DEBRIS')):
        baseline=PREV/f'HOLD_{site}'
        for label,run,c in [('Original',PREV/f'REF_{site}','#888888'),('Same head',baseline,'#cc8730'),('Body recovery',HERE/f'LIVE_{site}','#226e72')]:
            rows[run.name]=verify(run)
            with np.load(run/'BODY.npz') as b,np.load(baseline/'BODY.npz') as h,np.load(run/'TRAIN_MEASURES.npz') as m:
                if run!=baseline:
                    assert np.array_equal(b['initial'],h['initial']) and np.array_equal(b['trace'][:32],h['trace'][:32])
                t=b['trace']
                axes[0,col].plot(t[:,0],m['oral'][:,4],label=label,color=c)
                axes[1,col].plot(t[:,0],t[:,6],color=c)
            with np.load(run/'NECK.npz') as n,np.load(baseline/'NECK.npz') as v:
                assert np.array_equal(n['emitted'][:1600],v['emitted'][:1600])
        integrated=HERE/f'ALL_{site}'
        if (integrated/'CONFIG.json').exists():
            with np.load(integrated/'BODY.npz') as b,np.load(integrated/'TRAIN_MEASURES.npz') as m:
                t=b['trace']
                axes[0,col].plot(t[:,0],m['oral'][:,4],label='Body + mouth',color='#6b5890',lw=1.2)
                axes[1,col].plot(t[:,0],t[:,6],color='#6b5890',lw=1.2)
        passive=HERE/f'PASSIVE_{site}'
        if (passive/'CONFIG.json').exists():
            with np.load(passive/'BODY.npz') as b,np.load(passive/'TRAIN_MEASURES.npz') as m:
                t=b['trace']
                axes[0,col].plot(t[:,0],m['oral'][:,4],label='Mouth passive until touch',color='#323453',ls='--',lw=1)
                axes[1,col].plot(t[:,0],t[:,6],color='#323453',ls='--',lw=1)
        axes[0,col].set_title(site.title()+' · original start');axes[0,col].legend(fontsize=8)
        axes[0,col].set_ylabel('Mouth to food patch (mm)');axes[1,col].set_ylabel('Thorax upright cosine');axes[1,col].set_xlabel('Time (s)')
    fig.savefig(HERE/'BEHAVIOR.png',dpi=150);plt.close(fig)
    for d in sorted(HERE.glob('FOOD_*')):
        if d.is_dir() and (d/'CONFIG.json').exists():
            rows[d.name]=verify(d)
            tr=json.loads((d/'TRANSFER.json').read_text())
            if rows[d.name]['intake']>0:
                assert 0<=tr['min_supported_skin_gap_mm']<=.55 and tr['mouth_gate_samples']>0
            with np.load(d/'TARSAL_REFLEX.npz') as z:
                r=z['samples'];rows[d.name].update(taste_samples=int(r[:,1].sum()),max_reflex=float(r[:,2].max()),mouth_control_range=np.ptp(r[:,3:6],axis=0).tolist())
    for d in sorted(HERE.glob('NOVEL_*')):
        if (d/'CONFIG.json').exists():rows[d.name]=verify(d)
    for d in sorted(HERE.glob('ALL_*')):
        if not (d/'CONFIG.json').exists():
            continue
        rows[d.name]=verify(d)
        with np.load(d/'LOCAL_REFLEX.npz') as z:
            samples=z['samples']
            rows[d.name]['tarsal_active_samples']=int(np.sum(samples[:,2]>0))
            assert np.all(samples[samples[:,3]>=.18,2]==0)
    for d in sorted(HERE.glob('PASSIVE_*')):
        if not (d/'ORAL_ACTUATORS.json').exists():
            continue
        rows[d.name]=verify(d)
        meta=json.loads((d/'ORAL_ACTUATORS.json').read_text())
        rows[d.name]['oral_enabled_times']=[t for t,on in meta['switches'] if on]
        site=d.name.split('_')[1]
        if site in ('CLEAR','DEBRIS'):
            base=HERE/f'LIVE_{site}'
            for name,key in [('BODY','poses'),('NECK','arrived')]:
                with np.load(d/(name+'.npz')) as a,np.load(base/(name+'.npz')) as b:
                    rows[d.name]['identical_to_body_only_'+key]=np.array_equal(a[key],b[key])
    for site in ('CLEAR','DEBRIS'):
        old,new=[HERE/f'NOVEL_{arm}_{site}' for arm in ('OFF','ON')]
        if all(p.name in rows for p in (old,new)):
            with np.load(old/'BODY.npz') as a,np.load(new/'BODY.npz') as b:
                assert np.array_equal(a['initial'],b['initial']) and np.array_equal(a['trace'][:32],b['trace'][:32])
            with np.load(old/'NECK.npz') as a,np.load(new/'NECK.npz') as b:
                assert np.array_equal(a['emitted'][:1600],b['emitted'][:1600])
    for start in ('FOOD_','FOOD_release_'):
        old,new=[HERE/(start+arm) for arm in ('off','on')]
        if all(p.name in rows for p in (old,new)):
            with np.load(old/'BODY.npz') as a,np.load(new/'BODY.npz') as b:
                assert np.array_equal(a['initial'],b['initial'])
    (HERE/'AUDIT.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
