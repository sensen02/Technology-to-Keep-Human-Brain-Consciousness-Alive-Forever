"""Compare all three reward stages and independently match intact held-out pairs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rewards import analyze

HERE=Path(__file__).resolve().parent

def main():
    groups={}
    for config in sorted(HERE.glob('*/CONFIG.json')):
        folder=config.parent
        _,row=analyze(folder,3)
        with np.load(folder/'NECK.npz') as z:
            assert z['arrived'].shape[1]==17 and len(z['arrived'])*2==row['native_steps']
            for j,delay in enumerate(z['delays']):
                assert not z['arrived'][:delay,j].any()
                assert np.array_equal(z['arrived'][delay:,j],z['emitted'][:-delay,j])
            row['arrived_dn']=int(z['arrived'].sum())
        config=json.loads(config.read_text())
        row['seed']=config['seed']; row['start_x_cm']=config['start_x']
        with np.load(folder/'BODY.npz') as b:
            assert not b['observation'][:,1].any()
        if row['intake']>0:
            r=json.loads((folder/'TRANSFER.json').read_text())
            assert r['mouth_gate_samples']>0 and r['min_supported_skin_gap_mm']<=r['skin_reach_mm']
        groups[folder.name]=row
    figures,ax=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    for j,site in enumerate(('CLEAR','DEBRIS')):
        for key,color,label in [('REF','#777777','Original'),('HOLD','#246393','Head SAC'),
                                ('COMBINED','#c57624','Head + gait'),('REPLAY','#42834b','Head + gait + food replay')]:
            p=HERE/f'{key}_{site}'; ref=HERE/f'REF_{site}'
            if not p.exists(): continue
            with np.load(p/'BODY.npz') as b,np.load(p/'TRAIN_MEASURES.npz') as z,np.load(ref/'BODY.npz') as other:
                assert np.array_equal(b['initial'],other['initial'])
                assert np.array_equal(b['trace'][:32],other['trace'][:32])
                dist=z['oral']
                ax[0,j].plot(dist[:,0],dist[:,4],label=label,color=color,lw=1.4)
                t=b['trace']
                ax[1,j].plot(t[:,0],t[:,6],label=label,color=color,lw=1.4)
            with np.load(p/'NECK.npz') as n,np.load(ref/'NECK.npz') as v:
                assert np.array_equal(n['emitted'][:1600],v['emitted'][:1600])
        ax[0,j].set_title(site.title()+' · original start')
        ax[0,j].set_ylabel('Mouth to food patch (mm)')
        ax[1,j].set_ylabel('Thorax upright cosine')
        ax[1,j].set_xlabel('Time (s)')
        ax[0,j].legend(frameon=False)
    figures.savefig(HERE/'HELDOUT.png',dpi=165)
    (HERE/'AUDIT.json').write_text(json.dumps(groups,indent=2)+'\n')
    comparison = {}
    for site in ('CLEAR', 'DEBRIS'):
        ordinary = HERE / f'COMBINED_{site}'
        replay = HERE / f'REPLAY_{site}'
        values = {}
        with np.load(ordinary/'BODY.npz') as a, np.load(replay/'BODY.npz') as b:
            values['same_head_actions'] = np.array_equal(a['observation'][:,15:17], b['observation'][:,15:17])
            values['same_body_trajectory'] = np.array_equal(a['poses'], b['poses'])
        with np.load(ordinary/'NECK.npz') as a, np.load(replay/'NECK.npz') as b:
            for name in ('emitted', 'arrived', 'drive'):
                values['same_' + name] = np.array_equal(a[name], b[name])
        comparison[site] = values
    (HERE/'FOOD_REPLAY_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
    for k,r in groups.items():
        print(k,r['walk_gate'],r['rings'],r['intake'],r['fall_s'],r['standing'])

if __name__=='__main__':
    main()
