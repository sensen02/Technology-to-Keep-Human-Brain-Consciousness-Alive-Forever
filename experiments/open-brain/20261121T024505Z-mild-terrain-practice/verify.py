import json
from train_practice import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    results=json.loads((OUT/'evaluation.json').read_text())
    old=json.loads((PARENT/'evaluation.json').read_text())
    for a,b in zip(results,old):
        assert a['episode']==b['name'] and abs(a['previous']['step_mm']-b['new']['step_mm'])<1e-6
        assert abs(a['previous']['self_tail_rms200_mm_s']-b['new']['self_floor_last200ms_xy_rms_speed_mm_s'])<.002
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'episodes').glob('*/receipt.json'))]
    assert len(receipts)==20
    assert len({digest(p) for p in (OUT/'episodes').glob('*/terrain.npz')})==20
    assert all(r['supported_foot_max_surface_um']>25 and r['min_upright']>.8 for r in receipts)
    base=ObservedBody(PARENT/'weights.pt')
    for arm in ['flat_control','expanded']:
        net=ObservedBody(OUT/arm/'weights.pt');record=json.loads((OUT/arm/'training.json').read_text())
        assert record['weights_sha256']==digest(OUT/arm/'weights.pt')
        assert all(torch.equal(u,v) for u,v in zip(net.raw.feedback.parameters(),base.raw.feedback.parameters()))
        assert any(not torch.equal(u,v) for u,v in zip(net.raw.motion.parameters(),base.raw.motion.parameters()))
        assert not any(x.startswith(('11701','11903','terrain16','terrain17','terrain18','terrain19')) for x in record['rows'])
    grouped={}
    for key,selector in [('rough',lambda n:n.startswith('terrain')),('flat',lambda n:not n.startswith('terrain'))]:
        selection=[r for r in results if selector(r['episode'])]
        grouped[key]={arm:{k:float(np.mean([r[arm][k] for r in selection])) for k in selection[0][arm]} for arm in ['previous','flat_control','expanded']}
    write(OUT/'summary.json',dict(episodes=receipts,averages=grouped,verified_parent_flat_replay=7))
    fig,ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    sample=OUT/'episodes/terrain16'
    with np.load(sample/'terrain.npz') as z:
        x,y,h=[z[k] for k in ('x','y','z')]
    with np.load(sample/'external.npz') as z: center=z['markers_mm'][:,0]
    im=ax[0].pcolormesh(x,y,h*1000,shading='auto',cmap='viridis')
    ax[0].plot(*center[:,:2].T,color='white',lw=1.6)
    ax[0].set(xlim=(-2,15),ylim=(-5,5),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='Held-out terrain and teacher path')
    fig.colorbar(im,ax=ax[0],label='Surface height (µm)',orientation='horizontal',shrink=.7,pad=.18)
    x=np.arange(2)
    for i,(arm,label,color) in enumerate([('previous','Before practice','#667789'),('flat_control','Flat practice','#bd7c33'),('expanded','Expanded practice','#267b68')]):
        vals=[1000*grouped[key][arm]['roll50_mm'] for key in ('flat','rough')]
        ax[1].bar(x+(i-1)*.23,vals,width=.22,label=label,color=color)
    ax[1].set(xticks=x,xticklabels=['Flat (7 trials)','Rough (4 trials)'],ylabel='50 ms marker error (µm)',title='Prediction with measured pressure')
    ax[1].legend(loc='lower right',fontsize=8)
    fig.savefig(OUT/'practice.png',dpi=145)
    print('PASS',len(results),'evaluation episodes',len(receipts),'unique terrains')
if __name__=='__main__':main()
