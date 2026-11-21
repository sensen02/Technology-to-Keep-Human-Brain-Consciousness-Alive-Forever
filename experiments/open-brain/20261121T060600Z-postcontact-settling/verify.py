"""Paired evaluation from saved trajectories; all criteria match the earlier food task."""
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from settling import OUT,SRC,OLD,DT,T,local_axes,load_floor
from practice import digest
def read(p):return json.loads(p.read_text())
def evaluate(path):
    r=read(path/'result.json')
    with np.load(path/'replay.npz') as z:
        p=z['markers']; gate=z['gate'] if 'gate' in z else np.zeros(len(z['mouth']),bool)
        meal=z['intake']; contacts=z['mouth_contact'];end=z['brain_end'];encoded=z['brain_input']
        support=(load_floor(OLD/'floor.pt')(T(p[:-2]),T(p[1:-1])).detach().numpy()>=.5).sum(1)
        upright=local_axes(T(p[1:-1]))[:,2,2].numpy()
        speed=np.linalg.norm(np.diff(p[:,0,:2],axis=0),axis=1)/DT
        assert abs(meal[-1]-r['stock_loss'])<1e-8 and np.all(np.diff(meal,prepend=0)[~contacts]==0)
        assert np.array_equal(end.sum(0),r['arrivals']) and np.array_equal(encoded.sum(0),r['encoded'])
        assert np.isfinite(p).all()
        threshold=np.flatnonzero(meal>=.18)
        stance=bool(min(support[-28:])>=4 and min(upright[-28:])>.94 and max(speed[-28:])<=.8)
        # At the first meal target use the same 28 complete frames, including that transition.
        target_stance=None
        if len(threshold) and threshold[0]>=27:
            k=int(threshold[0]);w=slice(k-27,k+1)
            target_stance=bool(min(support[w])>=4 and min(upright[w])>.94 and max(speed[1:][w])<=.8)
        mm=dict(target_stance=target_stance,intake=float(meal[-1]),gate=int(gate.sum()),release=int(np.sum(gate[:-1]&~gate[1:])),
            min_support=int(min(support[-28:])),min_upright=float(min(upright[-28:])),
            tail_max_speed=float(max(speed[-28:])),tail_rms=float(np.mean(speed[-40:]**2)**.5),
            contact_seconds=float(contacts.sum()*DT),reached_meal_s=float((threshold[0]+1)*DT) if len(threshold) else None,
            end_standing=stance,task_pass=bool(len(threshold) and stance and min(upright)>=.55),
            arrived=end.sum(0).tolist(),encoded=encoded.sum(0).tolist())
        return mm,dict(speed=speed[1:],meal=meal,gate=gate,load=support)
def check_prefix(old,new):
    with np.load(old/'replay.npz') as a,np.load(new/'replay.npz') as b:
        g=np.flatnonzero(b['gate']);n=int(g[0])+1 if len(g) else len(b['gate'])
        keys={'markers':n+1 if len(g) else n+2,'mouth':n,'dn_read':n,'intake':n,'mouth_contact':n,'brain_dn':n*50,'brain_input':n*50,'brain_end':n*50}
        for k,c in keys.items():assert np.array_equal(a[k][:c],b[k][:c]),(k,new,n)
        return n
def main():
    seeds=[88207,88224,88225,91001]
    rows=[]
    fig,axs=plt.subplots(2,4,figsize=(13,5.5),layout='constrained',sharex=True)
    for col,seed in enumerate(seeds):
        baseline=OUT/f'live-{seed}-on-previous' if seed==91001 else SRC/f'live-{seed}-on-previous'
        new=OUT/f'live-{seed}-on-settling'
        row={'seed':seed,'identical_prefix_frames':check_prefix(baseline,new)}
        responsive=OUT/f'live-{seed}-on-responsive'
        row['responsive_prefix_frames']=check_prefix(baseline,responsive)
        for tag,path,color in [('old',baseline,'#b4622e'),('contact',new,'#176f82'),('responsive',responsive,'#6951ad')]:
            row[tag],curve=evaluate(path)
            t=(np.arange(len(curve['meal']))+1)*DT
            axs[0,col].plot(t,curve['speed'],lw=.7,c=color,alpha=.85,label=tag)
            axs[1,col].plot(t,curve['meal'],lw=1.4,c=color)
        switch=np.flatnonzero(curve['gate'])
        if len(switch):
            for ax in axs[:,col]:ax.axvline(t[switch[0]],c='#888',ls=':',lw=1)
        axs[0,col].set(title=str(seed),ylim=(-.4,None));axs[1,col].axhline(.18,c='#888',ls='--',lw=.8)
        axs[1,col].set(xlabel='Time (s)',ylim=(0,.95));rows.append(row)
    axs[0,0].set_ylabel('Thorax speed (mm/s)');axs[1,0].set_ylabel('Cumulative intake');axs[0,0].legend()
    fig.savefig(OUT/'comparison.png',dpi=160)
    off,_=evaluate(OUT/'live-88207-off-settling')
    assert sum(off['arrived'])==sum(off['encoded'])==0
    responsive_off,_=evaluate(OUT/'live-88207-off-responsive')
    assert sum(responsive_off['arrived'])==sum(responsive_off['encoded'])==0
    hashes={str(p):digest(p) for p in [OUT/'settling.pt',OUT/'responsive.pt',OUT/'conditioned.py',OLD/'weights.pt',OLD/'floor.pt',OLD/'skeleton.json',OLD/'mouth_offset.json',SRC/'array_loop.py',SRC/'eon_worker.py',OUT/'live.py',OUT/'settling.py']}
    (OUT/'acceptance.json').write_text(json.dumps(dict(rows=rows,contact_current_off=off,responsive_current_off=responsive_off,sha256=hashes),indent=2))
    for row in rows:print(row['seed'],row['identical_prefix_frames'],row['old']['task_pass'],row['responsive'])
if __name__=='__main__':main()
