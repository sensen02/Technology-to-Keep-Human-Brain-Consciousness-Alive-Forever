"""Paired effect of the additional DN volley on the 17 command channels, bilateral write."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
import numpy as np
OUT=Path(__file__).resolve().parent
PRIOR=OUT.parent/'20261121T005613Z-patch226-write'
sys.path.insert(0,str(PRIOR))
from common import geom,tips,paths,roots,IO,patterns,contacts,FULL,read,ElectrodeObservation
from readback import align,expected_evoked
from omnisim.modules.flex_depth import extracellular_traces
from omnisim.modules.cmos_readout import FrontEnd
from omnisim.modules.spike_sorting import ArraySorter

def local_waves(I,keys,sensors,length):
    with np.load(FULL/'anatomy.npz') as z:
        labels=z['labels']; lower=z['lower']; names=z['roots'].astype(str)
    ind=np.array([names.tolist().index(k) for k in keys])+1
    n=len(names)+1
    slices=[]
    for lo in range(0,labels.shape[1]-9,10):
        slab=labels[:,lo:lo+10,:]
        xyz=np.nonzero(slab)
        ids=slab[xyz]
        count=np.bincount(ids,minlength=n)[ind]
        xyz=np.column_stack([np.bincount(ids,weights=v,minlength=n)[ind] for v in xyz])
        cent=np.divide(xyz,count[:,None],where=count[:,None]>0,out=np.full_like(xyz,np.nan))
        slices.append((cent+lower+[.5,lo+.5,.5])*float(geom['cell_m']))
    del labels
    centers=np.stack(slices,1)
    lead=ElectrodeObservation(sensors,[],contact_radius_m=.15e-6)
    nonzero=I!=0
    waves=[];any_cut=0
    for name,row in zip(keys,centers):
        good=np.flatnonzero(np.isfinite(row[:,0]))
        candidates=[]
        for run in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
            if len(run)<3:continue
            q=row[run];arc=np.r_[0.,np.linalg.norm(np.diff(q,axis=0),axis=1).cumsum()]
            if arc[-1]<4e-6:continue
            grid=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/2e-6))+1)
            v=lead._lead(lead._at(q,arc,grid),tips[nonzero])@I[nonzero]
            kernel=np.diff(v,n=2)*(2e-6/grid[1])**2
            for phase,sign in enumerate([1,-1]):
                k=int(np.argmax(sign*kernel))
                if sign*kernel[k]>=.004:candidates.append((phase,float(-sign*kernel[k]),k+1,grid,q,arc))
        assert candidates,name
        phase,_,k,grid,q,arc=min(candidates,key=lambda r:r[:2])
        start=lead._at(q,arc,[grid[k]])
        branches=[SimpleNamespace(root=name,points_m=np.vstack((start,q[m])),speed_m_s=.1)
                  for m in (np.flatnonzero(arc<grid[k])[::-1],np.flatnonzero(arc>grid[k]))]
        wave=extracellular_traces(branches,sensors,.15e-6).sum(0)
        offset=phase*100
        out=np.zeros((length,len(sensors)),np.float32)
        size=min(len(wave),length-offset)
        any_cut+=int(size<len(wave))
        out[offset:offset+size]=wave[:size]
        waves.append(out)
    return waves,any_cut

def main():
    arms=patterns();I=arms[3];mask=~np.any(arms!=0,axis=0)
    sensors=tips[mask]
    front=FrontEnd(np.sqrt(2*np.pi)*.15e-6)
    raw=extracellular_traces(paths,sensors,.15e-6)
    actual=front.response(raw)
    expect=front.response(extracellular_traces(paths,contacts('nominal')[mask],.15e-6))
    evt=read(PRIOR/'registered.json')['arms']['both']['accepted']
    registered=front.response(expected_evoked(evt,sensors,raw.shape[1],evt)[None])[0]
    extra=read(PRIOR/'background.json')['2']['both']['roots']
    labels=dict(zip(geom['material_roots'].astype(str),geom['material_classes']))
    is_dn=np.array([labels[k]=='descending' for k in extra])
    assert sum(is_dn)==44 and len(extra)==144 and not set(extra)&set(IO['dn_roots'])
    waves,cut=local_waves(I,extra,sensors,raw.shape[1])
    signals=np.concatenate([front.response(np.stack(waves[i:i+5])) for i in range(0,len(waves),5)])
    del waves
    obs=ElectrodeObservation(sensors,[],contact_radius_m=.15e-6);active=I!=0
    raw[:]=0
    raw[0,:100]=obs._lead(sensors,tips[active])@I[active]
    raw[0,100:200]=-raw[0,:100]
    art=front.response(raw[:1])[0]
    testroots=[roots.index(r) for r in IO['dn_roots']]
    n=len(art);ch=len(sensors); results=[]
    for seed in (713,1211,1847):
        rng=np.random.default_rng(seed);true=np.repeat(testroots,6)
        noise=front.noise(n*(544+len(true)),ch,rng)[0].reshape(-1,n,ch)
        sorter=ArraySorter(expect,list(range(len(roots))),noise[:32],half_width=28)
        threshold=np.quantile(np.concatenate([sorter.strongest(noise[i:i+16])[0] for i in range(32,288,16)]),.99)
        blank=256
        clean=noise[288:].copy()
        # First 256 windows have no spontaneous command.
        for j,r in enumerate(true):clean[blank+j]+=align(actual[r],int(rng.integers(-12,13)))*geom['route_gain'][r]*rng.uniform(.85,1.15)
        count=len(clean)
        jitter=rng.integers(-1,2,count); gain=rng.normal(1,.02,count);offset=rng.integers(-6,7,count)
        cal_noise=front.noise(n*64,ch,rng)[0].reshape(64,n,ch).mean(0)
        for prob in (1.,.8):
            fire=(rng.random((count,len(extra)))<prob).astype('f')
            cal_fire=(rng.random((64,len(extra)))<prob).mean(0).astype('f')
            cal_gain=rng.normal(1,.02,64).mean()
            for dn_on in (False,True):
                keep=~is_dn if not dn_on else np.ones(len(is_dn),bool)
                bank=signals[keep].reshape(sum(keep),-1)
                template=(cal_fire[keep]@bank).reshape(n,ch)+cal_noise+cal_gain*art+registered
                guesses=[]
                for j in range(0,count,10):
                    end=min(j+10,count)
                    group=(fire[j:end,keep]@bank).reshape(-1,n,ch)
                    residual=[]
                    for idx,b in enumerate(group,j):
                        t=int(offset[idx])
                        w=clean[idx]+align(b+registered+gain[idx]*art,t+int(jitter[idx]))
                        cand=np.stack([align(template,t+d) for d in (-1,0,1)])
                        r=slice(62+t,78+t)
                        short=cand[:,r]
                        beta=((short*w[r]).sum((1,2))/(short**2).sum((1,2))).clip(.8,1.2)
                        best=np.argmin(((short*beta[:,None,None]-w[r])**2).sum((1,2)))
                        residual.append(w-beta[best]*cand[best])
                    scores,pred=sorter.strongest(np.stack(residual))
                    guesses.extend(np.where(scores>threshold,pred,-1).tolist())
                guesses=np.array(guesses)
                results.append(dict(seed=seed,p=prob,dn_on=dn_on,
                      correct=int((guesses[blank:]==true).sum()),total=len(true),
                      false_command=int(np.isin(guesses[:blank],testroots).sum()),blank=blank,predicted=guesses.tolist()))
        print(seed,flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(roots=extra,DN_count=int(is_dn.sum()),unregistered_up=int((~is_dn).sum()),
            cropped_local_waveforms=cut,rows=results),indent=2))
if __name__=='__main__':main()
