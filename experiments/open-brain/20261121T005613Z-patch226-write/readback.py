"""Paired recording during a write, with the same extra stimulation-only waveform in blanks."""
from types import SimpleNamespace
from common import *
from omnisim.modules.flex_depth import extracellular_traces
from omnisim.modules.cmos_readout import FrontEnd
from omnisim.modules.spike_sorting import ArraySorter

def align(w,offset):
    out=np.zeros_like(w)
    left=max(0,offset); right=min(len(w),len(w)+offset)
    if left<right:out[left:right]=w[left-offset:right-offset]
    return out

def expected_evoked(accepted, sensors, n, exposures):
    branches=[];starts=[];gains=[]
    for e in accepted:
        if e['kind']!='axon_fired':continue
        p=paths[roots.index(e['axon'])];points=p.points_m
        arc=np.r_[0.,np.linalg.norm(np.diff(points,axis=0),axis=1).cumsum()]
        grid=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/2e-6-1e-12))+1)
        loc=grid[e['node']]
        pt=ElectrodeObservation._at(points,arc,[loc])
        for end,seg in [('a',points[arc<loc][::-1]),('b',points[arc>loc])]:
            destination=next((x for x in exposures if x['kind']=='axon_arrived' and
                              x['axon']==p.root and x['end']==end),None)
            if destination is None:continue
            branches.append(SimpleNamespace(root=p.root,points_m=np.vstack((pt,seg)),speed_m_s=p.speed_m_s))
            starts.append(round(e['time_s']*1e6))
            gains.append(destination['gain'])
    out=np.zeros((n,len(sensors)),np.float32)
    if branches:
        waves=extracellular_traces(branches,sensors,.15e-6)
        for w,j,g in zip(waves,starts,gains):
            size=min(len(w),n-j)
            out[j:j+size]+=w[:size]*g
    return out

def main():
    pat=patterns(); runs=read(HERE/'registered.json')['arms']
    allowed=~np.any(pat!=0,axis=0)
    sensors=tips[allowed]
    front=FrontEnd(np.sqrt(2*np.pi)*.15e-6)
    expected=front.response(extracellular_traces(paths,contacts('nominal')[allowed],.15e-6))
    raw=extracellular_traces(paths,sensors,.15e-6)
    actual=front.response(raw)
    n=actual.shape[1]
    dn=[roots.index(r) for r in IO['dn_roots']]
    ss=ElectrodeObservation(sensors,[],contact_radius_m=.15e-6)
    transfer=ss._lead(sensors,tips[:, :])
    artifact={};evoked={}
    for k,I in zip(arms,pat):
        wave=np.zeros(raw.shape[1:],dtype=np.float32)
        wave[:100]=transfer@I;wave[100:200]=-(transfer@I)
        artifact[k]=front.response(wave[None])[0]
        active=runs[k]['accepted']
        evoked[k]=front.response(expected_evoked(active,sensors,len(wave),active)[None])[0]
    out=[]
    for seed in (713,1211,1847):
        rng=np.random.default_rng(seed)
        repeats=6
        probes=np.repeat(dn,repeats)
        noise,_=front.noise(n*(544+len(probes)),len(sensors),rng)
        windows=noise.reshape(-1,n,len(sensors))
        sorter=ArraySorter(expected,list(range(len(roots))),windows[:32],half_width=28)
        def score(w):
            result=[sorter.strongest(w[i:i+12]) for i in range(0,len(w),12)]
            return [np.concatenate(x) for x in zip(*result)]
        threshold=np.quantile(score(windows[32:288])[0],.99)
        inputs=windows[544:].copy()
        gains=rng.uniform(.85,1.15,len(probes))*geom['route_gain'][probes]
        offsets=rng.integers(-12,13,len(probes))
        for i,p in enumerate(probes):inputs[i]+=align(actual[p],int(offsets[i]))*gains[i]
        stim_offsets=rng.integers(-6,7,len(inputs))
        blank_offsets=rng.integers(-6,7,256)
        for arm in arms:
            for residual in ([0] if arm=='silent' else [1,.01,0]):
                stim=artifact[arm]*residual+evoked[arm]
                correct=[];false=[]
                for i in range(0,len(inputs),12):
                    end=min(i+12,len(inputs))
                    waves=inputs[i:end]+np.stack([align(stim,int(t)) for t in stim_offsets[i:end]])
                    v,p=sorter.strongest(waves)
                    correct.extend(((v>threshold)&(p==probes[i:end])).tolist())
                for i in range(0,256,12):
                    end=min(i+12,256)
                    waves=windows[288+i:288+end]+np.stack([align(stim,int(t)) for t in blank_offsets[i:end]])
                    v,p=sorter.strongest(waves)
                    false.extend(((v>threshold)&np.isin(p,[j for j,path in enumerate(paths) if path.direction=='DN'])).tolist())
                out.append(dict(seed=seed,arm=arm,artifact_residual=residual,correct=sum(correct),total=len(inputs),
                           blank_dn=sum(false),blanks=256,peak_artifact_uV=float(np.max(abs(artifact[arm]))*1e6),
                           peak_evoked_uV=float(np.max(abs(evoked[arm]))*1e6),correct_flags=correct))
        # Extra diagnostic. Estimate the entire stimulus-locked waveform from 64 independent
        # calibration windows with no spontaneous DN. The detector never sees a test waveform
        # when building the template. The clock reference is the electrical trigger.
        cal_rng=np.random.default_rng(seed+37201)
        for arm in arms[1:]:
            nc=64
            extra_noise,_=front.noise(n*nc,len(sensors),cal_rng)
            stim=artifact[arm]
            calib=(extra_noise.reshape(nc,n,len(sensors)).mean(0)+evoked[arm]+
                   cal_rng.normal(1,.02,nc).mean()*stim)
            for lag in (0,1):
                counts=[]; aligned_counts=[]
                for batch, offsets2, labels in ((inputs,stim_offsets,probes),(windows[288:544],blank_offsets,None)):
                    factors=cal_rng.normal(1,.02,len(batch))
                    delays=cal_rng.integers(-lag,lag+1,len(batch))
                    flags=[]; aligned_flags=[]
                    for i in range(0,len(batch),12):
                        end=min(i+12,len(batch))
                        waves=batch[i:end]+np.stack([
                            align(stim*factors[j]+evoked[arm],int(offsets2[j]+delays[j]))-
                            align(calib,int(offsets2[j])) for j in range(i,end)])
                        if lag:
                            # Estimate onset lag and gain from the observation only. With the
                            # trigger time known, the high-amplitude onset dominates this fit.
                            raw_w=waves+np.stack([align(calib,int(t)) for t in offsets2[i:end]])
                            residual=[]
                            for w,t in zip(raw_w,offsets2[i:end]):
                                region=slice(62+int(t),78+int(t))
                                candidates=np.stack([align(calib,int(t)+dt) for dt in (-1,0,1)])
                                ref=candidates[:,region]
                                beta=((ref*w[region]).sum(axis=(1,2))/(ref**2).sum(axis=(1,2))).clip(.8,1.2)
                                fit=np.argmin(((ref*beta[:,None,None]-w[region])**2).sum(axis=(1,2)))
                                residual.append(w-beta[fit]*candidates[fit])
                            v2,p2=sorter.strongest(np.stack(residual))
                            match=(p2==labels[i:end]) if labels is not None else np.isin(p2,[j for j,path in enumerate(paths) if path.direction=='DN'])
                            aligned_flags.extend(((v2>threshold)&match).tolist())
                        val,pred=sorter.strongest(waves)
                        expected_dn=(pred==labels[i:end]) if labels is not None else np.isin(pred,[j for j,path in enumerate(paths) if path.direction=='DN'])
                        flags.extend(((val>threshold)&expected_dn).tolist())
                    counts.append(sum(flags)); aligned_counts.append(sum(aligned_flags))
                if lag:
                    out.append(dict(seed=seed,arm=arm,artifact_residual='align_calibration64',jitter_samples=lag,
                        correct=aligned_counts[0],total=len(inputs),blank_dn=aligned_counts[1],blanks=256))
                out.append(dict(seed=seed,arm=arm,artifact_residual='calibration64',jitter_samples=lag,
                    correct=counts[0],total=len(inputs),blank_dn=counts[1],blanks=256))
        print(seed,'complete',flush=True)
    write('readback.json',dict(undriven_channels=len(sensors),rows=out))
if __name__=='__main__':main()
