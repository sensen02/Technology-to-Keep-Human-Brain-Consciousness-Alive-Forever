"""Linear voltage compression of independent, equal-PSD electrode channels before sorting."""
from pathlib import Path
import sys, json
from dataclasses import replace
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
OLD=BASE/'20261120T155833Z-body-surrogate'
sys.path.insert(0,str(BASE/'20261121T005613Z-patch226-write'))
from common import geom,contacts,patterns,paths,roots,IO,live
from omnisim.modules.flex_depth import extracellular_traces
from omnisim.modules.cmos_readout import FrontEnd
read=lambda f:json.loads(f.read_text())
def write(name,v): (OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
CLOCK=4
WIDTH=20
def raw_bank(selected, sites, front):
    response=front.response(extracellular_traces(selected,sites,.15e-6))
    return response[:,64:64+CLOCK*WIDTH]*1e6

def mix(src,bank):
    out=np.zeros((len(src)*CLOCK,bank.shape[-1]),np.float32)
    for tick,r in zip(*src.nonzero()):
        offset=tick*CLOCK
        n=min(len(bank[r]),len(out)-offset)
        out[offset:offset+n]+=bank[r,:n]
    return out

def scores(voltage,unit):
    frames=voltage.reshape(-1,unit.shape[-1])
    score=np.zeros((len(frames),len(unit)),np.float32)
    for lag in range(WIDTH):
        score[:len(frames)-lag]+=frames[lag:]@unit[:,lag].T
    return score

def sort(score,gram,sigma,criterion):
    # Score candidates only once a complete waveform is available to the recording packet.
    events=np.zeros_like(score,dtype='B')
    available=np.zeros_like(score,dtype='B')
    for end in range(50,len(score)+1,50):
        hi=end-WIDTH+1
        lo=max(0,hi-100)
        fit=score[lo:hi].copy()
        forbidden=np.zeros_like(fit,bool)
        while True:
            normalized=fit/sigma
            normalized[forbidden]=-np.inf
            t,r=np.unravel_index(normalized.argmax(),fit.shape)
            if normalized[t,r]<criterion: break
            if t+lo>=hi-50:
                events[t+lo,r]=1
                available[t+lo+WIDTH-1,r]=1
            forbidden[max(0,t-2):t+3,r]=True
            a=max(0,t-WIDTH+1); b=min(len(fit),t+WIDTH)
            amp=fit[t,r]
            fit[a:b]-=amp*gram[WIDTH-1+a-t:WIDTH-1+b-t,r]
    return available,events

def main():
    selected=[roots.index(r) for r in IO['dn_roots']+IO['an_roots_left_right']]
    keep=~patterns().any(0)
    front=FrontEnd(np.sqrt(2*np.pi)*.15e-6)
    selected_paths=[paths[i] for i in selected]
    nominal=raw_bank(selected_paths,contacts('nominal')[keep],front)
    actual=raw_bank(selected_paths,contacts('actual')[keep],front)
    # All input DNs propagate toward the cut; only append source-SWC points shared
    # by every traced continuation. Ambiguous branches supply no invented dipoles.
    cut_paths=[]
    with np.load(OUT/'extensions.npz') as ext:
        for path in selected_paths[:17]:
            chains=[ext[k] for k in ext.files if k.startswith(path.root+'_')]
            segment=[]
            if chains:
                for points in zip(*chains):
                    if not np.allclose(points,points[0],atol=1e-13,rtol=0):break
                    segment.append(points[0])
            xyz=path.points_m
            assert xyz[-1,1]>xyz[0,1]
            if len(segment)>1:xyz=np.vstack((xyz,np.asarray(segment)[1:]))
            cut_paths.append(replace(path,points_m=xyz))
    actual_cut=raw_bank(cut_paths,contacts('actual')[keep],front)
    # Equalize per-root influence when choosing the compression basis.
    balanced=nominal/np.linalg.norm(nominal.reshape(len(nominal),-1),axis=1)[:,None,None]
    matrix=balanced.reshape(-1,nominal.shape[-1])
    _,vectors=np.linalg.eigh(matrix.T@matrix)
    basis=np.ascontiguousarray(vectors[:,-96:])
    assert np.allclose(basis.T@basis,np.eye(96),atol=1e-5)
    projected=nominal@basis
    physical=(actual@basis)*geom['route_gain'][selected,None,None].astype('f')
    physical_cut=(actual_cut@basis)*geom['route_gain'][selected[:17],None,None].astype('f')
    waves=projected.reshape(len(selected),WIDTH,-1)
    unit=waves/np.linalg.norm(waves.reshape(len(waves),-1),axis=1)[:,None,None]
    gram=np.zeros((2*WIDTH-1,len(unit),len(unit)),'f')
    for lag in range(WIDTH):
        dot=unit[:,lag:].reshape(len(unit),-1)@unit[:,:WIDTH-lag].reshape(len(unit),-1).T
        gram[WIDTH-1+lag]=dot
        gram[WIDTH-1-lag]=dot.T
    # Generate the full physical input for calibration and each trial before projection.
    def noise(n,rng):
        frames=[]
        for block in range(0,n,8000):
            frames.append((front.noise(min(8000,n-block),int(keep.sum()),rng)[0]@basis)*1e6)
        return np.concatenate(frames)
    blank=noise(80000,np.random.default_rng(448))
    s=scores(blank,unit)
    sigma=s[:-WIDTH].std(0)
    threshold=float(np.quantile((s[:-50]/sigma).reshape(-1,50,len(unit)).max((1,2)),.99))
    rng=np.random.default_rng(773)
    src=np.zeros((20000,19),'B')
    bins=np.arange(50,19900,52);src[bins,np.arange(len(bins))%2+17]=1
    result=sort(scores(noise(80000,rng)+mix(src,physical),unit),gram,sigma,threshold)[1]
    sensitivity=[float(np.mean([bool(result[t-2:t+3,r].any()) for t in np.flatnonzero(src[:,r])])) for r in (17,18)]
    write('calibration.json',dict(read_channels=int(keep.sum()),latent_dimensions=96,registered_candidates=len(selected),sensitivity=sensitivity,threshold=threshold,projection_energy=float((projected**2).sum()/(nominal**2).sum()),noise_block_samples=8000))
    np.savez_compressed(OUT/'calibration.npz',basis=basis,unit=unit,sigma=sigma,read_site_ids=live[keep],roots=np.array(IO['dn_roots']+IO['an_roots_left_right']))
    print('calibrated',sensitivity,threshold,flush=True)
    for i,folder in enumerate(sorted((OLD/'data').iterdir())):
        seed,mode=folder.name.split('_',1)
        archive=BASE/'20261118T051014Z-neck-fault-assay/FORMAL'/('seed'+seed)
        if seed=='11903':
            archive=Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T052426Z-event-anchored-repair/FORMAL/seed11903')
        meta=read(archive/'INPUTS.json');lane=meta['modes'].index(mode)
        assert meta['dn_roots']==IO['dn_roots']
        with np.load(archive/'NEURAL.npz') as z:
            dn=z['dn_emitted'][:,lane].astype(bool);an=z['an_emitted'][:,lane].astype(bool)
        with np.load(folder/'nerve.npz') as z:
            assert z['roots'].astype(str).tolist()==IO['dn_roots']+IO['an_roots_left_right'] and np.array_equal(an,z['check_an_emitted'])
        blank=noise(80000,np.random.default_rng(7951+i))
        voltage=blank+mix(dn,physical_cut)
        downs=sort(scores(voltage,unit),gram,sigma,threshold)[0][:,:17]
        intact=blank+mix(dn,physical[:17])+mix(an,physical[17:])
        ups=sort(scores(intact,unit),gram,sigma,threshold)[1][:,17:]
        dest=OUT/'data'/folder.name
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'observed.npz',dn=downs,an=ups)
        if i==0: np.savez_compressed(OUT/'voltage_sample.npz',cut_uV=voltage[:2000],intact_uV=intact[:2000],sample_rate_hz=40000)
        print(folder.name,int(downs.sum()),int(ups.sum()),flush=True)

if __name__=='__main__':main()
