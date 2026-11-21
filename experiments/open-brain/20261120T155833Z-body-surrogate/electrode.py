"""Continuous matrix observation and causal blockwise matched-pulse extraction."""
from scipy.signal import resample_poly
from paths import *

SAMPLES = 5   # 20 us on the matrix, 100 us on the archived neck clock
WIDTH = 17    # maximum template support in neck ticks


def library(roots):
    with np.load(ARRAY) as z:
        order=z['roots'].tolist()
        traces=np.split(z['traces'],np.cumsum(z['lengths'])[:-1])
        waves=[resample_poly(traces[order.index(r)],1,5,axis=0).astype('f') for r in roots]
        sites=z['contacts']
    dist=np.linalg.norm(sites[:,None]-sites[None],axis=-1)*1e6
    chol=np.linalg.cholesky(.2+.65*np.exp(-dist/32)+.15*np.eye(64)).astype('f')
    templates=np.zeros((len(roots),WIDTH*SAMPLES,64),dtype='f')
    for i,w in enumerate(waves):
        templates[i,:len(w)]=w
    return templates,chol


def render(src, templates, chol, rng):
    raw=(rng.normal(size=(len(src)*SAMPLES,64)).astype('f')@chol.T)*2
    for t,i in zip(*np.nonzero(src)):
        w=templates[i]
        shift=int(rng.integers(0,3))
        start=t*SAMPLES+shift
        n=min(len(w),len(raw)-start)
        raw[start:start+n]+=w[:n]*(rng.uniform(.87,1.13)*rng.uniform(.92,1.08,64))
    return np.rint(raw/.02).astype(np.int16)


def extract(raw,templates,chol):
    # Same calibration for every trial. Fit every candidate clock tick, never consult source events.
    white=np.linalg.inv(chol.T)/2
    v=(raw.astype('f')*.02)@white
    # Calibration mean over small unknown phase shifts; only prior voltage enters each packet.
    t=(templates+np.pad(templates[:,:-1],((0,0),(1,0),(0,0)))+
       np.pad(templates[:,:-2],((0,0),(2,0),(0,0))))/3
    wave=(t@white).reshape(len(t),WIDTH,-1)
    norms=np.sqrt(np.sum(wave*wave,axis=(1,2)))
    unit=wave/norms[:,None,None]
    v=v.reshape(-1,wave.shape[-1])
    scores=np.zeros((len(v),len(t)),np.float32)
    for lag in range(WIDTH):
        scores[:len(v)-lag]+=v[lag:]@unit[:,lag].T
    # Cross correlations for greedy subtraction on the observed voltage interval.
    gram=np.zeros((WIDTH*2-1,len(t),len(t)),np.float32)
    for lag in range(WIDTH):
        dot=unit[:,lag:].reshape(len(t),-1) @ unit[:,:WIDTH-lag].reshape(len(t),-1).T
        gram[WIDTH-1+lag]=dot
        gram[WIDTH-1-lag]=dot.T
    out=np.zeros_like(scores,dtype=np.uint8)
    # Each decision for start t becomes available no later than its packet's end.
    # One extra preceding packet of candidate starts accounts for overlaps across boundaries.
    for end in range(50,len(v)+1,50):
        hi=end-WIDTH+1
        lo=max(0,hi-100)
        if hi<=lo:continue
        fit=scores[lo:hi].copy()
        mask=np.zeros_like(fit,bool)
        while True:
            idx=np.argmax(fit)
            tick,i=np.unravel_index(idx,fit.shape)
            val=fit[tick,i]
            if val<6.:break
            if tick+lo>=hi-50:
                out[tick+lo,i]=1
            mask[max(0,tick-1):tick+2,i]=True
            a=max(0,tick-WIDTH+1); b=min(len(fit),tick+WIDTH)
            amp=min(val,1.35*norms[i])
            fit[a:b]-=amp*gram[WIDTH-1+a-tick:WIDTH-1+b-tick,i]
            fit[mask]=-np.inf
    return out


if __name__=='__main__':
    binding=read(BASE/'20261117T235930Z-ascending-reference-fit/BINDING.json')
    for seed in SEEDS:
        folder=ARCHIVE/f'seed{seed}'
        meta=read(folder/'INPUTS.json')
        roots=meta['dn_roots']+[r['banc_root'] for r in binding['records']]
        wave,chol=library(roots)
        with np.load(folder/'NEURAL.npz') as z:
            downs,ups=z['dn_arrived'],z['an_emitted']
        for lane,mode in enumerate(meta['modes']):
            dest=HERE/'data'/f'{seed}_{mode}'
            dest.mkdir(exist_ok=True,parents=True)
            src=np.column_stack((downs[:,lane],ups[:,lane]))
            raw=render(src,wave,chol,np.random.default_rng(seed*17+lane))
            calls=extract(raw,wave,chol)
            reported=np.zeros_like(calls)
            reported[WIDTH-1:]=calls[:-WIDTH+1]  # recognition delay; bin only completed packets
            np.savez_compressed(dest/'matrix.npz',adc=raw,uv_per_code=.02,dt_us=20)
            np.savez_compressed(dest/'nerve.npz',roots=roots,dn=reported[:,:17],
                                an=calls[:,17:],check_dn_arrived=downs[:,lane],check_an_emitted=ups[:,lane])
            print(seed,mode,raw.shape,reported[:,:17].sum(),calls[:,17:].sum(),flush=True)
