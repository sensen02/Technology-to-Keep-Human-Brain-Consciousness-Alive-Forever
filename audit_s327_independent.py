"""Independent second implementation for S327: global-mode peak and its Poisson null."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
LEGS=('lf','lm','lh','rf','rm','rh');SEED=515151;POIS=200
def smooth(x,w=20):
    k=np.ones(w)/w;return np.convolve(x.astype(float),k,'same')
def dtr(y):
    t=np.arange(y.size);return y-np.polyval(np.polyfit(t,y,1),t)
def peak(y):
    n=y.size;fr=np.fft.rfftfreq(n,1e-3);pw=np.abs(np.fft.rfft(y))**2;df=fr[1]-fr[0]
    ok=(fr>=4+2*df)&(fr<=48)
    if not ok.any(): return None
    tot=pw[(fr>=0.3)&(fr<=50)].sum();pk=fr[int(np.argmax(np.where(ok,pw,0)))]
    band=(fr>=pk*0.85)&(fr<=pk*1.15)
    return float(pk),float(pw[band].sum()/max(tot,1e-30))
def modind(y,f0,bias):
    n=y.size;t=np.arange(n)*1e-3
    A=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)])
    c,*_=np.linalg.lstsq(A,y,rcond=None);return float(np.hypot(c[0],c[1])/max(abs(bias),1e-12))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    out={}
    srcs={'L1':S324/'SPK_L1_k17_seed0_10s.npz','L2':S324/'SPK_L2_k17abl_seed0_10s.npz','K17':S322/'SPK_K17.npz','K17_abl':S322/'SPK_K17_abl.npz','K10':S322/'SPK_K10.npz'}
    for nm,f in srcs.items():
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        raw=smooth(A.sum(1));bias=float(raw.mean());y=dtr(raw)
        p=peak(y)
        if p is None: continue
        rng=np.random.default_rng(SEED);n_bins=A.shape[0];lam=float(A.sum())/n_bins;conc=[]
        for _ in range(POIS):
            s=(rng.random(n_bins)<lam).astype(float)
            q=peak(dtr(smooth(s)))
            if q: conc.append(q[1])
        out[nm]=dict(peak_hz=round(p[0],2),conc=round(p[1],4),mod_index=round(modind(y,p[0],bias),4),
          poisson_p95_conc=round(float(np.percentile(conc,95)),4) if conc else None,n_bins=int(n_bins))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(nulls=POIS,seed=SEED,result=out),indent=2)+'\n')
    print(json.dumps(out))
    return 0
if __name__=='__main__': raise SystemExit(main())
