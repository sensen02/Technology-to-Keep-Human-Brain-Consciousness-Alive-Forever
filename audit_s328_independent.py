"""Independent second implementation for S328: autocorrelation descriptor and band ratio."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
S327=Path('/home/sensen/Desktop/engine修复验收/s327-global-mode-and-motif-influence/20261027T004101Z')
SEED=616161
def smooth(x,w=20):
    return np.convolve(x.astype(float),np.ones(w)/w,'same')
def dtr(y):
    t=np.arange(y.size);return y-np.polyval(np.polyfit(t,y,1),t)
def acf(y):
    c=y-y.mean();a=np.correlate(c,c,'full')[y.size-1:];return a/max(a[0],1e-30)
def desc(y):
    a=acf(y)
    seg=a[8:61];neg=None
    if seg.size and seg.min()<0: neg=int(8+int(np.argmin(seg)))
    rec=None
    if neg is not None:
        tail=a[neg:neg+40];pos=np.flatnonzero(tail>0)
        if pos.size: rec=int(neg+pos[0])
    lags={k:round(float(a[k]),4) for k in (5,10,20,25,40,50) if k<a.size}
    return dict(lags=lags,neg=neg,rec=rec)
def band(y,lo=18.0,hi=26.0):
    n=y.size;f=np.fft.rfftfreq(n,1e-3);P=np.abs(np.fft.rfft(y))**2
    return float(P[(f>=lo)&(f<=hi)].sum()/max(P[(f>=0.3)&(f<=50)].sum(),1e-30))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    out={}
    srcs={'B0':o/'SPK_G_tier_gate.npz','B1':S327/'SPK_B1_motif_no_out.npz','B2':S327/'SPK_B2_no_motif_drive.npz',
          'K17':S322/'SPK_K17.npz','K17_abl':S322/'SPK_K17_abl.npz','A0_gate':S322/'SPK_A0_gate.npz',
          'L1':S324/'SPK_L1_k17_seed0_10s.npz','L2':S324/'SPK_L2_k17abl_seed0_10s.npz'}
    for nm,f in srcs.items():
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        y=dtr(smooth(A.sum(1)))
        d=desc(y);d['band_18_26']=round(band(y),4);d['n_bins']=int(A.shape[0])
        out[nm]=d
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(seed=SEED,result=out),indent=2)+'\n')
    print(json.dumps({k:(v['neg'],v['rec'],v['band_18_26']) for k,v in out.items()}))
    return 0
if __name__=='__main__': raise SystemExit(main())
