"""Independent second implementation for S335: slow-envelope phases and added edge counts."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
BIN=100
def slow(x):
    n=len(x);r=np.array([x[i*BIN:(i+1)*BIN].sum() for i in range(n//BIN)],float);r=r-r.mean()
    b,a=sgn.butter(3,[0.5/(5.0),min(8.0/5.0,0.99)],btype='band')
    return sgn.filtfilt(b,a,r) if len(r)>12 else r
def ph(y): return float(np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(y))))))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    out={}
    for nm in ('C0_gate','R2_1p0','S2_same_1p0','R4_4p0'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        E={lg:slow(A[:,by[lg]].sum(1).astype(float)) for lg in LEGS}
        p={f'{a}|{b}':abs(np.degrees((ph(E[a])-ph(E[b])+np.pi)%(2*np.pi)-np.pi)) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        same=[v for k,v in p.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        cross=[v for k,v in p.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        out[nm]=dict(cross=float(np.mean(cross)),same=float(np.mean(same)),n_anti=int(sum(1 for v in cross if v>=135)),
          pool_median_rate_hz=float(np.median(A[:,[j for (_l,_g),j in pool]].sum(0)*1000.0/A.shape[0])))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['cross'],1),round(v['same'],1),v['n_anti'],round(v['pool_median_rate_hz'],1)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
