"""Independent second implementation for S332: Welch-segment cross-spectral phase."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S330=Path('/home/sensen/Desktop/engine修复验收/s330-leg-specific-budget-and-decoupling/20261027T011205Z')
S331=Path('/home/sensen/Desktop/engine修复验收/s331-antiphase-capability/20261027T012220Z')
LEGS=('lf','lm','lh','rf','rm','rh');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def sm(x):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    return (y-np.polyval(np.polyfit(t,y,1),t))[::2]
def csd_phase(x,y,fs=500.0,nper=1024,band=(16.0,26.0)):
    n=min(len(x),len(y));x=np.asarray(x)[:n];y=np.asarray(y)[:n]
    nper=min(nper,len(x));step=max(1,nper//2);win=np.hanning(nper);acc=np.zeros(nper//2+1,dtype=complex);cnt=0
    f=np.fft.rfftfreq(nper,1/fs)
    for s in range(0,len(x)-nper+1,step):
        X=np.fft.rfft(x[s:s+nper]*win);Y=np.fft.rfft(y[s:s+nper]*win)
        acc+=X*np.conj(Y);cnt+=1
    if cnt==0: return None
    S=acc/max(cnt,1);m=(f>=band[0])&(f<=band[1])
    if not m.any(): return None
    w=np.abs(S[m]);return float(np.angle(np.sum(w*np.exp(1j*np.angle(S[m])))))
def ad(a,b):
    d=(a-b+np.pi)%(2*np.pi)-np.pi;return abs(np.degrees(d))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    srcs={'K17':S322/'SPK_K17.npz','Za_pi_30':S331/'SPK_Za_pi_30.npz','Z0_gate':S331/'SPK_Z0_gate.npz',
          'G0_gate':o/'SPK_G0_gate.npz','D4_dI040':o/'SPK_D4_dI040.npz','C1_common020':o/'SPK_C1_common020.npz'}
    out={}
    for nm,f in srcs.items():
        if not Path(f).is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        E={lg:sm(A[:,by[lg]].sum(1)) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                p=csd_phase(E[a],E[b])
                pairs[f'{a}|{b}']=None if p is None else float(np.degrees(abs(p)))
        same=[v for k,v in pairs.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        cross=[v for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        out[nm]=dict(same=float(np.mean(same)),cross=float(np.mean(cross)),sel=float(np.mean(cross)-np.mean(same)),pairs=pairs)
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(result=out),indent=2)+'\n')
    print(json.dumps({k:(round(v['same'],1),round(v['cross'],1),round(v['sel'],1)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
