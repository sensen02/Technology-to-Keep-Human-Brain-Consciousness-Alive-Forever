"""Independent second implementation for S334: band scoring and ladder phases."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S333=Path('/home/sensen/Desktop/engine修复验收/s333-input-channel-relay/20261027T014011Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def csr(z):
    n=int(z['indptr'].size-1)
    return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
def env(x):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    return (y-np.polyval(np.polyfit(t,y,1),t))[::2]
def ph(x,y,band,fs=500.0,nper=1024):
    n=min(len(x),len(y));x=np.asarray(x)[:n];y=np.asarray(y)[:n];nper=min(nper,n)
    acc=np.zeros(nper//2+1,complex);c=0;w=np.hanning(nper);f=np.fft.rfftfreq(nper,1/fs)
    for s in range(0,n-nper+1,nper//2):
        X=np.fft.rfft(x[s:s+nper]*w);Y=np.fft.rfft(y[s:s+nper]*w);acc+=X*np.conj(Y);c+=1
    if c==0: return None
    S=acc/c;m=(f>=band[0])&(f<=band[1])
    if not m.any(): return None
    ww=np.abs(S[m])
    if ww.sum()<=0: return None
    return float(np.degrees(np.angle(np.sum(ww*np.exp(1j*np.angle(S[m]))))))
def sums(pairs):
    same=[abs(v) for k,v in pairs.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[abs(v) for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    return dict(same=float(np.mean(same)) if same else None,cross=float(np.mean(cross)) if cross else None,
      n_anti=int(sum(1 for v in cross if v>=135)),cross_min=float(min(cross)) if cross else None)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18={lg:[loc[int(c)] for r in ROLES for c in mod[lg][r]] for lg in LEGS};Mset=set(j for lg in LEGS for j in M18[lg])
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS};cols=[j for (_l,_g),j in pool];poolset=set(cols)
    W=csr(np.load(S303/'EDGES_w_2.5e-5_We.npz'))
    per={lg:np.asarray(W[cols,:][[cols.index(j) for j in by[lg]],:].sum(0)).ravel() for lg in LEGS}
    tot=sum(per[lg] for lg in LEGS);SE={}
    for lg in LEGS:
        frac=np.where(tot>1e-12,per[lg]/np.maximum(tot,1e-12),0.0)
        SE[lg]=np.array([int(j) for j in np.flatnonzero((frac>=0.90)&(per[lg]>0)) if int(j) not in poolset and int(j) not in Mset],int)
    out={}
    for nm in ('A0_gate','T1_in_diff010','T2_in_diff020','T3_in_diff040','T5_in_common020'):
        f=S333/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        A=np.unpackbits(np.load(f)['packed'],axis=1)[:,:12000]
        EN={lg:env(A[:,SE[lg]].sum(1)) for lg in LEGS};EP={lg:env(A[:,by[lg]].sum(1)) for lg in LEGS}
        pi_={f'{a}|{b}':ph(EN[a],EN[b],(160.0,190.0)) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        pp={f'{a}|{b}':ph(EP[a],EP[b],(160.0,190.0)) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        out[nm]=dict(injected=sums(pi_),pool_drive_band=sums(pp))
    arms={}
    for nm in ('G0_gate','F1_12p5','F2_20','F3_25p39','F4_40','F5_a010_25p39','F6_seed1_25p39','K1_off_2500','K2_common_off_2500'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        arms[nm]=dict(n_bins=int(A.shape[0]))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(reanalysis=out,arms=arms),indent=2)+'\n')
    print(json.dumps({k:(round(v['injected']['cross'],1),round(v['pool_drive_band']['cross'],1),v['pool_drive_band']['n_anti']) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
