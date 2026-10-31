"""Independent second implementation for S331: leg phases, selectivity and circular dispersion."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
LEGS=('lf','lm','lh','rf','rm','rh');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rm')}
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def sm(x,ds=2):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds]
def ph(y,f0,dt=2e-3):
    # y is already 2x downsampled (2 ms grid), so dt must be 2e-3
    t=np.arange(y.size)*dt
    z=np.sum(y*np.exp(-2j*np.pi*f0*t));return float(np.angle(z))
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
    K=np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000]
    idx=np.flatnonzero(K[:,loc[int(mod['lf']['E1'][0])]])
    d=np.diff(idx);f0=float(1000.0/max(np.mean(d),1e-9))
    out={}
    for nm,f in (('K17',S322/'SPK_K17.npz'),('Z0_gate',o/'SPK_Z0_gate.npz'),('Za_pi_10',o/'SPK_Za_pi_10.npz'),('Za_pi_30',o/'SPK_Za_pi_30.npz'),('Zb_same_30',o/'SPK_Zb_same_30.npz'),('Zc_pi_30_swap',o/'SPK_Zc_pi_30_swap.npz'),('K17_abl',S322/'SPK_K17_abl.npz')):
        if not Path(f).is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        legph={lg:ph(sm(A[:,by[lg]].sum(1)),f0) for lg in LEGS}
        ps=[];pc=[]
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                dd=ad(legph[a],legph[b]);same=any(a in TRIPOD[t] and b in TRIPOD[t] for t in TRIPOD)
                (ps if same else pc).append(dd)
        ph_all=[ph(sm(A[:,j]),f0) for j in [j for (_l,_g),j in pool]]
        z_=np.exp(1j*np.array(ph_all));R=float(abs(z_.mean()))
        anti=float(np.mean([ad(legph[a],legph[b])>=135 for i,a in enumerate(LEGS) for b in LEGS[i+1:]]))
        out[nm]=dict(f0=f0,pool_R=R,same=float(np.mean(ps)),cross=float(np.mean(pc)),
          sel=float(np.mean(pc)-np.mean(ps)),anti_frac=anti,
          pool_median_rate_hz=float(np.median(A[:, [j for (_l,_g),j in pool]].sum(0)*1000.0/A.shape[0])))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(result=out),indent=2)+'\n')
    print(json.dumps({k:(round(v['same'],1),round(v['cross'],1),round(v['anti_frac'],3),round(v['pool_R'],3)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
