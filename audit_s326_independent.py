"""Independent second implementation for S326: coherence via direct DFT segments, different seed."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
LEGS=('lf','lm','lh','rf','rm','rh');NULLS=1000;SEED=424242
def bst(x):
    idx=np.where(x>0)[0]
    if idx.size<2: return []
    g=np.diff(idx);cut=max(3*np.median(g),5.0);st=[idx[0]]
    for d,c in zip(g,idx[1:]):
        if d>cut: st.append(c)
    return st
def sm(x,w=20):
    k=np.ones(w)/w;return np.convolve(x.astype(float),k,'same')
def segdft(x,f0,dt=1e-3,L=1000,step=500):
    t=np.arange(L)*dt;w=np.hanning(L);out=[]
    for s in range(0,len(x)-L+1,step):
        out.append(np.sum(x[s:s+L]*w*np.exp(-2j*np.pi*f0*t)))
    return np.array(out)
def coh(X,Y):
    return float(abs(np.sum(X*np.conj(Y)))**2/max(np.sum(abs(X)**2)*np.sum(abs(Y)**2),1e-30))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    E1={lg:loc[int(mod[lg]['E1'][0])] for lg in LEGS}
    import csv
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    A1=np.unpackbits(np.load(S324/'SPK_L1_k17_seed0_10s.npz')['packed'],axis=1)[:,:12000].astype(bool)
    A2=np.unpackbits(np.load(S324/'SPK_L2_k17abl_seed0_10s.npz')['packed'],axis=1)[:,:12000].astype(bool)
    f0={}
    for lg in LEGS:
        st=bst(A1[:,E1[lg]].astype(float))
        f0[lg]=(1000.0/float(np.mean(np.diff(st)))) if len(st)>=10 else None
    rng=np.random.default_rng(SEED);res={'cells':[],'legs':{}}
    for (lg,gi),j in sorted(seen.items()):
        if f0[lg] is None: continue
        Y1=segdft(sm(A1[:,j].astype(float)),f0[lg]);X1=segdft(sm(A1[:,E1[lg]].astype(float)),f0[lg])
        c=coh(X1,Y1);nl=[]
        for _ in range(NULLS):
            th=rng.uniform(0,2*np.pi,X1.size);nl.append(coh(X1*np.exp(1j*th),Y1))
        nl=np.array(nl)
        res['cells'].append(dict(leg=lg,gi=gi,coh=c,null_p95=float(np.percentile(nl,95)),
          p=float((np.sum(nl>=c)+1)/(len(nl)+1)),passes=bool(c>np.percentile(nl,95) and (np.sum(nl>=c)+1)/(len(nl)+1)<0.05)))
    cols={lg:[j for (l2,gi),j in seen.items() if l2==lg] for lg in LEGS}
    for lg in LEGS:
        if f0[lg] is None or not cols[lg]: continue
        Y=segdft(sm(A1[:,cols[lg]].sum(1).astype(float)),f0[lg]);X=segdft(sm(A1[:,E1[lg]].astype(float)),f0[lg])
        Y2=segdft(sm(A2[:,cols[lg]].sum(1).astype(float)),f0[lg]);X2=segdft(sm(A2[:,E1[lg]].astype(float)),f0[lg])
        res['legs'][lg]=dict(coh_L1=coh(X,Y),coh_L2=coh(X2,Y2),f0=f0[lg])
    res['n_pass']=sum(1 for c in res['cells'] if c['passes']);res['n_cells']=len(res['cells'])
    res['nulls']=NULLS;res['seed']=SEED
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(dict(n_pass=res['n_pass'],n=res['n_cells'],legs=res['legs']),default=float))
    return 0
if __name__=='__main__': raise SystemExit(main())
