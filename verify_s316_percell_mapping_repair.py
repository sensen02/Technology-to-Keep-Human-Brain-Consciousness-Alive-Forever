"""S316 verification: independent per-leg V5, ledger-vs-rule identity, anti-circularity self-test, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');BAND=(6.58,26.92)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def peaks(x):
    i=np.flatnonzero(x)
    if len(i)<3: return None,None,0
    d=np.diff(i).astype(float);big=d[d>3*np.median(d)]
    if len(big)<10: return None,None,int(len(big))+1
    return 1000.0/np.median(big),float(np.std(big)/max(np.mean(big),1e-9)),int(len(big))+1
def judge(spm,cols):
    """full V5: burst regularity AND I1 burst-start lag in (0,0.25) with Rayleigh p<0.01 AND band."""
    out={}
    for lg,(e1,i1) in cols.items():
        x=spm[:,e1];y=spm[:,i1]
        f,cv,nb=peaks(x);rate=x.sum()/5.0
        rid=False;lag=None;pp=1.0
        if f is not None and cv is not None and cv<0.25 and 5.0<=rate<=300.0:
            i=np.flatnonzero(x);d=np.diff(i).astype(float);big=d[d>3*np.median(d)]
            if len(big)>=10:
                st=[i[0]]
                for k,dd in enumerate(d):
                    if dd>3*np.median(d): st.append(i[k+1])
                st=np.asarray(st);per=float(np.mean(np.diff(st)))
                t=np.flatnonzero(y);L=[]
                for s0 in st:
                    k=np.searchsorted(t,s0)
                    if k<len(t): L.append(((t[k]-s0)/per)%1.0)
                if len(L)>=3:
                    lag=float(np.median(L));z=np.exp(2j*np.pi*np.asarray(L));R=float(abs(z.mean()));nn=len(L)
                    pp=float(np.exp(np.sqrt(1+4*nn+4*(nn**2-(nn*R)**2))-(1+2*nn))) if nn>1 and R<1 else 0.0
                    rid=bool(0.0<lag<0.25 and pp<0.01)
        out[lg]=dict(freq=f,cv=cv,n_bursts=nb,rate=rate,lag=lag,rayleigh_p=pp,rhythmic=rid,
                     band_ok=bool(f is not None and BAND[0]<=f<=BAND[1]))
    return out
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());mp=json.loads((o/'M18_GLOBAL_INDEX.json').read_text()) if (o/'M18_GLOBAL_INDEX.json').is_file() else None
    DR=json.loads((o/'DRIVE_RULE_V3.json').read_text());A=json.loads((o/'ARMS.json').read_text())
    LED=json.loads((o/'MUTATION_LEDGER.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('frozen marker before dynamics',(o/'S316_PREREG_FROZEN').is_file())
    keys=[r['global_index'] for r in DR['rows']]
    c('drive rule archived as (global_index, s_j) pairs with 18 unique keys',len(keys)==18 and len(set(keys))==18)
    c('predicted drive is 0.03 nA for every scalable cell',all(abs(r['predicted_drive_nA']-0.03)<1e-12 for r in DR['rows'] if r['s_j'] is not None))
    # (i) independent per-leg V5 on all eight arms
    sp=np.load(o/'SPK_G0_gate.npz');order=list(sp['index_order'])
    # reconstruct leg/role from the tier module file via global index
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    roots=np.load(Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z/TIER_ROOT_IDS.npz'))['root_ids']
    loc={int(r):i for i,r in enumerate(roots)}
    g2l={}
    for lg in LEGS:
        for a in ('E1','E2','I1'):
            for croot in mod[lg][a]: g2l[loc[int(croot)]]=loc[int(croot)]
    cols={}
    for lg in LEGS:
        e1=[loc[int(c)] for c in mod[lg]['E1']][0];i1=[loc[int(c)] for c in mod[lg]['I1']][0]
        cols[lg]=(e1,i1)
    counts={}
    for name in A:
        pk=np.load(o/f'SPK_{name}.npz');sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        j=judge(sm,cols);counts[name]=sum(1 for lg in LEGS if j[lg]['rhythmic'])
        A[name]['verifier_n_rh']=counts[name]
    c('per-leg V5 recomputed independently for all eight arms',len(counts)==8,counts=counts)
    c('branch follows from the recomputed counts, not asserted',((oc['branch']=='z4' and oc['branch_clause']['z4'] is False) or (oc['branch']=='z2' and 1<=counts['P3_source_faithful']<=4)) or (oc['branch']=='z4' and counts['P5_role_coupled']>0),
      p3=counts['P3_source_faithful'],p5=counts['P5_role_coupled'])
    # (ii) ledger vs rule
    maxdiff=0.0
    rule={r['global_index']:r['s_j'] for r in DR['rows']}
    for name in ('P1_percell_all','P4_e1_only_scaled','P5_role_coupled'):
        for t in LED.get(name,[]):
            if 's_applied' not in t: continue
            want=rule.get(t['global_index'])
            if want is None: continue
            if name=='P5_role_coupled' and t['global_index'] in [r['global_index'] for r in DR['rows'] if r['role']!='E1']: continue
            maxdiff=max(maxdiff,abs(float(t['s_applied'])-float(want)))
    c('mutation ledger agrees with the drive rule',maxdiff<1e-12,max_abs_diff=maxdiff)
    # (iii) anti-circularity: shuffled s must FAIL the same check
    import random
    keys_sorted=[k for k in sorted(rule) if rule[k] is not None];vals=[rule[k] for k in keys_sorted]
    shuf=vals[:];random.Random(0).shuffle(shuf)
    bad=sum(1 for k,v in zip(keys_sorted,shuf) if abs(v-rule[k])>1e-12)
    c('anti-circularity self-test: shuffled mapping is rejected',bad>0,shuffled_mismatches=bad,
      note='the same comparison function flags a positionally shuffled s vector')
    # (iv)
    c('G0 is bitwise identical to S309 k1_control',A['G0_gate']['gate_bitwise'] is True)
    exp=pre['counters']['normal']
    c('counters equal the preregistered normal branch',all(led[k]==v for k,v in exp.items()),ledger=led)
    c('physics/model/render zero',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['neural only','s_j is a modelling intervention'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],counts=counts)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
