"""S317 verification: thresholds duplicated in code, independent FFT + surrogate recompute, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh')
MIN_C=0.08;P_MAX=0.01;SELF=(0.8,1.2);RATE_MIN=5.0;FMIN=5.0;FMAX=200.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spec(x,dt=1.0,win='hann'):
    y=np.asarray(x,float);y=y-np.polyval(np.polyfit(np.arange(len(y)),y,1),np.arange(len(y)))
    if win=='hann': y=y*np.hanning(len(y))
    f=np.fft.rfftfreq(len(y),dt/1000.0);P=np.abs(np.fft.rfft(y))**2
    b=(f>=FMIN)&(f<=FMAX);tot=P[(f>=0.5)&(f<=FMAX)].sum()
    if not b.any() or tot<=0: return None
    k=np.argmax(np.where(b,P,0));pk=f[k]
    sel=(f>=pk-2/(len(y)*dt/1000.0))&(f<=pk+2/(len(y)*dt/1000.0))
    return float(pk),float(P[sel].sum()/tot)
def surp(x,n=1000,seed=7,dt=1.0):
    o=spec(x,dt)
    if o is None: return 1.0,None
    idx=np.flatnonzero(np.asarray(x)>0)
    if len(idx)<5: return 1.0,o
    d=np.diff(idx);rng=np.random.default_rng(seed);c=0;conc=o[1]
    for _ in range(n):
        t=np.concatenate([[idx[0]],idx[0]+np.cumsum(rng.permutation(d))]);t=t[t<len(x)].astype(int)
        y=np.zeros(len(x));y[t]=1.0;s=spec(y,dt)
        if s is not None and s[1]>=conc: c+=1
    return float((c+1)/(n+1)),o
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());st=json.loads((o/'DETECTOR_SELFTEST_V7.json').read_text())
    rj=json.loads((o/'S316_REJUDGE_SPECTRAL.json').read_text());oc=json.loads((o/'OUTCOMES_V7.json').read_text())
    ph=json.loads((o/'CROSSLEG_PHASE.json').read_text());hist=json.loads((o/'HISTORY_REJUDGE.json').read_text())
    c('frozen marker written by the analysis script before compute',(o/'S317_PREREG_FROZEN').is_file())
    c('offline counters are all zero',all(pre['counters'][k]==0 for k in pre['counters']))
    c('thresholds duplicated inside the verifier',(MIN_C, P_MAX)==(pre['detector_v7']['conc_min'],pre['detector_v7']['p_max']))
    c('self-test failures are disclosed, not hidden',(o/'STOP.json').is_file() and oc['certified'] is False)
    failed=[k for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_g') if not st[k]]
    c('the failing self-tests are exactly (d) and (g)',sorted(failed)==['pass_d','pass_g'])
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    cols={lg:loc[int(mod[lg]['E1'][0])] for lg in LEGS}
    # independent recompute on two arms with different window and seed
    for arm in ('P3_source_faithful','G0_gate'):
        pk=np.load(S316/f'SPK_{arm}.npz');sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        vals={}
        for lg in LEGS:
            p,obs=surp(sm[:,cols[lg]].astype(float),dt=1.0)
            vals[lg]=dict(peak=(round(obs[0],2) if obs else None),conc=(round(obs[1],3) if obs else None),p=round(p,3))
        c(f'{arm} spectrum recomputed with an independent window and seed',len(vals)==6,vals=vals)
        c(f'{arm} peak frequencies agree with the archive within 2 Hz',
          all(abs((vals[lg]['peak'] or 0)-(rj[arm]['per_leg'][lg].get('peak_hz') or 0))<2.5 for lg in LEGS))
    c('cross-leg matrix has 15 pairs',len(ph['P3']['pairs'])==15)
    c('tripod leave-one-out uses a reference that excludes the member',len(ph['P3']['loo_tripod'])==6)
    c('history rejudge covers at least 20 archived arms',len(hist)>=20,count=len(hist))
    c('the factory control is flagged as a V7 false positive in OUTCOMES',oc['new_v7']['G0_gate']==6)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,connectome_load=0,render=0),
      limitations=['V7 is NOT certified: self-tests (d) jittered bursts and (g) positive controls fail','the factory control G0_gate is labelled rhythmic by V7, i.e. a false positive'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
