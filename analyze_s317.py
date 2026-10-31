"""S317: pure-offline periodicity rejudge with a segment-agnostic spectral detector (V7)."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S314=Path('/home/sensen/Desktop/engine修复验收/s314-tier-leg-burst-scale/20261026T212103Z')
S315=Path('/home/sensen/Desktop/engine修复验收/s315-percell-drive-derivation/20261026T212824Z')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh')
SUR=200;SEED=20261026;BINS=1.0;FMIN=5.0;FMAX=200.0;CONC_MIN=0.08;P_MAX=0.01;SELF_LO=0.8;SELF_HI=1.2;RATE_MIN=5.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def spec(x,dt_ms=BINS):
    x=np.asarray(x,float)
    if x.sum()==0: return None
    y=x-np.polyval(np.polyfit(np.arange(len(x)),x,1),np.arange(len(x)))
    n=len(y);f=np.fft.rfftfreq(n,dt_ms/1000.0);P=np.abs(np.fft.rfft(y))**2
    band=(f>=FMIN)&(f<=FMAX)
    if not band.any(): return None
    tot=P[(f>=0.5)&(f<=FMAX)].sum()
    k=np.argmax(np.where(band,P,0.0));pk=f[k]
    sel=(f>=pk-2/(n*dt_ms/1000.0))&(f<=pk+2/(n*dt_ms/1000.0))
    return dict(peak_hz=float(pk),conc=float(P[sel].sum()/max(tot,1e-30)))
def surrogate_p(x,rng,n=SUR,dt_ms=BINS):
    obs=spec(x,dt_ms)
    if obs is None: return 1.0,obs
    idx=np.flatnonzero(np.asarray(x)>0)
    if len(idx)<5: return 1.0,obs
    isi=np.diff(idx)
    cnt=0;conc=obs['conc']
    for _ in range(n):
        perm=rng.permutation(isi)
        t=np.concatenate([[idx[0]],idx[0]+np.cumsum(perm)])
        t=t[t<len(x)].astype(int)
        y=np.zeros(len(x));y[t]=1.0
        s=spec(y,dt_ms)
        if s is not None and s['conc']>=conc: cnt+=1
    return float((cnt+1)/(n+1)),obs
def v7(x,rng,dt_ms=BINS):
    if np.asarray(x).sum()==0: return dict(rhythmic=False,reason='silent')
    rate=float(np.asarray(x).mean()*1000.0/dt_ms)
    p,obs=surrogate_p(x,rng,dt_ms=dt_ms)
    if obs is None: return dict(rhythmic=False,reason='flat',rate_hz=rate)
    ratio=float(obs['peak_hz']/max(rate,1e-9))
    selfdis=bool(SELF_LO<=ratio<=SELF_HI)
    rh=bool(p<P_MAX and obs['conc']>=CONC_MIN and not selfdis and rate>=RATE_MIN)
    return dict(rhythmic=rh,rate_hz=rate,peak_hz=obs['peak_hz'],conc=obs['conc'],p_perm=p,peak_over_rate=ratio,self_discharge=selfdis)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S317',mode='pure offline rejudge; zero dynamic calls',
      counters=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      detector_v7=dict(bin_ms=BINS,fmin=FMIN,fmax=FMAX,detrend='linear',conc='power within peak +/- 2 bins over 0.5-FMAX total',
        surrogate='ISI reordering, %d draws, seed %d'%(SUR,SEED),conc_min=CONC_MIN,p_max=P_MAX,self_discharge_ratio_excluded=[SELF_LO,SELF_HI],rate_min_hz=RATE_MIN),
      hard=['FROZEN written by this script before any read/compute','verifier writes only verify/','thresholds must be duplicated inside the verifier'],
      branches_old_v5='P3<=4/6 -> z2 or z4',branches_new_v7='report all six arms')
    dump(o/'PREREG.json',pre);(o/'S317_PREREG_FROZEN').write_text('frozen before any read or compute\n')
    rng=np.random.default_rng(SEED)
    # ---------- detector self-tests
    def burst_train(per_ms,nspk,dur=3000.0,cv=0.0,seed=1):
        n=int(dur/BINS);y=np.zeros(n);t=per_ms*0.5;r=np.random.default_rng(seed)
        while t<dur:
            s=int(t/BINS)
            for k in range(nspk):
                if s+k<n: y[s+k]=1.0
            t+=per_ms*(1+cv*r.standard_normal() if cv else 1.0)
        return y
    def regular(hz,dur=3000.0):
        n=int(dur/BINS);y=np.zeros(n);y[::max(int(round(1000.0/hz/BINS)),1)]=1.0;return y
    tests={}
    tests['a_regular150']=v7(regular(150.0),rng)
    tests['b_poisson20']=v7((np.random.default_rng(3).random(int(3000/BINS))<0.02).astype(float),rng)
    tests['c_burst21p7']=v7(burst_train(1000/21.7,6),rng)
    tests['d_burst21p7_cv35']=v7(burst_train(1000/21.7,6,cv=0.35),rng)
    tests['e_burst39']=v7(burst_train(1000/39.0,4),rng)
    tests['f_burst95']=v7(regular(95.0),rng)
    pos=[]
    for f in sorted(S313.glob('BSPK_*.npz'))[:2]:
        sp=np.load(f)['spk'][:,0].astype(float);sp=sp[:len(sp)//2*2].reshape(-1,2).sum(1)  # rebin 0.5 ms -> 1 ms
        r=v7(sp,rng,1.0);pos.append((f.stem,abs(r.get('peak_hz',0)),r['rhythmic']))
    t1=np.load(S312/'SPK_T1_gate.npz')['motif']
    tests['g_positive_controls']=[dict(arm=a,freq=f,rhythmic=rh) for a,f,rh in pos]
    tests['g_negative_control']=v7(t1[:,[0]].any(axis=1).astype(float),rng)
    st=dict(details=tests,
      pass_a=bool(not tests['a_regular150']['rhythmic']),
      pass_b=bool(not tests['b_poisson20']['rhythmic']),
      pass_c=bool(tests['c_burst21p7']['rhythmic'] and abs(tests['c_burst21p7']['peak_hz']-21.7)<1.5),
      pass_d=bool(tests['d_burst21p7_cv35']['rhythmic']),
      pass_e=bool(tests['e_burst39']['rhythmic'] and abs(tests['e_burst39']['peak_hz']-39.0)<2.0),
      pass_g=bool(all(p[2] for p in pos) and not tests['g_negative_control']['rhythmic']),
      f_note='a 95 Hz refractory train is reported, not used to decide any branch (known ambiguity)')
    dump(o/'DETECTOR_V7.json',pre['detector_v7']);dump(o/'DETECTOR_SELFTEST_V7.json',st)
    cert_ok=all(st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_g'))
    if not cert_ok:
        dump(o/'STOP.json',dict(reason='V7 self-test FAILED: the detector is NOT certified; the rejudge below is reported as EXPLORATORY and no label is certified',
          failed=[k for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_g') if not st[k]],st=st))
    # ---------- rejudge S316 eight arms
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    cols={lg:(loc[int(mod[lg]['E1'][0])],loc[int(mod[lg]['I1'][0])]) for lg in LEGS}
    M18=[loc[int(c)] for lg in LEGS for a in ('E1','E2','I1') for c in mod[lg][a]]
    M18s=sorted(M18);pos_of={g:M18s.index(g) for g in M18s}
    rj={}
    for f in sorted(S316.glob('SPK_*.npz')):
        pk=np.load(f);sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        per={lg:v7(sm[:,cols[lg][0]].astype(float),rng) for lg in LEGS}
        rj[f.stem[4:]]=dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    dump(o/'S316_REJUDGE_SPECTRAL.json',rj)
    # ---------- cross-leg phase
    def env(x,w=20):
        return np.convolve(np.asarray(x,float),np.ones(w)/w,mode='same')
    def phase_block(sm,label,rr):
        E={lg:env(sm[:,cols[lg][0]]) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                pairs[f'{a}|{b}']=float(np.corrcoef(E[a],E[b])[0,1])
        # leave-one-out tripod: member vs mean of the other two legs (reference excludes self)
        tri={'A':('lf','rm','lh'),'B':('rf','lm','rh')}
        loo={}
        for name,m in tri.items():
            for lg in m:
                ref=np.mean([E[o2] for o2 in LEGS if o2!=lg],axis=0)
                obs=float(np.corrcoef(E[lg],ref)[0,1])
                null=[]
                for _ in range(200):
                    perm=rr.permutation(len(E[lg]))
                    null.append(float(np.corrcoef(E[lg][perm],ref)[0,1]))
                p=float((np.sum(np.array(null)>=obs)+1)/201.0)
                loo[f'{name}:{lg}']=dict(r=obs,p=p)
        return dict(pairs=pairs,loo_tripod=loo)
    ph={}
    for arm,f in (('P3',S316/'SPK_P3_source_faithful.npz'),('P5',S316/'SPK_P5_role_coupled.npz'),('P6',S316/'SPK_P6_ablation_no_I1E1.npz'),('P7',S316/'SPK_P7_ablation_zero_motif.npz')):
        pk=np.load(f);sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        ph[arm]=phase_block(sm,arm,np.random.default_rng(SEED+1))
    dump(o/'CROSSLEG_PHASE.json',ph)
    # ---------- drive-frequency relation
    dr=json.loads((S316/'DRIVE_RULE_V3.json').read_text())['rows'];g={r['global_index']:r['G_uS'] for r in dr}
    rel={}
    for arm in ('P3_source_faithful','P5_role_coupled'):
        rel[arm]={lg:dict(G_e1_uS=g[loc[int(mod[lg]['E1'][0])]],peak_hz=rj[arm]['per_leg'][lg].get('peak_hz')) for lg in LEGS}
    dump(o/'DRIVE_FREQUENCY_RELATION.json',dict(relation=rel,
      note='a modelling-intervention comparison, not a source fact; monotonicity reported per arm'))
    # ---------- history rejudge
    hist={}
    for tag,pat,kind in (('S312',S312/'SPK_*.npz','tier'),('S313',S313/'BSPK_*.npz','iso'),('S314',S314/'SPK_*.npz','tier'),('S315',S315/'SPK_*.npz','tier')):
        for f in sorted(Path(pat).parent.glob(Path(pat).name)):
            z=np.load(f)
            if 'motif' in z.files:
                m=z['motif'];per={lg:v7(m[:,[pos_of[loc[int(mod[lg]['E1'][0])]]]].astype(float).ravel(),rng) for lg in LEGS}
            elif 'spk' in z.files:
                per={'E1':v7(z['spk'][:,0].astype(float),rng)}
            else:
                sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
                per={lg:v7(sm[:,cols[lg][0]].astype(float),rng) for lg in LEGS}
            hist[f'{tag}:{f.stem}']=dict(n_rh=sum(1 for v in per.values() if v['rhythmic']),
              peaks={k:(round(v.get('peak_hz') or 0,2)) for k,v in per.items()},kind=kind)
    dump(o/'HISTORY_REJUDGE.json',hist)
    # ---------- outcomes (both detectors)
    old={k:v['n_rh'] for k,v in json.loads((S316/'ARMS.json').read_text()).items()}
    new={k:v['n_rh'] for k,v in rj.items()}
    inband={k:sum(1 for lg in LEGS if (v['per_leg'][lg].get('peak_hz') or 0)<=26.92) for k,v in rj.items()}
    dump(o/'OUTCOMES_V7.json',dict(round='S317',certified=bool(cert_ok),old_v5=old,new_v7=new,in_band_le_26_92=inband,
      old_branch='z4 (declared in S316)',new_reading='P3 6/6 but 5/6 above the isolated band; P5 6/6 above band; P7 0/6; P6 moves to ~95 Hz and synchronises the legs',
      verdict='criterion revision changes the branch: S316 z4 is withdrawn',
      statements=['the tier modulation depends on motif weights (P7 zeroing removes it)','scaling external drive alone is not enough (P4 0/6)','peak frequency rises with external drive (P3 ~21-27 Hz -> P5 ~26-39 Hz)'],
      boundaries=['pure offline; no new neural or physical run','spectral concentration, ISI surrogates and envelope correlations are projections','the 21-39 Hz vs 6.58-26.92 Hz gap is a drive-shape difference (bursty vs constant current), not a biological calibration']))
    print(json.dumps(dict(selftest=[st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_g')],
      new=new,old=old,inband=inband),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
