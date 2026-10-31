"""S318: certify a burst-envelope detector (V8), rejudge all archives, external-envelope control."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S314=Path('/home/sensen/Desktop/engine修复验收/s314-tier-leg-burst-scale/20261026T212103Z')
S315=Path('/home/sensen/Desktop/engine修复验收/s315-percell-drive-derivation/20261026T212824Z')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
SMOOTH_MS=20.0;FLO,FHI=5.0,45.0;CONC_MIN=0.15;P_MAX=0.01;NULLS=200;SEED=20261026
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def bursts_of(x,dt):
    idx=np.flatnonzero(x)
    if len(idx)<2: return np.array([],int),np.array([],int)
    isi=np.diff(idx)*dt;thr=max(3*np.median(isi),5.0)
    st=[idx[0]];cn=[1]
    for k,d in enumerate(isi):
        if d>thr: st.append(idx[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def env_series(starts,weights,n,dt):
    y=np.zeros(n);y[np.clip(starts,0,n-1)]+=weights
    w=int(round(SMOOTH_MS/dt));y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def spec_env(y,dt):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2
    df=f[1]-f[0];interior=(f>=FLO+2*df)&(f<=FHI-2*df)
    if not interior.any(): return None
    tot=P[(f>=0.3)&(f<=50.0)].sum()
    k=np.argmax(np.where(interior,P,0));pk=float(f[k])
    sel=(f>=pk-2*df)&(f<=pk+2*df)
    return dict(peak_hz=pk,conc=float(P[sel].sum()/max(tot,1e-30)))
def v8(x,dt,rng):
    x=np.asarray(x,float)
    if x.sum()==0: return dict(rhythmic=False,reason='silent',dt_ms=dt)
    rate=float(x.mean()*1000.0/dt);st,cn=bursts_of(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0.0
    base=dict(rate_hz=rate,nb=int(nb),spb=spb,dt_ms=dt)
    if nb<10 or spb>12 or rate<5.0: return dict(**base,rhythmic=False,reason='burst_gate')
    y=env_series(st,cn,len(x),dt);obs=spec_env(y,dt)
    if obs is None: return dict(**base,rhythmic=False,reason='no_interior')
    dur=len(x)*dt;cv=float(np.std(np.diff(st)*dt)/max(np.mean(np.diff(st)*dt),1e-9))
    cnt=0
    for _ in range(NULLS):
        nss=np.sort(rng.integers(0,len(x),size=nb))
        nw=cn[rng.integers(0,len(cn),size=nb)]
        s2=spec_env(env_series(nss,nw,len(x),dt),dt)
        if s2 is not None and s2['conc']>=obs['conc']: cnt+=1
    p=float((cnt+1)/(NULLS+1))
    ratio=obs['peak_hz']/max(rate,1e-9)
    selfdis=bool(spb<=1.5 and 0.8<=ratio<=1.2)
    rh=bool(p<P_MAX and obs['conc']>=CONC_MIN and not selfdis)
    return dict(**base,rhythmic=rh,peak_hz=obs['peak_hz'],conc=obs['conc'],p_null=p,burst_cv=cv,self_discharge=selfdis,peak_over_rate=ratio)
def leg_level(cells,dt,rng):
    res={r:v8(cells[r],dt,rng) for r in ROLES if r in cells}
    ok=[r for r in res if res[r].get('rhythmic')]
    peaks=[res[r]['peak_hz'] for r in ok]
    shared=bool(len(ok)>=2 and (max(peaks)-min(peaks))/max(np.mean(peaks),1e-9)<=0.20)
    return dict(roles=res,n_roles_ok=len(ok),leg_rhythmic=shared,
                leg_peak=float(np.mean(peaks)) if ok else None,
                e1_anchored=bool(res.get('E1',{}).get('rhythmic')))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S318',mode='pure offline detector certification; zero dynamic calls',
      counters=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      detector_v8=dict(dt='explicit per arm: tier 1 ms, isolated 0.5 ms; no default',
        burst_split='ISI > max(3 x median, 5 ms)',gates='nb>=10, spb<=12, rate>=5 Hz',
        envelope='burst starts weighted by burst size, %d ms box smooth, detrend, peak interior of [%g,%g] Hz'%(SMOOTH_MS,FLO,FHI),
        conc='power within peak +/-2 bins over 0.3-50 Hz total, conc_min=%g'%CONC_MIN,
        null='rate-matched random burst placement, %d draws, seed %d, p<%g'%(NULLS,SEED,P_MAX),
        self_discharge='reject only if spb<=1.5 and peak/rate in [0.8,1.2]',
        leg_rule='>=2/3 roles pass all cell criteria and peaks within 20 percent'),
      branches=dict(e1='V8 certified and P3 >=3/6 legs with >=2 in 6.58-26.92 Hz and no 20-40 Hz external envelope component',
        e2='certified but P3 <=2/6',e3='external envelope has a 20-40 Hz component at the output peak',e4='any V8 self-test fails'),
      freeze='S318_PREREG_FROZEN must contain sha256(PREREG.json), sha256(analyze_s318.py) and the UTC time, written before any read')
    dump(o/'PREREG.json',pre)
    (o/'S318_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s318.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(__file__),datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')))
    rng=np.random.default_rng(SEED)
    # ---- self-tests
    def burst(per,nspk,dur=3000.0,cv=0.0,dt=1.0,seed=1):
        n=int(dur/dt);y=np.zeros(n);t=per*0.5;r=np.random.default_rng(seed)
        while t<dur:
            s=int(t/dt)
            for k in range(nspk):
                if 0<=s+k<n: y[s+k]=1
            t+=per*(1+cv*r.standard_normal() if cv else 1.0)
        return y
    def reg(hz,dur=3000.0,dt=1.0):
        n=int(dur/dt);y=np.zeros(n);y[::max(int(round(1000.0/hz/dt)),1)]=1;return y
    T={}
    T['a']=v8(reg(150.0),1.0,rng);T['b']=v8((np.random.default_rng(3).random(3000)<0.02).astype(float),1.0,rng)
    T['c']=v8(burst(1000/21.7,6),1.0,rng);T['d']=v8(burst(1000/21.7,6,cv=0.20),1.0,rng)
    T['d35']=v8(burst(1000/21.7,6,cv=0.35),1.0,rng)
    T['e']=v8(burst(1000/39.0,4),1.0,rng);T['f']=v8(reg(95.0),1.0,rng)
    iso={}
    for nm,truth in (('cap8_full_0.0165',6.58),('cap8_full_0.0253',15.39),('cap8_zero_weights_0.0165',None)):
        z=np.load(S313/f'BSPK_{nm}.npz')['spk']
        e=v8(z[:,0].astype(float),0.5,rng);i=v8(z[:,2].astype(float),0.5,rng)
        iso[nm]=dict(truth=truth,e1=e,i1=i)
    g0=np.load(S316/'SPK_G0_gate.npz');g0m=np.unpackbits(g0['packed'],axis=1)[:,:int(g0['shape'][1])].astype(bool)
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    cols={lg:{r:[loc[int(c)] for c in mod[lg][r]] for r in ROLES} for lg in LEGS}
    g0legs={lg:leg_level({r:g0m[:,cols[lg][r]].any(axis=1).astype(float) for r in ROLES},1.0,rng) for lg in LEGS}
    T['g']=iso;T['h_g0_legs']={lg:dict(leg_rhythmic=g0legs[lg]['leg_rhythmic'],peaks={r:(g0legs[lg]['roles'][r].get('peak_hz')) for r in ROLES}) for lg in LEGS}
    st=dict(details=T,
      pass_a=bool(not T['a']['rhythmic']),pass_b=bool(not T['b']['rhythmic']),
      pass_c=bool(T['c']['rhythmic'] and abs(T['c']['peak_hz']-21.7)<1.0),
      pass_d=bool(T['d']['rhythmic']),
      pass_e=bool(T['e']['rhythmic'] and abs(T['e']['peak_hz']-39.0)<2.0),
      pass_f=bool(not T['f']['rhythmic']),
      pass_g=bool(iso['cap8_full_0.0165']['e1']['rhythmic'] and abs(iso['cap8_full_0.0165']['e1']['peak_hz']-6.58)<1.0
                  and iso['cap8_full_0.0253']['e1']['rhythmic'] and abs(iso['cap8_full_0.0253']['e1']['peak_hz']-15.39)<1.0
                  and not iso['cap8_zero_weights_0.0165']['e1']['rhythmic']),
      pass_h=bool(not any(g0legs[lg]['leg_rhythmic'] for lg in LEGS)),
      d35_note='cv 0.35 is reported but does not participate in certification (detection floor)')
    dump(o/'DETECTOR_V8.json',pre['detector_v8']);dump(o/'DETECTOR_SELFTEST_V8.json',st)
    sens={}
    for cvv in (0.10,0.15,0.20,0.25,0.30,0.35):
        r=v8(burst(1000/21.7,6,cv=cvv),1.0,rng);sens[str(cvv)]=dict(p=r.get('p_null'),conc=r.get('conc'),rhythmic=r['rhythmic'],peak=r.get('peak_hz'))
    dump(o/'SENSITIVITY_V8.json',dict(sweep=sens,note='below this contrast the detector cannot judge; not-detectable is not absent'))
    cert_ok=all(st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_f','pass_g','pass_h'))
    if not cert_ok:
        dump(o/'STOP.json',dict(reason='V8 self-test failed',failed=[k for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_f','pass_g','pass_h') if not st[k]]))
    # ---- rejudge S316 arms
    rj={}
    for f in sorted(S316.glob('SPK_*.npz')):
        pk=np.load(f);sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        per={lg:leg_level({r:sm[:,cols[lg][r]].any(axis=1).astype(float) for r in ROLES},1.0,rng) for lg in LEGS}
        rj[f.stem[4:]]=dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['leg_rhythmic']),
                            n_e1=sum(1 for lg in LEGS if per[lg]['e1_anchored']))
    dump(o/'REJUDGE_V8.json',dict(s316=rj))
    # ---- history rejudge
    hist={}
    for tag,pd_ in (('S312',S312),('S314',S314),('S315',S315)):
        for f in sorted(pd_.glob('SPK_*.npz')):
            z=np.load(f)
            if 'motif' in z.files:
                m=z['motif'];M18=[]
                for lg in LEGS:
                    for r in ROLES:
                        for c in mod[lg][r]: M18.append(loc[int(c)])
                ms=sorted(M18);pos={g:ms.index(g) for g in ms}
                per={lg:leg_level({r:m[:,pos[[loc[int(c)] for c in mod[lg][r]][0]]].astype(float) for r in ROLES},1.0,rng) for lg in LEGS}
            elif 'packed' in z.files:
                sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
                per={lg:leg_level({r:sm[:,cols[lg][r]].any(axis=1).astype(float) for r in ROLES},1.0,rng) for lg in LEGS}
            elif 'spk' in z.files:
                e=v8(z['spk'][:,0].astype(float),0.5,rng);per=None
                hist[f'{tag}:{f.stem}']=dict(n_rh=1 if e['rhythmic'] else 0,peaks={'E1':round(e.get('peak_hz') or 0,2)},dt_ms=0.5);continue
            else:
                hist[f'{tag}:{f.stem}']=dict(n_rh=0,note='no recognised array');continue
            hist[f'{tag}:{f.stem}']=dict(n_rh=sum(1 for lg in LEGS if per[lg]['leg_rhythmic']),
              peaks={lg:(None if per[lg]['leg_peak'] is None else round(per[lg]['leg_peak'],2)) for lg in LEGS})
    for f in sorted(S313.glob('BSPK_*.npz')):
        z=np.load(f)['spk'];e=v8(z[:,0].astype(float),0.5,rng)
        hist[f'S313:{f.stem}']=dict(n_rh=1 if e['rhythmic'] else 0,peaks={'E1':round(e.get('peak_hz') or 0,2)},dt_ms=0.5)
    dump(o/'HISTORY_REJUDGE_V8.json',hist)
    # ---- external envelope control
    def split_env(smatch,We,Wi):
        M=set(M18);ext=np.zeros((smatch.shape[0],18));inn=np.zeros_like(ext)
        for jj,j in enumerate(ms): 
            for W,sg in ((We,1.0),(Wi,-1.0)):
                s,e=W['indptr'][j],W['indptr'][j+1];src=W['indices'][s:e];dat=np.asarray(W['data'][s:e],float)
                msk=np.array([int(x) in M for x in src])
                if (~msk).any(): ext[:,jj]+=sg*(smatch[:,src[~msk]].astype(float)@dat[~msk])
                if msk.any(): inn[:,jj]+=sg*(smatch[:,src[msk]].astype(float)@dat[msk])
        return ext,inn
    We=np.load(S303/'EDGES_w_5e-4_We.npz');Wi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    M18=[]
    for lg in LEGS:
        for r in ROLES:
            for c in mod[lg][r]: M18.append(loc[int(c)])
    ms=sorted(M18)
    base=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    ext0,inn0=split_env(base,We,Wi)
    def envq(ext):
        vals=[]
        for jj in range(ext.shape[1]):
            st,cn=bursts_of(np.abs(ext[:,jj])>0,1.0)
            sm=np.convolve(ext[:,jj],np.ones(20)/20,mode='same');sm=sm-np.polyval(np.polyfit(np.arange(len(sm)),sm,1),np.arange(len(sm)))
            sp=spec_env(sm,1.0)
            vals.append(dict(mean=float(np.mean(ext[:,jj])),p10=float(np.percentile(ext[:,jj],10)),
              p50=float(np.percentile(ext[:,jj],50)),p90=float(np.percentile(ext[:,jj],90)),
              peak=(sp['peak_hz'] if sp else None),conc=(sp['conc'] if sp else None)))
        return vals
    labels=[f'{lg}_{r}' for lg in LEGS for r in ROLES]
    order=sorted(M18)
    lab_of={g:labels[[i for i in range(18) if False][:1][0]] if False else None for g in order}
    # label rows by the cell whose global index that sorted position holds
    gi2name={loc[int(mod[lg][r][0])]:f'{lg}_{r}' for lg in LEGS for r in ROLES}
    lab_sorted=[gi2name[g] for g in order]
    extq=envq(ext0);inq=envq(inn0)
    ctrl=dict(external={lab:q for lab,q in zip(lab_sorted,extq)},internal={lab:q for lab,q in zip(lab_sorted,inq)},
      external_by_position=extq,internal_by_position=inq,label_order=lab_sorted,
      ext_q_all=dict(p10=float(np.percentile(ext0,10)),p50=float(np.percentile(ext0,50)),p90=float(np.percentile(ext0,90))),
      internal_q_all=dict(p10=float(np.percentile(inn0,10)),p50=float(np.percentile(inn0,50)),p90=float(np.percentile(inn0,90))),
      note='per-cell constant scaling does not change the input spectrum, which holds only if the external source population firing is not significantly reshaped by motif feedback')
    band_conc=[0.0 if e['peak'] is None or not (20<=e['peak']<=40) else e['conc'] for e in extq]
    ctrl['has_20_40_component']=bool(any((e['peak'] is not None and 20<=e['peak']<=40 and (e['conc'] or 0)>=CONC_MIN) for e in extq))
    ctrl['per_cell']=ctrl['external']
    dump(o/'DRIVE_ENVELOPE_CONTROL.json',ctrl)
    # ---- cross-leg
    def env(x,w=20): return np.convolve(np.asarray(x,float),np.ones(w)/w,mode='same')
    def cross(sm):
        E={lg:env(sm[:,cols[lg]['E1']].any(axis=1).astype(float)) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                obs=float(np.corrcoef(E[a],E[b])[0,1])
                null=[]
                for k in range(1,11):
                    null.append(float(np.corrcoef(np.roll(E[a],k*50),E[b])[0,1]))
                pairs[f'{a}|{b}']=dict(r=obs,null_max=float(np.max(null)))
        return pairs
    cx={}
    for arm in ('P3_source_faithful','P5_role_coupled','P6_ablation_no_I1E1','P7_ablation_zero_motif'):
        pk=np.load(S316/f'SPK_{arm}.npz');sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        cx[arm]=cross(sm)
    dump(o/'CROSSLEG_V8.json',cx)
    # ---- branch
    p3=rj['P3_source_faithful'];p3band=[lg for lg in LEGS if p3['per_leg'][lg]['leg_peak'] and 6.58<=p3['per_leg'][lg]['leg_peak']<=26.92]
    if not cert_ok: branch='e4'
    elif p3['n_rh']<=2: branch='e2'
    elif ctrl['has_20_40_component']: branch='e3'
    elif p3['n_rh']>=3 and len(p3band)>=2: branch='e1'
    else: branch='e2'
    dump(o/'OUTCOMES_v8.json',dict(round='S318',certified=bool(cert_ok),branch=branch,
      arms={k:dict(n_rh=v['n_rh'],n_e1=v['n_e1'],peaks={lg:(None if v['per_leg'][lg]['leg_peak'] is None else round(v['per_leg'][lg]['leg_peak'],2)) for lg in LEGS}) for k,v in rj.items()},
      p3_in_band=len(p3band),external_20_40=ctrl['has_20_40_component'],
      counts=pre['counters'],
      unresolved=['rf E2 external net drive is negative','rm E2 requires an out-of-family scale','rf does not pass in either P3 or P5',
                  'ConductanceParams remains illustrative/uncalibrated','the 21-39 Hz vs 6.58-26.92 Hz difference is a drive-shape difference'],
      one_line=('e1' if branch=='e1' else branch)))
    print(json.dumps(dict(cert=cert_ok,passes={k:st[k] for k in st if k.startswith('pass')},
      branch=branch,p3=p3['n_rh'],p5=rj['P5_role_coupled']['n_rh'],p3band=len(p3band),
      iso={k:(v['e1'].get('peak_hz'),v['e1']['rhythmic']) for k,v in iso.items()}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
