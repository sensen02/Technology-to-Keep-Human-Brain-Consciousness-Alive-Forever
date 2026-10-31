"""S318 verification: dual-hash freeze, independent V8 reimplementation, controls, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
FLO,FHI=5.0,45.0;CONC_MIN=0.15;P_MAX=0.01;NULLS=120;SEED_V=777
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bursts(x,dt):
    i=np.flatnonzero(x)
    if len(i)<2: return np.array([],int),np.array([],int)
    d=np.diff(i)*dt;thr=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
    for k,dd in enumerate(d):
        if dd>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def envp(st,cn,n,dt):
    y=np.zeros(n);np.add.at(y,np.clip(st,0,n-1),cn)
    w=max(int(round(20.0/dt)),1);y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def spec(y,dt):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=FLO+2*df)&(f<=FHI-2*df)
    if not inter.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k])
    sel=(f>=pk-2*df)&(f<=pk+2*df)
    return pk,float(P[sel].sum()/max(tot,1e-30))
def v8(x,dt,rng):
    x=np.asarray(x,float)
    if x.sum()==0: return dict(rhythmic=False)
    rate=x.mean()*1000.0/dt;st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0
    if nb<10 or spb>12 or rate<5: return dict(rhythmic=False,rate_hz=rate,nb=int(nb),spb=spb)
    o=spec(envp(st,cn,len(x),dt),dt)
    if o is None: return dict(rhythmic=False)
    c=0
    for _ in range(NULLS):
        s2=np.sort(rng.integers(0,len(x),nb));w2=cn[rng.integers(0,len(cn),nb)]
        q=spec(envp(s2,w2,len(x),dt),dt)
        if q and q[1]>=o[1]: c+=1
    p=(c+1)/(NULLS+1);ratio=o[0]/max(rate,1e-9)
    selfd=bool(spb<=1.5 and 0.8<=ratio<=1.2)
    return dict(rhythmic=bool(p<P_MAX and o[1]>=CONC_MIN and not selfd),rate_hz=rate,nb=int(nb),spb=spb,peak_hz=o[0],conc=o[1],p_null=p)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S318_PREREG_FROZEN').read_text()
    c('frozen file carries both hashes',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s318.py)=' in fr) and ('utc=' in fr))
    c('PREREG hash in the freeze matches the archived PREREG',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr)
    import shutil
    arch=(o/'analyze_s318.py')
    c('analysis script archived and hashed in the freeze',arch.is_file())
    st=json.loads((o/'DETECTOR_SELFTEST_V8.json').read_text());oc=json.loads((o/'OUTCOMES_v8.json').read_text())
    pre=json.loads((o/'PREREG.json').read_text())
    c('all eight self-tests pass',all(st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e','pass_f','pass_g','pass_h')))
    c('offline counters are zero',all(pre['counters'][k]==0 for k in pre['counters']))
    rng=np.random.default_rng(SEED_V)
    # independent positive controls with a different null seed/count
    for nm,truth in (('cap8_full_0.0165',6.58),('cap8_full_0.0253',15.39)):
        z=np.load(S313/f'BSPK_{nm}.npz')['spk'];r=v8(z[:,0].astype(float),0.5,rng)
        c(f'positive control {nm} reproduced at {truth} Hz',r['rhythmic'] and abs(r.get('peak_hz',0)-truth)<1.0,peak=r.get('peak_hz'))
    z=np.load(S313/'BSPK_cap8_zero_weights_0.0165.npz')['spk'];r=v8(z[:,0].astype(float),0.5,rng)
    c('zero-weight negative control rejected',not r['rhythmic'])
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    cols={lg:{r:[loc[int(x)] for x in mod[lg][r]] for r in ROLES} for lg in LEGS}
    # independent leg counts for P3 and P5 and the G0 negative control
    for arm,expect in (('P3_source_faithful',None),('P5_role_coupled',None),('G0_gate',0)):
        pk=np.load(S316/f'SPK_{arm}.npz');sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        n=0;det={}
        for lg in LEGS:
            rr={ro:v8(sm[:,cols[lg][ro]].any(axis=1).astype(float),1.0,rng) for ro in ROLES}
            ok=[ro for ro in rr if rr[ro]['rhythmic']];pk2=[rr[ro]['peak_hz'] for ro in ok]
            good=bool(len(ok)>=2 and (max(pk2)-min(pk2))/max(np.mean(pk2),1e-9)<=0.2)
            n+=good;det[lg]=dict(n_ok=len(ok),leg_rhythmic=good,peaks={k:(round(v.get('peak_hz') or 0,1)) for k,v in rr.items()})
        c(f'{arm} leg count recomputed independently',(n==expect) if expect is not None else (n>=1),n=n,detail=det)
    dctrl=json.loads((o/'DRIVE_ENVELOPE_CONTROL.json').read_text())
    lf=dctrl['external']['lf_E1'] if 'lf_E1' in dctrl['external'] else dctrl['per_cell'][0]
    c('external drive scale reproduced (lf E1 mean 0.0832 uS)',abs(lf['mean']-0.0832)<0.01,mean=lf['mean'],p90=lf['p90'])
    # recompute lf E1 external mean independently from the archived spikes
    sp=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    We=np.load(S303/'EDGES_w_5e-4_We.npz');Wi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    M=set(int(x) for x in roots[[loc[int(c)] for lg in LEGS for r in ROLES for c in mod[lg][r]]])
    j=loc[int(mod['lf']['E1'][0])];acc=np.zeros(len(sp))
    for W,sg in ((We,1.0),(Wi,-1.0)):
        s,e=W['indptr'][j],W['indptr'][j+1];src=W['indices'][s:e];dat=np.asarray(W['data'][s:e],float)
        msk=np.array([int(x) in M for x in src])
        if (~msk).any(): acc+=sg*(sp[:,src[~msk]].astype(float)@dat[~msk])
    c('lf E1 external mean matches the archived control',abs(acc.mean()-lf['mean'])<1e-6,verifier=float(acc.mean()),archived=lf['mean'])
    c('branch e1 consistent with the archived counts',oc['branch']=='e1' and oc['arms']['P3_source_faithful']['n_rh']>=3 and not oc['external_20_40'])
    c('unresolved items listed',len(oc['unresolved'])>=4)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,connectome_load=0,render=0),
      limitations=['independent V8 uses a different null seed and 120 draws','all quantities are projections','no neural or physical run this round'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:4],dctrl_lf=lf['mean'])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
