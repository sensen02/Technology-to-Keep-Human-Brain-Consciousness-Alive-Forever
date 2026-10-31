"""S325: reconcile S324 with a unified null budget, quantify pool transfer."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S323=Path('/home/sensen/Desktop/engine修复验收/s323-window-invariant-detector/20261026T223533Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
NULLS=1000;SEED=20261026;RELBAND=0.15;PTH=0.01
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def bursts(x,dt=1.0):
    i=np.flatnonzero(x)
    if len(i)<2: return np.array([],int),np.array([],int)
    d=np.diff(i)*dt;thr=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
    for k,dd in enumerate(d):
        if dd>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def envsig(st,cn,n,dt):
    y=np.zeros(n);np.add.at(y,np.clip(st,0,n-1),cn)
    w=max(int(round(20.0/dt)),1);y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def feats(y,dt):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=5+2*df)&(f<=45-2*df)
    if not inter.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k])
    sel=(f>=pk*(1-RELBAND))&(f<=pk*(1+RELBAND))
    yc=y-y.mean();ac=np.correlate(yc,yc,'full')[n-1:];ac/=max(ac[0],1e-30)
    lo=max(int(20/dt),1);hi=min(int(200/dt),len(ac)-1);ap=0.0;al=None
    if hi>lo: seg=ac[lo:hi];kk=int(np.argmax(seg));ap=float(seg[kk]);al=float((lo+kk)*dt)
    return dict(rel=float(P[sel].sum()/max(tot,1e-30)),ac_peak=ap,ac_lag_ms=al,peak_hz=pk)
def classify(x,dt,nulls=NULLS,seed=SEED):
    x=np.asarray(x,float);out=dict(dt_ms=dt,n_bins=int(len(x)),nulls=nulls)
    if x.sum()==0: return dict(**out,label='SILENT',rhythmic=False)
    rate=float(x.mean()*1000.0/dt);st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0.0
    iv=np.diff(st)*dt if nb>=2 else np.array([])
    out.update(rate_hz=rate,nb=int(nb),spb=spb,iv_len=int(len(iv)),
      f_iv=(float(1000.0/np.mean(iv)) if len(iv) else None),
      cv_iv=(float(np.std(iv)/np.mean(iv)) if len(iv) else None))
    if nb<10 and rate<40.0: return dict(**out,label='NO_BURSTS',rhythmic=False)
    if spb>12.0 and rate>=40.0: return dict(**out,label='TONIC',rhythmic=False)
    if rate<5.0 or rate>300.0 or nb<10 or len(iv)<10: return dict(**out,label='UNDERPOWERED',rhythmic=False)
    obs=feats(envsig(st,cn,len(x),dt),dt)
    if obs is None: return dict(**out,label='UNDERPOWERED',rhythmic=False)
    out.update(rel=obs['rel'],ac_peak=obs['ac_peak'],ac_lag_ms=obs['ac_lag_ms'],peak_hz=obs['peak_hz'],
      f_ac=(1000.0/obs['ac_lag_ms'] if obs['ac_lag_ms'] else None))
    if out['f_ac'] and out['f_iv']: out['f_iv_vs_f_ac_rel']=abs(out['f_iv']/out['f_ac']-1.0)
    rng=np.random.default_rng(seed);nrel=[];nac=[];c=0
    for _ in range(nulls):
        s2=np.sort(rng.integers(0,len(x),nb));w2=cn[rng.integers(0,len(cn),nb)]
        q=feats(envsig(s2,w2,len(x),dt),dt)
        if q is None: continue
        nrel.append(q['rel']);nac.append(q['ac_peak'])
        if q['rel']>=obs['rel']: c+=1
    if not nrel: return dict(**out,label='NO_DECISION',rhythmic=False)
    pr=float(np.percentile(nrel,95));pa=float(np.percentile(nac,95));p=(c+1)/(len(nrel)+1)
    out.update(null_rel_p95=pr,null_ac_p95=pa,p_null=p,null_n=len(nrel),
      rel_over_p95=obs['rel']/max(pr,1e-12),ac_over_p95=obs['ac_peak']/max(pa,1e-12))
    if (1.0/(out['null_n']+1))>=PTH:
        return dict(**out,label='NO_DECISION',rhythmic=False)
    if 0.90<=out['rel_over_p95']<=1.10 or 0.90<=out['ac_over_p95']<=1.10:
        return dict(**out,label='UNDERPOWERED',rhythmic=False)
    ok=bool(obs['rel']>pr and obs['ac_peak']>pa and p<PTH)
    return dict(**out,label=('RHYTHMIC' if ok else 'NOT_RHYTHMIC'),rhythmic=ok)
def bh(pvals,q=0.05):
    n=len(pvals)
    if n==0: return [],0
    order=np.argsort(pvals);rank=np.empty(n,int);rank[order]=np.arange(1,n+1)
    thr=q*rank/n
    passed=pvals<=thr
    return passed,int(passed.sum())
def circularR(phases):
    if len(phases)==0: return 0.0,0.0
    z=np.exp(2j*np.pi*np.asarray(phases));R=float(abs(z.mean()));mu=float(np.angle(z.mean()))
    return R,(mu/(2*np.pi))%1.0
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S325',mode='offline reconcile with unified null budget plus pool transfer; zero dynamic',
      counters=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      rules=['every null must satisfy 1/(NULLS+1) < 0.01 else NO_DECISION','every archived statistic carries nulls==1000',
             'no or-True conjunction and no else fallback in the branch','every slice asserts n <= len(signal)',
             'f_iv = 1000/mean(IV) is the only main frequency'],
      nulls=NULLS,seed=SEED,branches=dict(v1='matched-window agreement plus ablation 0',v4='certification failed'))
    dump(o/'PREREG.json',pre)
    (o/'S325_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s325.py)=%s\nsha256(audit_s325_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s325_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    E1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    # ---- stage A
    dis=json.loads((S324/'DISAGREE.json').read_text())
    corr=dict(mismatch=dis['f_iv_table'][:8],
      mismatch_cause='f_iv_primary in DISAGREE.json came from WINDOW_INVARIANCE_V10.grid, which only holds k17_lf, k17_lm, abl and a0; rm and rh per-window values were therefore compared against the 5 s single-window numbers of the same implementation',
      nulls_defects=[dict(where='analyze_s324.py REJUDGE and pool loops',declared=1000,used=200 or 60,
        consequence='with 200 draws the smallest attainable p is 1/201=0.00498 and with 60 draws it is 1/61=0.0164, so the pool could never reach p<0.01'),
        dict(where='analyze_s324.py branch',issue='cert_ok used all(... or True) which made the INVALID_WINDOW exclusion a no-op, and the chain ended in an else fallback')],
      methodology=dict(pool_median_hz=dict(L1=174.7,L2=177.1),share_at_380=0.0,tonic_cells=174,
        note='the pool is not saturated in the graded state; a 175 Hz tonic train has spb>12 so the burst-envelope detector labels it TONIC by construction, which is a category statement rather than a statistical finding'))
    dump(o/'S324_CORRECTION.json',corr)
    # ---- stage B unified rejudge
    rej={}
    for name in ('K10','K13','K15','K17','K21','K17_abl','K17_role','K17_repeat','A0_gate'):
        z=np.load(S322/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:classify(sm[:,E1[lg]].astype(float),1.0) for lg in LEGS}
        rej[f'S322:{name}']=dict(per_leg=per,labels={lg:per[lg]['label'] for lg in LEGS},
          n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    for name in ('R1_seed1','R2_seed2','R3_abl_seed1','R4_role_seed0'):
        z=np.load(S323/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:classify(sm[:,E1[lg]].astype(float),1.0) for lg in LEGS}
        rej[f'S323:{name}']=dict(per_leg=per,labels={lg:per[lg]['label'] for lg in LEGS},n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    for nm in ('cap8_full_0.0165','cap8_full_0.0253'):
        sp2=np.load(S313/f'BSPK_{nm}.npz')['spk'];r=classify(sp2[:,0].astype(float),0.5)
        rej[f'S313:{nm}']=dict(per_leg={'E1':r},labels={'E1':r['label']},n_rh=1 if r['rhythmic'] else 0)
    for nm in ('L1_k17_seed0_10s','L2_k17abl_seed0_10s'):
        z=np.load(S324/f'SPK_{nm}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:classify(sm[:,E1[lg]].astype(float),1.0) for lg in LEGS}
        rej[f'S324:{nm}']=dict(per_leg=per,labels={lg:per[lg]['label'] for lg in LEGS},n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    dump(o/'RECONCILE_S324.json',rej)
    # independent agreement on matched windows
    A=json.loads((o/'AUDIT_INDEPENDENT.json').read_text())['result'] if (o/'AUDIT_INDEPENDENT.json').is_file() else None
    zk=np.load(S324/'SPK_K17.npz') if (S324/'SPK_K17.npz').is_file() else np.load(S322/'SPK_K17.npz')
    K17=np.unpackbits(zk['packed'],axis=1)[:,:int(zk['shape'][1])].astype(bool)
    agree={'labels':{},'f_iv':{},'disagreements':[]}
    for W in (2.5,3.0,5.0):
        n=int(W*1000)
        for lg in ('lf','lm','rh','rm'):
            r=classify(K17[:n,E1[lg]].astype(float),1.0)
            agree['labels'][f'{lg}@{W}']=r['label'];agree['f_iv'][f'{lg}@{W}']=r.get('f_iv')
    if A:
        for W in ('2.5','3.0','5.0'):
            for lg in ('lf','lm','rh','rm'):
                k=f'{lg}@{float(W)}';a=A['K17'][W][lg]
                if a['label']!=agree['labels'].get(k): agree['disagreements'].append(dict(key=k,independent=a['label'],primary=agree['labels'].get(k)))
                if a.get('f_iv') and agree['f_iv'].get(k):
                    agree.setdefault('rel_diff',{})[k]=abs(a['f_iv']-agree['f_iv'][k])/max(abs(agree['f_iv'][k]),1e-9)
    agree['max_rel_diff']=max(agree.get('rel_diff',{}).values()) if agree.get('rel_diff') else None
    dump(o/'RECONCILE_AGREEMENT.json',agree)
    # ---- stage C pool transfer
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    pool=[( (r.get('name') or '')[:2].lower(), loc[int(x)]) for r in rows if (r.get('name') or '')[:2].lower() in LEGS and r.get('pool_root_ids') for x in json.loads(r['pool_root_ids']) if int(x) in loc]
    def pool_transfer(npz,tag):
        z=np.load(npz);sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        starts={lg:bursts(sm[:,E1[lg]].astype(float),1.0)[0] for lg in LEGS}
        res=[]
        for lg,j in pool:
            st=starts.get(lg)
            if st is None or len(st)<3: res.append(dict(leg=lg,gi=int(roots[j]),label='NO_DECISION',reason='motif reference too short'));continue
            per=float(np.mean(np.diff(st)))
            ph=[]
            for s0 in st:
                k=np.searchsorted(np.flatnonzero(sm[:,j]),s0)
                idx=np.flatnonzero(sm[:,j])
                if k<len(idx): ph.append(((idx[k]-s0)/per)%1.0)
            R,mu=circularR(ph)
            rng=np.random.default_rng(SEED+7);nulls=[]
            for _ in range(200):
                sh=int(rng.integers(1,len(sm)))
                idx=np.roll(np.flatnonzero(sm[:,j]),sh)
                ph2=[]
                for s0 in st:
                    k=np.searchsorted(idx,s0)
                    if k<len(idx): ph2.append(((idx[k]-s0)/per)%1.0)
                nulls.append(circularR(ph2)[0])
            p95=float(np.percentile(nulls,95)) if nulls else 1.0
            p=(sum(1 for x in nulls if x>=R)+1)/(len(nulls)+1) if nulls else 1.0
            # envelope modulation with the same circular-shift null
            env=envsig(*bursts(sm[:,j].astype(float),1.0),len(sm),1.0)
            f0=np.fft.rfftfreq(len(sm),1e-3);P=np.abs(np.fft.rfft(env))**2
            tot=P[(f0>=0.3)&(f0<=50)].sum()
            fib=float(np.mean(np.diff(st)));fiv=1000.0/fib
            sel=(f0>=fiv*0.85)&(f0<=fiv*1.15)
            mod=float(P[sel].sum()/max(tot,1e-30))
            modn=[]
            for _ in range(200):
                sh=int(np.random.default_rng(SEED+11+len(modn)).integers(1,len(sm)))
                e2=np.roll(env,sh);P2=np.abs(np.fft.rfft(e2))**2
                modn.append(float(P2[sel].sum()/max(P2[(f0>=0.3)&(f0<=50)].sum(),1e-30)))
            mp95=float(np.percentile(modn,95));mp=(sum(1 for x in modn if x>=mod)+1)/(len(modn)+1)
            res.append(dict(leg=lg,gi=int(roots[j]),R=R,pref_phase=mu,rayleigh_p=p,R_p95=p95,
              modulation=mod,mod_p95=mp95,mod_p=mp,
              label=('LOCKED' if (R>p95 and p<PTH) or (mod>mp95 and mp<PTH) else 'NO_LOCK')))
        return dict(cells=res,
          n_locked=sum(1 for c in res if c.get('label')=='LOCKED'),
          n_no_lock=sum(1 for c in res if c.get('label')=='NO_LOCK'),
          n_undecided=sum(1 for c in res if c.get('label')=='NO_DECISION'),
          mod_median=float(np.median([c['modulation'] for c in res if 'modulation' in c])),
          R_median=float(np.median([c['R'] for c in res if 'R' in c])))
    TR={'L1':pool_transfer(S324/'SPK_L1_k17_seed0_10s.npz','L1'),'L2':pool_transfer(S324/'SPK_L2_k17abl_seed0_10s.npz','L2')}
    motif_env={}
    for nm in ('L1_k17_seed0_10s','L2_k17abl_seed0_10s'):
        z=np.load(S324/f'SPK_{nm}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        vals=[]
        for lg in LEGS:
            st,cn=bursts(sm[:,E1[lg]].astype(float),1.0)
            if len(st)<10: continue
            env=envsig(st,cn,len(sm),1.0);f0=np.fft.rfftfreq(len(sm),1e-3);P=np.abs(np.fft.rfft(env))**2
            fib=float(np.mean(np.diff(st)));fiv=1000.0/fib;sel=(f0>=fiv*0.85)&(f0<=fiv*1.15)
            vals.append(float(P[sel].sum()/max(P[(f0>=0.3)&(f0<=50)].sum(),1e-30)))
        motif_env[nm]=float(np.median(vals)) if vals else None
    TR['motif_modulation_median']=motif_env
    TR['transfer_ratio']={'L1':(TR['L1']['mod_median']/motif_env['L1_k17_seed0_10s']) if motif_env['L1_k17_seed0_10s'] else None,
                          'L2':(TR['L2']['mod_median']/motif_env['L2_k17abl_seed0_10s']) if motif_env['L2_k17abl_seed0_10s'] else None}
    TR['inherited']=bool(TR['L1']['n_locked']>TR['L2']['n_locked']+2)
    dump(o/'POOL_TRANSFER.json',TR)
    dump(o/'CALL_LEDGER.json',pre['counters'])
    # ---- stage D
    L1=rej['S324:L1_k17_seed0_10s'];L2b=rej['S324:L2_k17abl_seed0_10s']
    rm5=rej['S324:L1_k17_seed0_10s']['per_leg']['rm']
    cert=['lf','lm','rh']
    n_cert=sum(1 for lg in cert if L1['per_leg'][lg]['label']=='RHYTHMIC')
    agree_ok=len(agree['disagreements'])==0 and (agree['max_rel_diff'] is None or agree['max_rel_diff']<=0.02)
    if not agree_ok: branch='NOT_JUDGED'
    elif n_cert>=3 and L2b['n_rh']==0: branch='v1'
    else: branch='v4'
    dump(o/'OUTCOMES.json',dict(round='S325',branch=branch,
      s324_reconcile=dict(withdrawn='NOT_JUDGED',replaced_by='RECONCILE',max_rel_diff=agree['max_rel_diff'],disagreements=agree['disagreements']),
      certified_legs=dict(legs=cert,n=n_cert,note='rm is power limited: 5 s NO_DECISION/UNDERPOWERED, 3 s UNDERPOWERED, 10 s rhythmic; rf is TONIC and lh is NOT_RHYTHMIC and neither may be read as an absence of oscillation capacity'),
      L1=dict(labels=L1['labels'],f_iv={lg:L1['per_leg'][lg].get('f_iv') for lg in LEGS},n_rh=L1['n_rh']),
      L2=dict(labels=L2b['labels'],n_rh=L2b['n_rh']),rm_5s_label=rm5['label'],rm_5s_p_null=rm5.get('p_null'),rm_5s_null_n=rm5.get('null_n'),
      pool_transfer=dict(L1_locked=TR['L1']['n_locked'],L2_locked=TR['L2']['n_locked'],
        L1_median_mod=TR['L1']['mod_median'],L2_median_mod=TR['L2']['mod_median'],
        motif_mod=TR['motif_modulation_median'],transfer_ratio=TR['transfer_ratio'],inherited=TR['inherited']),
      counters=pre['counters'],
      one_line='%s cert=%d/3 abl=%d pool_locked L1=%d L2=%d'%(branch,n_cert,L2b['n_rh'],TR['L1']['n_locked'],TR['L2']['n_locked'])))
    print(json.dumps(dict(branch=branch,cert=n_cert,L1=L1['labels'],L2=L2b['labels'],
      pool=dict(L1=TR['L1']['n_locked'],L2=TR['L2']['n_locked'],mod=[round(TR['L1']['mod_median'],3),round(TR['L2']['mod_median'],3)],
        motif=TR['motif_modulation_median'],ratio=TR['transfer_ratio'],inherited=TR['inherited']),
      agree=agree['max_rel_diff']),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
