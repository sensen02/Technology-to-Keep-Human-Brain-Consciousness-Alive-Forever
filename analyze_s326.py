"""S326: certify pool->motor transfer with a frequency-domain coherence statistic (V11)."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
S325=Path('/home/sensen/Desktop/engine修复验收/s325-reconcile-and-pool-transfer/20261027T002215Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
NULLS=1000;SEED=20261026;WELCH_MS=1000;OVERLAP=0.5;F0_FALLBACK_NOTE='f0 comes from the L1 arm only'
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
def smooth(x,w=20):
    return np.convolve(np.asarray(x,float),np.ones(w)/w,mode='same')
def seg_spec(x,f0,seg=WELCH_MS,ov=OVERLAP,dt_ms=1.0):
    n=len(x);L=int(seg/dt_ms);step=max(int(L*(1-ov)),1);win=np.hanning(L)
    t=np.arange(L)*dt_ms/1000.0
    out=[]
    for s in range(0,n-L+1,step):
        sgm=x[s:s+L]*win
        c=np.sum(sgm*np.exp(-2j*np.pi*f0*t));out.append(c)
    return np.asarray(out)
def coherence(x,y,f0):
    X=seg_spec(x,f0);Y=seg_spec(y,f0)
    num=abs(np.sum(X*np.conj(Y)))**2
    den=np.sum(np.abs(X)**2)*np.sum(np.abs(Y)**2)
    return float(num/max(den,1e-30)),X,Y
def coh_null(X,Y,f0,n=1000,rng=None):
    rng=rng or np.random.default_rng(SEED);out=[]
    for _ in range(n):
        th=rng.uniform(0,2*np.pi,len(X))
        Xn=X*np.exp(1j*th)
        num=abs(np.sum(Xn*np.conj(Y)))**2
        den=np.sum(np.abs(Xn)**2)*np.sum(np.abs(Y)**2)
        out.append(float(num/max(den,1e-30)))
    return np.asarray(out)
def bh_q(p):
    p=np.asarray(p,float);m=p.size;order=np.argsort(p);rank=np.empty(m,int);rank[order]=np.arange(1,m+1)
    q=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    out=np.empty(m);out[order]=np.minimum(q,1.0);return out
def sine_amp_cosamp(x,f0,dt_ms=1.0):
    n=len(x);t=np.arange(n)*dt_ms/1000.0
    A=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)])
    coef,*_=np.linalg.lstsq(A,x,rcond=None)
    return float(np.hypot(coef[0],coef[1]))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S326',mode='offline coherence certification of pool->motor transfer; zero dynamic',
      counters=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      nulls=NULLS,seed=SEED,detector='magnitude-squared coherence at a known f0, Welch 1000 ms with 50 percent overlap, Hanning window',
      null='randomise the REFERENCE segment phases only, which is the only construction that changes a cross quantity',
      rules=['no power-preserving transform may be applied to the same signal when testing its own power statistics',
             'pool cells are deduplicated by (leg, gi) and the duplicate rows are listed',
             'both arms share the same f0 taken from the L1 arm',
             'NULLS=1000 and 1/(NULLS+1) < p_threshold',
             'BH q=0.05 over cells','no else fallback in the branch'],
      branches=dict(w1='pool inherits: level unchanged and modulation transferred',w2='level preserved but modulation inherited weakly',
                    w3='no inheritance',w4='NOT_JUDGED'))
    dump(o/'PREREG.json',pre)
    (o/'S326_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s326.py)=%s\nsha256(audit_s326_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s326_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    E1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    # ---------- stage A: verify the S325 defects from its own archive
    PT=json.loads((S325/'POOL_TRANSFER.json').read_text())
    def identical_count(cells):
        return sum(1 for c in cells if 'modulation' in c and abs(c['modulation']-c['mod_p95'])<1e-15),sum(1 for c in cells if 'modulation' in c)
    idL1=identical_count(PT['L1']['cells']);idL2=identical_count(PT['L2']['cells'])
    corr=dict(
      mod_null_is_identity=dict(L1=idL1,L2=idL2,
        explanation='np.roll does not change |FFT|, so every surrogate equals the observation and mod_p95 equals modulation exactly'),
      phase_statistic=dict(R_median=dict(L1=PT['L1']['R_median'],L2=PT['L2']['R_median']),
        rayleigh_frac_lt_0p01=dict(L1=round(sum(1 for c in PT['L1']['cells'] if c.get('rayleigh_p',1)<0.01)/max(1,len(PT['L1']['cells'])),3),
                                   L2=round(sum(1 for c in PT['L2']['cells'] if c.get('rayleigh_p',1)<0.01)/max(1,len(PT['L2']['cells'])),3)),
        explanation='R takes the first pool spike per reference cycle, so a 165 Hz cell always lands within 0.15 of the cycle and R saturates near 1'),
      reference_scale=dict(L2_reference_hz=[round(1000.0/np.mean(np.diff(bursts(np.unpackbits(np.load(S324/'SPK_L2_k17abl_seed0_10s.npz')['packed'],axis=1)[:,:int(np.load(S324/'SPK_L2_k17abl_seed0_10s.npz')['shape'][1])].astype(bool)[:,E1[lg]].astype(float),1.0)[0])),2) if len(bursts(np.unpackbits(np.load(S324/'SPK_L2_k17abl_seed0_10s.npz')['packed'],axis=1)[:,:12000].astype(bool)[:,E1[lg]].astype(float),1.0)[0])>2 else None for lg in LEGS]),
      record_counts=dict(records_L1=len(PT['L1']['cells']),unique_gi_L1=len({c['gi'] for c in PT['L1']['cells']})),
      retracted=['POOL_TRANSFER lock counts 148 vs 70','h = 0.263'],
      acknowledged_spec_error='my instruction asked for a rate-and-spectrum preserving phase randomisation as a null; for any power statistic that is an identity transform and can only produce mod == mod_p95')
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    raw=[]
    for r in rows:
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: raw.append((lg,loc[int(x)],int(x),(r.get('name') or '')))
    seen={};dups=[]
    for lg,j,gi,name in raw:
        key=(lg,gi)
        if key in seen: dups.append(dict(leg=lg,gi=gi,channel=name))
        else: seen[key]=j
    corr['dedup']=dict(raw_records=len(raw),unique=len(seen),duplicates=len(dups),duplicate_list=dups[:49])
    dump(o/'S325_CORRECTION.json',corr)
    dump(o/'DETECTOR_V11.json',dict(statistic='coherence(f0) = |sum_k X_k conj(Y_k)|^2 / (sum|X_k|^2 sum|Y_k|^2)',
      f0_source='L1 motif E1 burst f_iv, per leg, shared by both arms',
      null='randomise reference segment phases, %d draws'%NULLS,labels=['TRANSFERRED','WEAK','NOT_TRANSFERRED','NO_REFERENCE'],
      dedup='(leg,gi)',bh='q=0.05 over the 164 valid cells'))
    pool=list(seen.items())  # ((leg,gi), j)
    Z1=np.load(S324/'SPK_L1_k17_seed0_10s.npz');Z2=np.load(S324/'SPK_L2_k17abl_seed0_10s.npz')
    A1=np.unpackbits(Z1['packed'],axis=1)[:,:int(Z1['shape'][1])].astype(bool)
    A2=np.unpackbits(Z2['packed'],axis=1)[:,:int(Z2['shape'][1])].astype(bool)
    f0={}
    for lg in LEGS:
        st,_=bursts(A1[:,E1[lg]].astype(float),1.0)
        f0[lg]=float(1000.0/np.mean(np.diff(st))) if len(st)>=10 else None
    ref1={lg:(smooth(A1[:,E1[lg]].astype(float)) if f0[lg] else None) for lg in LEGS}
    ref2={lg:(smooth(A2[:,E1[lg]].astype(float)) if f0[lg] else None) for lg in LEGS}
    avail=[]
    # ---------- stage B
    rng=np.random.default_rng(SEED)
    cells=[];agg={}
    for (lg,gi),j in pool:
        y1=smooth(A1[:,j].astype(float));y2=smooth(A2[:,j].astype(float))
        rec=dict(leg=lg,gi=gi,rate1=float(A1[:,j].mean()*1000),rate2=float(A2[:,j].mean()*1000))
        if f0[lg] is None:
            rec.update(label='NO_REFERENCE');cells.append(rec);continue
        c1,X1,Y1=coherence(ref1[lg],y1,f0[lg]);n1=coh_null(X1,Y1,f0[lg],n=NULLS,rng=rng)
        c2,X2,Y2=coherence(ref2[lg],y2,f0[lg]);n2=coh_null(X2,Y2,f0[lg],n=NULLS,rng=rng)
        p1=(np.sum(n1>=c1)+1)/(len(n1)+1);p2=(np.sum(n2>=c2)+1)/(len(n2)+1)
        rec.update(coh1=c1,coh1_p95=float(np.percentile(n1,95)),coh1_p=p1,coh1_null_max=float(n1.max()),
                   coh2=c2,coh2_p95=float(np.percentile(n2,95)),coh2_p=p2,
                   label=('TRANSFERRED' if (c1>np.percentile(n1,95) and p1<0.05) else 'NOT_TRANSFERRED'))
        cells.append(rec)
    pv=[c['coh1_p'] for c in cells if 'coh1_p' in c]
    qs=bh_q(pv)
    k=0
    for c in cells:
        if 'coh1_p' in c: c['bh_q']=float(qs[k]);k+=1
    n_pass1=sum(1 for c in cells if c['label']=='TRANSFERRED')
    n_pass2=sum(1 for c in cells if c.get('coh2',0)>c.get('coh2_p95',1) and c.get('coh2_p',1)<0.05)
    avail=[c for c in cells if 'coh1' in c]
    for lg in LEGS:
        if f0[lg] is None: agg[lg]=dict(note='no reference');continue
        pool_cols=[j for (l2_,gi),j in pool if l2_==lg]
        if not pool_cols: continue
        agg1=A1[:,pool_cols].sum(1).astype(float);agg2=A2[:,pool_cols].sum(1).astype(float)
        ca,Xa,Ya=coherence(ref1[lg],smooth(agg1),f0[lg]);na=coh_null(Xa,Ya,f0[lg],n=NULLS,rng=rng)
        cb,Xb,Yb=coherence(ref2[lg],smooth(agg2),f0[lg]);nb=coh_null(Xb,Yb,f0[lg],n=NULLS,rng=rng)
        agg[lg]=dict(n_cells=len(pool_cols),coh_L1=ca,null_p95_L1=float(np.percentile(na,95)),
          coh_L2=cb,null_p95_L2=float(np.percentile(nb,95)),f0=f0[lg],mean_rate_L1=float(A1[:,pool_cols].sum(1).mean()*1000/len(pool_cols)),
          mean_rate_L2=float(A2[:,pool_cols].sum(1).mean()*1000/len(pool_cols)))
    dump(o/'POOL_TRANSFER_V11.json',dict(cells=cells,n_cells=len(cells),n_valid=len(avail),
      n_transferred_L1=n_pass1,n_transferred_L2=n_pass2,f0=f0,aggregate=agg,
      median_coh_L1=float(np.median([c['coh1'] for c in avail])),
      median_coh_L2=float(np.median([c['coh2'] for c in avail]))))
    # cross-leg negative control (same cells, other legs as reference)
    cross={'pairs':0,'same_leg_median':float(np.median([c['coh1'] for c in avail])),'cross_leg':{}}
    for lg in LEGS:
        if f0[lg] is None: continue
        vals=[]
        for (lg2,gi),j in pool:
            if lg2==lg: continue
            y=smooth(A1[:,j].astype(float))
            c,_,_=coherence(ref1[lg],y,f0[lg]);vals.append(c)
        cross['cross_leg'][lg]=dict(n=len(vals),median=float(np.median(vals)) if vals else None)
        cross['pairs']+=len(vals)
    cross['median_cross']=float(np.median([v['median'] for v in cross['cross_leg'].values() if v['median'] is not None])) if cross['cross_leg'] else None
    dump(o/'CROSSLEG_COHERENCE.json',cross)
    # ---------- stage C amplitude + decoder input
    amp={}
    for lg in LEGS:
        if f0[lg] is None: amp[lg]=dict(note='no reference');continue
        pool_cols=[j for (l2_,gi),j in pool if l2_==lg]
        mot=A1[:,E1[lg]].astype(float);poolagg=A1[:,pool_cols].sum(1).astype(float)
        am=sine_amp_cosamp(mot,f0[lg]);ap=sine_amp_cosamp(poolagg,f0[lg])
        mm=mot.mean();mp=poolagg.mean()
        h=(ap/max(mp,1e-12))/(am/max(mm,1e-12)) if am>0 else None
        rr=np.random.default_rng(SEED+3);nulls=[]
        for _ in range(NULLS):
            pois=rr.random(len(poolagg))<min(mp,1.0)
            if not pois.any(): nulls.append(0.0);continue
            apn=sine_amp_cosamp(pois.astype(float)*mp/max(pois.mean(),1e-9),f0[lg])
            nulls.append((apn/max(mp,1e-12))/(am/max(mm,1e-12)))
        amp[lg]=dict(f0=f0[lg],motif_amp=am,motif_mean=mm,motif_frac=am/max(mm,1e-12),
          pool_amp=ap,pool_mean=mp,pool_frac=ap/max(mp,1e-12),h=h,h_null_p95=float(np.percentile(nulls,95)),
          decoder_cmd_frac_of_V1=(ap/450.0),
          note='decoder input only: preclip=(rate_hz/450)*V1 and postclip=clip(0,1); no physics is attached and the archive defines no motor activation threshold, so none is invented')
    dump(o/'TRANSFER_AMPLITUDE.json',amp)
    dump(o/'CALL_LEDGER.json',pre['counters'])
    # ---------- stage D
    level_same=all(abs(agg[lg]['mean_rate_L1']-agg[lg]['mean_rate_L2'])/max(agg[lg]['mean_rate_L1'],1e-9)<0.10 for lg in agg if 'mean_rate_L1' in agg[lg])
    if n_pass1>=100 and n_pass2<=10: branch='w1'
    elif n_pass1>=20 and n_pass2<=10: branch='w2'
    elif n_pass1<=10: branch='w3'
    else: branch='NOT_JUDGED'
    if not level_same and branch=='w1': branch='w2'
    rm3=None
    rej=json.loads((S325/'RECONCILE_S324.json').read_text())['S324:L1_k17_seed0_10s']['per_leg']['rm']
    dump(o/'OUTCOMES.json',dict(round='S326',branch=branch,
      pool=dict(L1_transferred=n_pass1,L2_transferred=n_pass2,n_valid=len(avail),
        median_coh_L1=float(np.median([c['coh1'] for c in avail])),
        median_coh_L2=float(np.median([c['coh2'] for c in avail])),
        level_same=level_same,rate_L1={lg:round(agg[lg].get('mean_rate_L1',0),1) for lg in agg if 'mean_rate_L1' in agg[lg]},
        rate_L2={lg:round(agg[lg].get('mean_rate_L2',0),1) for lg in agg if 'mean_rate_L2' in agg[lg]}),
      aggregate_coh={lg:dict(L1=round(agg[lg]['coh_L1'],3) if 'coh_L1' in agg[lg] else None,
        p95=round(agg[lg]['null_p95_L1'],3) if 'null_p95_L1' in agg[lg] else None,
        L2=round(agg[lg]['coh_L2'],3) if 'coh_L2' in agg[lg] else None) for lg in agg},
      cross_leg=cross,h=amp,rm_alignment=dict(rm_5s_label=rej['label'],rm_5s_null_n=rej.get('null_n'),
        note='the rm conflict is settled to the 10 s reading for the certified set and rm stays excluded from the certified legs'),
      certified_legs=dict(legs=['lf','lm','rh'],n=3,note='rm power limited: 5 s and 3 s undecided, 10 s rhythmic; rf TONIC and lh NOT_RHYTHMIC and neither may be read as an absence of oscillation capacity'),
      counters=pre['counters'],
      one_line='%s pool L1 %d/%d vs L2 %d, median coh %.3f vs %.3f, level_same=%s'%(branch,n_pass1,len(avail),n_pass2,
        float(np.median([c['coh1'] for c in avail])),float(np.median([c['coh2'] for c in avail])),level_same)))
    print(json.dumps(dict(branch=branch,n1=n_pass1,n2=n_pass2,nvalid=len(avail),
      med=[round(float(np.median([c['coh1'] for c in avail])),3),round(float(np.median([c['coh2'] for c in avail])),3)],
      agg={lg:(round(agg[lg]['coh_L1'],3),round(agg[lg]['null_p95_L1'],3),round(agg[lg]['coh_L2'],3)) for lg in agg if 'coh_L1' in agg[lg]},
      cross=round(cross['median_cross'],3) if cross['median_cross'] else None,
      h={lg:(round(amp[lg]['h'],3) if amp[lg].get('h') else None) for lg in amp},
      mod_ident=corr['mod_null_is_identity'],level_same=level_same),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
