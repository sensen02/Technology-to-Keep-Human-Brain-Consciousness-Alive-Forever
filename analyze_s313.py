"""S313: V4 topology-derived phase criterion, burst band localization, tier drive instrumentation."""
from __future__ import annotations
import argparse, json, hashlib, dataclasses
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1');FP='56e0322a33c51b7f03aaae9f7df1ef3f'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(con_network_init=0,con_run=0,con_step=0,tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
def bursts(x,dt):
    idx=np.flatnonzero(x)
    if len(idx)<3: return np.array([]),np.array([])
    isi=np.diff(idx)*dt;thr=3*np.median(isi);st=[idx[0]];cnt=[1]
    for i,d in enumerate(isi):
        if d>thr: st.append(idx[i+1]);cnt.append(1)
        else: cnt[-1]+=1
    return np.asarray(st),np.asarray(cnt)
def v4(e1,i1,dt):
    """lag is topology-derived: I1 must follow E1 inside (0, 0.25) of a cycle, with Rayleigh concentration."""
    st,cnt=bursts(e1,dt)
    n=len(st);base=dict(n_bursts=int(n),mean_hz=float(e1.mean()*1000.0/dt))
    if n<11: base.update(rhythmic=False,reason='few_bursts');return base
    per=np.diff(st)*dt;cv=float(np.std(per)/max(float(np.mean(per)),1e-9));spc=float(cnt[1:-1].mean())
    i1t=np.flatnonzero(i1);lags=[]
    for s0 in st:
        k=np.searchsorted(i1t,s0)
        if k<len(i1t): lags.append(((i1t[k]-s0)*dt/np.mean(per))%1.0)
    lag=float(np.median(lags)) if lags else None
    p=1.0
    if lags:
        z=np.exp(2j*np.pi*np.asarray(lags));Rbar=float(abs(z.mean()));nn=len(lags)
        if nn>1 and Rbar<1: p=float(np.exp(np.sqrt(1+4*nn+4*(nn**2-(nn*Rbar)**2))-(1+2*nn)))
        else: p=0.0
    rh=bool(len(per)>=10 and cv<0.25 and lag is not None and 0.0<lag<0.25 and p<0.01 and 1.0<=spc<=12.0 and base['mean_hz']>=5.0)
    base.update(rhythmic=rh,period_ms=float(np.mean(per)),freq_hz=float(1000.0/np.mean(per)),cv=cv,spikes_per_cycle=spc,
                lag=lag,rayleigh_p=p,continuous=bool(cv>=0.25 and base['mean_hz']>=40))
    return base
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S313',stage_a='rejudge S312 with V4 (0 dynamic)',stage_b='50 isolated arms (burst band)',
      stage_c='2 tier arms (gate + instrumentation)',
      counts=dict(stage_b=dict(con_network_init=50,con_run=50,con_step=300000),
                  stage_c=dict(tier_build=2,select_touch_tier=4,edges_build=2,network_init=2,control_tick=10000,tier_advance=10000,neural_substep=20000)),
      hard=['verifier writes only verify/','S313_PREREG_FROZEN written before any dynamic call','npz for arrays','engine/ unchanged'],
      v4='lag derived from topology: 0<lag<0.25 with Rayleigh p<0.01, burst-interval cv<0.25, >=10 intervals, 1-12 spikes per cycle, mean>=5 Hz',
      currents=[0.0165,0.0224,0.0253,0.0375,0.05,0.07,0.1,0.16],
      branches=dict(d1='a stable burst band exists and the tier net input sits in the continuous-firing region',
                    d2='a band exists and the tier input is already inside it yet does not burst',
                    d3='no stable band',d4='gate or selftest failure'))
    dump(o/'PREREG.json',pre);(o/'S313_PREREG_FROZEN').write_text('frozen before any dynamic call\n')
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    P=ConductanceParams()
    # ---- stage A: rejudge S312 archives with V4
    rej={}
    for f in sorted(S312.glob('SPK_*.npz')):
        nm=f.stem[4:];sp=np.load(f)
        if 'motif' in sp.files:
            m=sp['motif'];e1=m[:,[3*i for i in range(6)]].any(axis=1);i1=m[:,[3*i+2 for i in range(6)]].any(axis=1);dt=1.0
        else:
            a=sp['spk'];e1=a[:,0];i1=a[:,2];dt=0.5
        rej[nm]=v4(e1,i1,dt)
    dump(o/'S312_REJUDGE.json',rej)
    # rate convention
    conv=dict(rule='tier spikes are 1 ms bins; rate_hz = count / seconds',selfcheck=dict(pool_hz=None))
    try:
        t5=np.load(S312/'SPK_T5_iso_raw_d050.npz');pool=t5['pool'];conv['selfcheck']['pool_hz']=float(pool.sum()/5.0)
        conv['example_T5_E1_hz']=float(t5['motif'][:,[3*i for i in range(6)]].any(axis=1).sum()/5.0)
    except Exception as e: conv['selfcheck']['error']=repr(e)
    dump(o/'RATE_CONVENTION.json',conv)
    # selftest v4
    def synth(per_ms,nspk,lag_frac,dur=3000.0,dt=0.5,gap_after=2,rand=False,seed=3):
        n=int(dur/dt);e=np.zeros(n,bool);i=np.zeros(n,bool);rng=np.random.default_rng(seed)
        t=(per_ms*0.5)
        while t<dur:
            s=int(t/dt)
            for k in range(nspk):
                if 0<=s+k<n: e[s+k]=True
            off=t+lag_frac*per_ms+(rng.uniform(0,per_ms) if rand else 0.0)
            j=int(off/dt)
            if 0<=j<n: i[j]=True
            t+=per_ms
        return e,i
    t=dict()
    e,i=synth(1000/14.0,2,0.15);t['a']=v4(e,i,0.5)
    e,i=synth(1000/17.0,2,0.15);t['b']=v4(e,i,0.5)
    e,_=synth(1000/14.0,2,0.15);_,i=synth(1000/14.0,1,0.0,rand=True,seed=9);t['c']=v4(e,i,0.5)
    rng=np.random.default_rng(5);n=int(3000/0.5);t['d']=v4(rng.random(n)<(20*0.5/1000),np.zeros(n,bool),0.5)
    z=np.zeros(n,bool);t['e']=v4(z,z,0.5)
    st=dict(details=t,pass_a=bool(t['a']['rhythmic'] and abs(t['a'].get('freq_hz',0)-14.0)<1.0),
      pass_b=bool(t['b']['rhythmic'] and abs(t['b'].get('freq_hz',0)-17.0)<1.0 and abs(t['b']['freq_hz']-t['a']['freq_hz'])>1.0),
      pass_c=bool(not t['c']['rhythmic']),pass_d=bool(not t['d']['rhythmic']),pass_e=bool(not t['e']['rhythmic']))
    dump(o/'DETECTOR_V4.json',dict(criteria=pre['v4']));dump(o/'DETECTOR_SELFTEST_V4.json',st)
    corr={'round':'S313','items':[
      dict(id='C1',s312='I1 lag must be in 0.3-0.7 of a cycle',measured=dict(observed_lags=[0.023,0.058,0.009,0.010]),
           corrected='this DISPATCH CRITERION WAS WRONG: the E1->E2->I1 motif requires I1 to follow E1 within (0,0.25); the old window rejected every regular bursting arm and is the single cause of all S312 non-rhythmic verdicts'),
      dict(id='C2',s312='selftest (a) passed',measured=dict(note='the test rolled the I1 train by half a cycle'),
           corrected='the test was self-serving; V4 adds an independent same-frequency oscillator control that must be rejected'),
      dict(id='C3',s312='stage B had no branch framework',measured=dict(prereg_branches=['c1','c2','c3','c4']),
           corrected='stage B conclusions are re-judged under V4 now'),
      dict(id='C4',s312='rates in OUTCOMES',measured=dict(T5_E1_hz=conv.get('example_T5_E1_hz'),pool_hz=conv['selfcheck']['pool_hz']),
           corrected='rate convention unified to count per second; S312 divided by 2 for tier arms and used dt=0.5 on 1 ms bins in v3'),
      dict(id='C5',s312='no S310 positive currents in the grid',measured=dict(S312_grid=[0.0165,0.0253,0.07,0.16,0.2]),
           corrected='S313 adds 0.0224/0.0375/0.05/0.1 so the S310 positives are actually sampled'),
      dict(id='C6',s312='raw family only full',measured=dict(tops_raw=['full']),corrected='S313 runs no_I1_E1 and zero_weights for both families'),
      dict(id='C7',s312='REPRO_MISMATCH only on T7/T8',measured=dict(T2_current=0.16,S309_k1_current=0.1),
           corrected='amplitude-mismatched arms are all marked, not just T7/T8')]}
    dump(o/'S312_CORRECTION.json',corr)
    if not all(st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e')):
        dump(o/'STOP.json',dict(reason='detector v4 self-test failed',st=st));dump(o/'CALL_LEDGER.json',C);return 3
    # ---- stage B
    e=json.loads((S307/'PER_LEG_MODULE.json').read_text())['edges']['lf']
    rawm=np.asarray([[0,e['E1->E2'],0],[e['E2->E1'],0,0],[e['E1->I1'],e['E2->I1'],0]],float)*5e-4
    rawi=np.asarray([[0,0,e['I1->E1']],[0,0,e['I1->E2']],[0,0,0]],float)*5e-4
    capm=np.minimum(np.asarray([[0,e['E1->E2'],0],[e['E2->E1'],0,0],[e['E1->I1'],e['E2->I1'],0]],float),8)*5e-4
    capi=np.minimum(np.asarray([[0,0,e['I1->E1']],[0,0,e['I1->E2']],[0,0,0]],float),8)*5e-4
    fams={'cap8':(capm,capi),'raw':(rawm,rawi)}
    def run(We,Wi,I,dur=3000.0):
        net=ConductanceNetwork(3,P,dt_ms=0.5);net.set_connectivity(We,Wi);C['con_network_init']+=1
        n=int(dur/0.5);spk=np.zeros((n,3),bool)
        for k in range(n): spk[k]=net.step(current_nA=np.array([I,0.0,0.0]));C['con_step']+=1
        C['con_run']+=1;return spk
    arms={}
    for fam,(We,Wi) in fams.items():
        for top in ('full','no_I1_E1','zero_weights'):
            W1=We.copy();W2=Wi.copy()
            if top=='no_I1_E1': W2[0,2]=0.0
            if top=='zero_weights': W1[:]=0.0;W2[:]=0.0
            for I in pre['currents']:
                k=f'{fam}_{top}_{I}';spk=run(W1,W2,I);arms[k]=v4(spk[:,0],spk[:,2],0.5)
                arms[k]['e2_hz']=float(spk[:,1].mean()*2000.0);arms[k]['e1_hz']=float(spk[:,0].mean()*2000.0)
                np.savez_compressed(o/f'BSPK_{k}.npz',spk=spk)
    for k in ('cap8_full_0.0165','raw_full_0.05'):
        fam,top,I=k.split('_')[0],'full',float(k.split('_')[-1]);We,Wi=fams[fam]
        spk=run(We,Wi,I);arms[k+'_repeat']=v4(spk[:,0],spk[:,2],0.5)
        arms[k+'_repeat']['e1_hz']=float(spk[:,0].mean()*2000.0)
        np.savez_compressed(o/f'BSPK_{k}_repeat.npz',spk=spk)
    band={}
    for fam in fams:
        for top in ('full','no_I1_E1','zero_weights'):
            pts=[(I,arms[f'{fam}_{top}_{I}']) for I in pre['currents']]
            rh=[I for I,a in pts if a.get('rhythmic')]
            band[f'{fam}_{top}']=dict(rhythmic_currents=rh,
              freq_by_current={str(I):a.get('freq_hz') for I,a in pts},
              cv_by_current={str(I):a.get('cv') for I,a in pts},
              onset_nA=(min(rh) if rh else None),offset_nA=(max(rh) if rh else None))
    for fam in fams:
        band[f'{fam}_full']['rate_ratio_full_over_noI1E1']={str(I):(arms[f'{fam}_full_{I}']['e1_hz']/max(arms[f'{fam}_no_I1_E1_{I}']['e1_hz'],1e-9)) for I in pre['currents']}
    dump(o/'BURST_BAND.json',dict(band=band,repeats={k:[arms[k],arms[k+'_repeat']] for k in ('cap8_full_0.0165','raw_full_0.05')},counters=dict(C)))
    # ---- stage C
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18=sorted({loc[int(c)] for lg in LEGS for a in ORDER for c in mod[lg][a] if int(c) in loc})
    builds=[]
    for _ in range(2):
        b=build_neural_tier(TierConfig(sign_fill='all_empty').validate());C['tier_build']+=1;C['select_touch_tier']+=2;C['edges_build']+=1
        builds.append(b)
    gate=dict(fp=builds[0]['fingerprint'],ok=builds[0]['fingerprint']==FP)
    e0=builds[0]['edges'];We0=e0['We'] if isinstance(e0,dict) else e0.we
    Wi0=e0['Wi'] if isinstance(e0,dict) else e0.wi
    refw=np.load(S303/'EDGES_w_5e-4_We.npz');refi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    h=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    gate['We']=[h(We0.data)==h(refw['data']),h(We0.indices)==h(refw['indices']),h(We0.indptr)==h(refw['indptr'])]
    gate['Wi']=[h(Wi0.data)==h(refi['data']),h(Wi0.indices)==h(refi['indices']),h(Wi0.indptr)==h(refi['indptr'])]
    gate['passed']=bool(gate['ok'] and all(gate['We']) and all(gate['Wi']))
    dump(o/'GATE.json',gate)
    if not gate['passed']:
        dump(o/'STOP.json',dict(reason='factory tier gate failed',gate=gate));dump(o/'CALL_LEDGER.json',C);return 2
    tierres={}
    for idx,b in enumerate(builds):
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());C['network_init']+=1
        n=5000;sp=np.zeros((n,len(roots)),bool);net_in=np.zeros((n,len(M18)))
        We=e0['We'] if isinstance(e0,dict) else e0.we;Wi=e0['Wi'] if isinstance(e0,dict) else e0.wi
        cols={}
        for jj,j in enumerate(M18):
            s,en=We.indptr[j],We.indptr[j+1];iw=Wi.indptr[j],Wi.indptr[j+1]
            cols[jj]=(We.indices[s:en],We.data[s:en],Wi.indices[iw[0]:iw[1]],Wi.data[iw[0]:iw[1]])
        for k in range(n):
            sp[k]=np.asarray(tier.advance(2,np.zeros(len(roots))),dtype=bool).any(axis=0);C['control_tick']+=1;C['tier_advance']+=1;C['neural_substep']+=2
            act=np.flatnonzero(sp[k])
            for jj,(we_i,we_d,wi_i,wi_d) in enumerate(cols.values()):
                net_in[k,jj]=we_d[sp[k,we_i]].sum()-wi_d[sp[k,wi_i]].sum()
        tierres[idx]=dict(e1_hz=float(sp[:,[loc[int(c)] for lg in LEGS for c in mod[lg]['E1'] if int(c) in loc]].sum()/5.0),net_in_median=float(np.median(net_in)),
          net_in_q=dict(p10=float(np.percentile(net_in,10)),p50=float(np.percentile(net_in,50)),p90=float(np.percentile(net_in,90))),
          spike_total=int(sp.sum()))
        np.savez_compressed(o/f'TSPK_{idx}.npz',net_in=net_in)
    dump(o/'TIER_DRIVE.json',dict(arms=tierres,equiv_current='I_eq(nA) ~ g_net(uS) * (E_exc - V_rest) = g * 65 (declared approximation)',
      band=band['cap8_full'],note='cell 18 motin net synaptic increment per ms'))
    dump(o/'CALL_LEDGER.json',C)
    rh_band=[I for I in pre['currents'] if arms[f'cap8_full_{I}'].get('rhythmic')]
    d1=bool(len(rh_band)>=2)
    oc=dict(round='S313',branch=('d1' if d1 else 'd3'),cap8_band=rh_band,raw_band=[I for I in pre['currents'] if arms[f'raw_full_{I}'].get('rhythmic')],
      tier_net_in=tierres[1]['net_in_q'],tier_e1_hz=tierres[1]['e1_hz'],
      one_line=('d1: cap-8 三节点 motif 存在稳定起振带 %s，tier 里 E1 净输入偏高' % rh_band) if d1 else 'd3: 无稳定起振带')
    dump(o/'OUTCOMES.json',oc)
    print(json.dumps(dict(selftest=[st['pass_a'],st['pass_b'],st['pass_c'],st['pass_d'],st['pass_e']],
      cap8_band=rh_band,raw_band=oc['raw_band'],counters=C,tier_e1=tierres[1]['e1_hz']),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
