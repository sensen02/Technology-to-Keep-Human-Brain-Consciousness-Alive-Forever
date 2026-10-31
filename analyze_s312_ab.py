"""S312 stage A (instrument calibration) and stage B (isolated substrate re-measure)."""
from __future__ import annotations
import argparse, json, hashlib, dataclasses
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S311=Path('/home/sensen/Desktop/engine修复验收/s311-cpg-anchor-and-drive/20261026T210656Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(con_network_init=0,con_run=0,con_step=0,model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0)
DT=0.5
def rate_hz(spk,dt=DT): return float(spk.mean()*1000.0/dt)
def bursts(spk,dt=DT):
    idx=np.flatnonzero(spk)
    if len(idx)<3: return np.array([]),np.array([])
    isi=np.diff(idx)*dt
    thr=3*np.median(isi)
    starts=[idx[0]];counts=[1];cur=1
    for i,d in enumerate(isi):
        if d>thr: starts.append(idx[i+1]);counts.append(1)
        else: counts[-1]+=1
    return np.asarray(starts),np.asarray(counts)
def v3(spk_e1,spk_i1,dt=DT):
    st,cnt=bursts(spk_e1,dt)
    if len(st)<11: return dict(rhythmic=False,reason='few_bursts',n_bursts=int(len(st)),mean_hz=rate_hz(spk_e1,dt))
    periods=np.diff(st)*dt
    cv=float(np.std(periods)/max(np.mean(periods),1e-9))
    spc=float(cnt[1:-1].mean())
    mean=rate_hz(spk_e1,dt)
    i1t=np.flatnonzero(spk_i1)
    lag_ok=False;lag=None
    if len(i1t)>=3 and len(st)>=3:
        per=float(np.mean(periods));cand=[]
        for s0 in st:
            k=np.searchsorted(i1t,s0)
            if k<len(i1t):
                off=(i1t[k]-s0)*dt
                cand.append((off % per)/per)
        if len(cand)>=3:
            lag=float(np.median(cand));lag_ok=bool(0.3<=lag<=0.7)
    rh=bool(len(periods)>=10 and cv<0.25 and lag_ok and 1.0<=spc<=12.0 and mean>=5.0)
    return dict(rhythmic=rh,n_bursts=int(len(st)),period_ms=float(np.mean(periods)),cv=cv,spikes_per_cycle=spc,
                mean_hz=mean,e1_i1_phase_lag_frac=lag,phase_ok=lag_ok)
def synth(period_ms,n_spk,dur_ms=3000.0,dt=DT,jitter=0.0,seed=1):
    n=int(dur_ms/dt);spk=np.zeros(n,dtype=bool);rng=np.random.default_rng(seed)
    t=period_ms*(1.0)
    while t<dur_ms:
        start=int(t/dt)
        for k in range(n_spk):
            j=int(start+k*rng.integers(1,3))
            if 0<=j<n: spk[j]=True
        t+=period_ms*(1+jitter*rng.standard_normal())
    return spk
def const_rate(hz,dur_ms=3000.0,dt=DT,seed=2):
    n=int(dur_ms/dt);p=hz*dt/1000.0;rng=np.random.default_rng(seed)
    return rng.random(n)<p
def selftest():
    per14=1000.0/14.0;per17=1000.0/17.0
    a=synth(per14,2);b=synth(per17,2);c=const_rate(20.0);z=np.zeros(int(3000/DT),dtype=bool)
    i1a=synth(per14,1); i1a=np.roll(i1a,int(0.5*per14/DT))
    i1b=synth(per17,1); i1b=np.roll(i1b,int(0.5*per17/DT))
    ra=v3(a,i1a);rb=v3(b,i1b);rc=v3(c,synth(1000.0/20.0,1));rz=v3(z,z)
    return dict(a=ra,b=rb,c=rc,d=rz,pass_a=bool(ra['rhythmic']),
      pass_b=bool(abs(1000.0/ra['period_ms']-17.0)<1.0 and abs(ra['period_ms']-1000.0/17.0)>1.0) if False else bool(rb['period_ms'] and abs(1000.0/rb['period_ms']-17.0)<1.0 and abs(rb['period_ms']-ra['period_ms'])>1.0),
      pass_c=bool(not rc['rhythmic']),pass_d=bool(not rz['rhythmic']))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    pre=dict(round='S312',stage_a='instrument calibration (0 dynamic tier calls)',stage_b='isolated substrate 20 runs',
      stage_c='tier motif input isolation 8 arms',
      hard=['verifier writes only verify/','S312_PREREG_FROZEN written before any dynamic call','traces stored as npz'],
      counters_branch=dict(normal=dict(banc_load=8,tier_build=8,select_touch_tier=16,edges_build=8,network_init=8,control_tick=40000,tier_advance=40000,neural_substep=80000,con_network_init=20,con_run=20,con_step=120000)),
      detector_v3='burst intervals: ISI>3x median defines bursts; need >=11 bursts (>=10 intervals), cv<0.25, I1 lag fraction in 0.3-0.7, E1 spikes per cycle in [1,12], mean rate >=5 Hz',
      drive_grid_b='cap8 currents {0.0165,0.0253,0.07,0.16,0.2} x {full,no_I1_E1,zero} = 15; raw family x same currents x {full} = 5',
      branches=dict(c1='any isolated tier arm rhythmic',c2='no isolated arm rhythmic but some I1 conditional rate ratio <=0.75',c3='no rhythmic and all ratios >0.90',c4='gate or selftest failure'))
    dump(o/'PREREG.json',pre);(o/'S312_PREREG_FROZEN').write_text('frozen before any dynamic call\n')
    # ---- stage A
    P=ConductanceParams();R=1.0/P.leak_uS;dV=P.threshold_mV-P.rest_mV
    sp=synth(1000.0/28.0,1);kr_mean=float((lambda s: (lambda a: (np.array([a:=a*np.exp(-DT/10.0)+x for x in s]) if False else None))(0))(sp) or 0)
    # kernel mean computed properly
    a=np.exp(-DT/10.0);acc=0.0;vals=[]
    for x in sp: acc=a*acc+x; vals.append(acc*DT/10.0)
    kr_mean=float(np.mean(vals));ratio=float(kr_mean and (sp.mean()*1000.0/DT)/max(kr_mean,1e-30))
    st=selftest()
    dump(o/'DETECTOR_V3.json',dict(criteria=pre['detector_v3'],burst_definition='ISI>3x median starts a new burst'))
    dump(o/'DETECTOR_SELFTEST_V3.json',st)
    corr={'round':'S312','items':[
      dict(id='C1',s311='mean_hz needs x1000',measured=dict(kernel_mean_28hz=kr_mean,spike_mean_hz=float(sp.mean()*1000.0/DT),ratio=ratio),
           corrected='the factor is 1000/dt_ms = 2000, not 1000; a 28 Hz synthetic burst gives spk.mean()*1000/dt = 28.0 against a kernel mean of %.5f (ratio %.1f)'%(kr_mean,ratio)),
      dict(id='C2',s311='0.2<=m/peak<=2.0 is the rhythm test',measured=dict(example_ratio_after_fix=147.7/17.58),
           corrected='after the unit fix that window means 0.2-2 spikes per cycle, so it only accepts spontaneous discharge; replaced by the burst-interval criteria in DETECTOR_V3'),
      dict(id='C3',s311='attenuation sweep including 0.01x',measured=dict(rheobase_nA=dV/R,frac_0p05_nA=0.16*0.05),
           corrected='0.05x, 0.025x and 0.01x of I_plateau are below the single-cell rheobase (%.3f nA), so those arms are silent for a trivial reason and are excluded from any motif conclusion'%(dV/R)),
      dict(id='C4',s311='anchor table usable directly',measured=dict(plateau_hz=333.5,plateau_currents_nA=[0.4,0.5,0.60753]),
           corrected='the measured table saturates at 333.5 Hz (2 ms refractory + 1 bin) for 0.4-0.60753 nA, so any anchor claim must name the exact column and row'),
      dict(id='C5',s311_branch='a3 (no oscillation)',measured=dict(s310_raw_shared_peaks=[9.77,13.67,15.62,18.55]),
           corrected='S310 branch 1 is PROVISIONAL: the three cells share one peak, which is a network signature, but that family is the raw counts at 40x the tier weights, so it does not transfer to the cap-8 substrate')]}
    dump(o/'S311_CORRECTION.json',corr)
    grid=dict(usable_current_floor_nA=dV/R,cap8_currents=[0.0165,0.0253,0.07,0.16,0.2],raw_currents=[0.0165,0.0253,0.07,0.16,0.2],
              anchor_column='ANCHOR_MEASURED.json adapt_on column',plateau_current_nA=0.16,plateau_hz_measured=333.5,
              note='the 0.05x/0.025x/0.01x arms of S311 are sub-rheobase and excluded')
    dump(o/'DRIVE_GRID.json',grid)
    if not (st['pass_a'] and st['pass_b'] and st['pass_c'] and st['pass_d']):
        dump(o/'STOP.json',dict(reason='detector v3 self-test failed',selftest=st));dump(o/'CALL_LEDGER.json',C);return 3
    # ---- stage B
    e=json.loads((S307/'PER_LEG_MODULE.json').read_text())['edges']['lf']
    NM={'E1':0,'E2':1,'I1':2}
    def build(scale):
        We=np.zeros((3,3));Wi=np.zeros((3,3))
        for k,(post,pre,t) in {'E1->E2':(1,0,'e'),'E2->E1':(0,1,'e'),'E2->I1':(2,1,'e'),'E1->I1':(2,0,'e'),'I1->E1':(0,2,'i'),'I1->E2':(1,2,'i')}.items():
            v=float(e[k])*scale
            (We if t=='e' else Wi)[post,pre]=v
        return We,Wi
    def run(We,Wi,I,dur=3000.0):
        net=ConductanceNetwork(3,P,dt_ms=DT);net.set_connectivity(We,Wi);C['con_network_init']+=1
        n=int(dur/DT);spk=np.zeros((n,3),dtype=bool)
        for k in range(n): spk[k]=net.step(current_nA=np.array([I,0.0,0.0]));C['con_step']+=1
        C['con_run']+=1;return spk
    cap=5e-4;raw=5e-4
    Wcap,Wi_cap=build(cap)          # counts x 5e-4 but capped at 8
    We_c=np.minimum(np.asarray([[0,e['E1->E2'],0],[e['E2->E1'],0,0],[e['E1->I1'],e['E2->I1'],0]],float),8)*cap
    Wi_c=np.minimum(np.asarray([[0,0,e['I1->E1']],[0,0,e['I1->E2']],[0,0,0]],float),8)*cap
    We_r=np.asarray([[0,e['E1->E2'],0],[e['E2->E1'],0,0],[e['E1->I1'],e['E2->I1'],0]],float)*raw
    Wi_r=np.asarray([[0,0,e['I1->E1']],[0,0,e['I1->E2']],[0,0,0]],float)*raw
    arms={}
    for fam,(We,Wi) in (('cap8',(We_c,Wi_c)),('raw',(We_r,Wi_r))):
        tops=('full','no_I1_E1','zero_weights') if fam=='cap8' else ('full',)
        for top in tops:
            W1=We.copy();W2=Wi.copy()
            if top=='no_I1_E1': W2[0,2]=0.0
            if top=='zero_weights': W1[:]=0.0;W2[:]=0.0
            for I in grid[f'{fam}_currents']:
                spk=run(W1,W2,I);key=f'{fam}_{top}_{I}'
                arms[key]=v3(spk[:,0],spk[:,2])
                np.savez_compressed(o/f'SPK_{key}.npz',spk=spk)
    dump(o/'STAGE_B.json',dict(arms=arms,counters=dict(C)))
    dump(o/'CALL_LEDGER.json',C)
    print(json.dumps(dict(selftest=[st['pass_a'],st['pass_b'],st['pass_c'],st['pass_d']],
      rhythmic=[k for k,v in arms.items() if v['rhythmic']],n_arms=len(arms),counters=C),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
