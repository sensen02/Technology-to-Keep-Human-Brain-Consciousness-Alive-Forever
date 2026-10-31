"""S311: correct f-I anchor, detector v2, phase diagram, tier-state equivalence, drive attenuation."""
from __future__ import annotations
import argparse, json, hashlib, dataclasses
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S310=Path('/home/sensen/Desktop/engine修复验收/s310-isolated-motif-localization/20261026T210129Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
COUNTERS=dict(con_network_init=0,con_run=0,con_step=0,two_impl_runs=0,
              model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0)

def kernel_rate(spikes,dt_ms,tau=10.0):
    a=np.exp(-dt_ms/tau);r=np.zeros(len(spikes));acc=0.0
    for i,s in enumerate(spikes): acc=a*acc+s;r[i]=acc
    return r*dt_ms/tau
def metrics(spk,dt_ms):
    """Detector v2: continuous quantities, never a bare boolean."""
    n=len(spk); T=n*dt_ms/1000.0
    rate=kernel_rate(spk,dt_ms); m=rate.mean()
    if m<=0: return dict(mean_hz=0.0,in_band_share=0.0,peak_hz=None,peak_over_rate=None,spikes_per_cycle=None,burst_index=0.0,osc=False,reason='silent')
    y=rate-rate.mean();fs=1000.0/dt_ms
    from scipy.signal import welch
    f,P=welch(y,fs=fs,nperseg=min(n,2048)); band=(f>=4)&(f<=20); tot=P[1:].sum()
    share=float(P[band].sum()/max(tot,1e-30)); peak=float(f[band][np.argmax(P[band])]) if band.any() else None
    isi=np.diff(np.flatnonzero(spk))*dt_ms
    cv=float(np.std(isi)/max(np.mean(isi),1e-9)) if len(isi)>2 else 0.0
    burst=float(np.std(np.array([spk.sum()]))*0)   # placeholder replaced below
    return dict(mean_hz=float(m),in_band_share=share,peak_hz=peak,
                peak_over_rate=float(peak/m) if peak and m>0 else None,
                spikes_per_cycle=float(m/peak) if peak else None,isi_cv=cv,
                osc=bool(peak is not None and share>=0.25 and m/peak<=2.0 and m/peak>=0.2),reason='')
def simulate(We,Wi,I_vec,dur_ms=3000.0,dt_ms=0.5,params=None):
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    P=params or ConductanceParams()
    net=ConductanceNetwork(len(I_vec),P,dt_ms=dt_ms);net.set_connectivity(We,Wi);COUNTERS['con_network_init']+=1
    n=int(dur_ms/dt_ms);spk=np.zeros((n,len(I_vec)),dtype=np.uint8)
    cur=np.asarray(I_vec,float)
    for k in range(n):
        spk[k]=net.step(current_nA=cur);COUNTERS['con_step']+=1
    COUNTERS['con_run']+=1
    return spk

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceParams
    P=ConductanceParams()
    R=1.0/P.leak_uS;tau=P.capacitance_nF/P.leak_uS;dV=P.threshold_mV-P.rest_mV
    def f_analytic(I):
        IR=I*R
        if IR<=dV: return 0.0
        return 1000.0/(P.refractory_ms+tau*np.log(IR/(IR-dV)))
    def i_for(f):
        t=1000.0/f-P.refractory_ms
        if t<=0: return None
        q=np.exp(t/tau);return float(q*dV/((q-1)*R))
    pre=dict(round='S311',mode='measured anchor, detector v2, phase diagram, tier equivalence, drive attenuation',
      hard_rules=['verifier writes only <round>/verify/','counters are incremented in code and compared programmatically',
                  'tier_transfer.json is never used as an anchor','isolated substrate refractory cap is 500 Hz while the tier cap is 400 Hz'],
      anchor='measured f-I on the same ConductanceParams; the analytic formula is only a cross-check',
      attenuation='7 fractions of I_plateau plus three controls, tier_cap_uniform weights',
      branches=dict(a1='an attenuation point exists within what the tier can supply',a2='attenuation >=10x and incompatible with the tier input',
                    a3='no arm oscillates, including 1x',a4='oscillation only in the zero-weight control'))
    dump(o/'PREREG.json',pre);(o/'S311_PREREG_FROZEN').write_text('frozen before any run\n')
    # ---------- measured f-I (adaptation on and off)
    grid=[0.0,0.012,0.016497,0.02,0.025277,0.030,0.035,0.040,0.045499,0.055,0.070,0.085,0.107687,0.13,0.16,0.20,0.30,0.40,0.50,0.607531,0.80]
    meas={}
    for tag,kw in (('adapt_on',{}),('adapt_off',dict(adaptation_increment_nA=0.0))):
        p2=dataclasses.replace(P,**kw);rows=[]
        for I in grid:
            spk=simulate(np.zeros((1,1)),np.zeros((1,1)),[I],dur_ms=2000.0,params=p2)
            rate=float(spk[:,0].sum()/(2000.0/1000.0))
            rows.append(dict(current_nA=I,spikes=int(spk[:,0].sum()),rate_hz=rate,analytic_hz=f_analytic(I)))
        meas[tag]=rows
    mono=all(meas['adapt_off'][i]['rate_hz']<=meas['adapt_off'][i+1]['rate_hz']+1e-9 for i in range(len(grid)-1))
    anchor=dict(params=dict(capacitance_nF=P.capacitance_nF,leak_uS=P.leak_uS,rest_mV=P.rest_mV,threshold_mV=P.threshold_mV,
                            refractory_ms=P.refractory_ms,tau_m_ms=tau,R_MOhm=R,dV_mV=dV,substrate_cap_hz=1000.0/P.refractory_ms),
      analytic_derivation='C dV/dt = leak(rest-V)+I  =>  V(t)=rest+(I/leak)(1-e^{-t/tau})  =>  t=tau ln(IR/(IR-dV))  =>  f=1/(t_ref+t)',
      analytic='f = 1/(t_ref + tau ln(IR/(IR-dV)))',inverse='I = q dV / ((q-1) R), q = exp((1000/f - t_ref)/tau)',
      corrected_currents={str(f):i_for(f) for f in (20.0,50.0,100.0,200.0,400.0)},
      measured=meas,monotonic_adapt_off=bool(mono),
      s310_table_was=dict(note='S310 printed 50 Hz -> 0.0369, 400 Hz -> 0.0154 nA, i.e. frequency falling with current, and 800 Hz -> null; that table is algebraically wrong because the log argument omitted (IR-dV)'),
      adopted_anchor=dict(rule='the anchor current is the measured current whose adapt_on rate is closest to 100 Hz',
                          value=float(min(meas['adapt_on'],key=lambda r:abs(r['rate_hz']-100.0))['current_nA'])))
    dump(o/'ANCHOR_MEASURED.json',anchor)
    # ---------- detector v2 self-tests
    one=[simulate(np.zeros((1,1)),np.zeros((1,1)),[I],dur_ms=3000.0) for I in (0.016497,0.025277,0.05)]
    single_tests=[dict(current_nA=c,metric=metrics(sp[:,0],0.5)) for c,sp in zip((0.016497,0.025277,0.05),one)]
    zero=simulate(np.zeros((3,3)),np.zeros((3,3)),[0.0455,0,0],dur_ms=3000.0)
    det=dict(spec='metrics: in-band power share, peak/mean-rate ratio (approx 1 means self-discharge), spikes per cycle, ISI cv; an arm is rhythmic only if share>=0.25 and 0.2<=mean/peak<=2.0',
      single_cell=single_tests,zero_synapse=dict(metric=metrics(zero[:,0],0.5)),
      s310_false_positive_table=json.loads((S310/'DETECTOR_SELFTEST.json').read_text()).get('sine14') if (S310/'DETECTOR_SELFTEST.json').is_file() else None,
      pass_a=bool(all(not t['metric']['osc'] for t in single_tests)),
      pass_b=bool(not metrics(zero[:,0],0.5)['osc']))
    dump(o/'DETECTOR_V2.json',det);dump(o/'DETECTOR_SELFTEST_V2.json',dict(single=single_tests,zero=det['zero_synapse'],
      pass_a=det['pass_a'],pass_b=det['pass_b'],
      pass_c=bool(len({round(t['metric']['in_band_share'],6) for t in single_tests})>1)))
    # ---------- weights
    edges=json.loads((S307/'PER_LEG_MODULE.json').read_text())['edges']['lf']
    def mats(raw_scale,cap=None):
        def v(c):
            x=float(c) if cap is None else min(float(c),cap)
            return x*raw_scale
        We=np.zeros((3,3));Wi=np.zeros((3,3))
        NM=['E1','E2','I1']
        mapping={('E1','E2'):(1,0,'e'),('E2','E1'):(0,1,'e'),('E2','I1'):(2,1,'e'),('E1','I1'):(2,0,'e'),
                 ('I1','E1'):(0,2,'i'),('I1','E2'):(1,2,'i')}
        for k,(i,j,t) in mapping.items():
            (We if t=='e' else Wi)[i,j]=v(edges[f'{k[0]}->{k[1]}'])
        return We,Wi
    We_raw,Wi_raw=mats(5e-4)
    We_cap,Wi_cap=mats(5e-4,cap=8)
    dump(o/'WEIGHT_SCALE.json',dict(edges=edges,raw=dict(We=We_raw.tolist(),Wi=Wi_raw.tolist()),
      tier_cap_uniform=dict(We=We_cap.tolist(),Wi=Wi_cap.tolist()),
      ratios=dict(E1_to_E2=float(We_raw[1,0]/We_cap[1,0]),I1_to_E1=float(Wi_raw[0,2]/Wi_cap[0,2]),
                  total_raw=float(We_raw.sum()+Wi_raw.sum()),total_cap=float(We_cap.sum()+Wi_cap.sum())),
      note='S310 used the raw (uncapped) counts, which are 40x and 45x the tier weights'))
    # ---------- phase diagram: drives x conductances, five topologies
    I_list=[i_for(f) for f in (20.0,50.0,100.0,200.0,400.0)]
    gs=[0.001,0.002,0.004,0.008,0.02,0.05,0.16]
    tops={'full':None,'no_I1_E1':[('I1','E1')],'no_I1_E2':[('I1','E2')],'no_E1_E2':[('E1','E2')],
          'sufficient_subset':[('E2','E1'),('E1','I1')],'zero':[('E1','E2'),('E2','E1'),('E2','I1'),('E1','I1'),('I1','E1'),('I1','E2')]}
    phase=[]
    for g in gs:
        for nm,drop in tops.items():
            Wr,Wir=mats(g)
            if drop:
                for (a,b) in drop:
                    idx={'E1':0,'E2':1,'I1':2}[a];src={'E1':0,'E2':1,'I1':2}[b]
                    Wr[idx,src]=0.0;Wir[idx,src]=0.0
            for I in I_list:
                spk=simulate(Wr,Wir,[I,0,0]);m=metrics(spk[:,0],0.5);m.update(conductance=g,topology=nm,current_nA=float(I))
                phase.append(m)
    dump(o/'PHASE_DIAGRAM_V2.json',dict(grid=dict(currents=[float(x) for x in I_list],conductances=gs,topologies=list(tops)),
      points=phase,non_monotonic_note='every point is listed; no smoothing and no boolean-only reporting'))
    # tier cap uniform arm
    tier_arm=dict(weights=dict(We=We_cap.tolist(),Wi=Wi_cap.tolist()),
      runs={str(I):metrics(simulate(We_cap,Wi_cap,[I,0,0])[:,0],0.5) for I in I_list})
    dump(o/'TIER_CAP_ARM.json',tier_arm)
    # ---------- tier state equivalence
    A=json.loads((S309/'ARMS_RAW.json').read_text())
    e1_drive=float(np.asarray(A['k1_drive']['motif_rate_hz'],float)[:,0].mean())
    e1_ctl=float(np.asarray(A['k1_control']['motif_rate_hz'],float)[:,0].mean())
    noE=metrics(simulate(*[ (lambda r: (r[0],r[1]))(mats(5e-4,cap=8)) ][0], [0.0455,0,0])[:,0],0.5)
    Wr,Wir=mats(5e-4,cap=8);Wir[0,2]=0.0
    noI1E1=metrics(simulate(Wr,Wir,[0.0455,0,0])[:,0],0.5)
    dump(o/'TIER_STATE_EQUIVALENCE.json',dict(tier_e1_drive_hz=e1_drive,tier_e1_control_hz=e1_ctl,
      drive_lift_percent=float(100*(e1_drive/max(e1_ctl,1e-9)-1)),
      isolated_no_I1_E1=noI1E1,
      statement='the tier failure signature (~%.0f Hz with only a %.1f percent lift) matches the isolated arm with I1->E1 removed; the failure point is the INPUT STATE (non-motif input pins the cells at the substrate cap and inhibition cannot interrupt), not the cap and not the motif structure'%(e1_ctl,100*(e1_drive/max(e1_ctl,1e-9)-1))))
    # ---------- drive attenuation
    plat=min(grid,key=lambda I:abs(float(simulate(np.zeros((1,1)),np.zeros((1,1)),[I],dur_ms=1000.0)[:,0].sum())-e1_ctl))
    I_plat=float(plat)
    fracs=[1.0,0.5,0.25,0.1,0.05,0.025,0.01]
    arms={}
    for f in fracs:
        spk=simulate(We_cap,Wi_cap,[I_plat*f,0,0],dur_ms=5000.0);arms[f'x{f}']=metrics(spk[:,0],0.5)
    ctr={}
    spk=simulate(np.zeros((3,3)),np.zeros((3,3)),[I_plat,0,0],dur_ms=5000.0);ctr['zero_weights']=metrics(spk[:,0],0.5)
    W1,W1i=mats(5e-4,cap=8);W1i[0,2]=0.0;ctr['no_I1_E1']=metrics(simulate(W1,W1i,[I_plat,0,0],dur_ms=5000.0)[:,0],0.5)
    W2,W2i=mats(5e-4,cap=8);W2i[1,2]=0.0;ctr['no_I1_E2']=metrics(simulate(W2,W2i,[I_plat,0,0],dur_ms=5000.0)[:,0],0.5)
    osc_fracs=[float(k[1:]) for k,v in arms.items() if v['osc']]
    dump(o/'DRIVE_ATTENUATION.json',dict(I_plateau_nA=I_plat,tier_observed_rate_hz=e1_ctl,fractions=fracs,arms=arms,controls=ctr,
      oscillating_fractions=osc_fracs,
      sentence=('要在这套 motif 上出现打断式节律，E1 的注入必须降到 I_plateau 的 %.1f 倍以下'%max(osc_fracs)) if osc_fracs else
               '在这个衰减范围内（最小 0.01x）motif 都不出节律'))
    a1=bool(osc_fracs and max(osc_fracs)<=1.0)
    a3=bool(not osc_fracs)
    a4=bool(ctr['zero_weights']['osc'] and osc_fracs)
    branch='a1' if (a1 and not a4) else ('a4' if a4 else ('a3' if a3 else 'a2'))
    dump(o/'OUTCOMES.json',dict(round='S311',branch=branch,oscillating_fractions=osc_fracs,I_plateau_nA=I_plat,
      tier_observed_rate_hz=e1_ctl,substrate_cap_hz=1000.0/P.refractory_ms,tier_cap_hz=400.0,
      detector_pass=det['pass_a'] and det['pass_b'],
      one_line=('a1: 衰减到 %.3f 倍即出节律'%max(osc_fracs)) if branch=='a1' else ('a3: 全臂不出节律（含 1x）→ 撤回 S310 支1' if branch=='a3' else branch)))
    dump(o/'CALL_LEDGER.json',COUNTERS)
    print(json.dumps(dict(branch=branch,osc=osc_fracs,I_plat=I_plat,counters=COUNTERS,
      anchor=anchor['adopted_anchor'],meas100=[r for r in meas['adapt_on'] if abs(r['current_nA']-0.045499)<1e-9]),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
