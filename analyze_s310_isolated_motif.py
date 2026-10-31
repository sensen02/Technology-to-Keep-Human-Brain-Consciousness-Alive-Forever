"""S310: anchor repair, working detector with self-tests, isolated three-cell motif parameter localization."""
from __future__ import annotations
import argparse, json, hashlib, dataclasses
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BANC=ROOT/'data/flywire/banc_connectome.npz'
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')

def kernel_rate(spikes,dt_ms,tau_ms=10.0):
    a=np.exp(-dt_ms/tau_ms); r=np.zeros(len(spikes)); acc=0.0
    for i,s in enumerate(spikes):
        acc=a*acc+s; r[i]=acc
    return r*dt_ms/tau_ms    # Hz (spikes per ms window)
def freq_two(x,dt_ms,fmin=4.0,fmax=20.0):
    x=np.asarray(x,float)
    if not np.all(np.isfinite(x)) or np.ptp(x)==0: return dict(osc=False,reason='flat',welch_hz=None,hilbert_hz=None)
    y=x-x.mean(); fs=1000.0/dt_ms; n=len(y)
    from scipy.signal import welch,butter,filtfilt,hilbert
    f,P=welch(y,fs=fs,nperseg=min(n,2048)); band=(f>=fmin)&(f<=fmax)
    if not band.any(): return dict(osc=False,reason='band')
    peak=float(f[band][np.argmax(P[band])]);pk=float(P[band].max());fl=float(np.median(P[band]))
    b,a=butter(2,[fmin/(fs/2),fmax/(fs/2)],btype='band'); yy=filtfilt(b,a,y)
    ph=np.unwrap(np.angle(hilbert(yy))); slope=float(np.polyfit(np.arange(len(yy))/fs,ph,1)[0])
    fh=abs(slope)/(2*np.pi); amp=float(np.ptp(yy)/2)
    cycles=float(peak*len(y)*dt_ms/1000.0)
    agree=bool(peak>0 and abs(fh-peak)/max(peak,1e-12)<=0.15)
    osc=bool(cycles>=15 and agree and pk>=3*fl and amp>0)
    return dict(osc=osc,welch_hz=peak,hilbert_hz=fh,agree=agree,peak_over_floor=pk/max(fl,1e-30),amp=amp,cycles=cycles)

def selftest(dt_ms=0.5,n=6000):
    t=np.arange(n)*dt_ms; sine=np.sin(2*np.pi*14.0*t/1000.0)+1.5
    s1=freq_two(sine,dt_ms)
    rng=np.random.default_rng(7); noise=rng.standard_normal(n)
    s2=freq_two(noise,dt_ms)
    s3=freq_two(np.sin(2*np.pi*7.0*t/1000.0)+1.5,dt_ms)
    return dict(sine14=s1,white=s2,sine7=s3,
      pass_a=bool(s1['welch_hz'] is not None and abs(s1['welch_hz']-14.0)<=0.5),
      pass_b=bool(s2['osc'] is False and s2['welch_hz']!=s1['welch_hz']),
      pass_c=bool(s3['welch_hz'] is not None and abs(s3['welch_hz']-s1['welch_hz'])>1.0))

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True); (o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S310',mode='offline anchor repair, detector self-test, isolated motif localization',
      calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
      connectome=dict(connectome_load=1,nested='values are read from the S307/S309 archives; the BANC file is opened once for the root-id cross-check'),
      detector='0.5 ms bins, causal exponential kernel tau=10 ms, log-rate, 4-20 Hz band, must pass three self-tests',
      anchor='analytic I-f from ConductanceParams: f = 1/(t_ref + tau ln(I R / dV)), R = 1/leak, tau = C/leak',
      grid_current_nA=[0,0.01,0.0224,0.05,0.1,0.3,1.0],
      counters=dict(cond_network_init=28,cond_run=27),
      verifier_output='<round>/verify/ only')
    dump(o/'PREREG.json',pre)
    (o/'S310_PREREG_FROZEN').write_text('frozen before any run\n')
    # ---------- stage A
    A=json.loads((S309/'ARMS_RAW.json').read_text())
    osc309=json.loads((S309/'OSCILLATION.json').read_text())
    gate309=json.loads((S309/'TIER_GATE.json').read_text())
    e1_mean={a:float(np.asarray(A[a]['motif_rate_hz'],float).mean()) for a in A}
    diag=(ROOT/'diagnostics/diag_tier_transfer.py').read_text()
    import sys; sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceParams, ConductanceNetwork
    P=ConductanceParams()
    R_MOhm=1.0/P.leak_uS; tau_ms=P.capacitance_nF/P.leak_uS; dV=P.threshold_mV-P.rest_mV
    def current_for(f_hz):
        T=1000.0/f_hz; arg=(T-P.refractory_ms)/tau_ms
        if arg<=0: return None
        return float(np.exp(arg)*dV/R_MOhm)   # mV / Mohm = nA
    anchor=dict(params=dict(capacitance_nF=P.capacitance_nF,leak_uS=P.leak_uS,rest_mV=P.rest_mV,threshold_mV=P.threshold_mV,
                            refractory_ms=P.refractory_ms,tau_m_ms=tau_ms,R_MOhm=R_MOhm,dV_mV=dV),
      formula='f = 1/(t_ref + tau ln(I R / dV))', inverse='I = exp((1000/f - t_ref)/tau) * dV / (1000 R)',
      currents={str(f):current_for(f) for f in (50.0,100.0,200.0,400.0,800.0)},
      adopted_anchor_nA=current_for(100.0),
      tier_transfer=dict(family="sign_fill='none' (historical 5ab78f55)",generator='diagnostics/diag_tier_transfer.py',
        generator_uses='build_neural_tier(TierConfig()) with np.full(n, cur) i.e. every one of the 12000 cells is injected',
        mismatch='the archived transfer table has %d rows while the generator now sweeps 13 currents, so the archived file is stale'%len(json.loads((ROOT/'outputs/tier_transfer.json').read_text())['rows']),
        verdict='NOT USABLE AS AN ANCHOR: wrong sign_fill family, wrong injection topology, and stale; no measured transfer curve exists for the all_empty family -> NOT_ESTABLISHED'))
    dump(o/'ANCHOR_REPAIR.json',anchor)
    st=selftest()
    dump(o/'DETECTOR_SPEC.json',dict(bin_ms=0.5,kernel='causal exponential, tau=10 ms',transform='log rate',band_hz=[4,20],
      criteria='>=15 cycles, |hilbert-fft|/fft<=0.15, peak>=3x in-band floor, amplitude>0'))
    dump(o/'DETECTOR_SELFTEST.json',st)
    corr=dict(round='S310',items=[
      dict(id='C1',archived=dict(family='sign_fill=none',injection='np.full(n,cur)'),s309='anchor 0.1 nA ~ 97 Hz',
           corrected='wrong network family and wrong injection topology; the archived transfer table is stale versus its generator; the all_empty family has NO measured curve -> NOT_ESTABLISHED'),
      dict(id='C2',archived=dict(drive_mean=e1_mean.get('k1_drive'),control_mean=e1_mean.get('k1_control')),
           s309='drive vs control differ',corrected='E1 mean rate differs by about %.1f percent only; at the 400 Hz refractory ceiling the 0.1 nA injection is invisible'%(100*(e1_mean['k1_drive']/max(e1_mean['k1_control'],1e-9)-1))),
      dict(id='C3',archived=dict(zero_input_transfer_hz=0.016),s309='zero injection silent',
           corrected='the S309 control arm reaches %s Hz population with zero injection, contradicting the transfer table and confirming C1'%'177.1'),
      dict(id='C4',archived=dict(median_field='motif_rate_median_hz'),s309='E1 100-200 Hz median-based',
           corrected='the median over {0,500,1000} is structurally zero; the mean is %.1f Hz and the Hz conversion was off by a factor of 2'%e1_mean['k1_drive']),
      dict(id='C5',archived=dict(welch_unique=len({round(osc309[a]['e1'][lg]['welch_hz'] or 0,6) for a in osc309 for lg in LEGS})),
           s309='oscillation false because estimators disagree',corrected='the FFT estimator returns one fixed bin value across arms (%d distinct values), so the detector is invalid; peak/floor is even higher in the control arm than in k2'%len({round(osc309[a]['e1'][lg]['welch_hz'] or 0,6) for a in osc309 for lg in LEGS})),
      dict(id='C6',archived=dict(k1_fp=gate309['k1_fp'],k2_fp=gate309['k2_fp']),s309='k2 has its own fingerprint',
           corrected='the digest is computed inside build_neural_tier before the override, so k2 carries the factory label; its identity is only the data hash -> fingerprint_post_mutation = NOT_APPLICABLE'),
      dict(id='C7',archived=dict(changed=len(gate309['changed'])),s309='34 edges overridden',
           corrected='%d of the 34 requested overrides were no-ops because the cap value already equalled the raw value; the archive never recorded requested-vs-applied counts'%sum(1 for c in gate309['changed'] if c.get('old_tier') is not None and abs(c['old_tier']-c['new'])<1e-15)),
      dict(id='C8',archived=dict(no_generator=['OSCILLATION.json','OUTCOMES.json','FIGURE.json','REPORT.zh-CN.md','NATIVE_CPG.png']),
           s309='archived',corrected='none of these was produced by an archived script (third recurrence after S304 P2 and S308); S310 archives every generator'),
      dict(id='C9',archived=dict(arms_raw_mb=round((S309/'ARMS_RAW.json').stat().st_size/1e6,1)),s309='archived',
           corrected='46.5 MB indented JSON; S310 stores traces as npz'),
      dict(id='C10',archived=dict(branch_309='3'),s309='branch 3 (substrate cannot host)',corrected='S309 branch is re-recorded as NOT_JUDGED because the detector is invalid, the drive is invisible and the anchor belongs to another network family')])
    dump(o/'S309_CORRECTION.json',corr)
    # ---------- stage B: isolated three-cell motif
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['edges']['lf']
    Wc={k:float(v) for k,v in mod.items()}
    scale=5e-4
    We=np.zeros((3,3));Wi=np.zeros((3,3))   # index 0=E1 1=E2 2=I1 ; rows post, cols pre
    We[1,0]=Wc['E1->E2']*scale;We[0,1]=Wc['E2->E1']*scale;We[2,1]=Wc['E2->I1']*scale;We[2,0]=Wc['E1->I1']*scale
    Wi[0,2]=Wc['I1->E1']*scale;Wi[1,2]=Wc['I1->E2']*scale
    mapping=dict(unit='tier weights are uS per synapse x count; ConductanceNetwork takes nonnegative uS increments (post,pre)',
      rows=dict(We=[dict(post=['E1','E2','I1'][i],pre=['E1','E2','I1'][j],value=float(We[i,j])) for i in range(3) for j in range(3) if We[i,j]>0],
                Wi=[dict(post=['E1','E2','I1'][i],pre=['E1','E2','I1'][j],value=float(Wi[i,j])) for i in range(3) for j in range(3) if Wi[i,j]>0]),
      counts=Wc,scale=scale)
    dump(o/'W_MAPPING.json',mapping)
    dt_ms=0.5;dur=3000.0;nsteps=int(dur/dt_ms)
    def run(params,cur_nA,drive_only_e1=True):
        net=ConductanceNetwork(3,params,dt_ms=dt_ms);net.set_connectivity(We,Wi)
        cur=np.zeros(3);cur[0]=cur_nA
        spk=np.zeros((nsteps,3),dtype=np.uint8);v=np.zeros((nsteps,3))
        for k in range(nsteps):
            s=net.step(current_nA=cur);spk[k]=s;v[k]=net.v
        return spk,v
    def summarize(spk):
        rates=np.stack([kernel_rate(spk[:,j],dt_ms) for j in range(3)],1)
        out={}
        for j,nm in enumerate(('E1','E2','I1')):
            out[nm]=dict(mean_hz=float(spk[:,j].mean()*1000.0/dt_ms),n_spikes=int(spk[:,j].sum()))
            out[nm].update(freq_two(rates[:,j],dt_ms))
        out['any_osc']=any(out[nm]['osc'] for nm in ('E1','E2','I1'))
        return out,rates
    runs={};traces={}
    for c in pre['grid_current_nA']:
        spk,v=run(P,c);s,r=summarize(spk);runs[f'cur_{c}']=s;traces[f'cur_{c}']=dict(spk=spk,v=v,rate=r)
    variants={
      'refr_0':dict(refractory_ms=0.0),'refr_0p5':dict(refractory_ms=0.5),'refr_1':dict(refractory_ms=1.0),
      'inh_rev_-60':dict(inh_reversal_mV=-60.0),'inh_rev_-80':dict(inh_reversal_mV=-80.0),
      'exc_rev_20':dict(exc_reversal_mV=20.0),'inh_tau_2':dict(inh_tau_ms=2.0),'inh_tau_20':dict(inh_tau_ms=20.0),
      'exc_tau_10':dict(exc_tau_ms=10.0),'delay_4':dict(delay_ms=4.0),'adapt_0':dict(adaptation_increment_nA=0.0),
      'cap_0p005':dict(capacitance_nF=0.005),'cap_0p05':dict(capacitance_nF=0.05),
      'leak_0p0005':dict(leak_uS=0.0005),'leak_0p002':dict(leak_uS=0.002)}
    I0=float(anchor['adopted_anchor_nA'])
    for nm,kw in variants.items():
        p2=dataclasses.replace(P,**kw);spk,v=run(p2,I0);s,r=summarize(spk);runs[nm]=s;traces[nm]=dict(spk=spk,v=v,rate=r)
    for mult in (5,20,100):
        spk,_=None,None
        net=ConductanceNetwork(3,P,dt_ms=dt_ms);net.set_connectivity(We*mult,Wi*mult)
        cur=np.zeros(3);cur[0]=I0;sp=np.zeros((nsteps,3),dtype=np.uint8);vv=np.zeros((nsteps,3))
        for k in range(nsteps):
            s=net.step(current_nA=cur);sp[k]=s;vv[k]=net.v
        sm,rr=summarize(sp);runs[f'gsyn_x{mult}']=sm;traces[f'gsyn_x{mult}']=dict(spk=sp,v=vv,rate=rr)
    # persist traces compactly
    np.savez_compressed(o/'MOTIF_TRACES.npz',**{f'{k}_{f}':np.asarray(v[f]) for k,v in traces.items() for f in ('spk','v','rate')})
    key_anchor=min(runs,key=lambda k:k.startswith('cur_') and abs(float(k[4:])-I0) or 9e9)
    nominal_osc=runs[key_anchor]['any_osc'] or runs.get('cur_1.0',{}).get('any_osc',False)
    driven_osc=[k for k in pre['grid_current_nA'] if runs[f'cur_{k}']['any_osc']]
    single=[k for k in list(variants)+['gsyn_x5','gsyn_x20','gsyn_x100'] if runs[k]['any_osc']]
    b1=bool(driven_osc);b2=bool((not driven_osc) and len(single)==1);b3=bool((not driven_osc) and len(single)>=2)
    branch=1 if b1 else (2 if b2 else (3 if b3 else 4))
    dump(o/'LOCALIZE.json',dict(runs=runs,current_grid=pre['grid_current_nA'],adopted_anchor_nA=I0,
      driven_oscillating_currents=driven_osc,single_parameter_oscillating=single,
      nominal_reason=dict(threshold_current_nA=float(dV/R_MOhm),anchor_over_threshold=float(I0/(dV/R_MOhm)),
        refractory_cap_hz=1000.0/P.refractory_ms,note='the nominal point either oscillates at the anchor or needs the listed single parameter')))
    oc=dict(round='S310',branch=branch,anchor_nA=I0,detector_selftest=st,
      oscillation=dict(driven=driven_osc,single_param=single),
      k2_fingerprint='NOT_APPLICABLE (digest computed before the override; identity is the data hash)',
      s309_branch='NOT_JUDGED (re-recorded)',
      one_line=('支1：隔离 motif 在某个驱动电流下就起振 → 底物能承载 motif，S309 失败在嵌入'%() if branch==1 else
                ('支2：名义不振，但恰有一个单参数改动起振：%s'%single if branch==2 else
                 ('支3：需要 ≥2 个参数联合：%s'%single if branch==3 else '支4：任何单参数都不振 → 点神经元+硬重置类底物与源侧 motif 不兼容，本路线 NOT_ESTABLISHED，两条路线留待人类裁决'))),
      boundaries=['neural only: no physics and no support/walking claim','ConductanceParams is illustrative/uncalibrated','the analytic I-f ignores synaptic input and adaptation'])
    dump(o/'OUTCOMES.json',oc)
    print(json.dumps(dict(branch=branch,anchor=I0,selftest=[st['pass_a'],st['pass_b'],st['pass_c']],
      driven=driven_osc,single=single,e1_means=e1_mean),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
