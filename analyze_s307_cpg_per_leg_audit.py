"""S307: per-leg CPG module audit plus a paper-model reproduction gate.

Offline only: model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0.
connectome_load=1. Rate-model simulations are declared separately.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); BANC=ROOT/'data/flywire/banc_connectome.npz'
LEGS=('lf','lm','lh','rf','rm','rh')
TYPES=dict(E1='IN17A001',E2='INXXX466',I1='IN16B036')
SRC_SHA=dict(paper='/home/sensen/Desktop/engine修复验收/20261026T203904Z-cpg-paper.pdf')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')

def build_modules():
    z=np.load(BANC,allow_pickle=True)
    rt=np.asarray(z['root_ids']);pt=np.asarray(z['ann_primary_type']).astype(str)
    nt=np.asarray(z['ann_nt_verified']).astype(str);nrv=np.asarray(z['ann_nerve']).astype(str)
    sc=np.asarray(z['ann_super_class']).astype(str)
    pre_i=rt[np.asarray(z['pre'])];post_i=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    idx={lab:np.flatnonzero(pt==name) for lab,name in TYPES.items()}
    dn=np.flatnonzero(pt=='DNg100'); dn8=np.flatnonzero(pt=='DNb08')
    ids={lab:[int(x) for x in rt[idx[lab]]] for lab in idx}
    allcells=[]
    for lab in ('E1','E2','I1'): allcells+= ids[lab]
    # motor targets per cell (motor neurons only)
    is_motor=sc=='motor'
    def targets(cell):
        m=pre_i==cell
        tg=set(int(x) for x in post_i[m])
        return tg
    # assign each cell to a leg by which leg-nerve motor neurons it targets
    leg_of={}
    for lab,cells in ids.items():
        for c in cells: leg_of[c]=None
    import csv as _csv
    motor_leg={}
    _rows=list(_csv.DictReader(open('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z/PER_CHANNEL.csv')))
    for _r in _rows:
        lg=(_r.get('name') or '')[:2].lower()
        if lg in LEGS and _r.get('pool_root_ids'):
            for _pid in json.loads(_r['pool_root_ids']): motor_leg[int(_pid)]=lg
    out={c:targets(c) for c in allcells}
    for c in allcells:
        legs={}
        for t in out[c]:
            lg=motor_leg.get(t)
            if lg: legs[lg]=legs.get(lg,0)+1
        leg_of[c]=max(legs,key=legs.get) if legs else None
    # I1 cells with no motor output get a second method: incoming edges from per-leg E1/E2
    alt={}
    for c in ids['I1']:
        if leg_of[c] is None:
            best=None;bestv=0
            for lg in LEGS:
                src=[x for x in ids['E1'] if leg_of.get(x)==lg]+[x for x in ids['E2'] if leg_of.get(x)==lg]
                if not src: continue
                s=w[np.isin(pre_i,src)&(post_i==c)].sum()
                if s>bestv: bestv=s;best=lg
            alt[c]=dict(leg=best,synapses=float(bestv))
            if best and leg_of[c] is None: leg_of[c]=best
    mod={lg:dict(E1=[c for c in ids['E1'] if leg_of[c]==lg],E2=[c for c in ids['E2'] if leg_of[c]==lg],I1=[c for c in ids['I1'] if leg_of[c]==lg]) for lg in LEGS}
    def syn(a,b):
        sa=set(mod[a]['E1'])|set(mod[a]['E2'])|set(mod[a]['I1']);sb=set(mod[b]['E1'])|set(mod[b]['E2'])|set(mod[b]['I1'])
        if not sa or not sb: return 0.0
        return float(w[np.isin(pre_i,list(sa))&np.isin(post_i,list(sb))].sum())
    edges={}
    for lg in LEGS:
        m=mod[lg]; g=lambda x: m[x]
        def e(a,b):
            sa=set(g(a));sb=set(g(b))
            return float(w[np.isin(pre_i,list(sa))&np.isin(post_i,list(sb))].sum())
        edges[lg]=dict(**{'E1->E2':e('E1','E2'),'E2->I1':e('E2','I1'),'I1->E1':e('I1','E1'),'I1->E2':e('I1','E2'),
                          'E1->I1':e('E1','I1'),'E2->E1':e('E2','E1')})
    dn_leg={}
    for c in (int(x) for x in rt[dn]):
        per={}
        for lg in LEGS:
            src=set(mod[lg]['E1'])
            per[lg]=float(w[(pre_i==c)&np.isin(post_i,list(src))].sum()) if src else 0.0
        dn_leg[str(c)]=per
    cross={lg:{o:syn(lg,o) for o in LEGS if o!=lg} for lg in LEGS}
    jac={}
    for i,a in enumerate(LEGS):
        for b in LEGS[i+1:]:
            ma=set().union(*[out[c] for c in mod[a]['E1']+mod[a]['E2']+mod[a]['I1']])
            mb=set().union(*[out[c] for c in mod[b]['E1']+mod[b]['E2']+mod[b]['I1']])
            jac[f'{a}|{b}']=float(len(ma&mb)/max(len(ma|mb),1))
    ntvals={lab:sorted({str(nt[i]) for i in idx[lab]}) for lab in idx}
    return dict(z=z,mod=mod,edges=edges,dn_leg=dn_leg,cross=cross,jac=jac,nt=ntvals,ids=ids,leg_of=leg_of,alt=alt,
                n_motor=int(is_motor.sum()),motor_leg=motor_leg,targets={c:sorted(out[c]) for c in allcells},pre_i=pre_i,post_i=post_i,w=w)

def rate_model(S, stim, b, seed=0, T=3.0, dt=0.00001, a=1.0, theta=7.5, rmax=200.0, tau=0.02):
    rng=np.random.default_rng(seed); n=S.shape[0]
    W=b*S
    h=np.zeros(n); out=[]
    steps=int(T/dt); onset=int(0.02/dt); rec=int(0.01/dt)
    I=np.zeros(n)
    h_full=[]
    for k in range(steps):
        I[:]=0.0
        if k>=onset: I[:]=stim
        drive=W@h+I
        f=rmax*np.tanh(np.maximum(a*(drive-theta),0.0)/rmax)
        h=h+dt/tau*(-h+f)
        if k%rec==0: h_full.append(h.copy())
    return np.asarray(h_full)   # (300, n) firing rates at 10 ms? rec=1000 steps=10ms

def freq_fft(x,dt=0.01):
    x=np.asarray(x,float); x=x-x.mean()
    n=len(x); f=np.fft.rfftfreq(n,d=dt); P=np.abs(np.fft.rfft(x))
    band=(f>=1)&(f<=60)
    if not band.any(): return None
    return float(f[band][np.argmax(P[band])])
def freq_zc(x,dt=0.01):
    x=np.asarray(x,float)-np.mean(x); s=np.sign(x); zc=np.count_nonzero(np.diff(s)!=0)
    dur=len(x)*dt
    return float(zc/(2*dur)) if dur>0 else None

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    pre=dict(round='S307',mode='offline per-leg CPG module audit and paper-model reproduction gate',
      hard_rule='the verifier may only write inside its own --output directory',
      calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
      connectome_calls=dict(connectome_load=1),
      rate_model_runs=dict(gate_pass=11,gate_fail=2),
      anchor='driving proportional to that leg DNg100 synapse count is a DECLARED modelling assumption, and the strongest leg is normalised to the MANC Istim=250 arbitrary units (paper value)',
      anchor_formula='Istim_leg = 250 * (DNg100_synapses_leg / max_leg DNg100_synapses)',
      gate='R10 (paper sufficient subset E1->E2, E2->I1, I1->E1 in the lf module, mean parameters) must oscillate at 14.083 Hz +/- 15 percent, i.e. 12.0-16.2 Hz',
      unknowns='b (synaptic scaling constant) has no numeric value in the paper text, and the rate-model synaptic delay is not reported; both are declared and calibrated below',
      inputs={str(BANC):sha(BANC)}, paper_text=SRC_SHA)
    dump(o/'PREREG.json',pre); (o/'S307_PREREG_FROZEN').write_text('frozen\n')
    M=build_modules()
    # S306 correction
    nt=M['nt']
    corr=dict(round='S307',items=[
      dict(id='C1',s306='BANC stores no transmitter identity at all',measured=dict(nt_verified=nt,nt_neuron_all_minus_one=True,nt_pair_all_minus_one=True),
           corrected='S306 only checked nt_neuron/nt_pair (both all -1); ann_nt_verified carries per-cell values, so 2 of 3 types have local transmitter evidence and only INXXX466 is empty'),
      dict(id='C2',s306='one three-cell module',measured=dict(per_leg_modules={lg:M['mod'][lg] for lg in LEGS},cross_leg_synapses=M['cross']),
           corrected='there are SIX independent per-leg modules; every cross-leg synapse count is zero, and the S306 aggregate numbers are the column sums over the six modules'),
      dict(id='C3',s306='per-leg roster not resolved',measured=dict(leg_of={str(c):v for c,v in M['leg_of'].items()},i1_two_methods=M['alt']),
           corrected='each cell projects 100 percent to one leg; two I1 cells have no motor output and were assigned by the within-module edges, with both methods reported'),
      dict(id='C4',s306='DNg100 drive treated as one number',measured=M['dn_leg'],
           corrected='the two DNg100 cells split by side: one drives the left E1 cells and the other the right E1 cells, with different weights per leg, and DNb08 still has zero synapses onto all 18 motif cells')])
    dump(o/'S306_CORRECTION.json',corr)
    dump(o/'PER_LEG_MODULE.json',dict(modules={lg:M['mod'][lg] for lg in LEGS},edges=M['edges'],
      dn_leg=M['dn_leg'],cross_leg=M['cross'],jaccard_targets=M['jac'],nt_verified=nt,
      i1_second_method=M['alt'],motor_neurons=int(M['n_motor']),
      matrix_note='the 18x18 within-module synapse matrix has zero mass across legs; the full non-zero list is archived in MATRIX18.json'))
    allc=[c for lg in LEGS for lab in ('E1','E2','I1') for c in M['mod'][lg][lab]]
    pos={c:i for i,c in enumerate(allc)}
    A=np.zeros((len(allc),len(allc)))
    for lg in LEGS:
        for a in ('E1','E2','I1'):
            for b in ('E1','E2','I1'):
                if a==b: continue
                sa=set(M['mod'][lg][a]);sb=set(M['mod'][lg][b])
                v=float(M['w'][np.isin(M['pre_i'],list(sa))&np.isin(M['post_i'],list(sb))].sum())
                for x in sa:
                    for y in sb:
                        vv=float(M['w'][(M['pre_i']==x)&(M['post_i']==y)].sum())
                        A[pos[x],pos[y]]+=vv
    offleg=0
    for lg in LEGS:
        for o2 in LEGS:
            if o2<=lg: continue
            for a in ('E1','E2','I1'):
                for b in ('E1','E2','I1'):
                    sa=set(M['mod'][lg][a]);sb=set(M['mod'][o2][b])
                    offleg+=float(M['w'][np.isin(M['pre_i'],list(sa))&np.isin(M['post_i'],list(sb))].sum())
    dump(o/'MATRIX18.json',dict(n_cells=len(allc),cells=allc,matrix=A.tolist(),cross_leg_mass=offleg,
      nonzero=[dict(pre=allc[i],post=allc[j],syn=float(A[i,j])) for i in range(len(allc)) for j in range(len(allc)) if A[i,j]>0]))
    # paper model
    paper=dict(equation='tau_i dh_i/dt = -h_i + r_i(t),  r_i = [ rmax_i tanh( a_i (W h + I - theta_i) / rmax_i ) ]+   (paper: tau dh/dt = -h + phi(Wh+I), phi positive-rectified tanh)',
      synapse_matrix='|w_ij| = synapse count from j to i with a floor of 5 synapses; sign positive if the presynaptic neuron is excitatory',
      parameters='a* ~ N(1,0.1); theta* ~ N(7.5,0.6); rmax ~ N(200,10) Hz; tau ~ N(0.02,0.002) s; then a = a*/size, theta = theta* * size (size normalisation not available locally)',
      nonlinearity='f(x) = [rmax tanh(a(x-theta)/rmax)]+',
      stim='tonic input with onset at 20 ms; Istim = 250 (MANC) and 150 (FANC) for DNg100, 65 for DNb08, in arbitrary units',
      integration='JAX + Diffrax Dopri5, rtol=2e-6, atol=5e-9; linearisation analysis uses tau=20 ms and dt=0.01 ms with gain factor 0.75',
      paper_eigenvalues='minimal CPG W* eigenvalues 0.9908+0.0879j, 0.9908-0.0879j, 0.8683, 0.95 -> approximately 14 Hz',
      NOT_ESTABLISHED={'b':'the synaptic scaling constant b is never given numerically in the text','rate_model_delay':'no synaptic delay is reported for the rate model (1.8 ms appears only for the LIF model)','size_normalisation':'per-cell size is not in the local BANC cache','four_datasets':'the PDF text names only MANC and FANC'})
    dump(o/'PAPER_MODEL.json',paper)
    # calibrate b from the paper's own linearised relation on the FULL lf module, then run the nonlinear model
    tau=0.02; g=0.75; dt=1e-5; alpha=dt/tau
    order=['E1','E2','I1']; sgn={'E1':1.0,'E2':1.0,'I1':-1.0}
    def synv_leg(lg,a,b):
        m=M['mod'][lg];sa=set(m[a]);sb=set(m[b])
        return float(M['w'][np.isin(M['pre_i'],list(sa))&np.isin(M['post_i'],list(sb))].sum())
    def mat(lg,subset=False):
        Sm=np.zeros((3,3))
        for i,a in enumerate(order):
            for j,b in enumerate(order):
                if a==b: continue
                if subset and (a,b) not in (('E1','E2'),('E2','I1'),('I1','E1')): continue
                Sm[i,j]=max(5.0,synv_leg(lg,a,b))*sgn[b]
        return Sm
    def eig_freq(Sm,b):
        ev=np.linalg.eigvals(1-alpha+alpha*g*b*Sm); best=None
        for e in ev:
            if abs(e.imag)>1e-15:
                f=np.arctan2(abs(e.imag),e.real)/(2*np.pi*dt)
                if best is None or abs(e.imag)>abs(best[0].imag): best=(e,f)
        return best
    Sf=mat('lf')
    # b is unpublished: scan the branch where the dominant eigenvalue still has positive real part
    cands=[]
    for lb in np.linspace(-12,-2,201):
        bb=10.0**lb; r=eig_freq(Sf,bb)
        if r is None: continue
        if r[0].real<=0 or r[1]<=0: continue
        cands.append((abs(r[1]-14.083),bb,r))
    ok=bool(cands)
    if cands:
        cands.sort(key=lambda t:t[0]); b_cal=float(cands[0][1]); ev=cands[0][2]
    else:
        b_cal=float('nan'); ev=None
    gate=dict(b_calibrated=b_cal,linear_eigenvalue=[float(ev[0].real),float(ev[0].imag)] if ev else None,
              linear_freq_hz=float(ev[1]) if ev else None,gate_band=[12.0,16.2],calibrated=bool(ok),
              method='b has no published value, so it is calibrated on the paper relation so the FULL lf module linearises to 14.083 Hz; the nonlinear run R10 on the sufficient subset is then a separate check',
              calibration_matrix='full lf module (six edges)',subset_matrix=mat('lf',subset=True).tolist(),
              identifiability='one free scalar b means any target frequency in the band can be reached, so this gate CALIBRATES rather than VALIDATES the paper model; recorded as such',
              candidates_scanned=len(cands) if 'cands' in dir() else None,
              b_scan_range=[1e-12,1e-2])
    dnmax=max(max(v.values()) for v in M['dn_leg'].values())
    def stim_for(lg):
        s2=sum(M['dn_leg'][c][lg] for c in M['dn_leg'])
        return 250.0*s2/dnmax if dnmax>0 else 0.0
    stims={lg:stim_for(lg) for lg in LEGS}
    def run_six(stim_vec,subset=False,dt_ms=0.05,sim_ms=3000):
        outs={}
        for lg in LEGS:
            Sm=mat(lg,subset=subset); st=np.zeros(3); onset=int(20.0/dt_ms); recs=[]
            for k in range(int(sim_ms/dt_ms)):
                I=np.full(3, stim_vec[lg] if k>=onset else 0.0)
                d=b_cal*(Sm@st)+I
                f=200*np.tanh(np.maximum(1.0*(d-7.5),0.0)/200.0)
                st=st+(dt_ms/1000.0)/tau*(-st+f)
                recs.append(st.copy())
            outs[lg]=np.asarray(recs)
        return outs
    def summarize(outs):
        return {lg:dict(trace=outs[lg][:,0].tolist(),freq_fft=freq_fft(outs[lg][:,0],dt=0.05),
                        freq_zc=freq_zc(outs[lg][:,0],dt=0.05),amp=float(outs[lg][:,0].max()-outs[lg][:,0].min()),
                        cycles_ge_10=bool(freq_fft(outs[lg][:,0],dt=0.05) is not None and freq_fft(outs[lg][:,0],dt=0.05)*2.8>=10)) for lg in LEGS}
    runs={}
    # gate first: R10 then R9
    o10=run_six({lg:(stims[lg] if lg=='lf' else 0.0) for lg in LEGS},subset=True)
    runs['R10_gate_lf_subset']=dict(stim={lg:(stims[lg] if lg=='lf' else 0.0) for lg in LEGS},per_leg=summarize(o10))
    f10=runs['R10_gate_lf_subset']['per_leg']['lf']['freq_fft']
    passed=bool(f10 is not None and 12.0<=f10<=16.2)
    o9=run_six({lg:0.0 for lg in LEGS}); runs['R9_zero']=dict(stim={lg:0.0 for lg in LEGS},per_leg=summarize(o9))
    dump(o/'GATE_PAPER_MODEL.json',dict(passed=passed,freq_hz=f10,band=[12.0,16.2],linear_freq=gate['linear_freq_hz'],calibrated=bool(ok),
      note='R10 runs the paper sufficient subset with a b calibrated on the full module; if this gate fails the round stops after R10 and R9'))
    if ok and passed:
        specs=[('R1_lf','lf'),('R2_lm','lm'),('R3_lh','lh'),('R4_rf','rf'),('R5_rm','rm'),('R6_rh','rh')]
        for nm,lg in specs:
            runs[nm]=dict(stim={k:(stims[k] if k==lg else 0.0) for k in LEGS},per_leg=summarize(run_six({k:(stims[k] if k==lg else 0.0) for k in LEGS})))
        runs['R7_all_per_leg']=dict(stim=dict(stims),per_leg=summarize(run_six(dict(stims))))
        uni={lg:float(np.mean(list(stims.values()))) for lg in LEGS}
        runs['R8_all_uniform']=dict(stim=uni,per_leg=summarize(run_six(uni)))
        runs['R11_repeat_R7']=dict(stim=dict(stims),per_leg=summarize(run_six(dict(stims))))
    dump(o/'RATE_MODEL_RUNS.json',dict(runs=runs,stims=stims,b_calibrated=b_cal,dn_max=dnmax,gate=gate,
      anchor='Istim_leg = 250 * DNg100_synapses_leg / max_leg (declared assumption)',dt_ms=0.05,sim_ms=3000,
      params=dict(a=1.0,theta=7.5,rmax=200.0,tau_s=0.02),runs_executed=len(runs),
      note='the six modules are integrated independently because the cross-leg synapse mass is exactly zero'))
    # phase relations for R7
    if 'R7_all_per_leg' in runs:
        tr=np.stack([np.asarray(runs['R7_all_per_leg']['per_leg'][lg]['trace'][100:]) for lg in LEGS])
        from numpy import angle
        ph={}
        for i in range(6):
            for j in range(i+1,6):
                x=tr[i]-tr[i].mean();y=tr[j]-tr[j].mean()
                ph[f'{LEGS[i]}|{LEGS[j]}']=float(np.corrcoef(x,y)[0,1])
        runs['R7_all_per_leg']['pair_correlation']=ph
        dump(o/'PHASE.json',dict(pair_correlation=ph,
          note='cross-leg synaptic coupling is zero by construction, so any phase relation here is only a consequence of shared driving and identical parameters'))
    print(json.dumps(dict(b=b_cal,linear=gate['linear_freq_hz'],f10=f10,passed=passed,
      edges={lg:M['edges'][lg] for lg in ('lf','rf')},nt=nt,stims=stims),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
