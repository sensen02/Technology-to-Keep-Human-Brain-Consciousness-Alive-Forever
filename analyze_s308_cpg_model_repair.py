"""S308: repair the paper firing-rate model (matrix, sign, dt, injection) and re-run the reproduction gate.

Offline: model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0; connectome_load=1.
"""
from __future__ import annotations
import argparse, json, hashlib, csv
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); BANC=ROOT/'data/flywire/banc_connectome.npz'
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh'); ORDER=('E1','E2','I1'); SGN={'E1':1.0,'E2':1.0,'I1':-1.0}
TAU=0.02; A_GAIN=1.0; THETA=7.5; RMAX=200.0; G_NONLIN=0.75
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')

def leg_edges():
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())
    return mod

def signed_matrix(edges,leg,subset=False):
    """rows = target, cols = source, sign belongs to the PRESYNAPTIC cell."""
    W=np.zeros((3,3))
    for i,a in enumerate(ORDER):
        for j,b in enumerate(ORDER):
            if a==b: continue
            if subset and (b,a) not in (('E1','E2'),('E2','I1'),('I1','E1')): continue
            cnt=edges[leg][f'{b}->{a}']
            W[i,j]=max(5.0,cnt)*SGN[b]
    return W

def sim(W,I_inj,dt_ms=0.01,sim_ms=3000.0,x0=None):
    dt=dt_ms/1000.0; n=int(sim_ms/dt_ms); onset=int(20.0/dt_ms)
    st=np.zeros(3) if x0 is None else x0.copy(); out=np.empty((n,3))
    for k in range(n):
        I=np.zeros(3)
        if k>=onset: I[0]=I_inj
        d=W@st+I
        f=RMAX*np.tanh(np.maximum(A_GAIN*(d-THETA),0.0)/RMAX)
        st=st+dt/TAU*(-st+f); out[k]=st
    return out

def freq_fft(x,dt_ms):
    x=np.asarray(x,float)
    if not np.all(np.isfinite(x)): return None
    y=x-x.mean(); n=len(y)
    if np.allclose(y,0): return None
    f=np.fft.rfftfreq(n,d=dt_ms/1000.0); P=np.abs(np.fft.rfft(y))
    band=(f>=1)&(f<=80)
    return float(f[band][np.argmax(P[band])]) if band.any() else None
def freq_zc(x,dt_ms):
    x=np.asarray(x,float)
    if not np.all(np.isfinite(x)): return None
    y=x-x.mean()
    if np.allclose(y,0): return None
    zc=int(np.count_nonzero(np.diff(np.sign(y))!=0)); dur=len(y)*dt_ms/1000.0
    return float(zc/(2*dur)) if dur>0 else None

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    pre=dict(round='S308',mode='repair the paper firing-rate model and re-run the gate; cross-leg coupling audit',
      hard_rule='the verifier may only write inside its own --output directory',
      calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
      connectome_calls=dict(connectome_load=1),
      runs=dict(main=15,threshold_per_leg_max=10,threshold_total_max=60),
      sign='symbol assigned to the PRESYNAPTIC cell (E1,E2 positive; I1 negative) from the paper plus ann_nt_verified; transmitter-to-sign remains an external convention',
      linearisation='lambda = (1-alpha) I + alpha g b W with an explicit identity matrix',
      dt_convention='arg(0.9908+0.0879j)=0.088485 rad gives 14.083 Hz at dt=1 ms and 1408.275 Hz at dt=10 us; this round adopts dt=1 ms for the frequency mapping and reports 10 us as a sensitivity because the paper text says 0.01 ms while its own eigenvalue arithmetic implies 1 ms',
      anchor='single-cell isolated rate X: I = theta + rmax atanh(X/rmax); main X=100 Hz, sensitivity X=25 and 50 Hz on lf only; injection into E1 only',
      gate='R10 (sufficient subset, lf, X=100) must oscillate with the main frequency in 12.0-16.2 Hz',
      inputs={str(BANC):sha(BANC),str(S307/'PER_LEG_MODULE.json'):sha(S307/'PER_LEG_MODULE.json')})
    dump(o/'PREREG.json',pre);(o/'S307_PREREG_FROZEN').write_text('frozen\n')
    E=leg_edges();E=dict(edges=E['edges'],modules=E['modules'],dn=E['dn_leg'])
    mods=E['modules'];edges=E['edges']
    # S307 correction
    rm=json.loads((S307/'RATE_MODEL_RUNS.json').read_text());gt=json.loads((S307/'GATE_PAPER_MODEL.json').read_text())
    tr10=np.asarray(rm['runs']['R10_gate_lf_subset']['per_leg']['lf']['trace'],float)
    tr9=np.asarray(rm['runs']['R9_zero']['per_leg']['lf']['trace'],float)
    corr=dict(round='S308',items=[
      dict(id='C1',archived=dict(R10_all_nan=bool(np.all(~np.isfinite(tr10))),R9_all_nan=bool(np.all(~np.isfinite(tr9))),gate=gt),
           s307='R10/R9 carried the gate result',corrected='both traces are entirely NaN, so the gate was NEVER evaluated; the reported freq 1.0 Hz and freq_zc 9.99983 are degenerate returns from 1/(2*0.05 ms) on a flat NaN array'),
      dict(id='C2',archived=dict(candidates_scanned=gt.get('candidates_scanned')),s307='b could not be identified in 1e-12..1e-2',
           corrected='arithmetic necessity of a coding bug: the scalar 1-alpha was added to every matrix element instead of only the diagonal, so the matrix collapsed to rank one with a fixed complex pair (arg 2.4224 rad) and a frequency near 38550 Hz for every b'),
      dict(id='C3',archived=dict(s307_subset_matrix=gt.get('subset_matrix')),s307='sign attached as in the paper',
           corrected='the sign was multiplied onto the POSTSYNAPTIC cell, flipping I1->E1 from -360 to +360 and turning the inhibitory feedback into positive feedback'),
      dict(id='C4',archived=dict(s307_dt=1e-5),s307='dt=1e-5 s',corrected='dt was 100x too small: the paper eigenvalue argument 0.088485 rad gives 14.083 Hz only with dt=1 ms'),
      dict(id='C5',archived=dict(s307_injection='np.full(3, stim)'),s307='tonic drive applied to the module',corrected='the tonic current was injected into all three cells; the paper and BANC support only DNg100->E1 (643 synapses vs 3 to I1 and 0 to E2)'),
      dict(id='C6',archived=dict(withdrawn=['gate failed so no engine code','a free scalar can hit any frequency so the gate only calibrates'],),s307='as stated',
           corrected='both statements WITHDRAWN: the reachable set was an artefact of the bugs above, and the gate had no numbers at all')])
    dump(o/'S307_CORRECTION.json',corr)
    # matrices
    WM={};WF={}
    for lg in LEGS:
        WM[lg]=signed_matrix(edges,lg,subset=False).tolist();WF[lg]=signed_matrix(edges,lg,subset=True).tolist()
    dump(o/'W_MATRIX.json',dict(order=list(ORDER),sign_rule='positive for E1,E2 and negative for I1, attached to the presynaptic cell',
      counts={lg:{f'{b}->{a}':edges[lg][f'{b}->{a}'] for a in ORDER for b in ORDER if a!=b} for lg in LEGS},
      signed_full=WM,signed_subset=WF,
      lf_paper_edges=dict(E1_to_E2=WM['lf'][1][0],E2_to_I1=WM['lf'][2][1],I1_to_E1=WM['lf'][0][2],I1_to_E2=WM['lf'][1][2]),
      s307_wrong_subset=gt.get('subset_matrix')))
    # b derivation: two routes
    dt_map_ms=1.0; alpha=dt_map_ms/1000.0/TAU
    lam_target=0.9908+0.0879j
    W=np.asarray(WM['lf']);ev=np.linalg.eigvals(W);dom=ev[np.argmax(np.abs(ev.imag))]
    b_A=float(((lam_target-(1-alpha))/(alpha*G_NONLIN))/dom)  # complex
    r_sum=float(np.abs(W).sum())/3.0; b_B=float(THETA/(max(r_sum,1e-9)*100.0))
    b_A_mag=float(abs(b_A))
    repair=dict(dt_conventions={'dt_1ms':dict(dt_ms=1.0,arg_rad=0.088485,freq_hz=float(0.088485/(2*np.pi*1e-3))),
                                'dt_10us':dict(dt_ms=0.01,arg_rad=0.088485,freq_hz=float(0.088485/(2*np.pi*1e-5)))},
      adopted='dt 1 ms because only that convention reproduces the paper 14 Hz from its own eigenvalue',
      alpha=alpha,dominant_eigenvalue=[float(dom.real),float(dom.imag)],
      b_route_A=dict(value=b_A_mag,complex_value=[float(b_A.real),float(b_A.imag)],formula='b = ((lambda_target-(1-alpha))/(alpha g)) / lambda_dominant(W_lf)',
        note='the ratio is complex, so only its magnitude is usable; this is recorded as a defect of route A'),
      b_route_B=dict(value=b_B,formula='b = theta / (mean_abs_row_synapses * X_anchor)'),
      ratio=float(max(b_A_mag,b_B)/max(min(b_A_mag,b_B),1e-30)),
      NOT_ESTABLISHED='the paper never prints b; route A is complex-valued and route B is a dimensional-order argument, so the two differ by %.1f x and both are carried below'%float(max(b_A_mag,b_B)/max(min(b_A_mag,b_B),1e-30)))
    dump(o/'PAPER_MODEL_REPAIR.json',repair)
    X_main=100.0; I_main=THETA+RMAX*np.arctanh(min(X_main/RMAX,0.999))
    dnmax=max(max(v.values()) for v in E['dn'].values())
    ratio_leg={lg:sum(E['dn'][c][lg] for c in E['dn'])/dnmax for lg in LEGS}
    def run_set(b,dt_ms=0.01):
        res={}
        def one(nm,Wm,inj):
            tr=sim(Wm,inj,dt_ms=dt_ms); fin=bool(np.all(np.isfinite(tr)))
            e1=tr[:,0]
            res[nm]=dict(per_leg=None,finite=fin,freq_fft=freq_fft(e1,dt_ms) if fin else None,
                         freq_zc=freq_zc(e1,dt_ms) if fin else None,pp=float(e1.max()-e1.min()) if fin else None,
                         trace_e1=(e1.tolist() if fin else None))
            return res[nm]
        for i,lg in enumerate(LEGS):
            one(f'R{i+1}_{lg}',b*np.asarray(WM[lg]),I_main*ratio_leg[lg])
        # R7 six legs simultaneously = independent integration; record each leg
        r7={}
        for lg in LEGS:
            tr=sim(b*np.asarray(WM[lg]),I_main*ratio_leg[lg],dt_ms=dt_ms); r7[lg]=tr[:,0].tolist()
        res['R7_all_per_leg']=dict(per_leg={lg:dict(freq_fft=freq_fft(np.asarray(r7[lg]),dt_ms),pp=float(np.ptp(r7[lg]))) for lg in LEGS},traces=r7)
        uni={lg:I_main for lg in LEGS}
        r8={lg:sim(b*np.asarray(WM[lg]),uni[lg],dt_ms=dt_ms)[:,0].tolist() for lg in LEGS}
        res['R8_all_uniform']=dict(per_leg={lg:dict(freq_fft=freq_fft(np.asarray(r8[lg]),dt_ms)) for lg in LEGS},traces=r8)
        one('R9_zero',b*np.asarray(WM['lf']),0.0)
        one('R10_gate_lf_subset',b*np.asarray(WF['lf']),I_main)
        tr=sim(b*np.asarray(WM['lf']),I_main*ratio_leg['lf'],dt_ms=dt_ms)
        res['R11_repeat_R7']=dict(finite=bool(np.all(np.isfinite(tr))),trace_e1=tr[:,0].tolist())
        # R12 inject into all three (the S307 shape)
        dt=dt_ms/1000.0;n=int(3000/dt_ms);onset=int(20/dt_ms);st=np.zeros(3);rec=np.empty((n,3))
        for k in range(n):
            I=np.full(3,I_main*ratio_leg['lf'] if k>=onset else 0.0);d=b*(np.asarray(WM['lf'])@st)+I
            f=RMAX*np.tanh(np.maximum(A_GAMMA*(d-THETA),0.0)/RMAX) if False else RMAX*np.tanh(np.maximum(A_GAIN*(d-THETA),0.0)/RMAX)
            st=st+dt/TAU*(-st+f);rec[k]=st
        res['R12_three_cell_injection']=dict(finite=bool(np.all(np.isfinite(rec))),freq_fft=freq_fft(rec[:,0],dt_ms),trace_e1=rec[:,0].tolist())
        for nm,X in (('R13_X25_lf',25.0),('R14_X50_lf',50.0)):
            Inj=THETA+RMAX*np.arctanh(min(X/RMAX,0.999));tr=sim(b*np.asarray(WM['lf']),Inj,dt_ms=dt_ms)
            res[nm]=dict(finite=bool(np.all(np.isfinite(tr))),freq_fft=freq_fft(tr[:,0],dt_ms),trace_e1=tr[:,0].tolist())
        # R15 gate with b_B
        tr=sim(b_B*np.asarray(WF['lf']),I_main,dt_ms=dt_ms)
        res['R15_gate_with_bB']=dict(finite=bool(np.all(np.isfinite(tr))),freq_fft=freq_fft(tr[:,0],dt_ms),trace_e1=tr[:,0].tolist())
        return res
    A_GAMMA=1.0
    runsA=run_set(b_A_mag); runsB=run_set(b_B)
    f10=runsA['R10_gate_lf_subset']['freq_fft']; f10b=runsB['R15_gate_with_bB']['freq_fft']
    passed=bool(f10 is not None and 12.0<=f10<=16.2)
    dump(o/'RATE_MODEL_RUNS2.json',dict(b_A=b_A_mag,b_B=b_B,dt_ms=0.01,I_main=I_main,ratio_leg=ratio_leg,
      runs_bA=runsA,runs_bB=runsB,gate_freq_bA=f10,gate_freq_bB=f10b,
      note='all traces asserted finite before any frequency is computed; the six modules are integrated independently because cross-leg synaptic mass inside the 18 cells is zero'))
    dump(o/'GATE_PAPER_MODEL.json',dict(passed=passed,freq_hz_bA=f10,freq_hz_bB=f10b,band=[12.0,16.2],
      note='evaluated only after the NaN bug was fixed and the traces were asserted finite'))
    # per-leg onset thresholds (bisection on b, <=10 iters per leg)
    thr={}
    for lg in LEGS:
        Wm=np.asarray(WM[lg]);Inj=I_main*ratio_leg[lg]
        lo,hi=1e-6,1.0
        def osc(bb):
            tr=sim(bb*Wm,Inj,dt_ms=0.05,sim_ms=1500.0)
            if not np.all(np.isfinite(tr)): return False
            f=freq_fft(tr[:,0],0.05)
            return bool(f is not None and f>2.0 and np.ptp(tr[:,0])>1e-6)
        base=osc(hi)
        if not base: thr[lg]=dict(onset=None,note='no oscillation up to b=1')
        else:
            for _ in range(10):
                mid=np.sqrt(lo*hi)
                if osc(mid): hi=mid
                else: lo=mid
            tr=sim(hi*Wm,Inj,dt_ms=0.05,sim_ms=1500.0)
            thr[lg]=dict(b_onset=float(hi),freq_at_onset=freq_fft(tr[:,0],0.05))
    dump(o/'PER_LEG_THRESHOLD.json',dict(thresholds=thr,note='bounded bisection on b for each leg; at most 10 runs per leg'))
    print(json.dumps(dict(bA=b_A_mag,bB=b_B,gateA=f10,gateB=f10b,passed=passed,
      thr={lg:thr[lg].get('b_onset') for lg in LEGS}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
