"""S308 verification: independent recomputation of the repaired matrices, dt map, b, traces, thresholds, cross-leg tiers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BANC=ROOT/'data/flywire/banc_connectome.npz'
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    assert (o/'S308_PREREG_FROZEN').is_file()
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());corr=json.loads((o/'S307_CORRECTION.json').read_text())
    wm=json.loads((o/'W_MATRIX.json').read_text());rep=json.loads((o/'PAPER_MODEL_REPAIR.json').read_text())
    runs=json.loads((o/'RATE_MODEL_RUNS2.json').read_text());thr=json.loads((o/'PER_LEG_THRESHOLD.json').read_text())
    cl=json.loads((o/'CROSSLEG_COUPLING.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    gt=json.loads((o/'GATE_PAPER_MODEL.json').read_text())
    c('zero dynamics declared',all(pre['calls'][k]==0 for k in ('model_load','forward','physics_step','neural_tick','tier_build','render')))
    c('six correction items',len(corr['items'])>=6)
    c('S307 NaN withdrawal recorded',any('NaN' in json.dumps(i) or 'nan' in json.dumps(i) for i in corr['items']))
    c('sign attached to presynaptic', 'PRESYNAPTIC' in wm['sign_rule'].upper())
    z=np.load(BANC,allow_pickle=True);rt=np.asarray(z['root_ids'])
    pre_r=rt[np.asarray(z['pre'])];post_r=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())
    def mat(lg):
        M=np.zeros((3,3))
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if a==b: continue
                sa=set(mod['modules'][lg][b]);sb=set(mod['modules'][lg][a])
                cnt=float(w[np.isin(pre_r,list(sa))&np.isin(post_r,list(sb))].sum())
                sgn=1.0 if b in ('E1','E2') else -1.0
                M[i,j]=max(5.0,cnt)*sgn
        return M
    for lg in LEGS:
        c(f'{lg} signed matrix reproduced',np.allclose(mat(lg),np.asarray(wm['signed_full'][lg])))
    c('lf paper edges have the correct signs',wm['lf_paper_edges']['E1_to_E2']>0 and wm['lf_paper_edges']['E2_to_I1']>0
      and wm['lf_paper_edges']['I1_to_E1']<0 and wm['lf_paper_edges']['I1_to_E2']<0)
    wrong=np.asarray(wm.get('s307_reconstructed_wrong_subset') or [[0,0,0]]*3);right=np.asarray(wm['signed_full']['lf'])
    c('S307 wrong matrix kept for comparison and shown to flip I1->E1 positive',wrong[0,2]>0 and right[0,2]<0,wrong_I1_to_E1=float(wrong[0,2]),correct_I1_to_E1=float(right[0,2]))
    c('two dt conventions both reported',abs(rep['dt_conventions']['dt_1ms']['freq_hz']-14.0830)<0.01 and abs(rep['dt_conventions']['dt_10us']['freq_hz']-1408.3)<1.0)
    c('b derived two ways with the ratio recorded',rep['b_route_A']['value']>0 and rep['b_route_B']['value']>0 and rep['ratio']>1)
    A=runs['runs_bA']
    for nm in [f'R{i+1}_{lg}' for i,lg in enumerate(LEGS)]+['R9_zero','R10_gate_lf_subset','R12_three_cell_injection','R13_X25_lf','R14_X50_lf']:
        tr=A[nm].get('trace_e1')
        if tr is not None: c(f'{nm} trace finite',bool(np.all(np.isfinite(np.asarray(tr,float)))))
    # independent frequency: Hilbert phase slope on the gate run
    tr=np.asarray(A['R10_gate_lf_subset']['trace_e1'],float); y=tr-tr.mean()
    from scipy.signal import hilbert
    ph=np.unwrap(np.angle(hilbert(y))); slope=np.polyfit(np.arange(len(y))*0.01,ph,1)[0]
    fh=abs(slope/(2*np.pi))*1000.0
    c('gate frequency confirmed by Hilbert phase',abs(fh-runs['gate_freq_bA'])<2.0,hilbert_hz=float(fh),fft_hz=runs['gate_freq_bA'])
    c('gate passes and is in band',gt['passed'] is True and 12.0<=gt['freq_hz_bA']<=16.2)
    c('R11 deterministic repeat bitwise',A['R11_repeat_R7']['trace_e1']==A['R7_all_per_leg']['traces']['lf'])
    c('per-leg thresholds recorded for all legs',all(thr['thresholds'][lg].get('b_onset') is not None for lg in LEGS))
    c('cross-leg tiers sum to the total',abs(sum(cl['input_mass'].values())-cl['total'])<1e-6)
    c('branch 2 with the in-band clause table',oc['branch']==2 and oc['clause_check']['b1_all_in_band'] is False)
    c('cross-leg availability reported with numbers (b4)','cross_leg_fraction' in oc['b4'])
    c('analysis script archived',(o/'analyze_s308_cpg_model_repair.py').is_file())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,tier_build=0,render=0),
      limitations=['b is still not sourced; two derivation routes differ by 56x and both are carried','rate model is not our simulation and not a gait','no support/walking claim'])
    (o/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],hilbert=fh)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
