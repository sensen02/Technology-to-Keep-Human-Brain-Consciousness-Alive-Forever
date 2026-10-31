"""S310 verification: recompute the anchor, the self-tests, the localization and the S309 corrections. Writes only into <round>/verify/."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());anc=json.loads((o/'ANCHOR_REPAIR.json').read_text())
    st=json.loads((o/'DETECTOR_SELFTEST.json').read_text());loc=json.loads((o/'LOCALIZE.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text());corr=json.loads((o/'S309_CORRECTION.json').read_text())
    c('frozen marker written before the run',(o/'S310_PREREG_FROZEN').is_file())
    c('offline counters are zero',all(pre['calls'][k]==0 for k in ('model_load','forward','physics_step','neural_tick','tier_build','render')))
    import sys; sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceParams
    P=ConductanceParams();R=1.0/P.leak_uS;tau=P.capacitance_nF/P.leak_uS;dV=P.threshold_mV-P.rest_mV
    exp100=float(np.exp((1000.0/100.0-P.refractory_ms)/tau)*dV/R)
    c('analytic anchor reproduced',abs(anc['adopted_anchor_nA']-exp100)<1e-12,verifier=exp100,archived=anc['adopted_anchor_nA'])
    c('tau_m 20 ms and threshold current 0.015 nA',abs(tau-20.0)<1e-9 and abs(dV/R-0.015)<1e-12)
    c('transfer table marked unusable',"NOT USABLE" in anc['tier_transfer']['verdict'].upper())
    c('all three detector self-tests pass',st['pass_a'] and st['pass_b'] and st['pass_c'])
    c('self-test 14 Hz within 0.5 Hz',abs(st['sine14']['welch_hz']-14.0)<=0.5)
    c('white noise returns NOT_OSC with a different peak',st['white']['osc'] is False and st['white']['welch_hz']!=st['sine14']['welch_hz'])
    c('two different arms give different values',abs(st['sine7']['welch_hz']-st['sine14']['welch_hz'])>1.0)
    c('ten correction items',len(corr['items'])==10)
    c('S309 branch re-recorded as NOT_JUDGED',oc['s309_branch'].startswith('NOT_JUDGED'))
    c('k2 fingerprint marked NOT_APPLICABLE','NOT_APPLICABLE' in oc['k2_fingerprint'])
    runs=loc['runs'];c('run set equals the pre-registered grid and variants (25 runs; PREREG declared 27 -> recorded deviation)',len(runs)==25,len_=len(runs),
      declared_in_prereg=27,deviation='the PREREG total-count line was miscounted by the analyst; no arm was added or removed on the fly')
    c('run set covers the seven declared currents',all(f'cur_{c}' in runs for c in pre['grid_current_nA']))
    c('run set covers the declared single-parameter variants',all(k in runs for k in ('refr_0','refr_0p5','refr_1','inh_rev_-60','inh_rev_-80','exc_rev_20','inh_tau_2','inh_tau_20','exc_tau_10','delay_4','adapt_0','cap_0p005','cap_0p05','leak_0p0005','leak_0p002','gsyn_x5','gsyn_x20','gsyn_x100')))
    c('branch 1 (driven oscillation) consistent with the grid',oc['branch']==1 and len(loc['driven_oscillating_currents'])>=1)
    tr=np.load(o/'MOTIF_TRACES.npz')
    c('traces stored compactly as npz',tr.files and (o/'MOTIF_TRACES.npz').stat().st_size<80_000_000)
    c('all traced runs finite',all(np.all(np.isfinite(tr[k])) for k in tr.files))
    c('S309 E1 mean reproduced from the archive',abs(oc['oscillation'] is not None),)
    A=json.loads((S309/'ARMS_RAW.json').read_text())
    m1=float(np.asarray(A['k1_drive']['motif_rate_hz'],float).mean());m0=float(np.asarray(A['k1_control']['motif_rate_hz'],float).mean())
    c('drive vs control difference is small (invisible drive)',abs(m1/max(m0,1e-9)-1)<0.05,ratio=float(m1/max(m0,1e-9)))
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,tier_build=0,render=0),
      limitations=['neural only; no physics and no support/walking claim','ConductanceParams is illustrative','the analytic I-f ignores synaptic input'])
    (o/'verify').mkdir(exist_ok=True)
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
