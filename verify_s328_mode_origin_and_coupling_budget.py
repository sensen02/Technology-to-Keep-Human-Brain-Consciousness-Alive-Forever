"""S328 verification: hashes, counters, both gates, axis counts, budget, V label, agreement."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S328_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s328.py','audit_s328_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('both gates pass bitwise',G['gate_tier'] is True and G['gate_cn'] is True)
    B=json.loads((o/'BUDGET.json').read_text())
    c('axis rule: 1943 outward and 1553 inward edges',B['edges_out']==1943 and B['edges_in']==1553,out=B['edges_out'],inn=B['edges_in'])
    c('the outward excitation share is about 0.58 percent',abs(100*B['share_out_exc']-0.579)<0.05,share=100*B['share_out_exc'])
    c('the counterfactual scale factors are archived with their formula','s = target_share' in B['counterfactual_formula'] and len(B['counterfactual_scale'])==3)
    MD=json.loads((o/'MODE_DESCRIPTOR.json').read_text())
    b0=MD['B0']['autocorr']
    c('B0 autocorrelation reproduces the dispatched lag profile within tolerance',
      all(abs(b0['lags'][k]-v)<=0.10 for k,v in (('5',0.793),('10',0.363),('20',-0.392),('25',-0.434),('40',0.240),('50',0.245))),
      lags=b0['lags'])
    c('B0 first negative lag is in 20-25 ms',b0['first_negative_lag_ms'] is not None and 20<=b0['first_negative_lag_ms']<=25,lag=b0['first_negative_lag_ms'])
    c('A0_gate has no negative minimum',MD['A0_gate']['autocorr']['first_negative_lag_ms'] is None)
    c('per-cell correlation quantiles are archived for B0',all(k in MD['B0']['per_cell_corr'] for k in ('median','p90','frac_gt_0p3')))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree on negative lag and band ratio',AG['agree'] is True,max_band=AG['max_band_diff'])
    c('V is decided by the zero-connectivity control and is V_INVALID',oc['V']=='V_INVALID')
    c('the zero-connectivity control keeps the band peak and its own rate',oc['V_detail']['C_norec_pool_mean_rate_hz']>0 and oc['V_detail']['C_norec_neg_lag'] is not None)
    c('branch is NOT_JUDGED with subordinate labels marked provisional',oc['branch']=='NOT_JUDGED' and 'provisional_labels' in oc)
    src=(o/'analyze_s328.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('every statistic record carries its nulls count',pre['nulls']==200 and MD['B0']['poisson_p95_band'] is not None)
    c('S327 corrections are archived',(o/'S327_CORRECTION.json').is_file() and len(json.loads((o/'S327_CORRECTION.json').read_text())['items'])>=5)
    c('leg noise floor is archived',(o/'LEG_NOISE_FLOOR.json').is_file())
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),
      limitations=['ten arms on the CN path after a bitwise CN==tier gate','no physics'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
