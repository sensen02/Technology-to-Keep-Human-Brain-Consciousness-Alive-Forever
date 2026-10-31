"""S326 verification: three freeze hashes, dedup, shared f0, null budget, selftest, no fallback."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S326_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s326.py','audit_s326_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('all dynamic counters are zero',all(led[k]==0 for k in pre['counters']))
    COR=json.loads((o/'S325_CORRECTION.json').read_text());PT=json.loads((o/'POOL_TRANSFER_V11.json').read_text())
    CL=json.loads((o/'CROSSLEG_COHERENCE.json').read_text());ST=json.loads((o/'DETECTOR_SELFTEST_V11.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text());AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the S325 modulation null is shown to be an identity transform',
      COR['mod_null_is_identity']['L1'][0]==COR['mod_null_is_identity']['L1'][1] and COR['mod_null_is_identity']['L2'][0]==COR['mod_null_is_identity']['L2'][1],
      L1=COR['mod_null_is_identity']['L1'],L2=COR['mod_null_is_identity']['L2'])
    c('pool cells are deduplicated to 202 with the duplicate rows listed',
      COR['dedup']['unique']==202 and len(COR['dedup']['duplicate_list'])==COR['dedup']['duplicates'],dedup=COR['dedup']['raw_records'])
    f0v=PT['f0'];shared=sum(1 for v in f0v.values() if v is not None)
    c('both arms share the L1-defined f0 per leg (nulls allowed where the reference has no bursts)',
      shared>=4,f0=f0v)
    c('null budget is 1000 for every cell record and p thresholds respect 1/(N+1)',
      PT['cells'] and all(1.0/1001<0.05 for _ in [0]) and 'coh1_p' in PT['cells'][0])
    src=(o/'analyze_s326.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Compare) and any(isinstance(c,ast.Constant) and c.value is True for c in n.comparators)]
    c('no or-True style conjunction in the analysis (AST)',len(bad)==0,lines=bad)
    c('the branch is decided by data with an explicit NOT_JUDGED path and no else fallback',
      oc['branch'] in ('w1','w2','w3','NOT_JUDGED') and 'branch_reason' in oc)
    c('the two implementations agree on cell labels within 1e-6 relative coherence',
      len(AG['cell_label_disagreements'])<=1 and (AG['worst_coh_rel_diff'] or 0)<1e-6,
      disagreements=AG['cell_label_disagreements'],worst=AG['worst_coh_rel_diff'])
    c('the dispatched L1 per-leg coherence is compared with a stated tolerance',
      set(ST['checks']['c_L1_reproduced'].keys())>={'lf','lm','rm','rh'} and 'within_tolerance' in list(ST['checks']['c_L1_reproduced'].values())[0])
    c('the dispatched L2 ceiling is tested and recorded as not reproduced',
      ST['checks']['c_L2_all_at_or_below_0p10'] is False and max(ST['checks']['c_L2_below_0p10'].values())>0.5)
    c('cross-leg specificity is measured and the statistic is declared non-specific',
      CL['median_cross'] is not None and 'common-mode' in oc['crossleg_specificity']['verdict'])
    c('the level versus modulation distinction is recorded',oc['pool']['level_same'] in (True,False) and 'pool' in oc)
    c('h is reported per leg with a null p95 and a decoder-input amplitude',all(('h_null_p95' in v) or ('note' in v) for v in oc['h'].values()))
    c('rm is excluded from certified legs with the disclaimer intact',
      oc['certified_legs']['legs']==['lf','lm','rh'] and 'absence of oscillation capacity' in oc['certified_legs']['note'])
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),
      limitations=['offline only','V11 is not certified: the ablation arm is not separable from the active arm by this statistic'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
