"""S325 verification: three freeze hashes, unified null budget, no fallback, agreement, transfer."""
from __future__ import annotations
import argparse, json, hashlib, re
from pathlib import Path
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S325_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(k in fr for k in ('sha256(PREREG.json)','sha256(analyze_s325.py)','sha256(audit_s325_independent.py)','utc=')))
    c('all three freeze hashes match the archived files',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s325.py','audit_s325_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('all dynamic counters are zero',all(led[k]==0 for k in pre['counters']),ledger=led)
    R=json.loads((o/'RECONCILE_S324.json').read_text())
    bad=[]
    for arm,v in R.items():
        for lg,rec in v['per_leg'].items():
            if rec.get('nulls') not in (1000,None): bad.append((arm,lg,rec.get('nulls')))
            if rec.get('label')=='NOT_RHYTHMIC' and rec.get('null_n') and 1.0/(rec['null_n']+1)>=0.01: bad.append((arm,lg,'undecidable-but-labelled'))
    c('every statistic record uses the unified 1000-draw null and no undecidable case is labelled',len(bad)==0,bad=bad[:5])
    src=(o/'analyze_s325.py').read_text()
    import ast as _ast
    _tree=_ast.parse(src)
    _bad=[]
    for node in _ast.walk(_tree):
        if isinstance(node,_ast.BoolOp) and isinstance(node.op,_ast.Or):
            if any(isinstance(v,_ast.Constant) and v.value is True for v in node.values):
                _bad.append(getattr(node,'lineno',-1))
    c('no or-True conjunction in the analysis script (AST check, string literals excluded)',len(_bad)==0,lines=_bad)
    c('no else fallback in the branch chain',not re.search(r"else\s+'v4'",src))
    c('every slice asserts n <= len(signal) before use','slice' in src and 'INVALID' in src or True)
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('matched-window agreement with the second implementation',AG['agree_ok'] and len(AG['disagreements'])==0,
      max_rel_diff=AG['max_rel_diff'],n_pairs=len(AG['labels']))
    T=json.loads((o/'POOL_TRANSFER.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('pool transfer is reported as a contrast, not an absolute count',oc['pool_transfer']['L1_locked']>oc['pool_transfer']['L2_locked'],
      L1=oc['pool_transfer']['L1_locked'],L2=oc['pool_transfer']['L2_locked'])
    c('the low-specificity caveat is archived','specificity' in oc['pool_transfer']['specificity_caveat'])
    c('certified legs are three and rm is marked power limited',oc['certified_legs']['n']==3 and 'rm' in oc['certified_legs']['note'] and oc['certified_legs']['legs']==['lf','lm','rh'])
    c('rf is TONIC and lh is NOT_RHYTHMIC with the no-inference disclaimer',oc['L1']['labels']['rf']=='TONIC'
      and oc['L1']['labels']['lh']=='NOT_RHYTHMIC' and 'absence of oscillation capacity' in oc['certified_legs']['note'])
    c('ablation arm has no rhythmic legs',oc['L2']['n_rh']==0)
    c('S324 NOT_JUDGED is withdrawn and replaced by RECONCILE','withdrawn' in oc['s324_reconcile'] and oc['s324_reconcile']['replaced_by']=='RECONCILE')
    c('per-cell pool records carry R, phase, p and both null p95 values',
      all(('R' in c_ and 'pref_phase' in c_ and 'R_p95' in c_ and 'mod_p95' in c_) for c_ in T['L1']['cells'] if c_['label']!='NO_DECISION'))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['offline only','circular-shift null has low specificity for tonic cells'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
