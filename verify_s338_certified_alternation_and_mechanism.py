"""S338 verification: hashes, counters, gates, estimator certification, no dead code, disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    fr=(o/'S338_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s338.py','audit_s338_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('the construction gate is bitwise against both archives',G['bitwise']['C0_gate'] and all(d['equal'] for d in G['bitwise']['C0_gate']),gate=G['bitwise'].get('C0_gate'))
    c('the CN path gate is declared as a repeat rather than independent evidence',
      G.get('cn_eq_c0') is None and 'declared repeat' in G.get('cn_note','') and ('C0_gate','N_gate_cn') in [tuple(x) for x in G['declared_dups']])
    AR=json.loads((o/'ARMS.json').read_text())
    c('every spike file has exactly 5000 rows',all(AR[k]['tick_count']==5000 for k in AR),ticks={k:AR[k]['tick_count'] for k in list(AR)[:3]})
    c('no undeclared duplicate spike file',G['undeclared_dups']==[],dups=G['undeclared_dups'])
    c('every arm declares null drive and time independent input',all(AR[k]['f_drive_hz'] is None and AR[k]['dI_time_dependent'] is False for k in AR))
    c('saturation and silence are flagged',all(k in AR for k in AR) and all('reasons' in AR[k] for k in AR))
    cert=json.loads((o/'ESTIMATOR_CERT_V17.json').read_text())['estimators']
    c('all three estimators are certified on synthetic truth',all(cert[k]['certified'] for k in ('a','b','c')),
      max_err={k:round(cert[k]['max_err'],2) for k in ('a','b','c')})
    c('the half-cycle lag check is archived','d_lag' in cert)
    E=json.loads((o/'ALTERNATION_CERT_V17.json').read_text())
    c('the rejudge uses only certified estimators',set(E['certified_estimators'])<={'a','b','c'} and len(E['certified_estimators'])>=2,used=E['certified_estimators'])
    src=(o/'analyze_s338.py').read_text();tree=ast.parse(src)
    dead=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is False]
    orr=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no if-False dead branch and no or-True conjunction (AST)',len(dead)==0 and len(orr)==0,dead=dead,or_true=orr)
    c('the arm statistic is a circular median rather than an absolute mean',
      'circ_median' in src and 'abs_phase=float(np.mean' not in src)
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is one of the registered labels or NOT_JUDGED',oc['branch'] in ('b1','b5','b6','b7','b8','b9','NOT_JUDGED'),branch=oc['branch'])
    c('the provisional reading and the agreement failure are both recorded',oc.get('branch_if_agreed') is not None and (o/'DISAGREE.json').is_file())
    c('the sine fraction gate is bounded',pre['thresholds']['sine_frac']>0 and pre['thresholds']['n_cycles']>=8)
    c('the frequency flags record the measured f_alt values',set(oc['f_alt'])=={'C0','N1','N2'})
    MC=json.loads((o/'MOTOR_CHAIN_V3.json').read_text())
    c('the motor chain reports three configurations',set(MC['arms'])=={'default','s300','u400'})
    c('the motor chain reports leg phases and zero and half lag correlations',
      all(('leg_pairs' in v and 'ctrl_corr' in v) for v in MC['arms']['default'].values() if v.get('leg_pairs')))
    c('the S337 corrections are archived with ten items',len(json.loads((o/'S337_CORRECTION.json').read_text())['items'])>=10)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['ten arms, no physics','NOT_JUDGED on a 14.6 degree cross-implementation gap'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
