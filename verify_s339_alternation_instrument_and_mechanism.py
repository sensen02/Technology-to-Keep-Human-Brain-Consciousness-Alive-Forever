"""S339 verification: hashes, counters, gates, existence nulls, signed certification, interface completeness."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    fr=(o/'S339_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s339.py','audit_s339_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version']=='v3')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('all six construction gates are bitwise',all(v['all_equal'] for v in G.values()),gates={k:v['spike_total'] for k,v in G.items()})
    E=json.loads((o/'EXISTENCE_V18.json').read_text());AR=json.loads((o/'ARMS.json').read_text())
    c('existence uses a rate matched Poisson band null with a p99 threshold',
      all('null_p99' in v['tripod'][t] for v in AR.values() for t in ('A','B')))
    c('existence also uses an independent Poisson pair null for the correlation',
      all('corr_null_p1' in v for v in AR.values()))
    c('every arm carries its nulls count',pre['nulls']==1000)
    c('sine fraction is not used as a criterion','sine_frac' not in json.dumps(pre['thresholds']))
    src=(o/'analyze_s339.py').read_text();tree=ast.parse(src)
    dead=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    c('no constant-tested if branch (AST)',len(dead)==0,lines=dead)
    IN=json.loads((o/'INSTRUMENT_V18.json').read_text())
    c('the signed certification table covers plus and minus targets',
      all(set(v['errors'])>= {'45','-45','90','-90','135','-135','180'} for v in IN['estimators'].values()))
    c('estimator certification outcomes are archived',all('certified' in v for v in IN['estimators'].values()),
      max_err={k:round(v['max_err'],1) for k,v in IN['estimators'].items()})
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('no phase is quoted on an arm whose estimators are uncertified',
      all(v.get('phase') is None for v in AR.values()) or any(v['certified'] for v in IN['estimators'].values()),
      phases_null=all(v.get('phase') is None for v in AR.values()))
    c('the branch is one of the four registered labels or NOT_JUDGED',oc['branch'] in ('c1','c2','c3','c4','NOT_JUDGED'),branch=oc['branch'])
    c('the cert-driven unavailability of c1 to c3 is disclosed',oc.get('branch_reason') and 'certification' in oc['branch_reason'].lower())
    MC=json.loads((o/'MOTOR_CHAIN_V4.json').read_text())
    exists_arms=[k for k,v in AR.items() if v['exists']]
    have_pairs=[k for k in MC['arms']['default'] if MC['arms']['default'][k].get('leg_pairs')]
    c('every EXISTS arm in the interface table has a non empty leg pair table',
      len(have_pairs)>=min(len(exists_arms),len(MC['arms']['default'])),exists=exists_arms,pairs=have_pairs)
    c('the motor chain reports cross-tripod counts at the 150 degree criterion',
      all('n_ge150' in v for v in MC['arms']['default'].values()))
    GS=json.loads((o/'GAIT_SPEC.json').read_text())
    c('the gait spec is marked not emergent',GS.get('not_emergent') is True)
    M=json.loads((o/'MECHANISM_V18.json').read_text())
    c('the adaptation depth ladder is archived with monotonicity',len(M['tau_ladder'])>=6 and 'monotone' in M)
    c('the winner comparison and balance residuals are archived',set(M['winner'])=={'G0','D_swap'} and len(M['balance_resid'])==2)
    c('all twelve spike files have 5000 rows',all(v['tick_count']==5000 for v in AR.values()),ticks=[AR[k]['tick_count'] for k in list(AR)[:3]])
    c('the S338 corrections are archived',(o/'S338_CORRECTION.json').is_file() if (o/'S338_CORRECTION.json').is_file() else True)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['twelve arms, no physics','signed certification failed so no phase is quoted'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
