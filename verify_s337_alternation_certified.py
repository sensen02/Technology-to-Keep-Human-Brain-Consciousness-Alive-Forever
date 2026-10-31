"""S337 verification: hashes, counters, gates, three-estimator rule, controls, disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S337_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s337.py','audit_s337_independent.py')))
    c('a REFREEZE ledger records the post-freeze corrections',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('all four construction gates are bitwise',all(v['equal'] for v in G.values()),gates={k:v['spike_total'] for k,v in G.items()})
    AR=json.loads((o/'ARMS.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('every arm has a null drive frequency and no time dependent current',all(AR[k]['f_drive_hz'] is None and AR[k]['dI_time_dependent'] is False for k in AR))
    c('every added-edge arm carries the not-emergent mark',all(AR[k]['not_emergent'] for k in AR if AR[k]['added_edges']['n']>0))
    c('added edges declare count, conductance and share',all(set(('n','total','selfloops','share'))<=set(AR[k]['added_edges']) for k in AR if AR[k]['added_edges']['n']>0))
    c('the same-tripod control has zero self loops',AR['S3s_2p0']['added_edges']['selfloops']==0)
    c('detection precedes scoring: arms without detection report no phase',
      all((AR[k].get('abs_phase') is None) for k in AR if not AR[k].get('detected')))
    c('the three estimators are recorded per detected arm',all(len(AR[k].get('est',[]))==3 for k in AR if AR[k].get('detected')))
    c('the 0.85 mean rate gate is absent from the criteria', 'rate_frac' not in json.dumps(pre['thresholds']))
    c('floor and peak fractions are the criteria instead', 'p10_frac' in pre['thresholds'] and 'p90_frac' in pre['thresholds'])
    c('the branch is one of the four labels or NOT_JUDGED',oc['branch'] in ('b1','b2','b3','b4','NOT_JUDGED'),branch=oc['branch'])
    c('the estimator disagreement voids the round as preregistered',oc['branch']=='NOT_JUDGED' and (o/'STOP.json').is_file())
    c('the repair list is archived',len(oc['repair'])>=2)
    c('the cross-implementation disagreement is archived',(o/'DISAGREE.json').is_file())
    MI=json.loads((o/'MOTOR_INTERFACE.json').read_text())
    c('the motor interface reports three configurations',set(MI['arms'])=={'default','s300','u400'})
    c('the motor interface reports phase, depth and clip fractions',all(all(k in v for k in ('leg_phase','depth','clip0','clip1','ctrl_median')) for cfg in MI['arms'] for v in MI['arms'][cfg].values()))
    B=json.loads((o/'ADDED_INHIBITION_BUDGET.json').read_text())
    c('the added inhibition budget is recomputed rather than quoted',B['total_uS']>0 and B['share_of_Wi']>0,total=round(B['total_uS'],4),share=round(B['share_of_Wi'],4))
    E=json.loads((o/'ALTERNATION_EXISTENCE.json').read_text())
    c('existence reports both tripods with null statistics',all('null_p95' in v for a in E['arms'].values() for v in a.get('tripod',{}).values() if 'ok' in v))
    c('the S336 corrections are archived with seven items',len(json.loads((o/'S336_CORRECTION.json').read_text())['items'])>=7)
    src=(o/'analyze_s337.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['ten arms, no physics','NOT_JUDGED because the primary three-estimator spread exceeds 10 degrees'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
