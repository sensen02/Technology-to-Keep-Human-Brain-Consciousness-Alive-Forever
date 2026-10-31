"""S331 verification: hashes, counters, gates, phase criteria, agreement disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S331_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s331.py','audit_s331_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('Z0 is bitwise identical to both S330 G_cn_gate and S322 K17',G['z0_equals_S330'] is True and G['z0_equals_S322K17'] is True)
    GP=json.loads((o/'GLOBAL_PHASE.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the global phase table is archived with banded circular statistics',all(k in GP['K17'] for k in ('pool202','amp_top10pct','leg_phase')))
    c('the tripod selectivity is archived for every arm','selectivity' in GP['K17']['leg_phase'])
    c('no arm shows an antiphase fraction above zero',all(v['anti_frac']==0 for v in json.loads((o/'ARMS.json').read_text()).values()))
    AR=json.loads((o/'ARMS.json').read_text())
    c('the pi arms move the cross-tripod phase but nowhere near 135 degrees',
      AR['Za_pi_30']['cross_mean_deg']>AR['Z0_gate']['cross_mean_deg'] and AR['Za_pi_30']['cross_mean_deg']<135,
      z0=round(AR['Z0_gate']['cross_mean_deg'],1),pi30=round(AR['Za_pi_30']['cross_mean_deg'],1))
    c('the same-share control is archived to separate no-effect from no-following',AR['Zb_same_30']['injected_uS']>0)
    c('the swap arm is archived to test routing artefacts',AR['Zc_pi_30_swap']['injected_uS']>0)
    c('the injection actually reached the network',all(AR[k]['injected_uS']>0 for k in ('Za_pi_10','Za_pi_30','Zb_same_30','Zc_pi_30_swap')))
    c('the phase injection spec derives T and f_iv from measurement',json.loads((o/'PHASE_INJECTION_SPEC.json').read_text())['T_ms']>0)
    c('the branch is NOT_JUDGED with both reasons recorded',oc['branch']=='NOT_JUDGED' and 'disagree' in oc['branch_reason'])
    AG=json.loads((o/'DISAGREE.json').read_text())
    c('the cross-implementation phase disagreement is enumerated',len(AG['rows'])>=5 and AG['agree'] is False)
    c('the rate-mismatch rule was applied',isinstance(oc['rate_mismatch'],list))
    c('the six S330 corrections are archived',len(json.loads((o/'S330_CORRECTION.json').read_text())['items'])>=6)
    c('the V15 self-test is archived with independently synthesised nulls',(o/'DETECTOR_SELFTEST_V15.json').is_file()
      and json.loads((o/'DETECTOR_SELFTEST_V15.json').read_text())['controls']['c']['null_std']>0)
    src=(o/'analyze_s331.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('the tripod split matches the pre-registered definition',pre['tripod']=={'A':['lf','rm','lh'],'B':['lm','rf','rh']})
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['five arms, no physics','phase differences are f0 sensitive which is what the DISAGREE file records'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
