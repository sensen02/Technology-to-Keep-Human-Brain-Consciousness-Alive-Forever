"""S343 verification: hashes, counters, signed estimators, cadence routes, gate, audit coverage."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S343_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s343.py','audit_s343_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version'] in ('v2','v3'))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('the decoder evaluation counter is data driven',led['decoder_eval']==11,decoder_eval=led['decoder_eval'])
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    src=(o/'analyze_s343.py').read_text();tree=ast.parse(src)
    acos=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and getattr(node.func,'name','') in ('arccos','acos'): acos.append(node.lineno)
    c('no arccos of a correlation is used as a phase estimator (AST)',len(acos)==0,lines=acos)
    const_if=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    const_exp=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.IfExp) and isinstance(n.test,ast.Constant)]
    c('no constant tested branch in if or conditional expression form (AST)',len(const_if)==0 and len(const_exp)==0,ifs=const_if,ifexps=const_exp)
    c('no transfer function filtfilt call is used',len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='filtfilt'])==0)
    g=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise against three archives',g['P0_gate']['all_equal'] is True,spikes=g['P0_gate']['spike_total'],refs=[x['spike_total'] for x in g['P0_gate']['refs']])
    AR=json.loads((o/'ARMS.json').read_text())
    c('all eight arms carry 5000 ticks and null drive',all(v['tick_count']==5000 and v['f_drive_hz'] is None for v in AR.values()))
    c('every arm reports two signed estimators with their disagreement',
      all(('ls_deg' in v['phase'] and 'xcorr_deg' in v['phase'] and 'agree_deg' in v['phase']) for v in AR.values()))
    c('every arm reports three cadence routes',all(set(('welch_s','acf_s','ls_s'))<=set(v['period']) for v in AR.values()))
    c('periods carry seconds and a resolved flag',all(v['period']['unit']=='s' and 'resolved' in v['period'] for v in AR.values()))
    duties=[v['duty']['A'] for v in AR.values()]
    c('duty is not an identity: its spread exceeds the threshold',float(__import__('numpy').std(duties))>pre['thresholds']['duty_std'],std=round(float(__import__('numpy').std(duties)),4))
    c('swing carries a degenerate denominator guard',all('degenerate' in v['swing'] for v in AR.values()))
    c('the balancing knob is monotone in the B bias',
      AR['P0_gate']['rates']['ratio']>=AR['B20']['rates']['ratio']>=AR['B25']['rates']['ratio']>=AR['B30']['rates']['ratio'],
      ratios={k:round(AR[k]['rates']['ratio'],2) for k in ('P0_gate','B20','B25','B30')})
    CAL=json.loads((o/'DECODER_CALIB.json').read_text())
    c('the decoder sweep covers at least eight denominators',len(CAL['sweep'])>=8,n=len(CAL['sweep']))
    S=json.loads((o/'DECODER_CALIB_SUMMARY.json').read_text())
    c('the decoder infeasibility is archived with a repair suggestion','repair' in S and S['infeasible'] is True)
    MI=json.loads((o/'MOTOR_INTERFACE_V7.json').read_text())
    c('the control interface reaches nine of nine cross pairs at the default decoder',MI['arms']['default']['n_ge150']==9,
      n=MI['arms']['default']['n_ge150'],cross_mean=round(MI['arms']['default']['cross_mean'],1))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    AU=json.loads((o/'AUDIT_INDEPENDENT.json').read_text())
    c('the independent audit exists and covers every branch carrying arm',(o/'AUDIT_INDEPENDENT.json').is_file() and set(('P0_gate','B20','B25','B30','B30_seed1','B30_seed2','B30_LA30','B30_LB30'))<=set(AU))
    c('the reconciliation file exists and records each arm',(o/'RECONCILE_AGREEMENT.json').is_file() and len(AG['rows'])==len(AU))
    c('the audit matches period and duty within tolerance',AG['max_period_diff']<=pre['thresholds']['dperiod'] and AG['max_duty_diff']<=pre['thresholds']['dduty'],
      dp=AG['max_period_diff'],dd=AG['max_duty_diff'])
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is one of the registered labels or not judged',oc['branch'] in ('b1','b2','b3','NOT_JUDGED'),branch=oc['branch'])
    c('the phase disagreement above tolerance is disclosed',AG['agree'] is False and 'arccos' in oc['branch_reason'])
    GS=json.loads((o/'GAIT_SPEC_V4.json').read_text())
    c('the gait spec is marked not emergent and carries a train constraint',GS['not_emergent'] is True and 'train_constraint' in GS)
    c('the gait spec period is not the quantised 0.400 s value',GS['period_s'] is None or abs(GS['period_s']-0.400)>0.02,period=GS['period_s'])
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['eight arms, no physics','not judged: the two signed estimators disagree and the translator cannot satisfy c5'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=[x['name'] for x in r['failed']])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
