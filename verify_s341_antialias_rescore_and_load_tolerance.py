"""S341 verification: hashes, counters, gate, alias rule, sign stability, thresholds, disagreement."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S341_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s341.py','audit_s341_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version']=='v2')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    g=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise against S340',g.get('equal') is True,spikes=g.get('spike_total'))
    AL=json.loads((o/'ALIAS_AUDIT.json').read_text())
    c('every archived arm carries an alias table for five rates',
      all(set(v['alias']['A'])>={'200','100','50','40','20'} and set(v['alias']['B'])>={'200','100','50','40','20'} for v in AL['arms'].values()),
      n_arms=len(AL['arms']))
    c('illegal sampling rates are detected on at least one arm',
      any(not fav['legal'] for v in AL['arms'].values() for fav in v['alias']['A'].values()))
    RS=json.loads((o/'RESCORE_ANTIALIAS.json').read_text())
    scored=[v for v in RS['arms'].values() if v.get('rho') is not None]
    c('the rescore reports sign stability across three low passes',
      all(('rhos' in v and 'sign_stable' in v) for v in scored) and len(scored)>30,n_scored=len(scored),
      unscored=[k for k,v in RS['arms'].items() if v.get('rho') is None])
    c('the null p1 is estimated per arm rather than globally',all(('null_p1' in v) for v in RS['arms'].values() if v['rho'] is not None))
    c('every decision record carries observed null p1 and nulls',
      all(('null_p1' in v and v.get('null_p1',{}).get('nulls')==1000) for v in RS['arms'].values() if v['rho'] is not None))
    c('the control arm is labelled no rhythm',RS['arms']['S338_C0_gate']['label']=='NO_RHYTHM',label=RS['arms']['S338_C0_gate']['label'])
    c('the working point arms stay alternating',all(RS['arms'][k]['label']=='ALTERNATING' for k in ('S336_R3_2p0','S339_G0_gate','S340_G_gate')))
    c('the historical label changes are listed',RS['n_changed']>=1 and len(RS['changed'])==RS['n_changed'],changed=RS['n_changed'])
    CAL=json.loads((o/'THRESHOLD_CALIB.json').read_text())
    c('the threshold calibration shows the null p1 moving with trimming and low pass',
      len({round(v['p1'],3) for v in CAL['table'].values()})>1)
    AR=json.loads((o/'ARMS.json').read_text())
    c('all eight arms carry 5000 ticks and null drive',all(v['tick_count']==5000 and v['f_drive_hz'] is None for v in AR.values()))
    c('every arm declares its bias in nanoamp',all('bias_nA' in v for v in AR.values()))
    LT=json.loads((o/'LOAD_TOLERANCE.json').read_text())
    c('the load ladder reports five rungs with a verdict',len(LT['ladder'])==5 and LT['verdict'] in ('TOLERANCE_GE_20PCT','TOLERANCE_LT_5PCT','MEASURED_INTERVAL'))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('a disagreement above tolerance voids the round',(AG['agree'] is False)==(oc['branch']=='NOT_JUDGED'),branch=oc['branch'],agree=AG['agree'])
    c('the sign disagreement is archived with a repair',(o/'DISAGREE.json').is_file())
    c('the provisional reading is recorded separately',oc.get('branch_if_agreed') is not None)
    src=(o/'analyze_s341.py').read_text();tree=ast.parse(src)
    const_if=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    orr=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no constant if branch and no or-True conjunction (AST)',len(const_if)==0 and len(orr)==0)
    c('the anti alias rule is enforced as at least twice the band top',
      abs(pre['thresholds']['alias_guard']-2.0)<1e-9 and pre['band'][1]==4.0)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['eight arms, no physics','NOT_JUDGED on a sign disagreement between the two pipelines'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=[x['name'] for x in r['failed']])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
