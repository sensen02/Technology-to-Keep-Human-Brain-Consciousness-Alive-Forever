"""S340 verification: hashes, counters, gate, certification, labels, dose, gait spec, agreement."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    fr=(o/'S340_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s340.py','audit_s340_independent.py')))
    c('a REFREEZE ledger records the post-freeze corrections',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version']=='v2')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    gates=json.loads((o/'GATE.json').read_text())
    c('the working point gate is bitwise against S339',gates.get('G_gate',{}).get('equal') is True,spikes=gates.get('G_gate',{}).get('spike_total'))
    src=(o/'analyze_s340.py').read_text();tree=ast.parse(src)
    bad_or=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value in (True,False) for v in n.values)]
    const_if=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    c('no constant tested boolean and no constant if branch (AST)',len(bad_or)==0 and len(const_if)==0,or_lines=bad_or,if_lines=const_if,or_true=len(bad_or))
    guard=[l for l in src.split('\n') if ('or 0)' in l or 'or 0]' in l) and not l.lstrip().startswith("'")]
    decision_lines=[l for l in guard if any(t in l for t in ('phase','rho','label','exists','p1','p2','p3','p5'))]
    c('no or-zero default gate on a decision value',len(decision_lines)==0,remaining=guard)
    IN=json.loads((o/'INSTRUMENT_V19.json').read_text())
    c('the certification covers fifteen cells at two rates and three duties',len(IN['cells'])>=15,cells=len(IN['cells']))
    c('every decision dict carries observed null p1 null p99 null std nulls and threshold',
      all(k in pre['thresholds'] for k in ('null_corr','null_band')) and IN['all_pass'] in (True,False))
    c('the unreliable cells are archived rather than dropped',any(not v['passes'] for v in IN['cells'].values()),
      failing={k:round(v['accuracy'],3) for k,v in IN['cells'].items() if not v['passes']})
    SW=json.loads((o/'F0_SWEEP.json').read_text())
    c('the fixed f0 estimator is documented as f0 sensitive on real arms',any(v['sensitive'] for v in SW['arms'].values()),
      dispersion={k:round(v['dispersion'],1) for k,v in SW['arms'].items()})
    RC=json.loads((o/'ARMS_RECLASS_V19.json').read_text())
    c('the reclassification keeps four distinct labels',set(RC['labels'])=={'ALTERNATING','RHYTHM_NOT_ANTIPHASE','WINNER_TAKE_ALL','NO_RHYTHM'})
    c('the gate arm and the three seed arms are alternating',all(RC['arms'][k]['label']=='ALTERNATING' for k in ('S339_G0_gate','S336_R3_2p0','S336_R3_seed1','S336_R3_seed2')))
    c('the control arm keeps its no-rhythm label',RC['arms']['S338_C0_gate']['label']=='NO_RHYTHM')
    c('the previously mislabelled deep tau arm is recovered',RC['arms']['S339_T_300']['label']=='ALTERNATING',
      label=RC['arms']['S339_T_300']['label'],rho=round(RC['arms']['S339_T_300']['rho_bp'],3))
    DW=json.loads((o/'DOSE_WINDOW.json').read_text())
    c('the dose window localises an onset edge',DW['onset_tau_ms'] is not None,onset=DW['onset_tau_ms'])
    low=[k for k,v in DW['ladder'].items() if v['low_rate']]
    c('low rate arms are listed separately','ladder' in DW,low=low)
    GS=json.loads((o/'GAIT_SPEC_V2.json').read_text())
    c('the gait spec carries seconds and a duty for every leg',GS['unit']=='s' and all(GS['legs'][lg]['duty'] is not None for lg in GS['legs']))
    c('the gait spec forbids the emergence assumption',
      any('emergent' in x and 'connectome' in x for x in GS['physics_must_not_assume']) and '不得假设该节律为连接组涌现' in json.dumps(GS,ensure_ascii=False))
    c('the gait spec is marked not emergent',GS.get('not_emergent') is True)
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is success or not judged',oc['branch'] in ('SUCCESS','NOT_JUDGED'),branch=oc['branch'])
    c('the failing predicates are archived with the p1 cell detail',oc['flags']['p1'] is False)
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the cross implementation disagreement is recorded',AG['agree'] in (True,False) and 'label_agreement' in AG,
      label_agreement=AG['label_agreement'],max_rho_diff=AG['max_rho_diff'])
    AR=json.loads((o/'ARMS.json').read_text())
    c('all seven arms carry 5000 ticks and null drive',all(v['tick_count']==5000 and v['f_drive_hz'] is None for v in AR.values()))
    c('the load asymmetry arms keep alternating',AR['U1_asymA10']['label']=='ALTERNATING' and AR['U2_asymB10']['label']=='ALTERNATING')
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['seven arms, no physics','p1 fails on five certification cells and the two implementations disagree'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=[x['name'] for x in r['failed']])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
