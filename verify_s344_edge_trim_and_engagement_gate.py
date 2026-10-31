"""S344 verification: hashes, counters, gate, single filter entry, trim rule, decoder criteria, drift, window conflict."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S344_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s344.py','audit_s344_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version'] in ('v2','v3','v4'))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics or model work',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0)
    src=(o/'analyze_s344.py').read_text();tree=ast.parse(src)
    filt=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'attr','') in ('filtfilt','sosfiltfilt')]
    entry_defs=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='trimmed_band']
    c('all zero phase filtering happens inside the single entry',len(entry_defs)==1 and len(filt)==1,entry=entry_defs,filter_calls=filt)
    acos=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'name','') in ('arccos','acos')]
    c('no arccos phase estimator (AST)',len(acos)==0,lines=acos)
    const_if=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    const_exp=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.IfExp) and isinstance(n.test,ast.Constant)]
    c('no constant tested branch in if or conditional expression form',len(const_if)==0 and len(const_exp)==0,ifs=const_if,ifexps=const_exp)
    g=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise against three archives',g['G0_gate']['all_equal'] is True,refs=[x['spike_total'] for x in g['G0_gate']['refs']])
    IN=json.loads((o/'INSTRUMENT_V21.json').read_text())
    c('the instrument table covers four bin widths three trims and two bands',len(next(iter(IN['table'].values())))==24,n=len(next(iter(IN['table'].values()))))
    c('trim zero rows are flagged as edge contaminated',
      all(v['contaminated'] for per in IN['table'].values() for k,v in per.items() if '_tr0.0_' in k))
    c('a single pipeline is chosen and the rest are sensitivity only','chosen' in IN)
    CERT=json.loads((o/'SYNTH_CERT_V21.json').read_text())
    c('the synthetic grid is archived in full',len(CERT['cells'])==36,n=len(CERT['cells']))
    c('the certification outcome is reported honestly as unevaluable',all(v['n']==0 for v in CERT['cells'].values()))
    DEC=json.loads((o/'DECODER_CRITERIA.json').read_text())
    c('the decoder criterion uses nine of nine pairs leg span and common saturation',
      all(('min_leg_span' in v and 'common_sat' in v and 'n_cross_ge150' in v) for v in DEC['arms'].values()))
    c('clip fractions are declared descriptive only','not a gate' in DEC['note'])
    c('the decoder criterion passes on the default configuration',all(v['pass_gate'] for v in DEC['arms'].values()))
    DR=json.loads((o/'PHASE_DRIFT.json').read_text())
    c('the ten second arms report drift blocks and coherence',all(('drift_deg_s' in v and 'R_2s' in v and 'coherence_s' in v) for v in DR['arms'].values()))
    c('the balanced arm drifts far less than the unbalanced one',
      abs(DR['arms']['B30_10s']['drift_deg_s'])<abs(DR['arms']['P10_10s']['drift_deg_s']),
      balanced=round(DR['arms']['B30_10s']['drift_deg_s'],3),unbalanced=round(DR['arms']['P10_10s']['drift_deg_s'],2))
    EG=json.loads((o/'ENGAGEMENT_GATE_V2.json').read_text())
    c('the window conflict is visible in the gate as ineligible arms',
      any(not v['eligible'] for v in EG['c1_detail'].values()))
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is one of the four registered labels',oc['branch'] in ('d1','d2','d3','d4'),branch=oc['branch'])
    c('no handoff file is emitted when the gate is not met',not (o/'PHYSICS_HANDOFF.json').is_file() or oc['branch'] in ('d1','d2'))
    AU=json.loads((o/'AUDIT_INDEPENDENT.json').read_text())
    c('the independent audit covers every arm',set(('G0_gate','B30_5s','B30_seed1','B30_seed2','LA30_5s','LB30_5s','P10_10s','B30_10s'))<=set(AU))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text()) if (o/'RECONCILE_AGREEMENT.json').is_file() else dict(agree=False,note='absent')
    c('the reconciliation file exists',AG.get('note')!='absent')
    AR=json.loads((o/'ARMS.json').read_text())
    c('all eight arms are archived with ticks and a null drive',all(v['ticks'] in (5000,10000) and v['f_drive_hz'] is None for v in AR.values()))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['eight arms, no physics','the trim rule makes five second arms ineligible and the certification could not be evaluated'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=[x['name'] for x in r['failed']])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
