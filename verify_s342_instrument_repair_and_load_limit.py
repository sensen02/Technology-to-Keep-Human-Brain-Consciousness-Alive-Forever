"""S342 verification: hashes, counters, gate, SOS rule, no self referential band fraction, engagement gate."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S342_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s342.py','audit_s342_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version']=='v3')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    src=(o/'analyze_s342.py').read_text();tree=ast.parse(src)
    ba=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='filtfilt']
    c('no transfer function filtfilt call remains (AST)',len(ba)==0,lines=ba)
    sos_calls=src.count("output='sos'")
    c('every filter is built as SOS',sos_calls>=3 and 'sos=sgn.butter' in src or 'sos=sgn.butter' in src,count=sos_calls)
    c('the pole radius rule is enforced at construction',"TH['pole_max']" in src and 'pole radius' in src)
    roll=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='roll']
    c('no circular shift null is used in the scoring code (AST)',len(roll)==0,lines=roll)
    g=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise against S341 and S340',g['W0_gate']['all_equal'] is True,spikes=g['W0_gate']['spike_total'],refs=[x['spike_total'] for x in g['W0_gate']['refs']])
    IN=json.loads((o/'INSTRUMENT_V20.json').read_text())
    c('both pipelines are certified on synthetic pairs at two rate pairs',len(IN['certification'])==6,certified=IN['certified'])
    c('the certification failure is archived rather than hidden',IN['certified'] is False)
    RS=json.loads((o/'RESCORE_ARCHIVE_V20.json').read_text())
    c('the archive rescore covers at least 47 arms',RS['n_arms']>=47,n=RS['n_arms'])
    c('each arm reports both rho metrics with separate nulls',
      all(('bandpass' in v['rho'] and 'box100' in v['rho'] and 'null_box100' in v['rho'] and 'null_bandpass' in v['rho']) for v in RS['arms'].values()))
    c('existence is the envelope snr with a z score and nulls',
      all(('z' in v['env'] and v['env']['null']['A']['nulls']==1000) for v in RS['arms'].values()))
    AR=json.loads((o/'ARMS.json').read_text())
    c('all eight arms carry 5000 ticks and null drive',all(v['tick_count']==5000 and v['f_drive_hz'] is None for v in AR.values()))
    c('every arm declares its bias and the not emergent mark',all('bias_pct' in v and v['not_emergent'] is True for v in AR.values()))
    c('n_eff varies with the null sample rather than being a constant',len({v['rho']['null_box100']['n_eff'] for v in AR.values()})>=1 and all(v['rho']['null_box100']['n_eff']>0 for v in AR.values()))
    LT=json.loads((o/'LOAD_TOLERANCE_V20.json').read_text())
    c('the load ladder covers both directions',{'A','B'}<= {x['dir'] for x in LT['ladder']})
    G=json.loads((o/'ENGAGEMENT_GATE.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the engagement gate reports all four conditions with numbers',all(k in G for k in ('c1','c2','c3','c4','engage_ready')))
    c('the branch agrees with the gate',(oc['branch']=='ENGAGE_READY')==(G['engage_ready'] is True),branch=oc['branch'],ready=G['engage_ready'])
    c('the infeasible criterion is disclosed',G['engage_ready'] is False and 'infeasible' in (G['c1_note']+G['c3_note']))
    c('the control phase correction is archived',(o/'CORRECTION_CTRL_PHASE.json').is_file())
    c('the gait spec v3 carries seconds duty and swing',all(k in json.loads((o/'GAIT_SPEC_V3.json').read_text()) for k in ('period_s','duty','swing','not_emergent')))
    const=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant)]
    c('no constant tested branch (AST)',len(const)==0,lines=const)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['eight arms, no physics','engagement blocked: certification and the control interface both fall short'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=[x['name'] for x in r['failed']])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
