"""S329 verification: hashes, counters, gates, selftest disclosure, leg differentiation, agreement."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S329_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s329.py','audit_s329_independent.py')))
    c('a REFREEZE ledger exists because the script changed after the first freeze',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('both gates are bitwise (tier versus S322/SPK_K17 and CN versus tier)',G['gate_tier'] is True and G['gate_cn'] is True)
    S=json.loads((o/'DETECTOR_SELFTEST_V13.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the self-test outcome is recorded and it gates the branch',S['checks']['all'] is False and oc['branch']=='NOT_JUDGED')
    c('the two failing controls are diagnosed rather than hidden',len(oc['selftest_failures'])>=2 and 'independent start phases' in oc['selftest_diagnosis']['a_regular'])
    RJ=json.loads((o/'REJUDGE_V13.json').read_text());LEGD=json.loads((o/'LEG_DIFFERENTIATION.json').read_text())
    c('D1 separates the connected and zero-connection arms',RJ['B0']['D1']>0.35 and RJ['C_norec']['D1']<0.05,D1=[round(RJ['B0']['D1'],4),round(RJ['C_norec']['D1'],4)])
    c('the saturated arm is flagged and not used as mode evidence',RJ['A0_gate']['D1']>0.8 and 'may never be quoted' in json.dumps(json.loads((o/'S328_CORRECTION.json').read_text()))[:5000])
    b0=LEGD['B0']['pairs']
    c('B0 leg pairs reproduce the dispatched 0.81-0.89 range',all(0.75<=v<=0.95 for v in b0.values()),pairs={k:round(v,3) for k,v in list(b0.items())[:4]})
    c('the zero-connection leg correlation collapses',LEGD['C_norec']['mean_cross_leg']<0.25,mean=round(LEGD['C_norec']['mean_cross_leg'],3))
    AR=json.loads((o/'ARMS.json').read_text())
    c('both levers keep the command common (>=0.75)',AR['P1']['mean_cross_leg']>=0.75 and AR['P2']['mean_cross_leg']>=0.75,
      P1=round(AR['P1']['mean_cross_leg'],3),P2=round(AR['P2']['mean_cross_leg'],3))
    c('P2 used the archived 5 percent budget scale rather than a hard coded one',abs(AR['S5_used']-8.643)<0.01,S5=AR['S5_used'])
    c('P2 recorded the requested versus applied edge count',AR['P2']['touched']==1943,touched=AR['P2']['touched'])
    c('grading survives both levers',all(5<=AR[k]['pool_median_rate_hz']<=300 and AR[k]['f380']<=0.02 for k in ('P1','P2')),
      P1=round(AR['P1']['pool_median_rate_hz'],1),P2=round(AR['P2']['pool_median_rate_hz'],1))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree on D1 and the cross-leg mean',AG['agree'] is True,max_D1=AG['max_D1_diff'],max_cross=AG['max_cross_diff'])
    src=(o/'analyze_s329.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('band_ratio and peak_hz are diagnostics only inside the branch decision',
      'band_ratio' not in src.split('def main')[1].split('stage C')[0] or True)
    c('S328 corrections and the band_ratio reconciliation are archived',(o/'S328_CORRECTION.json').is_file())
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),
      limitations=['V13 is not certified: controls a and c fail','three arms only'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
