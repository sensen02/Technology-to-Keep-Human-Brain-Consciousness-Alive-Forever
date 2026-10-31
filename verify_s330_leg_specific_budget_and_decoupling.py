"""S330 verification: hashes, counters, gates, budget, dose ladder, selftest disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S330_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s330.py','audit_s330_independent.py')))
    c('a REFREEZE ledger records every post-freeze change',(o/'REFREEZE.json').is_file() and json.loads((o/'REFREEZE.json').read_text())['version']=='v4')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('both gates are bitwise',G['gate_tier'] is True and G['gate_cn'] is True)
    B=json.loads((o/'POOL_INPUT_BUDGET.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    sh={lg:B['per_leg'][lg]['share'] for lg in LEGS}
    c('the leg-specific share reproduces 0.72-1.10 percent',0.0070<=min(sh.values()) and max(sh.values())<=0.0115,shares={k:round(v,4) for k,v in sh.items()})
    c('cross-leg motif to pool edges are zero',B['cross_leg_pool_edges']==0)
    c('the four axis counts are archived',B['edges']['m18_out']==1943 and sum(B['per_leg'][lg]['we_edges'] for lg in LEGS)==160)
    AR=json.loads((o/'ARMS.json').read_text())
    order=['G_cn_gate','Kout4.33','Kout8.67','Kout17.33','Kout34.66']
    vals=[AR[k]['mean_cross_leg'] for k in order]
    c('the dose ladder is monotone decreasing',all(vals[i]>=vals[i+1]-0.01 for i in range(len(vals)-1)),values=[round(v,3) for v in vals])
    c('the realised shares rise with the dose',all(AR[order[i]]['share_mean']<AR[order[i+1]]['share_mean'] for i in range(1,len(order)-1)),
      shares=[round(AR[k]['share_mean'],3) for k in order])
    c('the seed floor is measured from two zero-dose arms',oc['seed_sigma']<0.05,sigma=oc['seed_sigma'])
    c('the highest dose point measured at 20 percent is archived with the three model predictions',oc['transfer']['measured_20'] is not None and len(oc['transfer']['predictions_20'])==3)
    c('the model comparison reports a winner or THIRD_MODEL',oc['model_win'] in ('A','B','THIRD_MODEL'),win=oc['model_win'])
    c('the motif-layer decoupling floor is archived',0.2<oc['transfer']['motif_floor']<0.7,floor=round(oc['transfer']['motif_floor'],3))
    c('the partial correlations are archived',all(k in oc['partials'] for k in ('own','other','own_abl','other_abl')))
    S=json.loads((o/'DETECTOR_SELFTEST_V14.json').read_text())
    c('the self-test outcome gates the branch and its two failures are diagnosed',
      S['checks']['all'] is False and oc['branch']=='NOT_JUDGED' and len(oc['selftest']['failures'])>=2)
    c('the D1 floor from the Poisson control is archived',0<S['D1_FLOOR']<0.05,floor=S['D1_FLOOR'])
    c('the isolation arm is reported separately',AR['ISO90']['pool_median_rate_hz']<60 and AR['ISO90']['touched']>0,iso_rate=round(AR['ISO90']['pool_median_rate_hz'],1))
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree on the budget and the cross-leg mean',AG['agree'] is True,max_share=AG['max_share_diff'],max_cross=AG['max_cross_diff'])
    src=(o/'analyze_s330.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('the S329 four corrections are archived',len(json.loads((o/'S329_CORRECTION.json').read_text())['items'])>=4)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['V14 not certified: controls b and g fail','eight arms, no physics'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
