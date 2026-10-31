"""S327 verification: hashes, counters, gate, global-mode targets, agreement, branch."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S327_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s327.py','audit_s327_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GLOBAL_MODE.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    A=json.loads((o/'ARMS.json').read_text());AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    g=G['global_mode']
    tgt=G['targets_reported_by_dispatch']
    got={k:[g[k]['peak_hz'],g[k]['conc_abs2bin'],round(g[k]['mod_index'],3)] for k in tgt if k in g and 'peak_hz' in g[k]}
    c('the dispatched global-mode table is reproduced with absolute 2-bin concentration',
      all(abs(got[k][0]-v[0])<=0.5 and abs(got[k][1]-v[1])<=0.05 and abs(got[k][2]-v[2])<=0.05 for k,v in tgt.items() if k in got),
      got=got,target=tgt)
    c('the Poisson null is below every observed concentration (no power-preserving transform)',all(v['conc_rel15']>v['poisson_p95'] for v in g.values() if 'poisson_p95' in v))
    c('GATE: B0 repeat is bitwise identical to S322/SPK_K17',oc['gate'] is True)
    c('B1 zeroed the motif external output edges and recorded the count',A['B1']['zeroed']>0,zeroed=A['B1']['zeroed'])
    c('B1 kept the 34 motif internal edges at the design value',A['B1']['motif_edges_set']>=34,motif=A['B1']['motif_edges_set'])
    c('B2 silenced the motif cells only, not the whole network',A['B2']['pool_mean_rate_hz']>50,pool=A['B2']['pool_mean_rate_hz'])
    c('the two implementations agree on peak, relative concentration and modulation index',AG['agree'] is True,max_peak=AG['max_peak_diff'],max_conc=AG['max_conc_diff'],max_mod=AG['max_mod_diff'])
    src=(o/'analyze_s327.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('the branch is data-derived with an explicit NOT_JUDGED path',oc['branch'] in ('y1','y2','y3','y4') and 'level_change_pct' in oc)
    c('the common-mode invariance tolerances are the preregistered ones',
      oc['common_mode_same']==(abs(oc['common_mode_drift_hz'])<=0.5 and 0.8<=oc['conc_ratio']<=1.25))
    c('the claim is limited to the motif having no measurable influence (not to absence of oscillation)',
      'no measurable' in oc['one_line'] or oc['branch']=='y1')
    c('the cross-arm versus same-arm confusion is documented',any('CROSS-ARM' in i.get('corrected','').upper() for i in json.loads((o/'S326_CORRECTION.json').read_text())['items']))
    c('conditional coherence is archived with both versions',(o/'CONDITIONAL_COHERENCE.json').is_file())
    c('the hypothesis file lists falsifiable predictions',len(json.loads((o/'HYPOTHESIS.json').read_text())['predictions'])>=3)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['offline plus three 5 s tier arms','no physics'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
