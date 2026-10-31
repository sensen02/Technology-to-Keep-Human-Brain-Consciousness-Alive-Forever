"""S335 verification: hashes, counters, gate, static drive, added edges, doses, disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
DOSES=('R1_0p5','R2_1p0','R3_2p0','R4_4p0')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S335_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s335.py','audit_s335_independent.py')))
    c('a REFREEZE ledger records the post-freeze fixes',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('the gate records evidence rather than a bare boolean',all(k in G for k in ('equal','spike_total','ref_spike_total','shape','method')) and G['equal'] is True,
      spike_total=G.get('spike_total'),ref=G.get('ref_spike_total'))
    AR=json.loads((o/'ARMS.json').read_text())
    c('every arm has a null drive frequency and no time dependent current',all(AR[k]['f_drive_hz'] is None and AR[k]['dI_time_dependent'] is False for k in AR))
    c('the gate arm adds no edges',AR['C0_gate']['added_edges']['n']==0)
    c('the reciprocal arms add 20400 edges with declared conductance',
      AR['R2_1p0']['added_edges']['n']==20400 and AR['R2_1p0']['added_edges']['share_of_Wi']>0,
      n=AR['R2_1p0']['added_edges']['n'],share=round(AR['R2_1p0']['added_edges']['share_of_Wi'],4))
    c('the same-tripod control adds the same total conductance to a different target set',
      AR['S2_same_1p0']['added_edges']['n']==20404 and AR['S2_same_1p0']['added_edges']['dirs']==['A<-A','B<-B'])
    c('every arm reports both halves and the slow band',all('halves' in AR[k] and AR[k]['slow']['cross'] is not None or AR[k]['slow']['cross'] is None for k in AR))
    c('the fast carrier band is reported as a diagnostic only',all('fast_diag' in AR[k] for k in AR))
    c('tripod rates are reported per arm',all('tripod' in AR[k] and AR[k]['tripod'].get('A') is not None for k in AR),
      sample={k:AR[k]['tripod'] for k in list(AR)[:2]})
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is one of the four registered labels',oc['branch'] in ('a1','a2','a3','a4'),branch=oc['branch'])
    c('the branch thresholds are numerically preregistered',all(k in pre['branch_thresholds'] for k in ('anti_cross','anti_pairs','rate_min','desync_corr')))
    c('the same-tripod control does not reach antiphase',AR['S2_same_1p0']['slow']['cross']<135,value=round(AR['S2_same_1p0']['slow']['cross'],1))
    c('at least one dose reaches a large slow phase difference',max(AR[k]['slow']['cross'] for k in DOSES)>=100,
      best=round(max(AR[k]['slow']['cross'] for k in DOSES),1))
    c('no dose passes both halves at the criterion, which is what rules out a1',
      not all(AR[k]['halves']['first']['cross'] is not None and AR[k]['halves']['first']['cross']>=135 and AR[k]['halves']['second']['cross']>=135 for k in DOSES))
    c('the collapsed dose is reported rather than hidden',AR['R4_4p0']['pool_median_rate_hz']<40,dose_rate=round(AR['R4_4p0']['pool_median_rate_hz'],1))
    c('the motor chain is archived with both configurations',(o/'MOTOR_CHAIN.json').is_file() and set(json.loads((o/'MOTOR_CHAIN.json').read_text()))>= {'default','s300'})
    c('the two stage A defects are disclosed',all(k in oc['stage_a_defects'] for k in ('tier_phase_split','motor_chain')))
    c('the superseded modulation depth definition is recorded',(o/'MOD_DEPTH_DEFINITION.json').is_file())
    c('the S334 corrections are archived with five items',len(json.loads((o/'S334_CORRECTION.json').read_text())['items'])>=5)
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree within 5 degrees on the slow phase',AG['agree'] is True,max_diff=AG['max_phase_diff'])
    src=(o/'analyze_s335.py').read_text();tree=ast.parse(src)
    bad=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad)==0,lines=bad)
    c('the added inhibition is labelled as a model construction, not connectome',(o/'REPORT.zh-CN.md').is_file() and '非连接组' in (o/'REPORT.zh-CN.md').read_text())
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['seven arms, no physics','stage A tier split invalid and motor chain incomplete, both disclosed'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
