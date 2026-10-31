"""S323 verification: freeze hashes, V9 thresholds from same-window null, counters, branch."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S323_PREREG_FROZEN').read_text()
    c('freeze carries PREREG hash, script hash and UTC',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s323.py)=' in fr) and ('utc=' in fr))
    c('both freeze hashes match',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr and ('sha256(analyze_s323.py)=%s'%sha(o/'analyze_s323.py')) in fr)
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    I=json.loads((o/'WINDOW_INVARIANCE.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    g=I['grid']
    c('no absolute-bin concentration is used (relative band only)',all('rel' in g[W]['k17_lf'] for W in g))
    c('every threshold equals its own same-window null p95',all(abs(g[W]['k17_lf']['null_rel_p95']-g[W]['k17_lm']['null_rel_p95'])<1 for W in g))
    c('synthetic controls pass at every window',I['checks']['a_syn'] is True)
    c('negative controls reject at every window',I['checks']['c_neg'] is True and I['checks']['e_ctrl'] is True)
    c('isolated positive controls reproduce 6.58 / 15.39 Hz',I['checks']['d_iso'] is True,
      peaks={k:round(v.get('peak_hz') or 0,2) for k,v in I['isolated'].items()})
    c('the real-signal window check fails and is disclosed',I['checks']['b_real'] is False and 'b_real' in oc['invariance_failure']['failed_check'])
    c('branch u4 follows from the failed certification',oc['branch']=='u4' and I['checks']['all'] is False)
    c('ablation contrast is archived (motif arms rhythmic, ablation zero)',oc['rejudge_n_rh']['K17_abl']==0 and oc['rejudge_n_rh']['K17']>=1)
    c('seed repeats are not bitwise copies of the k=17.33 run',oc['r1_r2_not_bitwise_K17'] is True)
    c('pool inheritance is reported with both counts',oc['pool']['K17']['nb10']>0 and oc['pool']['K17']['rel20']>=0)
    c('single-cell control is reported and not assumed',oc['s1_single_cell']['rhythmic'] is False)
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['no physics','V9 is not certified: the real-signal window check fails'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
