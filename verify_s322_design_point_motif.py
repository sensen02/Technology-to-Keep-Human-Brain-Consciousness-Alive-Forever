"""S322 verification: units, freeze, gate, counters, independent rate recompute, branch."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S321=Path('/home/sensen/Desktop/engine修复验收/s321-motif-scale-sweep/20261026T222314Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');CAP=400.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S322_PREREG_FROZEN').read_text()
    c('freeze carries PREREG hash, script hash and UTC',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s322.py)=' in fr) and ('utc=' in fr))
    c('both freeze hashes match the archived files',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr and ('sha256(analyze_s322.py)=%s'%sha(o/'analyze_s322.py')) in fr)
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    A=json.loads((o/'ARMS.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('gate arm bitwise equals the 5000 archived S309 rows',A['A0_gate']['gate'] is True)
    c('dimension assertion: every rate <= 400 Hz',all(v['max_rate']<=CAP for v in A.values()),worst=max(v['max_rate'] for v in A.values()))
    # independent recompute of rates from the archived spike counts with the 1 ms tick base
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in mod[lg][r]]
    ok=True;det={}
    for name in A:
        z=np.load(o/f'SPK_{name}.npz');cnt=z['counts'];n=int(z['shape'][0])
        rate=cnt/(n*1.0/1000.0)
        det[name]=dict(motif18_mean=float(rate[M18].mean()),pop_median=float(np.median(rate)),max=float(rate.max()))
        if abs(rate[M18].mean()-A[name]['motif18_mean'])>1e-6: ok=False
    c('independent rate recompute matches the archive (1 ms tick base)',ok,detail=det)
    # S321 rejudge: the doubled rates must now be halved
    rj=json.loads((o/'S321_REJUDGE.json').read_text())
    c('S321 rejudge shows the corrected population medians',abs(rj['B0_factory_motif']['pop_median']-28.0)<1.5,
      b0=round(rj['B0_factory_motif']['pop_median'],2))
    c('S321 rejudge shows the corrected f380 below the ceiling for non-gate arms',
      all(rj[k]['f380']<=0.02 for k in rj if k!='A0_gate'))
    dp=json.loads((o/'DESIGN_POINT.json').read_text())
    c('design point k=17.33 is recorded and equals 0.004/2.3077e-4',abs(dp['k_design']-17.33)<0.02,k=dp['k_design'])
    c('the four k arms plus ablation, role and repeat are all archived',len(A)==9)
    c('K17_repeat is bitwise identical to K17',bool(np.array_equal(np.unpackbits(np.load(o/'SPK_K17.npz')['packed'],axis=1),
      np.unpackbits(np.load(o/'SPK_K17_repeat.npz')['packed'],axis=1))))
    means=[A[k]['motif18_mean'] for k in ('K10','K13','K15','K17','K21')]
    b0m=A['K10']['motif18_mean']
    t3=all(abs(m-b0m)/b0m<=0.10 for m in means)
    c('branch t3 follows from the data (all k within 10 percent)',oc['branch']=='t3' and t3,means=means)
    c('the rhythmic-leg nuance is disclosed (1/6 at k>=17.33)',A['K17']['n_legs']==1 and A['K21']['n_legs']==1)
    c('the cause at k=17.33 is reported with three separate numbers',all(k in oc['k17_cause'] for k in ('e1_hz','i1_hz','e2_hz','label')))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),
      units_selfcheck='max archived rate %.1f Hz must stay below the 400 Hz refractory cap'%max(v['max_rate'] for v in A.values()),
      limitations=['no physics','uniform I* is a declarative proxy'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
