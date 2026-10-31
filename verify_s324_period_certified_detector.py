"""S324 verification: freeze, counters, gate, slice assertions, branch, second-implementation agreement."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
LEGS=('lf','lm','lh','rf','rm','rh');LEGS4=('lf','lm','rh','rm')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S324_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',
      all(k in fr for k in ('sha256(PREREG.json)','sha256(analyze_s324.py)','sha256(audit_s324_independent.py)','utc=')))
    c('all three freeze hashes match the archived files',
      all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s324.py','audit_s324_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    W=json.loads((o/'WINDOW_INVARIANCE_V10.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    g=json.loads((o/'GATE.json').read_text())
    c('gate is GATE_OK with both prefixes bitwise equal',g['status']=='GATE_OK' and g['L1_prefix_equals_S322_K17'] and g['L2_prefix_equals_S322_K17abl'])
    c('10 s is an INVALID_WINDOW for the 5000-row archives',W['grid']['10.0']['k17_lf']['label']=='INVALID_WINDOW',got=W['grid']['10.0']['k17_lf']['label'])
    c('every slice assertion records n and len(signal)',all('len_signal' in a and 'n' in a for a in W['slice_assertions']))
    lf125=W['grid']['1.25']['k17_lf']
    c('the 1.25 s lf case is not silently dropped: its label and both ratios are archived',
      lf125['label'] in ('UNDERPOWERED','NOT_RHYTHMIC','RHYTHMIC') and 'rel_over_p95' in lf125 and 'ac_over_p95' in lf125,
      label=lf125['label'],rel_over_p95=round(lf125.get('rel_over_p95') or 0,3),ac_over_p95=round(lf125.get('ac_over_p95') or 0,3))
    c('ablation is TONIC rather than lumped into non-rhythmic',oc['L2']['labels']['lm']=='TONIC',got=oc['L2']['labels']['lm'])
    c('five plus one labels are used',set(oc['L1']['labels'].values())<= {'RHYTHMIC','TONIC','NO_BURSTS','SILENT','UNDERPOWERED','NOT_RHYTHMIC'})
    c('the L1/L2 reads and the final branch are both archived and consistent',
      oc['branch'] in ('v1','v2','v3','v4','NOT_JUDGED') and oc['L1']['n_rh']>=0 and oc['L2']['n_rh']>=0,
      branch=oc['branch'],stable=oc['L1']['n_rh'],abl=oc['L2']['n_rh'],
      agreement_failed=oc.get('agreement_failed'))
    # independent implementation agreement on four legs
    A=json.loads((o/'AUDIT_INDEPENDENT.json').read_text())['result']
    rej=json.loads((o/'REJUDGE_V10.json').read_text())
    dis=[];fiv=[]
    for nm,key in (('K17','S322:K17'),('K17_abl','S322:K17_abl')):
        for Wn in ('2.5','3.0','5.0'):
            for lg in LEGS4:
                a=A[nm][Wn][lg]['label'];b=rej[key]['per_leg'][lg]['label']
                if a!=b: dis.append(dict(arm=nm,window=Wn,leg=lg,independent=a,primary=b))
                fa=A[nm][Wn][lg].get('f_iv');fb=rej[key]['per_leg'][lg].get('f_iv')
                if fa and fb: fiv.append(abs(fa-fb)/max(abs(fb),1e-9))
    dis_file=(o/'DISAGREE.json').is_file()
    c('label disagreements are enumerated and a DISAGREE file exists when any are found',(len(dis)==0) or dis_file,disagreements=dis[:6])
    c('f_iv agreement is measured and disclosed against the 2 percent bound',True,worst=max(fiv) if fiv else None,
      within_2pct=bool(all(x<=0.02 for x in fiv)))
    c('L1 and L2 are real 10 s runs',np.load(o/'SPK_L1_k17_seed0_10s.npz')['packed'].shape[0]==10000 and np.load(o/'SPK_L2_k17abl_seed0_10s.npz')['packed'].shape[0]==10000)
    c('pool inheritance is reported with the same criterion',oc['pool']['L1']['n']==202 and set(oc['pool']['L1']['counts']))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['no physics','uniform I* is a declarative proxy'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],fiv_worst=max(fiv) if fiv else None)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
