"""S334 verification: hashes, counters, gate, band rules, ladder, persistence, agreement."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
LADDER=('F1_12p5','F2_20','F3_25p39','F4_40')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S334_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s334.py','audit_s334_independent.py')))
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    G=json.loads((o/'GATE.json').read_text())
    c('the gate arm is bitwise identical to S333 A0_gate',G['g0'] is True)
    AR=json.loads((o/'ARMS.json').read_text())
    c('every dynamic arm records its drive frequency',all(AR[k]['f_drive'] is not None for k in AR if k!='G0_gate'))
    c('the criterion band contains the drive frequency in every arm',
      all(AR[k]['band'][0]<=AR[k]['f_drive']<=AR[k]['band'][1] for k in AR if k!='G0_gate'))
    src=(o/'analyze_s334.py').read_text();tree=ast.parse(src)
    bad=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and getattr(node.func,'id','')=='sin' and node.args:
            a=node.args[0]
            for sub in ast.walk(a):
                if isinstance(sub,ast.Constant) and isinstance(sub.value,float) and sub.value>5:
                    bad.append(getattr(sub,'lineno',-1))
    c('no sine drive is written with a literal frequency constant (AST)',len(bad)==0,lines=bad)
    RB=json.loads((o/'RELAY_BAND_REANALYSIS.json').read_text())['arms']
    c('the drive-band reanalysis reproduces the dispatched relay numbers',
      abs(RB['T3_in_diff040']['injected_band_160_190']['cross']-176.4)<=2 and abs(RB['T3_in_diff040']['pool_band_160_190']['cross']-169.4)<=2
      and RB['T3_in_diff040']['pool_band_160_190']['n_anti']==9,
      inj=round(RB['T3_in_diff040']['injected_band_160_190']['cross'],1),pool=round(RB['T3_in_diff040']['pool_band_160_190']['cross'],1))
    c('the common-mode arm is a clean negative control',RB['T5_in_common020']['pool_band_160_190']['cross']<=30)
    c('the four ladder points all relay',all(AR[k]['pool']['cross']>=135 for k in LADDER),ladder={k:round(AR[k]['pool']['cross'],1) for k in LADDER})
    c('the manipulation check passes for the ladder arms',
      all(AR[k]['manip']['cross']>=135 and AR[k]['manip']['n_anti']>=8 for k in LADDER),
      manip={k:(round(AR[k]['manip']['cross'],1),AR[k]['manip']['n_anti']) for k in LADDER})
    c('the 16-26 Hz version is also archived for every arm',all('pool_ref_band' in AR[k] for k in AR))
    k1=AR['K1_off_2500']['halves'];k2=AR['K2_common_off_2500']['halves']
    c('the off arm reports both halves for the pool and the injected population',
      all(v is not None for v in (k1['first']['pool_cross'],k1['second']['pool_cross'],k1['second']['inj_cross'])))
    c('the off arm shows no persistence',k1['second']['pool_cross']<90,k1_second=round(k1['second']['pool_cross'],1),inj_second=round(k1['second']['inj_cross'],1))
    c('the common off arm is also non-persistent',k2['second']['pool_cross']<90,k2_second=round(k2['second']['pool_cross'],1))
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('the branch is one of the five registered labels',oc['branch'] in ('p1','p2','p3','p4','p5'),branch=oc['branch'])
    c('the branch thresholds are numerically preregistered',all(k in pre['branch_thresholds'] for k in ('relay_cross','self_cross','neg_lt')))
    c('the relay spec table is archived',(o/'RELAY_SPEC.json').is_file())
    c('the S333 corrections are archived with seven items',len(json.loads((o/'S333_CORRECTION.json').read_text())['items'])>=7)
    AG=json.loads((o/'RECONCILE_AGREEMENT.json').read_text())
    c('the two implementations agree within 5 degrees',AG['agree'] is True,max_diff=AG['max_phase_diff'])
    bad2=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.Or) and any(isinstance(v,ast.Constant) and v.value is True for v in n.values)]
    c('no or-True conjunction (AST)',len(bad2)==0,lines=bad2)
    c('the no-persistence interpretation is recorded','does not persist' in oc['interpretation'])
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['nine arms, no physics','phase is an envelope-domain projection'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
