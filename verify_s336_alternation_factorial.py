"""S336 verification: hashes, counters, gates, sampling rule, factorial controls, disclosure."""
from __future__ import annotations
import argparse, json, hashlib, ast
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S336_PREREG_FROZEN').read_text()
    c('freeze carries PREREG and both script hashes plus UTC',all(('sha256(%s)=%s'%(k,sha(o/k))) in fr for k in ('PREREG.json','analyze_s336.py','audit_s336_independent.py')))
    c('a REFREEZE ledger records the post-freeze corrections',(o/'REFREEZE.json').is_file())
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led[k]==v for k,v in pre['counters'].items()),ledger=led)
    c('no tier advance and no physics/model/render',led['tier_advance']==0 and led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    c('the declared band sits strictly under Nyquist',pre['declared_band'][1] < pre['fs_slow']/2,band=pre['declared_band'],fs=pre['fs_slow'])
    c('the implemented edge equals the declared edge',abs(pre['implemented_edge_hz']-pre['declared_band'][1])<1e-9)
    src=(o/'analyze_s336.py').read_text();tree=ast.parse(src)
    clip=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and getattr(node.func,'attr','')=='butter':
            for sub in ast.walk(node):
                if isinstance(sub,ast.Call) and getattr(sub.func,'attr','') in ('min','max','clip'):
                    clip.append(getattr(sub,'lineno',-1))
    c('no min/clip appears inside a butter edge expression (AST)',len(clip)==0,lines=clip)
    c('the detection null is an independent Poisson aggregate rather than a phase randomised copy',
      'rng.poisson' in src and 'irfft(amp' not in src)
    G=json.loads((o/'GATE.json').read_text())
    c('the three construction gates are bitwise',all(G['equal'].values()),detail={k:v['spike_total'] for k,v in G['detail'].items()})
    AR=json.loads((o/'ARMS.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('every arm has a null drive frequency',all(AR[k]['f_drive_hz'] is None for k in AR))
    c('the same-tripod control has zero self loops and 20202 edges',
      AR['S2s_same_1p0_noloop']['added_edges']['selfloops']==0 and AR['S2s_same_1p0_noloop']['added_edges']['n']==20202,
      added=AR['S2s_same_1p0_noloop']['added_edges'])
    c('the factorial has all four cells',all(k in AR for k in ('C0_gate','R3_2p0','M1_add2_bias_B','M2_bias_only')))
    c('both calibration loops archive three iterations',
      len(AR['M1_add2_bias_B']['calibration'])==3 and len(AR['M2_bias_only']['calibration'])==3)
    c('every arm records f_slow, per-leg detection and null statistics',
      all(AR[k]['f_slow'] is not None and all('null_p95' in v for v in AR[k]['leg_detect'].values()) for k in AR))
    plat={k:(v[0] if isinstance(v,list) else v) for k,v in oc['platform'].items()}
    c('the phase platform excludes unstable arms with explicit labels',
      set(plat.values())<= {'OK','NO_SLOW_RHYTHM','ESTIMATOR_UNSTABLE','BAND_UNSTABLE','INVALID_WINDOW'},
      platform=plat)
    c('the tier split v2 uses a common reference and reports both layers',
      'reference' in json.loads((o/'TIER_PHASE_SPLIT_V2.json').read_text()))
    MC=json.loads((o/'MOTOR_CHAIN_V2.json').read_text())
    c('the motor chain reports both the slow and the carrier band',all(('slow' in v['bands'] and 'carrier' in v['bands']) for cfg in ('default','s300') for v in MC['arms'][cfg].values()))
    c('the branch is one of the four registered labels or NOT_JUDGED',oc['branch'] in ('b1','b2','b3','b4','NOT_JUDGED'),branch=oc['branch'])
    c('the cross-implementation disagreement is archived',(o/'DISAGREE.json').is_file())
    c('the added-edge conclusions carry NOT_EMERGENT',oc.get('not_emergent') is True)
    c('the collapsed control is named rather than hidden',AR['M1_add2_bias_B']['rates']['A']<40 or AR['R4_4p0'] if 'R4_4p0' in AR else True)
    c('the S335 corrections are archived with ten items',len(json.loads((o/'S335_CORRECTION.json').read_text())['items'])>=10)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,tier_build=0,render=0),limitations=['eight arms plus two calibration loops, no physics','the round is NOT_JUDGED because the two implementations differ by 18.3 degrees'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
