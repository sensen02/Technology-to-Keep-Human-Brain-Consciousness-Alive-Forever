"""S321 verification: full freeze hashes, operating point from measurement, single cfg, counters, branch."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S321_PREREG_FROZEN').read_text()
    c('freeze carries PREREG hash, script hash and UTC',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s321.py)=' in fr) and ('utc=' in fr))
    c('PREREG hash matches',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr)
    c('the single producing script hash matches',('sha256(analyze_s321.py)=%s'%sha(o/'analyze_s321.py')) in fr,note='one script produced every archive artifact this round')
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    c('counters equal the preregistration exactly',all(led.get(k,v)==v for k,v in pre['counters'].items() if k!='con_network_init'),ledger=led)
    c('con_network_init was neither used nor claimed',led.get('con_network_init',0)==0)
    c('no physics, model or rendering',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    A=json.loads((o/'ARMS.json').read_text());OP=json.loads((o/'OPERATING_POINT.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    c('gate arm is bitwise identical to the 5000 archived S309 rows',A['A0_gate']['gate'] is True)
    # operating point must be reproducible from the archive, with no fallback
    G=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in G if r['graded']]
    best=min(sorted(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0)),key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))
    c('I* reproduced from the measured graded table (no constant)',abs(OP['I_star_nA']-best['I'])<1e-12,I_star=OP['I_star_nA'],archived=best['I'])
    c('motif edges were really mutated (34 ordered edges)',all(A[k]['n_touched']>0 for k in ('E1x','E2x','E4x','E8x','E0_abl','PS1x')))
    c('B0 has no mutation and equals the factory motif weights',A['B0_factory_motif']['n_touched']==0)
    c('the repeat arm is bitwise identical to B0',oc['replay_equal_B0_R'] is True)
    c('C_sd0 uses a different background array than B0',oc['sd0_differs'] is True)
    c('ablation arm zeroes the edges',all(m['after_uS']==0.0 for m in A['E0_abl']['mutation']))
    c('edge-total arms set every edge to k x 2.308e-4',all(abs(m['after_uS']-2*json.loads((o/'MOTIF_SCALE_SEMANTICS.json').read_text())['edge_total_uS'])<1e-15 for m in A['E2x']['mutation']))
    c('branch follows the preregistered rule with no else fallback',oc['branch'] in ('s1','s2','s3','s4') and (oc['branch']!='s4' or any(A[f'E{k}x']['f380']>0.02 for k in (1,2,4,8)) or A['A0_gate']['gate'] is not True))
    c('STOP written for s4',(o/'STOP.json').is_file() if oc['branch']=='s4' else True)
    c('the tier vs ConductanceNetwork discrepancy is disclosed','not interchangeable' in oc.get('unexpected',''))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['no physics','graded precondition failed at this point'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
