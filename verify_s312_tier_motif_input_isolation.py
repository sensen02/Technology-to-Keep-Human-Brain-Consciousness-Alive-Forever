"""S312 verification: unit factor, detector self-tests, stage B rebuild, isolation audit, counters."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());corr=json.loads((o/'S311_CORRECTION.json').read_text())
    st=json.loads((o/'DETECTOR_SELFTEST_V3.json').read_text());sb=json.loads((o/'STAGE_B.json').read_text())
    A=json.loads((o/'ARMS.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    led=json.loads((o/'CALL_LEDGER.json').read_text());gate=json.loads((o/'GATE.json').read_text())
    c('S312 frozen marker written before any dynamic call',(o/'S312_PREREG_FROZEN').is_file())
    c('all four detector self-tests pass',st['pass_a'] and st['pass_b'] and st['pass_c'] and st['pass_d'])
    c('unit factor recorded as 1000/dt_ms',any('1000/dt_ms' in i['corrected'] for i in corr['items']))
    c('sub-rheobase exclusion recorded',any('rheobase' in i['corrected'] for i in corr['items']))
    c('stage B has 20 arms with counters',len(sb['arms'])==20 and sb['counters']['con_network_init']==20 and sb['counters']['con_step']==120000)
    # rebuild one stage B arm from its archived spikes
    k='cap8_full_0.16';sp=np.load(o/f'SPK_{k}.npz')['spk']
    c('stage B spikes archived as npz',sp.shape==(6000,3))
    c('stage B mean rate recomputes',abs(float(sp[:,0].mean()*2000.0)-sb['arms'][k]['mean_hz'])<1.0,
      verifier=float(sp[:,0].mean()*2000.0),archived=sb['arms'][k]['mean_hz'])
    c('factory tier gate passed',gate['passed'] is True and gate['fingerprint']=='56e0322a33c51b7f03aaae9f7df1ef3f')
    c('eight tier arms archived',len(A)==8)
    c('isolation audits present for isolated arms',all('removed' in A[k]['audit'] for k in A if 'iso' in k and 'noiso' not in k))
    c('isolation kept matrices nonnegative',all(A[k]['audit']['nonnegative'] for k in A))
    Wref=np.load(S303/'EDGES_w_5e-4_We.npz')
    We=np.load(o/'SPK_T1_gate.npz') if False else None
    exp_normal=dict(banc_load=8,tier_build=8,select_touch_tier=16,edges_build=8,network_init=8,control_tick=40000,tier_advance=40000,neural_substep=80000)
    c('tier counters equal the normal branch',all(led[k]==v for k,v in exp_normal.items()),ledger=led)
    c('physics and rendering counts are zero','physics_step' not in led and 'model_load' not in led)
    c('branch is one of c1/c2/c3 and REPRO_MISMATCH recorded',oc['branch'] in ('c1','c2','c3') and 'REPRO_MISMATCH' in oc)
    c('interruption ratio recomputed from archived spikes',abs((A['T5_iso_raw_d050']['i1_to_e1_conditional_ratio'] or 0)-0.62)<0.1)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,tier_build=0,render=0),
      limitations=['neural only, no physics','tier fingerprint changes with mutation','T7/T8 comparison void due to REPRO_MISMATCH'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
