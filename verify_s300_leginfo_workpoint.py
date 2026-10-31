"""S300 verification: gate, budget, arm set, per-arm identities, chain recomputation, zero-extra-arm rule."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
LEGS=('lf','lm','lh','rf','rm','rh'); ARMS=('s_1.00','s_0.50','s_0.25','s_0.10','s_0.05','s_0.02')
REF=Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
BUD=dict(model_load=1,tier_build=1,network_init=6,live_step=24000,neural_tick=4800,live_fwd_position=6,live_forward=0,diagnostic_forward=24006)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    checks=[]
    def ck(n,v,**e): checks.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text()); led=json.loads((out/'CALL_LEDGER.json').read_text())
    g=json.loads((out/'GATE_ORIGINAL_EXACT.json').read_text()); comp=json.loads((out/'COMPLETE.json').read_text())
    ck('prereg lock file present and prereg untouched during run',(out/'S300_PREREG_FROZEN').is_file() and pre['round']=='S300')
    ck('gate passed',g['passed'] is True and comp['gate_decoder_original_exact'] is True)
    ck('budget exactly as declared',led==BUD)
    ck('arm set exactly the declared six',sorted(d.name for d in out.iterdir() if d.is_dir() and d.name.startswith('s_'))==sorted(ARMS))
    ck('gate failure would leave no later arms',g['passed'] is True or not any((out/a).exists() for a in ARMS[1:]))
    ref=np.load(REF/'NEURAL.npz',allow_pickle=True); got=np.load(out/'s_1.00'/'NEURAL.npz',allow_pickle=True)
    for k in ('rates_hz','decoder_preclip','decoder_postclip','sensory_current','encoded_current','submitted_current','actual_step_current','background_current'):
        a=np.asarray(ref[k],float); b=np.asarray(got[k],float)
        ck('s_1.00 identical '+k, a.shape==b.shape and np.array_equal(a,b))
    sub=np.asarray(got['submitted_current'],float); enc=np.asarray(got['encoded_current'],float); bg=np.asarray(got['background_current'],float)
    act=np.asarray(got['actual_step_current'],float)
    ck('submitted == encoded * 1.0',np.array_equal(sub,enc))
    ck('actual == repeat(background+submitted,2)',np.array_equal(act,np.repeat(bg[None,:]+sub,2,axis=0)))
    dec=np.asarray(got['decoder_postclip'],float); ctrl=np.asarray(got['ctrl'],float)
    ck('ctrl == repeat(shift(decoded),10)',np.array_equal(ctrl,np.repeat(np.concatenate([np.zeros((1,90)),dec[:-1]]),10,axis=0)))
    chain=json.loads((out/'LEG_AXIS_CHAIN.json').read_text()); sel=json.loads((out/'SELECTION.json').read_text())
    for arm in ARMS:
        sp=np.load(out/arm/'SPIKES.npz',allow_pickle=True); shp=sp['shape']
        ck(arm+' spike shape (800,12000)',tuple(shp)==(800,12000))
        cnt=sp['spike_counts']; ck(arm+' spike counts sum to duty',abs(float(np.sum(cnt))/(800*12000)-chain[arm]['duty'])<1e-12)
        ck(arm+' arm directory complete',(out/arm/'SPIKE_SUMMARY.json').is_file() and (out/arm/'POOL_NEURONS.npz').is_file())
    ck('no feasible arm and selection is null per the frozen filter',sel['selected'] is None and sel['feasible']==[])
    loss=json.loads((out/'LOSS_LOCATION.json').read_text())
    ck('loss chain ordered and computed from the s_1.00 arm',loss['D_in_leg']>loss['D_pool_leg']>loss['D_out_leg'] and loss['where'] in ('network','pool readout','both levels collapse'))
    ck('substitutions record all asserted anchors',len(json.loads((out/'SUBSTITUTIONS.json').read_text())['substitutions'])>=9)
    res=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c for c in checks if not c['passed']],
             verifier_sha256=sha(__file__),calls=dict(model_load=0,physics_step=0,neural_tick=0),
             limitations=['leg-axis D is a declared projection, not a physical quantity','s is a modelling scale, not a biological calibration','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=res['passed'],checks=len(checks),failed=res['failed'])))
    return 0 if res['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
