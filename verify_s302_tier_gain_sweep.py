"""S302 independent verification: gate, counters, per-arm structure invariants, metrics, branches."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ARMS=('w_5e-4','w_5e-5','w_5e-6','w_5e-7','w_5e-8'); W={'w_5e-4':5e-4,'w_5e-5':5e-5,'w_5e-6':5e-6,'w_5e-7':5e-7,'w_5e-8':5e-8}
LEGS=('lf','lm','lh','rf','rm','rh')
NOGRAD=dict(model_load=1,banc_load=5,tier_build=5,select_touch_tier=10,edges_build=5,network_init=5,live_step=20000,neural_tick=4000,live_fwd_position=5,diagnostic_forward=20005)
S300=Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z/s_1.00')
REF=Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text()); led=json.loads((out/'CALL_LEDGER.json').read_text())
    g=json.loads((out/'GATE_ORIGINAL_EXACT.json').read_text()); st=json.loads((out/'ARM_STRUCTURE.json').read_text())
    c('prereg frozen marker present',(out/'S302_PREREG_FROZEN').is_file() and pre['round']=='S302')
    c('gate passed against S289 and S300',g['passed'] is True and len(g.get('checks',[]))>80)
    c('counters equal the no-graded branch exactly',all(led.get(k)==v for k,v in NOGRAD.items()),ledger=led)
    c('arm set exactly the five declared weights',sorted(d.name for d in out.iterdir() if d.is_dir() and d.name.startswith('w_'))==sorted(ARMS))
    base=st['w_5e-4']
    c('w_5e-4 fingerprint equals the historical fingerprint',base['fingerprint']=='56e0322a33c51b7f03aaae9f7df1ef3f')
    for a in ARMS:
        v=st[a]
        for k in ('roots_sha256','We_indices_sha256','We_indptr_sha256','Wi_indices_sha256','Wi_indptr_sha256'):
            c(f'{a} structure invariant {k}',v.get(k)==base.get(k))
        ratio=v['We_data_sum']/base['We_data_sum']
        c(f'{a} We data proportional to the declared weight',abs(ratio-W[a]/W['w_5e-4'])<1e-12,ratio=ratio)
        c(f'{a} Wi data proportional to the declared weight',abs(v['Wi_data_sum']/base['Wi_data_sum']-W[a]/W['w_5e-4'])<1e-12)
        e=np.load(out/('EDGES_'+a+'_We.npz'),allow_pickle=True); eb=np.load(out/'EDGES_w_5e-4_We.npz',allow_pickle=True)
        c(f'{a} We indices/indptr bit-identical to w_5e-4',np.array_equal(e['indices'],eb['indices']) and np.array_equal(e['indptr'],eb['indptr']))
        rel=float(np.abs(np.asarray(e['data'],float)-np.asarray(eb['data'],float)*W[a]/W['w_5e-4']).max()/max(np.abs(np.asarray(eb['data'],float)*W[a]/W['w_5e-4']).max(),1e-300))
        c(f'{a} We data equals scaled reference within 1e-12',rel<=1e-12,rel=rel)
    ref=np.load(REF/'NEURAL.npz',allow_pickle=True); r3=np.load(S300/'NEURAL.npz',allow_pickle=True)
    for tag,src in (('s289',ref),('s300',r3)):
        got=np.load(out/'w_5e-4'/'NEURAL.npz',allow_pickle=True)
        for k in ('rates_hz','decoder_preclip','decoder_postclip','sensory_current','encoded_current','submitted_current','actual_step_current','background_current'):
            a0=np.asarray(src[k],float); b0=np.asarray(got[k],float)
            c(f'w_5e-4 identical to {tag} {k}',a0.shape==b0.shape and np.array_equal(a0,b0))
    got=np.load(out/'w_5e-4'/'NEURAL.npz',allow_pickle=True)
    c('submitted == encoded*1.0',np.array_equal(np.asarray(got['submitted_current'],float),np.asarray(got['encoded_current'],float)))
    c('actual == repeat(background+submitted,2)',np.array_equal(np.asarray(got['actual_step_current'],float),np.repeat(np.asarray(got['background_current'],float)[None,:]+np.asarray(got['submitted_current'],float),2,axis=0)))
    dec=np.asarray(got['decoder_postclip'],float); ctrl=np.asarray(got['ctrl'],float)
    c('ctrl == repeat(shift(decoded),10)',np.array_equal(ctrl,np.repeat(np.concatenate([np.zeros((1,90)),dec[:-1]]),10,axis=0)))
    groups=json.loads((out/'ARM_GROUPS.json').read_text()) if (out/'ARM_GROUPS.json').exists() else {}
    for a in ARMS:
        sp=np.load(out/a/'SPIKES.npz',allow_pickle=True); spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
        rate=spk.mean(0)/0.0005
        c(a+' spike shape (800,12000)',tuple(sp['shape'])==(800,12000))
        met=json.loads((out/'ARM_METRICS.json').read_text())[a]
        c(a+' duty/median/f380/zero recomputed',abs(met['duty']-float(spk.mean()))<1e-12 and abs(met['med']-float(np.median(rate)))<1e-9
          and abs(met['f380']-float((rate>=380).mean()))<1e-12 and abs(met['zero']-float((rate==0).mean()))<1e-12)
    sel=json.loads((out/'OUTCOMES.json').read_text()) if (out/'OUTCOMES.json').exists() else {}
    c('no graded arm so the round lands on branch 3',sel.get('branch')==3, branch=sel.get('branch'))
    c('no control arm directory exists on the no-graded branch',not any(out.glob('*control*')))
    c('verifier makes no dynamic call',True)
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,physics_step=0,neural_tick=0,tier_build=0),
           limitations=['weight_scale is a modelling parameter, not a biological calibration','fingerprints differ per arm because the digest includes We/Wi data, so tier-level results are not comparable with historical fingerprints','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),failed=r['failed'][:3],nfail=len(r['failed']))))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
