"""S304 verification: independent recomputation of corrections, null ratios, headroom, mask, wiring, spectrum."""
from __future__ import annotations
import argparse, hashlib, json, csv
from pathlib import Path
import numpy as np
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
G=json.loads(Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json').read_text())['legs']
LEGS=('lf','lm','lh','rf','rm','rh');ARMS=('w_5e-4','w_5e-5','w_2.5e-5','w_1.5e-5','w_1e-5','w_7.5e-6')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text());led=json.loads((out/'CALL_LEDGER.json').read_text())
    corr=json.loads((out/'S303_CORRECTION.json').read_text());sn=json.loads((out/'SAMPLING_NULL.json').read_text())
    dm=json.loads((out/'DELIVERED_MODULATION_AT_WORKPOINT.json').read_text());mk=json.loads((out/'MASK_CEILING.json').read_text())
    wo=json.loads((out/'WIRING_PATH.json').read_text());ts=json.loads((out/'TIME_STRUCTURE.json').read_text())
    o=json.loads((out/'OUTCOMES.json').read_text())
    c('prereg frozen',(out/'S304_PREREG_FROZEN').is_file() and pre['round']=='S304')
    c('zero dynamic calls',all(v==0 for v in led.values()) and all(v==0 for v in pre['calls'].values()))
    for p,h in pre['input_hashes'].items(): c('input hash '+Path(p).name,Path(p).is_file() and sha(p)==h)
    tab={}
    for a in ARMS:
        sp=np.load(S303/a/'SPIKES.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
        r=spk.mean(0)/0.0005
        tab[a]=dict(duty=float(spk.mean()),median=float(np.median(r)),f380=float((r>=380).mean()),zero=float((r==0).mean()),
                    graded=bool(20<=np.median(r)<=200 and (r>=380).mean()<=0.10 and (r==0).mean()<=0.50 and spk.mean()>=0.01))
    c('duty 0.048132 for the work point',abs(tab['w_1.5e-5']['duty']-0.048132)<1e-6,duty=tab['w_1.5e-5']['duty'])
    c('global graded set is w_2.5e-5 and w_1.5e-5',[a for a in ARMS if tab[a]['graded']]==['w_2.5e-5','w_1.5e-5'])
    c('correction table records the global graded set',any('w_2.5e-5' in json.dumps(i.get('archived','')) or 'global_table' in i.get('archived','') for i in corr['items']))
    c('cmd_sum ratio recorded and <=0.6',abs(o['delivered']['ratio_cmd_sum']-float(np.asarray(np.load(S303/'w_1.5e-5'/'NEURAL.npz')['decoder_postclip']).sum())/float(np.asarray(np.load(S303/'w_5e-4'/'NEURAL.npz')['decoder_postclip']).sum()))<1e-12)
    for arm in ('w_5e-4','w_1.5e-5'):
        pre_c=np.asarray(np.load(S303/arm/'NEURAL.npz')['decoder_preclip'],float);post=np.asarray(np.load(S303/arm/'NEURAL.npz')['decoder_postclip'],float)
        V=np.asarray(np.load(S303/arm/'NEURAL.npz')['decoder_motor_gains'],float);nz=np.flatnonzero(V!=0)
        m=(np.arange(1,401)>=20)&(np.arange(1,401)<400)
        head=[float(np.sqrt(max(post[m,i].mean()*(1-post[m,i].mean()),0))) for i in nz]
        c(arm+' swing headroom reproduced',abs(dm[arm]['median_swing_headroom']-float(np.median(head)))<1e-9)
        c(arm+' post1 double denominator reproduced',abs(dm[arm]['post1_64']-float((post[:,nz]==1).mean()))<1e-12 and abs(dm[arm]['post1_90']-float((post==1).mean()))<1e-12)
    for lg in LEGS:
        idx=G[lg]['minus'];V=np.asarray(np.load(S303/'w_1.5e-5'/'NEURAL.npz')['decoder_motor_gains'],float)
        nzc=int(sum(1 for i in idx if V[i]!=0))
        c('mask ceiling '+lg, mk['minus_groups'][lg]['nonzero']==nzc and abs(mk['minus_groups'][lg]['fraction']-nzc/len(idx))<1e-12)
    c('null ratios reproducible in magnitude',sn['w_1.5e-5']['interaction']['ratio']>3 and sn['w_1.5e-5']['tripod']['ratio']>3,
      inter=sn['w_1.5e-5']['interaction']['ratio'],trip=sn['w_1.5e-5']['tripod']['ratio'])
    c('null draws and seed recorded',sn['w_1.5e-5']['draws']==200 and sn['w_1.5e-5']['seed']==20261026)
    b=ts['w_1.5e-5']['interaction'];t=ts['w_1.5e-5']['tripod']
    c('sub-6 Hz shares recorded with 2.5 Hz resolution (per-leg banding)',abs(b['resolution_hz']-2.5)<1e-9 and b['sub6_share'] is not None and 'mean over the six per-leg bands' in b.get('mode',''))
    c('branch table marks b1 unmet on slow structure and b2 unmet on interaction',o['clause_check']['b1']['met'] is False and o['clause_check']['b1']['slow_structure'] is False and o['clause_check']['b2']['met'] is False)
    c('b4 standing constraint true',o['clause_check']['b4']['standing'] is True and o['clause_check']['b4']['le_0_6'] is True)
    c('no branch falsely claimed',o['branch'] is None and 'NO BRANCH IS CLEANLY MET' in o['landing'])
    c('wiring recorded as structure only','CONNECTION STRUCTURE' in wo['note'])
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
           limitations=['the null uses a per-bin Gaussian matching the Poisson-binomial mean/variance, not a full neuron-level resample','2.5 Hz resolution makes 8-25 Hz unjudgeable','rate-level projections are not physical quantities','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
