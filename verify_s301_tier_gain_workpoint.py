"""S301 stage-1 verification: independent recomputation of the offline audits, zero dynamic calls."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
A=Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z/s_1.00')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    ck=[]; 
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text()); wp=json.loads((out/'TIER_WORKPOINT.json').read_text())
    hd=json.loads((out/'DRIVE_HEADROOM.json').read_text()); co=json.loads((out/'S300_CORRECTION.json').read_text())
    pd=json.loads((out/'POOL_DRIVE.json').read_text()); st=json.loads((out/'STAGE_STATUS.json').read_text())
    led=json.loads((out/'CALL_LEDGER.json').read_text())
    c('S301 prereg round and frozen marker',pre['round']=='S301' and (out/'S301_PREREG_FROZEN').is_file())
    c('zero dynamic calls recorded',all(v==0 for v in led.values()))
    for p,h in pre['input_hashes'].items(): c('input hash '+str(p),Path(p).is_file() and sha(p)==h)
    c('no missing inputs',pre['missing_inputs']==[])
    tr=json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/tier_transfer.json').read_text())
    c('transfer table copied verbatim',wp['transfer_table']==tr['rows'] and len(tr['rows'])==11)
    c('hard cap arithmetic 400 Hz',wp['hard_cap']['cap_hz']==400.0 and 'refractory' in wp['hard_cap']['source'])
    sp=np.load(A/'SPIKES.npz',allow_pickle=True); spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    rate=spk.mean(0)/0.0005; ne=np.load(A/'NEURAL.npz',allow_pickle=True); sc=np.asarray(ne['sensory_current'],float)
    s=wp['s_1_00']
    c('median/p90/max/frac380/zero/duty reproduced',abs(s['rate_median']-float(np.median(rate)))<1e-9 and abs(s['rate_p90']-float(np.percentile(rate,90)))<1e-9
      and abs(s['rate_max']-float(rate.max()))<1e-9 and abs(s['frac_ge_380']-float((rate>=380).mean()))<1e-12
      and abs(s['zero_rate_fraction']-float((rate==0).mean()))<1e-12 and abs(s['duty']-float(spk.mean()))<1e-12)
    drv=np.flatnonzero(np.abs(sc).max(0)>0); am=np.abs(sc[:,drv]).mean(0)
    c('driven cells and current headroom reproduced',hd['driven_cells']==len(drv)==3431 and abs(hd['mean_abs_current_median']-float(np.median(am)))<1e-12
      and abs(hd['mean_abs_current_p90']-float(np.percentile(am,90)))<1e-12)
    c('all S300 arms above the documented threshold',all(v['times_threshold']>50 for v in hd['grid'].values()))
    po=np.load(A/'POOL_NEURONS.npz',allow_pickle=True); loc=np.asarray(po['local_index']); keep=np.flatnonzero(loc>=0)
    c('pool direct drive is zero and coverage reproduced',pd['directly_driven']==int(np.count_nonzero(np.abs(sc[:,loc[keep]]).max(0)>0))==0 and pd['present_in_tier']==len(keep)==202)
    c('S300 correction lists the four items',len(co['items'])==4 and all('correction' in i for i in co['items']))
    c('leg spread below the 5 percent branch threshold',wp['per_leg_spread_percent']<5.0, spread=wp['per_leg_spread_percent'])
    c('stage 2 honestly not executed',st['stage1']=='COMPLETE' and st['stage2'].startswith('NOT_EXECUTED') and st['landing'].startswith('NOT_ESTABLISHED'))
    c('no simulator module import',not any(k.split('.')[0]=='mujoco' or k.startswith('engine.') for k in __import__('sys').modules))
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
           limitations=['stage 2 not executed so the four-way branch is NOT_ESTABLISHED','per-leg pool mapping affects the spread; both 3.21 and 1.25 percent are below the 5 percent threshold',
                        'weight_scale is a modelling parameter','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),failed=r['failed'])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
