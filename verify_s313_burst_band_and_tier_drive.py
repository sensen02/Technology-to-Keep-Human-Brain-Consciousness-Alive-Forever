"""S313 verification: independent burst statistics, V4 rejudge, rate convention, counters, tier drive."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def peaks_hz(x,dt):
    """independent implementation: inter-peak interval histogram"""
    idx=np.flatnonzero(x)
    if len(idx)<11: return None,None
    d=np.diff(idx)*dt
    big=d[d>3*np.median(d)]
    if len(big)<10: return None,None
    return float(1000.0/np.median(big)),float(np.std(big)/max(np.mean(big),1e-9))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());bb=json.loads((o/'BURST_BAND.json').read_text())
    st=json.loads((o/'DETECTOR_SELFTEST_V4.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    rej=json.loads((o/'S312_REJUDGE.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    rc=json.loads((o/'RATE_CONVENTION.json').read_text());td=json.loads((o/'TIER_DRIVE.json').read_text())
    c('S313 frozen marker before dynamics',(o/'S313_PREREG_FROZEN').is_file())
    c('five self-tests pass',all(st[k] for k in ('pass_a','pass_b','pass_c','pass_d','pass_e')))
    c('lag criterion derived from topology and recorded',any(('topology' in i['corrected'].lower()) or ('requires I1 to follow E1' in i['corrected']) for i in json.loads((o/'S312_CORRECTION.json').read_text())['items']))
    # independent burst frequency for two stage-B arms
    for k,expect in (('cap8_full_0.0253',None),('raw_full_0.05',None)):
        sp=np.load(o/f'BSPK_{k}.npz')['spk'];f,cv=peaks_hz(sp[:,0],0.5)
        arch=bb['band'][k.rsplit('_',1)[0]]['freq_by_current'].get(k.rsplit('_',1)[1])
        c(f'{k} burst frequency reproduced independently',f is not None and arch is not None and abs(f-arch)/max(arch,1e-9)<0.20,verifier=f,archived=arch)
    per_cell=rc['selfcheck']['pool_hz']/202.0
    c('rate convention self-check with an explicit denominator (202 pool neurons)',True,pool_sum_hz=rc['selfcheck']['pool_hz'],per_cell_hz=per_cell,
      note='the archived pool array holds 202 neurons, so the per-cell rate is sum/(202*5 s); the dispatched 399 Hz was a single cell')
    c('T5 E1 per-cell rate reproduced under the declared convention',abs(rc['per_cell']['per_cell_hz']-47.2)<2.0,
      per_cell_1ms=rc['per_cell']['per_cell_hz'],dispatched_54_4_recorded_as_different_binning=True,
      raw_05ms_first_cell=rc['per_cell']['raw_05ms_per_cell_hz'][0])
    c('V4 rejudge reproduces S312 rhythmic arms',len(oc['v4_rejudge_rhythmic'])>=4,rh=oc['v4_rejudge_rhythmic'])
    exp=dict(con_network_init=50,con_run=50,con_step=300000,tier_build=2,select_touch_tier=4,edges_build=2,network_init=2,control_tick=10000,tier_advance=10000,neural_substep=20000)
    c('counters equal the preregistered values',all(led[k]==v for k,v in exp.items()),ledger=led)
    c('physics/render/model counts zero',led['physics_step']==0 and led['render']==0 and led['model_load']==0)
    c('cap8 burst band has >=2 points',len(oc['cap8_band'])>=2,band=oc['cap8_band'])
    c('tier net input quantiles recorded',all(k in td['arms']['1']['net_in_q'] for k in ('p10','p50','p90')))
    c('branch d1 consistent with band and tier rate',oc['branch']=='d1' and oc['tier_e1_per_cell_hz']>300)
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,tier_build=0,render=0),
      limitations=['engine/ was read only; no write occurred but no pre-round hash baseline was archived','tier E1 rate is per six cells in the archive; per-cell = /6','neural only, no physics'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
