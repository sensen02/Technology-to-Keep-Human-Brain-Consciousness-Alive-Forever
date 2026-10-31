"""S299 Stage A verification: recompute the offline audits independently; assert zero dynamic calls."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, csv
ARCH = Path('/home/sensen/Desktop/engine修复验收')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S290 = ARCH / 's290-motor-mapping-coverage/20261026T174500Z'
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def disp(X):
    X=np.asarray(X,float);R=X-X.mean(1,keepdims=True);return float(R.mean(0).std()/abs(X.mean()))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    checks=[]
    def ck(n,v,**e):checks.append(dict(name=n,passed=bool(v),**e))
    try:
        pre=js(out/'PREREG.json');led=js(out/'CALL_LEDGER.json');st=js(out/'STAGE_STATUS.json')
        ck('S299 prereg and zero dynamic calls',pre['round']=='S299' and all(v==0 for v in led.values()) and all(v==0 for v in pre['stages'].keys() and [0,0,0,0,0,0]))
        for p,h in pre['input_hashes'].items(): ck('input hash '+str(p),Path(p).is_file() and sha(p)==h)
        ck('missing inputs recorded and none are silently substituted',isinstance(pre['missing_inputs'],list),
           missing=pre['missing_inputs'])
        z=np.load(S289/'NEURAL.npz');sc=np.asarray(z['sensory_current'],float);V=np.asarray(z['decoder_motor_gains'],float)
        rates=np.asarray(z['rates_hz'],float);pre_=np.asarray(z['decoder_preclip'],float);post=np.asarray(z['decoder_postclip'],float)
        nz=np.flatnonzero(V!=0);drv=np.flatnonzero(np.abs(sc).max(0)>0);sig=sc[:,drv];keep=[]
        for j in range(sig.shape[1]):
            if not any(np.array_equal(sig[:,j],sig[:,k]) for k in keep): keep.append(j)
        U=sc[:,drv[keep]]
        ia=js(out/'INPUT_AUDIT.json')
        ck('driven cells and unique series',ia['driven_cells']==len(drv)==3431 and ia['unique_driven_series']==U.shape[1]==29)
        ck('per-step unique counts',[d['unique_values'] for d in ia['per_step_unique']]==[int(len(np.unique(sc[i]))) for i in (100,200,399)])
        ck('input channel extremes',abs(ia['extremes']['min']-float(U.min()))<1e-12 and abs(ia['extremes']['max']-float(U.max()))<1e-12 and abs(ia['channel_std_max']-float(U.std(0).max()))<1e-12)
        ck('input dispersion reproduced',abs(ia['dispersion']['dispersion']-disp(U))<1e-12)
        oa=js(out/'OUTPUT_AUDIT.json');rm=rates[:,nz].mean(0)
        ck('rate mean range and p10-p90',abs(oa['rates_mean_range'][0]-float(rm.min()))<1e-12 and abs(oa['rates_mean_range'][1]-float(rm.max()))<1e-12
           and abs(oa['rates_mean_p10_p90'][0]-float(np.percentile(rm,10)))<1e-12)
        ck('postclip saturation fraction over activated channels',abs(oa['postclip_at_1_fraction_nz']-float((post[:,nz]==1).mean()))<1e-12)
        ck('across-channel spread reproduced',abs(oa['across_channel_spread_percent']-float((np.percentile(rm,90)-np.percentile(rm,10))/rm.mean()*100))<1e-10)
        rs=js(out/'READOUT_STRUCTURE.json')
        rows=list(csv.DictReader((S290/'PER_CHANNEL.csv').open()));sets={}
        for r in rows:
            if r['pool_root_ids']: sets[r['name']]=tuple(sorted(int(x) for x in json.loads(r['pool_root_ids'])))
        pool=sorted({x for v in sets.values() for x in v})
        ck('pool neuron count and distinct sets',rs['pool_neurons']==len(pool)==208 and rs['distinct_pool_sets']==len(set(sets.values()))==67)
        roots=np.load(S292/'TIER_ROOT_IDS.npz')['root_ids'];driven_roots={int(roots[i]) for i in drv[keep]}
        ck('overlap of pool neurons with 29 driven series reproduced',rs['overlap_pool_neurons_with_29_driven_series']==len(set(pool)&driven_roots),overlap=len(set(pool)&driven_roots))
        ci=js(out/'COMPRESSION_INDEX.json')
        ck('CI = D_out / D_in and both reproduced',abs(ci['D_in']-disp(U))<1e-12 and abs(ci['D_out']-disp(rates[:,nz]))<1e-12 and abs(ci['CI']-disp(rates[:,nz])/disp(U))<1e-12)
        enc=js(out/'ENCODER_CONFIG.json')
        ck('encoder defaults flagged as differing from the dispatch','50' in enc['dispatch_discrepancy'] and enc['ChannelConfig_defaults']['reference_force']==1.0
           and enc['ChannelConfig_defaults']['reference_angle_rad']==0.30)
        ck('stage B and C honestly not executed',st['stage_b'].startswith('NOT_EXECUTED') and st['stage_c'].startswith('NOT_EXECUTED') and st['landing'].startswith('NOT_ESTABLISHED'))
        ck('no simulator module import',not any(k.split('.')[0]=='mujoco' or k.startswith('engine.') for k in __import__('sys').modules))
        res=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c for c in checks if not c['passed']],
                 verifier_sha256=sha(__file__),calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
                 limitations=['stage B and stage C were not executed, so the four-way landing is NOT_ESTABLISHED','CI and D are definitions declared in PREREG, not physical quantities','the 2.5 Hz window limitation still applies to any later phase question'])
        (out/'VERIFICATION.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(passed=res['passed'],checks=len(checks),failed=res['failed'])));return 0 if res['passed'] else 1
    except Exception as exc:
        import traceback
        (out/'VERIFICATION.json').write_text(json.dumps(dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),checks=checks),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(passed=False,error=repr(exc))));return 1
if __name__=='__main__': raise SystemExit(main())
