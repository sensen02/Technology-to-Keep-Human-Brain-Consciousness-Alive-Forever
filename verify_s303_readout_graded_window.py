"""S303 verification: gates, counters, structure, pool-level criteria, hub/headroom recomputation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ARMS=('w_5e-4','w_5e-5','w_2.5e-5','w_1.5e-5','w_1e-5','w_7.5e-6')
W=dict(zip(ARMS,(5e-4,5e-5,2.5e-5,1.5e-5,1e-5,7.5e-6)))
S302=Path('/home/sensen/Desktop/engine修复验收/s302-tier-gain-sweep/20261026T200813Z')
BUD=dict(model_load=1,banc_load=6,tier_build=6,select_touch_tier=12,edges_build=6,network_init=6,live_fwd_position=6,live_step=24000,neural_tick=4800,live_forward=0,diagnostic_forward=24006)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text());led=json.loads((out/'CALL_LEDGER.json').read_text())
    g1=json.loads((out/'GATE_ORIGINAL_EXACT.json').read_text());g2=json.loads((out/'ARM_REPRO.json').read_text())
    o=json.loads((out/'OUTCOMES.json').read_text());M=json.loads((out/'ARM_METRICS.json').read_text());G=json.loads((out/'ARM_GROUPS.json').read_text())
    hub=json.loads((out/'HUB_STRUCTURE.json').read_text());hr=json.loads((out/'HEADROOM.json').read_text())
    c('prereg frozen marker',(out/'S303_PREREG_FROZEN').is_file() and pre['round']=='S303')
    c('gate1 passed (S289+S300)',g1['passed'] is True and len(g1.get('checks',[]))>80)
    c('gate2 w_5e-5 reproduces S302',g2['passed'] is True and g2['spikes_bitwise_equal'] is True)
    c('counters equal the six-arm budget',all(led.get(k)==v for k,v in BUD.items()),ledger=led)
    c('arm set exactly the six declared weights',sorted(x.name for x in out.iterdir() if x.is_dir() and x.name.startswith('w_'))==sorted(ARMS))
    st=json.loads((out/'ARM_STRUCTURE.json').read_text());base=st['w_5e-4']
    for a in ARMS:
        v=st[a]
        for k in ('roots_sha256','We_indices_sha256','We_indptr_sha256','Wi_indices_sha256','Wi_indptr_sha256'):
            c(f'{a} structure invariant {k}',v.get(k)==base.get(k))
        e=np.load(out/('EDGES_'+a+'_We.npz'));eb=np.load(out/'EDGES_w_5e-4_We.npz')
        rel=float(np.abs(np.asarray(e['data'],float)-np.asarray(eb['data'],float)*W[a]/W['w_5e-4']).max()/max(np.abs(np.asarray(eb['data'],float)*W[a]/W['w_5e-4']).max(),1e-300))
        c(f'{a} We data scales with the declared weight',rel<=1e-12,rel=rel)
    for a in ARMS:
        sp=np.load(out/a/'SPIKES.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool);rate=spk.mean(0)/0.0005
        po=np.load(out/a/'POOL_NEURONS.npz');pl=sorted({int(x) for x in np.asarray(po['local_index']) if x>=0})
        c(a+' duty recomputed',abs(M[a]['duty']-float(spk.mean()))<1e-12)
        c(a+' pool median recomputed',abs(M[a]['pool_median']-float(np.median(rate[pl])))<1e-9)
        c(a+' pool >=380Hz share recomputed',abs(M[a]['pool_f380']-float((rate[pl]>=380).mean()))<1e-12)
        c(a+' pool zero-rate share recomputed',abs(M[a]['pool_zero']-float((rate[pl]==0).mean()))<1e-12)
        c(a+' global median reported separately',abs(M[a]['global_median']-float(np.median(rate)))<1e-9)
        c(a+' pool neurons are 202 of 208',len(pl)==202 and int(np.sum(np.asarray(po['local_index'])<0))==6)
    graded=[a for a in ARMS if 20<=M[a]['pool_median']<=200 and M[a]['pool_f380']<=0.10 and M[a]['pool_zero']<=0.50 and M[a]['duty']>=0.01]
    c('pool-level graded set reproduced',graded==o['pool_graded_arms'],graded=graded)
    c('decision uses the pool level, not the global median',o['branch']==(1 if graded and G[max(graded,key=lambda a:G[a]["spread_A"])]['spread_A']>=5 else (2 if graded else 3)))
    if graded:
        best=max(graded,key=lambda a:G[a]['spread_A'])
        c('selected arm is the graded arm with the largest spread',o['selected']==best and best=='w_1.5e-5')
        c('selected arm leg spread >=5 percent',G[best]['spread_A']>=5.0,spread=G[best]['spread_A'])
        c('selected arm is not silent',M[best]['cmd_sum']>0 and M[best]['duty']>=0.01,cmd_sum=M[best]['cmd_sum'])
    we=np.load(S302/'EDGES_w_5e-5_We.npz');wi=np.load(S302/'EDGES_w_5e-5_Wi.npz')
    eind=np.diff(we['indptr']);iind=np.diff(wi['indptr'])
    po2=np.load(S302/'w_5e-5'/'POOL_NEURONS.npz');pl2=sorted({int(x) for x in np.asarray(po2['local_index']) if x>=0})
    c('hub pool in-degree median reproduced',abs(hub['pool']['exc_indegree_median']-float(np.median(eind[pl2])))<1e-9)
    c('hub rest in-degree median reproduced',abs(hub['rest']['exc_indegree_median']-float(np.median([v for i,v in enumerate(eind) if i not in set(pl2)])))<1e-9)
    c('hub pooled percentile reproduced',abs(hub['pool']['percentile_median']-float(np.median([np.mean(eind<=eind[p]) for p in pl2])))<1e-9)
    src_from=lambda W: np.repeat(np.arange(int(W['shape'][0]) if np.asarray(W['shape']).size else 12000),np.diff(W['indptr']))
    ne=np.load(S302/'w_5e-5'/'NEURAL.npz');sp=np.load(S302/'w_5e-5'/'SPIKES.npz')
    nrate=np.unpackbits(sp['packed'],axis=1)[:,:12000].astype(bool).mean(0)/0.0005
    sw=src_from(we); si=src_from(wi)
    net=np.bincount(sw,weights=np.asarray(we['data'],float)*nrate[sw],minlength=12000)-np.bincount(si,weights=np.asarray(wi['data'],float)*nrate[si],minlength=12000)
    c('headroom proxy reproduced',abs(hr['pool']['median']-float(np.median(net[pl2])))<1e-9)
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,physics_step=0,neural_tick=0,tier_build=0),
           limitations=['pool level is 202 of 208; six absent neurons never redistributed','spread and static drive are declared projections','weight_scale is a modelling parameter','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
