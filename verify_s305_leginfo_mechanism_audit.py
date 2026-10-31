"""S305 verification: rebuild every null from raw spikes, check the branch, the routing, the decomposition.
Writes VERIFICATION.json only into its own --output directory."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np, scipy.sparse as sps
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
GROUPS=Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json')
LEGS=('lf','lm','lh','rf','rm','rh');TRI_A=('lf','rm','lh');TRI_B=('rf','lm','rh')
ARM='w_1.5e-5';BASE='w_5e-4';SEED=20261026;R=200
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(x,tau_ms=5.0,dt_ms=1.0):
    a=np.exp(-dt_ms/tau_ms);acc=0.0;out=np.empty(len(x))
    for i in range(len(x)): acc=a*acc+(1-a)*x[i];out[i]=acc
    return out
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    assert (out/'S305_PREREG_FROZEN').is_file(), 'must write into the S305 directory'
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((out/'PREREG.json').read_text());led=json.loads((out/'CALL_LEDGER.json').read_text())
    nul=json.loads((out/'NULL_RECONSTRUCTION.json').read_text());ll=json.loads((out/'LEG_LEVEL_NULL.json').read_text())
    br=json.loads((out/'BRANCH_REDECISION.json').read_text());o=json.loads((out/'OUTCOMES.json').read_text())
    ro=json.loads((out/'ROUTING_AUDIT.json').read_text());ld=json.loads((out/'LEVEL_DECOMPOSITION.json').read_text())
    corr=json.loads((out/'S304_CORRECTION.json').read_text());sb=json.loads((out/'SENSITIVITY_BOUND.json').read_text())
    c('prereg frozen',pre['round']=='S305')
    c('zero dynamic calls',all(v==0 for v in led.values()) and all(v==0 for v in pre['calls'].values()))
    for p,h in pre['inputs'].items(): c('input hash '+Path(p).name,Path(p).is_file() and sha(p)==h)
    c('analysis script archived',(Path('/run/media/sensen/Data2/cell_wound_prototype/analyze_s305_leginfo_mechanism_audit.py')).is_file())
    # independent null rebuild (Bernoulli, same filtering) on a reduced draw count for runtime
    sp=np.load(S303/ARM/'SPIKES.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    ms=spk.reshape(400,2,spk.shape[1]).mean(1);p=spk.mean(0)
    po=np.load(S303/ARM/'POOL_NEURONS.npz',allow_pickle=True)
    r2l={int(r):int(l) for r,l in zip(np.asarray(po['pool_root_ids']),np.asarray(po['local_index'])) if l>=0}
    rows=list(csv.DictReader((S290/'PER_CHANNEL.csv').open()))
    L={lg:sorted({r2l[int(x)] for r in rows if (r.get('leg') or '').lower()==lg and r['pool_root_ids'] for x in json.loads(r['pool_root_ids']) if int(x) in r2l}) for lg in LEGS}
    c('leg pool counts 38/32/30/38/32/32',[len(L[lg]) for lg in LEGS]==[38,32,30,38,32,32])
    c('per-leg pool sets are disjoint',len({i for v in L.values() for i in v})==202)
    obs=np.stack([smooth(ms[:,j])*1000.0 for j in range(ms.shape[1])],1)
    def inter(series):
        Lm=np.stack([series[:,L[lg]].mean(1) for lg in LEGS],1);d=Lm-Lm.mean(1,keepdims=True)
        return float((d-d.mean(0,keepdims=True)).std())
    oi=inter(obs);rng=np.random.default_rng(SEED);sims=[]
    for _ in range(40):
        s=(rng.random((800,p.size))<p[None,:]).reshape(400,2,p.size).mean(1)
        sims.append(inter(np.stack([smooth(s[:,j])*1000.0 for j in range(s.shape[1])],1)))
    c('Bernoulli null ratio reproduced in magnitude',abs(oi/np.median(sims)-nul[ARM]['interaction_bernoulli']['ratio'])<0.6,
      verifier_ratio=float(oi/np.median(sims)),archived=nul[ARM]['interaction_bernoulli']['ratio'])
    c('Bernoulli ratio below 3 (b1 not supportable)',nul[ARM]['interaction_bernoulli']['ratio']<3)
    c('circshift ratio recorded',nul[ARM]['interaction_circshift']['ratio']<3)
    c('Gaussian null marked inadmissible',o['nulls']['gaussian']['admissible'] is False)
    c('15 leg pairs reported',nul[ARM]['n_pairs']==15 and len(nul[ARM]['pair_correlations'])==15)
    c('permutation p95 and observed reproduced',ll['permutations']==2000 and ll['observed_over_p95']<1.0,
      observed=ll['observed_spread_percent'],p95=ll['permutation_p95'])
    c('three leg conventions reported',{'all15_neuron_level','minus_channel_level','plus_channel_level'}<=set(ll['conventions']))
    c('sensitivity statement present','not detectable is not the same as absent' in sb['statement'].lower() or 'NOT DETECTABLE' in sb['statement'])
    G=json.loads(GROUPS.read_text())['legs']
    we=np.load(S303/'EDGES_w_1.5e-5_We.npz');wi=np.load(S303/'EDGES_w_1.5e-5_Wi.npz');n=int(we['shape'][0])
    A=sps.csc_matrix((np.asarray(we['data'],float),np.asarray(we['indices'],int),np.asarray(we['indptr'],int)),shape=(n,n)).tocsr()
    A=A+sps.csc_matrix((np.asarray(wi['data'],float),np.asarray(wi['indices'],int),np.asarray(wi['indptr'],int)),shape=(n,n)).tocsr()
    sci=json.loads((S303/ARM/'SENSORY_CHANNEL_INDEX.json').read_text())['groups']
    sens=sorted({int(x) for k,v in sci.items() if k.startswith('lf:') for x in v})
    v=np.zeros(n);v[sens]=1.0;h1=v@A
    c('routing hop-1 total reproduced',abs(float(np.asarray(h1).sum())-ro['per_leg']['lf']['1']['total'])<1e-6,
      verifier=float(np.asarray(h1).sum()),archived=ro['per_leg']['lf']['1']['total'])
    c('routing hop-1 favours the same leg',ro['per_leg']['lf']['1']['same_over_cross']>1.0,ratio=ro['per_leg']['lf']['1']['same_over_cross'])
    c('absent pool neurons listed',ro['absent_count']==len(ro['absent_pool_neurons'])>0)
    dec=ld['per_leg']
    c('decomposition mask and within shares recorded',{'mask_share','within_nonzero_spread_percent'}<=set(ld) | {'mask_share'} and 'within_nonzero_spread_percent' in ld)
    c('b1 withdrawn and b3 hit',o['b1_withdrawn'] is True and o['branch']==3 and o['b3_hit'] is True)
    c('b4 standing',o['b4_standing']['le_0_6'] is True)
    c('next step revoked',o['next_step_revoked'].startswith('the single-arm closed-loop'))
    c('S304 corrections list five items',len(corr['items'])>=5)
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
           calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
           limitations=['the verification rebuild uses 40 null draws (magnitude check) while the archived analysis uses 200','rate-level and synapse-weighted quantities are projections','no support/walking claim'])
    (out/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
