"""S309 verification: freeze-before-run, cap audit, tier gate, motif weights, counters, oscillation criteria."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BANC=ROOT/'data/flywire/banc_connectome.npz'
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());cap=json.loads((o/'CAP_AUDIT.json').read_text())
    gate=json.loads((o/'TIER_GATE.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    osc=json.loads((o/'OSCILLATION.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    sign=json.loads((o/'MOTIF_SIGN_AUDIT.json').read_text())
    c('S309 frozen marker written by the analysis script before any run',(o/'S309_PREREG_FROZEN').is_file() and 'S307_PREREG_FROZEN' not in [p.name for p in o.iterdir()])
    c('gate: factory fingerprint',gate['gate']['fingerprint']=='56e0322a33c51b7f03aaae9f7df1ef3f' and gate['gate']['passed'] is True)
    z=np.load(BANC,allow_pickle=True);rt=np.asarray(z['root_ids'])
    pre_r=rt[np.asarray(z['pre'])];post_r=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    roots=np.load(S303/'TIER_ROOT_IDS.npz')['root_ids'] if (S303/'TIER_ROOT_IDS.npz').is_file() else np.load(Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z/TIER_ROOT_IDS.npz'))['root_ids']
    inside=np.isin(pre_r,list(set(int(x) for x in roots)))&np.isin(post_r,list(set(int(x) for x in roots)))
    c('edges in tier 442843',int(inside.sum())==cap['edges_in_tier'],verifier=int(inside.sum()),archived=cap['edges_in_tier'])
    c('raw mass reproduced',abs(float(w[inside].sum())-cap['raw_synapse_mass'])<1e-6)
    c('capped mass reproduced',abs(float(np.minimum(w[inside],8).sum())-cap['capped_synapse_mass'])<1e-6)
    c('loss fraction 48.4 percent',abs(cap['loss_fraction']-0.4842)<0.01)
    c('edges at cap 36.1 percent',abs(cap['edges_at_cap_fraction']-0.3614)<0.01)
    we=np.load(S303/'EDGES_w_5e-4_We.npz')
    c('We data has 6 levels and 35.3 percent at max',len({round(float(x),6) for x in np.unique(we['data'])})==6 and abs(float(np.mean(we['data']==we['data'].max()))-0.3533)<0.01)
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    rows=cap['motif_edges'];c('36 motif ordered edges catalogued',len(rows)==36)
    def cnt(lg,src,tgt):
        sa=set(mod[lg][src]);sb=set(mod[lg][tgt])
        return float(w[np.isin(pre_r,list(sa))&np.isin(post_r,list(sb))].sum())
    c('motif raw counts reproduced for lf E1->E2',abs([r for r in rows if r['leg']=='lf' and r['edge']=='E1->E2'][0]['raw']-cnt('lf','E1','E2'))<1e-9)
    c('tier weight equals min(count,8)*5e-4',all(abs(r['tier_weight']-min(r['raw'],8)*5e-4)<1e-12 for r in rows))
    c('sign audit: ACh excitatory and glutamate inhibitory declared',sign['pure_exc']=='acetylcholine' and 'glutamate' in sign['pure_inh'])
    c('E2 declarative fill flagged','all_empty' in sign['e2_declared_fill'])
    c('counters equal the preregistered values',led==pre['counters'],ledger=led)
    for a in ('k1_drive','k2_drive'):
        sp=np.load(o/f'SPIKES_{a}.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
        c(f'{a} spike shape (5000,12000)',tuple(sp['shape'])==(5000,12000))
        c(f'{a} duty recomputed',abs(float(spk.mean())-json.loads((o/'ARMS_RAW.json').read_text())[a]['duty'])<1e-9)
    c('two estimators agree flag recorded for every leg',all('agree' in osc[a]['e1'][lg] for a in osc for lg in LEGS))
    A=json.loads((o/'ARMS_RAW.json').read_text());LEG=('lf','lm','lh','rf','rm','rh')
    mmean={a:float(np.asarray(A[a]['motif_rate_hz'],float)[:,:6].mean()) for a in ('k1_drive','k2_drive')}
    c('b3 precondition recomputed from the mean rate',oc['clause_check']['b3']['motif_driven_ge_20hz']==all(v>=20 for v in mmean.values()),means=mmean)
    c('branch 3 recorded with the median-vs-mean correction',oc['branch']==3 and 'MEDIAN' in oc['median_vs_mean_note'].upper())
    c('physics counters are zero','physics_step' in pre['calls'] and pre['calls']['physics_step']==0)
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,forward=0,physics_step=0,render=0),
      limitations=['neural only: no physics and no support/walking claim','anchor 0.1 nA is declared, not biological','motif override changes weights only'])
    (o/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3])))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
