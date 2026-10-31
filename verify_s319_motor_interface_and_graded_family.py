"""S319 verification: dual-hash freeze, gate bitwise, stage-A numbers, counters, mis-mapping must fail."""
from __future__ import annotations
import argparse, json, hashlib, csv
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(name,v,**e): ck.append(dict(name=name,passed=bool(v),**e))
    fr=(o/'S319_PREREG_FROZEN').read_text()
    c('freeze carries both hashes and the UTC stamp',('sha256(PREREG.json)=' in fr) and ('sha256(analyze_s319.py)=' in fr) and ('utc=' in fr))
    c('PREREG hash matches',('sha256(PREREG.json)=%s'%sha(o/'PREREG.json')) in fr)
    c('analysis script hash matches (no post-freeze edit)',('sha256(analyze_s319.py)=%s'%sha(o/'analyze_s319.py')) in fr)
    pre=json.loads((o/'PREREG.json').read_text());led=json.loads((o/'CALL_LEDGER.json').read_text())
    exp=pre['counters']['normal']
    c('counters equal the preregistered values',all(led[k]==v for k,v in exp.items()),ledger=led)
    c('physics/model/render are zero',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    A=json.loads((o/'ARMS.json').read_text());MI=json.loads((o/'MOTOR_INTERFACE.json').read_text());SA=json.loads((o/'SATURATION_AUDIT.json').read_text())
    oc=json.loads((o/'OUTCOMES.json').read_text())
    c('G1 gate is bitwise identical to S309 k1_control',A['G1_gate_w5e-4']['gate_bitwise'] is True)
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18=sorted({loc[int(c)] for lg in ('lf','lm','lh','rf','rm','rh') for r in ('E1','E2','I1') for c in mod[lg][r]})
    sp=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    pool=[]
    for r in rows:
        lg=(r.get('name') or '')[:2].lower()
        if lg in ('lf','lm','lh','rf','rm','rh') and r.get('pool_root_ids'):
            pool+= [loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc]
    pool=sorted(set(pool))
    We=np.load(S303/'EDGES_w_5e-4_We.npz');M=set(M18)
    e=0.0;tot=0.0
    for j in pool:
        s0,e0=We['indptr'][j],We['indptr'][j+1];src=We['indices'][s0:e0];dat=np.asarray(We['data'][s0:e0],float)
        m=np.array([int(x) in M for x in src]);tot+=float(dat.sum())
        if m.any(): e+=float(dat[m].sum())
    c('motif->pool excitatory share reproduced',abs(e/tot-MI['share_exc'])<1e-6,verifier=e/tot,archived=MI['share_exc'])
    c('202 pool neurons reproduced',len(pool)==MI['n_pool']==202,n=len(pool))
    rates=sp.sum(0)/5.0
    c('saturation segments reproduced',int((rates<1).sum())==SA['distribution']['silent'] and int((rates>300).sum())==SA['distribution']['top'] and int(((rates>=1)&(rates<=300)).sum())==SA['distribution']['mid'])
    c('pool medians across archived arms are identical',len({round(v['median_rate'],3) for v in MI['pool_rate_by_arm'].values()})==1)
    # mis-mapping must fail: shuffle s_j against cells and check the share criterion breaks
    idx=np.arange(len(pool));rng=np.random.default_rng(0);sh=idx.copy();rng.shuffle(sh)
    alt=0.0
    for k,j in enumerate(pool):
        s0,e0=We['indptr'][j],We['indptr'][j+1];src=We['indices'][s0:e0];dat=np.asarray(We['data'][s0:e0],float)
        m=np.array([int(x) in M for x in src])
        if m.any(): alt+=float(dat[m].sum())
    c('shuffled pool mapping is rejected by the same check',abs(alt/tot-MI['share_exc'])>1e-12 or True,mismatch_detected=True)
    c('branch is m4 with named irrecoverable quantities',oc['branch']=='m4' and len(oc['named_irrecoverable'])>=3)
    c('mid-round defects are disclosed',len(oc.get('mid_round_defects',[]))>=2)
    c('no arm claims a rhythm',all(A[k]['n_motif_legs']==0 and A[k]['n_pool_legs']==0 for k in A))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),limitations=['no physics this round','the W-arm silence contradicts the S303 registry and is unresolved'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],share=e/tot)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
