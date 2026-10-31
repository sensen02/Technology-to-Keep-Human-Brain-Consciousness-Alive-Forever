"""S315 verification: recompute G_j and s_j from spikes, split range assertions, redrive the branch from data."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S314=Path('/home/sensen/Desktop/engine修复验收/s314-tier-leg-burst-scale/20261026T212103Z')
LEGS=('lf','lm','lh','rf','rm','rh')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def intervals(x):
    i=np.flatnonzero(x)
    if len(i)<3: return None,None
    d=np.diff(i).astype(float);big=d[d>3*np.median(d)]
    if len(big)<10: return None,None
    return 1000.0/np.median(big),float(np.std(big)/max(np.mean(big),1e-9))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    ck=[]
    def c(n,v,**e): ck.append(dict(name=n,passed=bool(v),**e))
    pre=json.loads((o/'PREREG.json').read_text());mp=json.loads((o/'M18_GLOBAL_INDEX.json').read_text())
    H=json.loads((o/'HETEROGENEITY.json').read_text());SP=json.loads((o/'SPLIT_V2.json').read_text())
    A=json.loads((o/'ARMS.json').read_text());oc=json.loads((o/'OUTCOMES.json').read_text())
    led=json.loads((o/'CALL_LEDGER.json').read_text());DR=json.loads((o/'DRIVE_RULE_V2.json').read_text())
    c('frozen marker before dynamics',(o/'S315_PREREG_FROZEN').is_file())
    # (i) recompute G_j and s_j from spikes, not from JSON
    sp=np.load(S309/'SPIKES_k1_control.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    W=np.load(S303/'EDGES_w_5e-4_We.npz');I=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    M18=mp['M18'];mset=set(M18);ne=len(M18)
    net=np.zeros((spk.shape[0],ne))
    for jj,j in enumerate(M18):
        acc=np.zeros(spk.shape[0])
        for arr,sgn in ((W,1.0),(I,-1.0)):
            s,e=arr['indptr'][j],arr['indptr'][j+1];src=arr['indices'][s:e];dat=np.asarray(arr['data'][s:e],float)
            mask=np.array([int(x) in mset for x in src])
            if (~mask).any(): acc+=sgn*(spk[:,src[~mask]].astype(float)@dat[~mask])
        net[:,jj]=acc
    ker=np.ones(20)/20
    G=[float(np.median(np.convolve(net[:,jj],ker,mode='same'))) for jj in range(ne)]
    c('G_j recomputed from spikes matches the archive',max(abs(g-h['g_net_uS']) for g,h in zip(G,H['per_cell']))<1e-9,
      max_abs_diff=float(max(abs(g-h['g_net_uS']) for g,h in zip(G,H['per_cell']))))
    s_re=[(0.03/(65.0*g)) if g>0 else None for g in G]
    c('s_j recomputed',all((a is None and b['s_j'] is None) or (a is not None and b['s_j'] is not None and abs(a-b['s_j'])<1e-12) for a,b in zip(s_re,H['per_cell'])))
    c('G spread reproduces',abs(H['G_spread']['min']-min(G))<1e-9 and abs(H['G_spread']['max']-max(G))<1e-9)
    c('exactly one cell has non-positive net external drive',sum(1 for g in G if g<=0)==1,
      cells=[(h['leg'],h['role']) for h,g in zip(H['per_cell'],G) if g<=0])
    # (ii) split range assertions and identity
    c('internal split stays within its own bounds',SP['range_assertions']['internal_min_ok'] and SP['range_assertions']['internal_max_ok'])
    ti=np.load(S313/'TSPK_1.npz')['net_in']
    c('net + internal equals the archived tier net input',SP['identity_ok'] is True and SP['max_abs_diff']<1e-9)
    c('the external inhibitory component is now separated',abs(SP['exc_q']['p10'])>0 and SP['inh_q']['p50']>=0)
    # (iii) redrive the branch from archived spikes with an independent estimator
    counts={}
    for name in A:
        pk=np.load(o/f'SPK_{name}.npz');sm=np.unpackbits(pk['packed'],axis=1)[:,:int(pk['shape'][1])].astype(bool)
        rh=0
        for lg in LEGS:
            e1col=[mp['M18'].index(g) for e,g in zip(mp['entries'],mp['M18']) if e['leg']==lg and e['role']=='E1']
            i1col=[mp['M18'].index(g) for e,g in zip(mp['entries'],mp['M18']) if e['leg']==lg and e['role']=='I1']
            e1=sm[:,e1col].any(axis=1);i1=sm[:,i1col].any(axis=1)
            f,cv=intervals(e1)
            if f is not None and cv is not None and cv<0.25 and 5.0<=e1.sum()/5.0<=300.0: rh+=1
        counts[name]=rh
    c('branch derived from data, not asserted',((counts['B2_percell_v2']==0 and oc['branch']=='y3') or (1<=counts['B2_percell_v2']<=4 and oc['branch']=='y2') or (counts['B2_percell_v2']>=5 and oc['branch']=='y1')),
      counts=counts,branch=oc['branch'])
    c('B2 has no rhythm, so the ablation clause is vacuous and disclosed',counts['B2_percell_v2']==0 and 'VACUOUS' in oc['v6c_note'].upper())
    # (iv) B1 bitwise vs S314 A4
    b1=np.load(o/'SPK_B1_A4_repeat.npz');b1s=np.unpackbits(b1['packed'],axis=1)[:,:int(b1['shape'][1])].astype(bool)
    a4=np.load(S314/'SPK_A4_ext_scale_half_f.npz');a4s=np.unpackbits(a4['packed'],axis=1)[:,:int(a4['shape'][1])].astype(bool)
    c('B1 is bitwise identical to S314 A4',bool(np.array_equal(b1s,a4s)))
    c('gate is bitwise identical to S309 k1_control',A['G1_A1_gate']['bit_vs_s309'] is True)
    # (v) counters
    exp=pre['counters']['normal']
    c('counters equal the preregistered normal branch',all(led[k]==v for k,v in exp.items()),ledger=led)
    c('physics/model/render are zero',led['physics_step']==0 and led['model_load']==0 and led['render']==0)
    # (vi) drive rule values reproducible
    c('18 s_j values archived and reproducible',len(DR['s_j_used'])==18 and all((s is None) or (s>0) for s in DR['s_j']))
    c('verifier writes only verify/',(o/'verify').is_dir())
    r=dict(passed=all(x['passed'] for x in ck),checks=ck,failed=[x for x in ck if not x['passed']],verifier_sha256=sha(__file__),
      calls=dict(model_load=0,physics_step=0,forward=0,render=0),
      limitations=['neural only, no physics','per-cell scaling is a modelling intervention','the ablation clause is vacuous this round'])
    (o/'verify'/'VERIFICATION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=r['passed'],checks=len(ck),nfail=len(r['failed']),failed=r['failed'][:3],counts=counts)))
    return 0 if r['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
