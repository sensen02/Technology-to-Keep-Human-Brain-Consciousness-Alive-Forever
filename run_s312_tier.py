"""S312 stage C: tier motif input isolation, eight arms."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, csv
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1');FP='56e0322a33c51b7f03aaae9f7df1ef3f'
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(banc_load=0,tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    assert (o/'S312_PREREG_FROZEN').is_file()
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    z=np.load(ROOT/'data/flywire/banc_connectome.npz',allow_pickle=True);rt=np.asarray(z['root_ids'])
    pre_r=rt[np.asarray(z['pre'])];post_r=rt[np.asarray(z['post'])];wsyn=np.asarray(z['syn'],float)
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    cells={lg:{a:[loc[int(c)] for c in mod[lg][a] if int(c) in loc] for a in ORDER} for lg in LEGS}
    M18=sorted({c for lg in LEGS for a in ORDER for c in cells[lg][a]});mset=set(M18)
    rows=list(csv.DictReader(open('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z/PER_CHANNEL.csv')))
    pool=[]
    for r in rows:
        if r['pool_root_ids']:
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: pool.append(loc[int(x)])
    pool=sorted(set(pool))
    e1=[c for lg in LEGS for c in cells[lg]['E1']]
    def build():
        b=build_neural_tier(TierConfig(sign_fill='all_empty').validate());C['tier_build']+=1;C['banc_load']+=1;C['edges_build']+=1;C['select_touch_tier']+=2;return b
    def get_edges(b):
        e=b['edges'];return (e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    b0=build();We0,Wi0=get_edges(b0)
    gate=dict(fingerprint=b0['fingerprint'],ok=b0['fingerprint']==FP,
      We=[sha_arr(We0.data)==sha_arr(np.load(S303/'EDGES_w_5e-4_We.npz')['data']),
          sha_arr(We0.indices)==sha_arr(np.load(S303/'EDGES_w_5e-4_We.npz')['indices']),
          sha_arr(We0.indptr)==sha_arr(np.load(S303/'EDGES_w_5e-4_We.npz')['indptr'])],
      Wi=[sha_arr(Wi0.data)==sha_arr(np.load(S303/'EDGES_w_5e-4_Wi.npz')['data']),
          sha_arr(Wi0.indices)==sha_arr(np.load(S303/'EDGES_w_5e-4_Wi.npz')['indices']),
          sha_arr(Wi0.indptr)==sha_arr(np.load(S303/'EDGES_w_5e-4_Wi.npz')['indptr'])])
    gate['passed']=bool(gate['ok'] and all(gate['We']) and all(gate['Wi']))
    dump(o/'GATE.json',gate)
    if not gate['passed']:
        dump(o/'STOP.json',dict(reason='factory tier gate failed',gate=gate));dump(o/'CALL_LEDGER.json',C);return 2
    def isolate(W):
        removed=0;mass=0.0
        for j in M18:
            s,e=W.indptr[j],W.indptr[j+1]
            for t in range(s,e):
                src=int(W.indices[t])
                if src not in mset:
                    mass+=float(W.data[t]);W.data[t]=0.0;removed+=1
        return removed,mass
    def set_motif(W,scale,cap=None):
        n=0
        for lg in LEGS:
            for tgt in ORDER:
                for src in ORDER:
                    if src==tgt: continue
                    sa=set(mod[lg][src]);sb=set(mod[lg][tgt])
                    cnt=float(wsyn[np.isin(pre_r,list(sa))&np.isin(post_r,list(sb))].sum())
                    if cnt<=0: continue
                    v=cnt*scale if cap is None else min(cnt,cap)*scale
                    for tc in sb:
                        j=loc[int(tc)]
                        for sc in sa:
                            i=loc[int(sc)]
                            s,e=W.indptr[j],W.indptr[j+1]
                            k=np.flatnonzero(W.indices[s:e]==i)
                            if k.size: W.data[s+k[0]]=v;n+=1
        return n
    arms=[('T1_gate',False,'cap8',0.0),('T2_iso_cap8_d050',True,'cap8',0.05),('T3_iso_cap8_d160',True,'cap8',0.16),
          ('T4_iso_cap8_d000',True,'cap8',0.0),('T5_iso_raw_d050',True,'raw',0.05),('T6_iso_raw_d160',True,'raw',0.16),
          ('T7_noiso_cap8_d160',False,'cap8',0.16),('T8_noiso_raw_d160',False,'raw',0.16)]
    res={}
    for name,iso,fam,I in arms:
        b=build() if name!='T1_gate' else b0
        We,Wi=get_edges(b)
        audit=dict(isolated=iso,family=fam,current_nA=I)
        if fam=='raw': audit['motif_edges_set']=set_motif(Wi,5e-4) + set_motif(We,5e-4)
        if iso:
            r1,m1=isolate(We);r2,m2=isolate(Wi);audit.update(removed=r1+r2,removed_mass=m1+m2)
        audit['nonnegative']=bool(np.all(We.data>=0) and np.all(Wi.data>=0))
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());C['network_init']+=1
        bg=np.asarray(tier.background_nA,float);cur=np.zeros(len(roots))
        if I>0:
            for c in e1: cur[c]=I
        n=5000;sp=np.zeros((n,len(roots)),dtype=bool)
        for k in range(n):
            sp[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0);C['control_tick']+=1;C['tier_advance']+=1;C['neural_substep']+=2
        e1t=sp[:,e1];i1t=sp[:,[c for lg in LEGS for c in cells[lg]['I1']]]
        def rate(m): return float(m.mean()*2000.0)
        idx=np.flatnonzero(i1t.any(axis=1))
        base=float(e1t.mean())
        cond=[]
        for t in idx:
            hi=min(n,t+10)
            cond.append(float(e1t[t:hi].mean()))
        ratio=float(np.median(cond)/max(base,1e-12)) if cond and base>0 else None
        res[name]=dict(audit=audit,motif_rate_hz={f'{lg}_{a}':rate(sp[:,[c for c in cells[lg][a]]]) for lg in LEGS for a in ORDER},
          e1_rate_hz=rate(e1t),e2_rate_hz=rate(sp[:,[c for lg in LEGS for c in cells[lg]['E2']]]),
          i1_rate_hz=rate(i1t),pool_rate_hz=rate(sp[:,pool]),duty=float(sp.mean()),
          i1_to_e1_conditional_ratio=ratio,n_i1_spike_ms=int(len(idx)))
        np.savez_compressed(o/f'SPK_{name}.npz',motif=sp[:,M18],pool=sp[:,pool])
    dump(o/'ARMS.json',res);dump(o/'CALL_LEDGER.json',C)
    print(json.dumps(dict(gate=gate['passed'],counters=C,
      e1={k:round(v['e1_rate_hz'],1) for k,v in res.items()},ratio={k:(round(v['i1_to_e1_conditional_ratio'],3) if v['i1_to_e1_conditional_ratio'] else None) for k,v in res.items()}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
