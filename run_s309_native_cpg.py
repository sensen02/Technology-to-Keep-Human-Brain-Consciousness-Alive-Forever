"""S309 stage B: five neural-only arms on the engine substrate (no physics, no render)."""
from __future__ import annotations
import argparse, json, hashlib, dataclasses
from pathlib import Path
import numpy as np, scipy.sparse as sps

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1')
FACTORY_FP='56e0322a33c51b7f03aaae9f7df1ef3f'
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    assert (o/'S309_PREREG_FROZEN').is_file()
    import sys; sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier, TierConfig, NeuralTier, OwnershipLedger
    calls=dict(banc_load=0,tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0)
    _orig=build_neural_tier
    def counted(cfg=None,data_path=None):
        calls['banc_load']+=1;calls['edges_build']+=1;calls['select_touch_tier']+=int(getattr(cfg,'growth_iterations',2));calls['tier_build']+=1
        return _orig(cfg,data_path)
    built1=counted(TierConfig(sign_fill='all_empty').validate())
    built2=counted(TierConfig(sign_fill='all_empty').validate())
    roots=np.asarray(built1['tier']['root_ids']).astype(np.uint64)
    loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    z=np.load(ROOT/'data/flywire/banc_connectome.npz',allow_pickle=True);rt=np.asarray(z['root_ids'])
    pre_r=rt[np.asarray(z['pre'])];post_r=rt[np.asarray(z['post'])];wsyn=np.asarray(z['syn'],float)
    def edges_of(built):
        e=built['edges']
        if isinstance(e,dict): return e['We'],e['Wi']
        return e.we,e.wi
    We1,Wi1=edges_of(built1);We2,Wi2=edges_of(built2)
    h1=dict(We=[sha_arr(We1.data),sha_arr(We1.indices),sha_arr(We1.indptr)],Wi=[sha_arr(Wi1.data),sha_arr(Wi1.indices),sha_arr(Wi1.indptr)])
    refW=np.load(S303/'EDGES_w_5e-4_We.npz');refI=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    gate=dict(fingerprint=built1['fingerprint'],fingerprint_ok=built1['fingerprint']==FACTORY_FP,
              We_match=[h1['We'][0]==sha_arr(refW['data']),h1['We'][1]==sha_arr(refW['indices']),h1['We'][2]==sha_arr(refW['indptr'])],
              Wi_match=[h1['Wi'][0]==sha_arr(refI['data']),h1['Wi'][1]==sha_arr(refI['indices']),h1['Wi'][2]==sha_arr(refI['indptr'])])
    gate['passed']=bool(gate['fingerprint_ok'] and all(gate['We_match']) and all(gate['Wi_match']))
    # k2 override: 36 motif edges at raw counts x 5e-4
    changed=[]
    def set_edge(W,a,b,val):
        """a = target cell set, b = source cell set; sets every matching entry."""
        olds=[]
        for tc in a:
            for sc in b:
                j=loc[int(tc)];i=loc[int(sc)]
                s0,e0=W.indptr[j],W.indptr[j+1];rows=W.indices[s0:e0]
                k=np.flatnonzero(rows==i)
                if k.size:
                    olds.append(float(W.data[s0+k[0]]));W.data[s0+k[0]]=val
        return (max(olds) if olds else None)
    for lg in LEGS:
        for tgt in ORDER:
            for src in ORDER:
                if src==tgt: continue
                sa=set(mod[lg][src]);sb=set(mod[lg][tgt])
                cnt=float(wsyn[np.isin(pre_r,list(sa))&np.isin(post_r,list(sb))].sum())
                if cnt<=0: continue
                val=cnt*5e-4
                W=Wi2 if src=='I1' else We2
                old=set_edge(W,sb,sa,val)
                changed.append(dict(leg=lg,edge=f'{src}->{tgt}',raw=cnt,new=val,old_tier=old))
    h2=dict(We=[sha_arr(We2.data),sha_arr(We2.indices),sha_arr(We2.indptr)],Wi=[sha_arr(Wi2.data),sha_arr(Wi2.indices),sha_arr(Wi2.indptr)])
    np.savez_compressed(o/'TIER_HASHES.npz',k1_We=np.array(h1['We']),k1_Wi=np.array(h1['Wi']),k2_We=np.array(h2['We']),k2_Wi=np.array(h2['Wi']))
    json.dump(dict(gate=gate,k1=h1,k2=h2,changed=changed,k1_fp=built1['fingerprint'],k2_fp=built2['fingerprint']),open(o/'TIER_GATE.json','w'),indent=2)
    # pool neuron sets per actuator (90) from the manifest pools
    import csv
    rows=list(csv.DictReader(open('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z/PER_CHANNEL.csv')))
    pool={}
    for r in rows:
        if r['pool_root_ids']:
            pool[r['name']]=[loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc]
    e1cells={lg:[loc[int(c)] for c in mod[lg]['E1'] if int(c) in loc] for lg in LEGS}
    drive=dict(zip(LEGS,[190,122,72,114,84,61]))
    I0=0.1   # nA anchor: measured tier transfer gives ~97 Hz at 0.1 nA
    arms=[('k1_drive',built1,'drive'),('k1_control',built1,'zero'),('k2_drive',built2,'drive'),('k2_control',built2,'zero'),('k2_lf_only',built2,'lf')]
    out={}
    for name,built,mode in arms:
        tier=NeuralTier(built,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());calls['network_init']+=1
        bg=np.asarray(tier.background_nA,float)
        cur=np.zeros(len(roots))
        if mode=='drive':
            for lg in LEGS:
                for c in e1cells[lg]: cur[c]=I0*drive[lg]/max(drive.values())
        elif mode=='lf':
            for c in e1cells['lf']: cur[c]=I0
        sp_rows=[];e1all=[c for lg in LEGS for c in e1cells[lg]];motif_ts=np.zeros((5000,len(e1all)));pool_ts=np.zeros((5000,len(pool)))
        pool_names=list(pool);applied=np.zeros(len(roots))
        for k in range(5000):
            calls['control_tick']+=1
            a=tier.advance(2,cur);calls['tier_advance']+=1;calls['neural_substep']+=2
            a=np.asarray(a,dtype=bool)
            sp_rows.append(a.any(axis=0))
            motif_ts[k]=a[:,e1all].mean(axis=0)
            pool_ts[k]=[a[:,pool[n]].mean() for n in pool_names]
            applied=bg+cur
        sp=np.asarray(sp_rows,dtype=np.uint8)
        np.savez_compressed(o/f'SPIKES_{name}.npz',packed=np.packbits(sp,axis=1),shape=np.array(sp.shape),counts=sp.sum(0).astype(np.int32))
        rates=np.asarray(motif_ts,float)*1000.0     # Hz per 1 ms bin
        cmd=np.clip((np.asarray(pool_ts,float)*1000.0)/450.0,0.0,1.0)
        out[name]=dict(duty=float(sp.mean()),motif_rate_hz=rates.tolist(),pool_cmd=cmd.tolist(),pool_names=pool_names,
                       applied_injection_nA=float(cur.max()),background_mean=float(bg.mean()))
    json.dump(out,open(o/'ARMS_RAW.json','w'),indent=2)
    json.dump(calls,open(o/'CALL_LEDGER.json','w'),indent=2)
    print(json.dumps(dict(gate=gate['passed'],fp=gate['fingerprint'],changed=len(changed),duty={k:round(v['duty'],5) for k,v in out.items()}),ensure_ascii=False))
    print(json.dumps(calls,ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
