"""S306: resolve the sourced CPG circuit in the local BANC connectome. Connective-tissue audit only.

No physics, no neural dynamics: model_load=0, physics_step=0, forward=0, neural_tick=0, render=0.
Connectome work is declared separately: connectome_load and tier_build.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BANC=ROOT/'data/flywire/banc_connectome.npz'
TYPES=dict(E1='IN17A001',E2='INXXX466',I1='IN16B036',I2='IN19A007',E3='IN19B012',DNg100='DNg100',DNb08='DNb08')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--with-tier',action='store_true')
    a=ap.parse_args();out=a.output
    pre=dict(round='S306',mode='offline sourced-CPG audit; no dynamics',
             calls=dict(model_load=0,physics_step=0,forward=0,neural_tick=0,render=0),
             connectome_calls=dict(connectome_load=1 if a.with_tier else 0,tier_build=1 if a.with_tier else 0),
             hard_rule='the verifier may only write inside its own --output directory (S304 P1 lesson)',
             source=dict(doi='10.1101/2025.09.12.675944',title='Connectome simulations identify a central pattern generator circuit for fly walking',
                         authors='Pugliese SM, Chou GM, Abe ETT, Turcu D, Lancaster JK, Tuthill JC, Brunton BW',
                         pdf='https://faculty.washington.edu/tuthill/docs/Pugliese_cpg_2025.pdf',text_sha256=None),
             inputs={str(BANC):sha(BANC)})
    dump(out/'PREREG.json',pre);(out/'S306_PREREG_FROZEN').write_text('frozen\n')
    z=np.load(BANC,allow_pickle=True)
    keys=list(z.keys())
    rt=np.asarray(z['root_ids']);pt=np.asarray(z['ann_primary_type']).astype(str);sc=np.asarray(z['ann_super_class']).astype(str)
    ni={int(r):i for i,r in enumerate(rt)}
    # rows are neuron INDICES into root_ids, with 'syn' the synapse count
    pre_i=np.asarray(z['pre']).astype(np.int64)
    post_i=np.asarray(z['post']).astype(np.int64)
    w=np.asarray(z['syn'],float)
    pre_i[:]=rt[pre_i]; post_i[:]=rt[post_i]
    resolve={}
    for lab,name in TYPES.items():
        m=pt==name; idx=np.flatnonzero(m)
        ids=[int(x) for x in rt[idx]]
        resolve[lab]=dict(type=name,cells=int(m.sum()),root_ids=ids,super_class=sorted({str(x) for x in sc[idx]}),
                          total_synapses_out=float(w[np.isin(pre_i,ids)].sum()) if pre_i is not None and ids else 0.0,
                          total_synapses_in=float(w[np.isin(post_i,ids)].sum()) if post_i is not None and ids else 0.0,
                          in_tier=None)
    def syn(src_lab,dst_lab):
        s=set(resolve[src_lab]['root_ids']);d=set(resolve[dst_lab]['root_ids'])
        if not s or not d: return dict(count=0,synapses=0.0,measured_in_banc=False)
        m=np.isin(pre_i,list(s))&np.isin(post_i,list(d))
        return dict(count=int(m.sum()),synapses=float(w[m].sum()),measured_in_banc=bool(m.sum()>0))
    pairs={}
    for x in ('E1','E2','I1','I2','E3'):
        for y in ('E1','E2','I1','I2','E3'):
            if x!=y: pairs[f'{x}->{y}']=syn(x,y)
    dn={f'{dn}->{t}':syn(dn,t) for dn in ('DNg100','DNb08') for t in ('E1','E2','I1')}
    dump(out/'BANC_RESOLVE.json',dict(banc_keys=keys,n_neurons=int(len(rt)),n_synapses=int(len(pre_i)) if pre_i is not None else None,
        types=resolve,dn_resolution=dn,note='cell-type names are resolved by ann_primary_type; the paper defines its minimal circuit in MANC and FANC, so every BANC match is a cross-dataset mapping, not the paper circuit itself'))
    dump(out/'MOTIF_IN_BANC.json',dict(pairs=pairs,dn_edges=dn,
        paper=dict(circuit='E1=IN17A001, E2=INXXX466, I1=IN16B036 (MANC, 636/1024 pruning screens), FANC adds I2=IN19A007 and E3=IN19B012',
                   strong='E1->E2, E2->I1, I1->E1, I1->E2 are the particularly strong connections; all-to-all in the three-neuron circuit',
                   sufficient_subset='E1->E2, E2->I1, I1->E1 alone still oscillates',
                   drive='only E1 receives direct synaptic input from DNg100',oscillation_hz=14.083),
        comparison='each BANC edge is labelled measured_in_banc or not_measured; the ratio to the paper is not computed because the paper reports the motif in MANC/FANC and does not publish per-edge synapse counts in the main text'))
    nt=dict()
    for k in ('nt_neuron','nt_pair'):
        if k in z:
            v=np.asarray(z[k]); nt[k]=dict(dtype=str(v.dtype),shape=list(v.shape),unique=list(np.unique(v)[:5]),all_minus_one=bool(np.all(v==-1)))
    fafb=[]
    for name in ('fafb_consolidated_cell_types.csv.gz','fafb_classification.csv.gz'):
        p=ROOT/'data/flywire'/name
        if p.is_file(): fafb.append(dict(file=name,size=int(p.stat().st_size),sha256=sha(p),columns_scanned=False))
    dump(out/'SIGN_PROVENANCE.json',dict(banc_nt=nt,
        conclusion='NOT_ESTABLISHED locally: BANC stores no transmitter identity (nt_neuron/nt_pair are all -1), so the one-inhibitory/two-excitatory split cannot be read from this file',
        fafb_candidates=fafb,
        needed='an external transmitter prediction table (or the paper supplementary tables) mapping these cell types to cholinergic/GABAergic/glutamatergic'))
    if a.with_tier:
        import sys; sys.path.insert(0,str(ROOT))
        from engine.embodied.adapters import build_neural_tier, TierConfig
        built=build_neural_tier(TierConfig(sign_fill='all_empty').validate())
        roots=np.asarray(built['tier']['root_ids']).astype(np.uint64)
        np.savez_compressed(out/'TIER_ROOT_IDS.npz',root_ids=roots)
        tierset=set(int(x) for x in roots)
        for lab in resolve:
            resolve[lab]['in_tier']=[r for r in resolve[lab]['root_ids'] if r in tierset]
            resolve[lab]['in_tier_count']=len(resolve[lab]['in_tier'])
        dump(out/'BANC_RESOLVE.json',dict(banc_keys=keys,n_neurons=int(len(rt)),n_synapses=int(len(pre_i)),types=resolve,dn_resolution=dn,
            tier=dict(fingerprint=built['fingerprint'],n=int(built['n']),missing=[lab for lab in resolve if resolve[lab]['in_tier_count']<resolve[lab]['cells']]),
            note='TIER_ROOT_IDS.npz archived here (12000 ids) because S303-S305 never stored them; cell-type names resolved by ann_primary_type; the paper defines its circuit in MANC/FANC so BANC matches are cross-dataset'))
    print(json.dumps({lab:(TYPES[lab],resolve[lab]['cells'],resolve[lab].get('in_tier_count')) for lab in resolve},ensure_ascii=False))
    print(json.dumps({'pairs_nonzero':{k:v['synapses'] for k,v in pairs.items() if v['synapses']>0},'dn':{k:v['synapses'] for k,v in dn.items()}},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
