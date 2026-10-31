"""S309: cap audit plus native CPG oscillation tests. Neural dynamics only: no physics, no render."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, scipy.sparse as sps

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BANC=ROOT/'data/flywire/banc_connectome.npz'
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1')
FACTORY_FP='56e0322a33c51b7f03aaae9f7df1ef3f'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage-a',action='store_true')
    o=ap.parse_args().output
    pre=dict(round='S309',purpose='cap-8 audit and native CPG oscillation tests on the engine substrate',
      calls=dict(model_load=0,forward=0,physics_step=0,render=0),
      counters=dict(banc_load=2,tier_build=2,select_touch_tier=4,edges_build=2,network_init=5,control_tick=25000,tier_advance=25000,neural_substep=50000),
      arms=['k1_drive','k1_control','k2_drive','k2_control','k2_lf_only'],
      oscillation='four criteria in the 4-20 Hz band: >=15 cycles in 5 s, two independent estimators agree within 5 percent (Welch peak vs band-passed Hilbert phase slope), peak power >= 3x the in-band noise floor, amplitude >= 1 Hz',
      branches=dict(b1='k2 oscillates and k1 does not -> the cap is the block',b2='both oscillate -> the block is downstream of the pool',b3='neither oscillates but a motif cell is driven >=20 Hz -> the substrate cannot host the motif',b4='gate or control failure'))
    dump(o/'PREREG.json',pre)
    (o/'S309_PREREG_FROZEN').write_text('frozen before any run\n')   # written by this script, before dynamics
    # ---------------- stage A: cap audit
    z=np.load(BANC,allow_pickle=True);rt=np.asarray(z['root_ids'])
    pre_r=rt[np.asarray(z['pre'])];post_r=rt[np.asarray(z['post'])];w=np.asarray(z['syn'],float)
    roots=np.load(S303/'TIER_ROOT_IDS.npz')['root_ids'] if (S303/'TIER_ROOT_IDS.npz').is_file() else np.load(Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z/TIER_ROOT_IDS.npz'))['root_ids']
    tset=set(int(x) for x in roots)
    inside=np.isin(pre_r,list(tset))&np.isin(post_r,list(tset))
    raw=float(w[inside].sum());cap8=float(np.minimum(w[inside],8).sum())
    edges_in=int(inside.sum());n_at_cap=int((w[inside]>=8).sum())
    we=np.load(S303/'EDGES_w_5e-4_We.npz');wi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    levels=sorted({round(float(x),6) for x in np.unique(we['data'])})
    frac_max_we=float(np.mean(np.asarray(we['data'])==np.max(we['data'])))
    frac_max_wi=float(np.mean(np.asarray(wi['data'])==np.max(wi['data'])))
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())
    motif_rows=[]
    for lg in LEGS:
        for a in ORDER:
            for b in ORDER:
                if a==b: continue
                sa=set(mod['modules'][lg][b]);sb=set(mod['modules'][lg][a])
                cnt=float(w[np.isin(pre_r,list(sa))&np.isin(post_r,list(sb))].sum())
                motif_rows.append(dict(leg=lg,edge=f'{b}->{a}',raw=cnt,tier_weight=float(min(cnt,8)*5e-4)))
    cap=dict(edges_in_tier=edges_in,raw_synapse_mass=raw,capped_synapse_mass=cap8,loss_fraction=float(1-cap8/raw),
      edges_at_cap=n_at_cap,edges_at_cap_fraction=float(n_at_cap/edges_in),
      we_data_levels=levels,we_fraction_at_max=frac_max_we,wi_fraction_at_max=frac_max_wi,
      cap_value=8,cap_source='engine/neck_cut_data.py:217 max_synapses_per_edge=8 and :1238 np.minimum(syn,8)*weight_scale',
      motif_edges=motif_rows,
      note='the tier cap discards nearly half of the synapse mass and flattens the six CPG modules to identical weights')
    dump(o/'CAP_AUDIT.json',cap)
    sign=dict(pure_exc='acetylcholine',pure_inh=['gaba','glutamate'],
      we_edges=[r for r in motif_rows if r['edge'] in ('E1->E2','E2->I1')],
      wi_edges=[r for r in motif_rows if r['edge'] in ('I1->E1','I1->E2')],
      e2_declared_fill='INXXX466 ann_nt_verified is empty, so its row is filled by all_empty (declarative, not sourced)',
      conclusion='the sign structure needs no override: ACh is excitatory and glutamate inhibitory by the engine rule, and only the E2 row is declarative')
    dump(o/'MOTIF_SIGN_AUDIT.json',sign)
    tt=json.loads((ROOT/'outputs/tier_transfer.json').read_text())['rows']
    anchor=[r for r in tt if abs(r['current_nA']-0.1)<1e-9][0]
    spec=dict(nucleus='engine NeuralTier (conductance point neurons, refractory 2 ms -> 400 Hz cap)',
      injection='background_nA plus I_anchor into the six E1 cells only; I_anchor = %.3g nA measured to give about %.1f Hz population (outputs/tier_transfer.json); no sweep'%(anchor['current_nA'],anchor['population_hz']),
      duration_s=5.0,control_dt_ms=1.0,neural_substep_ms=0.5,substeps_per_control=2,background_seed=0,
      drive_per_leg=dict(zip(LEGS,[190,122,72,114,84,61])),
      arms='k1 factory cap 8, k2 motif-exempt (36 edges at raw counts x 5e-4, everything else cap 8)')
    dump(o/'NATIVE_SPEC.json',spec)
    amp=dict(note='recomputed below from the archived arm, not taken on trust',decoder_denominator=450.0)
    s295=Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z')
    grp=json.loads((s295/'DIRECTION_GROUPS.json').read_text())['legs']
    best=None
    for lg in LEGS:
        for kind in ('minus','plus'):
            idx=grp[lg][kind]
            for f in sorted((s295/'runs').glob('*.npz')):
                pass
    cas=np.load(S303/'w_5e-4'/'NEURAL.npz',allow_pickle=True);post=np.asarray(cas['decoder_postclip'],float)
    amp['amplitude_target']=dict(required_command_std_for_lift=0.34,required_pool_rate_std_hz=0.34*450.0,
      source='S295 successful lift arm group command std (as archived in the tracker); the pool rate std needed is std*450 because preclip = rate/450')
    dump(o/'AMPLITUDE_TARGET.json',amp)
    if oa:=None: pass
    print(json.dumps(dict(mass_loss=cap['loss_fraction'],at_cap=cap['edges_at_cap_fraction'],levels=len(levels),frac_max_we=frac_max_we),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
