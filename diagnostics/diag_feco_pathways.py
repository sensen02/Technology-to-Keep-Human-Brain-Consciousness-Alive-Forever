#!/usr/bin/env python3
"""Isolated fixed sensory stimulation experiment; NOT autonomous sensing or behavior.

No body instantiated, motor commands generated, subtype direction inferred or
weights tuned. Motor annotations remain candidate anatomical correspondence.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/autonomous_pathways_round4'
from engine.embodied.adapters import TierConfig,build_neural_tier,NeuralTier,OwnershipLedger

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=ROOT/'outputs/autonomous_evidence_round3/sensory_evidence.json'
    motorfile=ROOT/'outputs/autonomous_mapping_round2/mapping_manifest.json'
    ids=json.loads(evidence.read_text())['explicit_root_ids_by_subtype']
    manifest=json.loads(motorfile.read_text())
    cfg=TierConfig(background_mean_nA=0.,background_sd_nA=0.)
    built=build_neural_tier(cfg) # once; topology and ordering shared by ALL cases
    roots=np.asarray(built['tier']['root_ids']).astype(str)
    local={r:i for i,r in enumerate(roots)}
    global_roots=set(np.asarray(built['data']['root_ids']).astype(str))
    assert len(local)==len(roots)
    groups={};missing={}
    for kind,rr in ids.items():
        assert len(rr)==len(set(rr))
        missing[kind]=[r for r in rr if r not in local]
        assert all(r not in global_roots for r in missing[kind]),'present source neuron excluded from tier'
        groups[kind]=np.array([local[r] for r in rr if r in local],dtype=int)
    assert {k:len(v) for k,v in groups.items()}=={'claw':52,'hook':52,'club':96}
    all_sensory=np.concatenate(list(groups.values()))
    assert len(np.unique(all_sensory))==200
    gg=built['global_index'][all_sensory]
    assert np.all(built['data']['ann_class'][gg]=='chordotonal_organ_neuron')
    motors={}
    for row in manifest['actuators']:
        if row['resolution']['status']!='EXACT_ANNOTATION_CANDIDATE':continue
        rr=[x['root_id'] for x in row['candidate_neurons']]
        assert rr and all(r in local for r in rr)
        ix=np.array([local[r] for r in rr],int)
        assert len(np.unique(ix))==len(ix)
        assert np.all(built['data']['ann_super_class'][built['global_index'][ix]]=='motor')
        motors[row['actuator']]=ix
    assert len(motors)==22
    all_mn=np.unique(np.concatenate(list(motors.values())))
    assert not np.intersect1d(all_sensory,all_mn).size
    # Matrices multiply presynaptic spike column: rows postsynaptic, columns presynaptic.
    adjacency=(built['edges']['We']!=0).maximum(built['edges']['Wi']!=0).tocsr()
    connectivity={}
    for kind,ix in groups.items():
        reached=np.zeros(built['n'],bool);reached[ix]=True
        depth=0
        while True:
            nxt=reached | (np.asarray(adjacency@reached).reshape(-1)>0)
            if np.array_equal(nxt,reached):break
            reached=nxt;depth+=1
        connectivity[kind]={'reachable_neurons':int(reached.sum()),'closure_iterations':depth,
          'motors':{name:{'reachable_MNs':int(reached[jj].sum()),'total_MNs':len(jj),
                          'direct_nonzero_conductance_edges':int(adjacency[jj][:,ix].nnz)} for name,jj in motors.items()},
          'scope':'unsigned graph reachability; edge count is NOT synapse count, sign, activation or function'}
    cut=dict(built);cut['edges']=dict(built['edges'])
    for key in ('We','Wi'):
        mat=built['edges'][key].copy();mat.data[:]=0;mat.eliminate_zeros();cut['edges'][key]=mat
        assert mat.nnz==0
    dt=cfg.dt_ms/1000
    n_on=round(.05/dt);n_off=round(.1/dt)
    assert np.isclose(n_on*dt,.05) and np.isclose(n_off*dt,.1)
    traces={};cases={}
    for case in ('no_stim','claw','hook','club','claw_propagation_cut'):
        tier=NeuralTier(cut if case=='claw_propagation_cut' else built,cfg,OwnershipLedger())
        assert np.all(tier.background_nA==0)
        stimulus='claw' if case=='claw_propagation_cut' else case
        current=np.zeros(built['n'])
        ix=groups.get(stimulus,np.array([],int));current[ix]=.05
        support=np.zeros(built['n'],bool);support[ix]=True
        assert not np.any(current[~support])
        s=np.concatenate([tier.advance(n_on,current),tier.advance(n_off,np.zeros(built['n']))])
        assert s.shape==(n_on+n_off,built['n']) and s.dtype==bool
        traces[case]=s
        def counts(sel):return {'pulse':int(s[:n_on,sel].sum()),'release':int(s[n_on:,sel].sum()),'total':int(s[:,sel].sum()),'neurons':len(sel)}
        cases[case]={'stimulated_subtype':stimulus if len(ix) else None,'stimulated_local_indices':ix.tolist(),
          'injected_current_per_selected_neuron_nA':.05 if len(ix) else 0.,'background_max_nA':float(np.abs(tier.background_nA).max()),
          'outside_selected_injection_max_nA':float(np.max(np.abs(current[~support]),initial=0)),
          'network_counts':counts(np.arange(built['n'])),
          'sensory_counts':{k:counts(v) for k,v in groups.items()},
          'motor_counts':{k:counts(v) for k,v in motors.items()},'unique_MN_counts':counts(all_mn),
          'whole_network_count_per_step':s.sum(axis=1).tolist(),
          'selected_sensory_count_per_step':s[:,ix].sum(axis=1).tolist(),
          'unique_MN_count_per_step':s[:,all_mn].sum(axis=1).tolist()}
        assert cases[case]['network_counts']['total']==int(np.count_nonzero(s))
        print(case,cases[case]['network_counts'],cases[case]['unique_MN_counts'],flush=True)
    assert not traces['no_stim'].any()
    outside=np.ones(built['n'],bool);outside[groups['claw']]=False
    assert not traces['claw_propagation_cut'][:,outside].any(),'cut still transmits spikes'
    assert traces['claw_propagation_cut'][:,groups['claw']].any(),'stimulus failed to reach sensory cells'
    sources=[Path(__file__),evidence,motorfile,ROOT/'engine/embodied/adapters.py',ROOT/'engine/neural_cond.py',ROOT/'data/flywire/banc_connectome.npz',ROOT/'data/flywire/banc_neurons.csv.gz']
    report={'overall_status':'ISOLATED_STIMULUS_PATHWAY_TEST_NOT_AUTONOMOUS','description':__doc__,
      'config':asdict(cfg),'tier_fingerprint':built['fingerprint'],'build_calls':1,
      'source_sha256':{str(p):sha(p) for p in sources},'sensory_root_ids':{k:roots[v].tolist() for k,v in groups.items()},
      'absent_from_npz_roots':missing,'motor_root_ids':{k:roots[v].tolist() for k,v in motors.items()},
      'clock':{'dt_s':dt,'pulse_s':.05,'release_s':.1,'step_timestamp':'end of neural step'},
      'propagation_cut':'same node ordering/initial state; copied We and Wi all zero, no rebuild; all chemical network propagation removed, not neuron lesion',
      'connectivity':connectivity,'cases':cases,'assertions':'identity/counts/clock/background/support/no-stim/cut locality all passed',
      'limitations':['fixed experimental injection, not environmental sensing','default conductance weights are illustrative','no body or motor command generation','no directional sign inferred','22 actuator annotations remain anatomical candidates']}
    np.savez_compressed(OUT/'spike_traces.npz',root_ids=roots,dt_s=dt,**traces)
    report['spike_trace_sha256']=sha(OUT/'spike_traces.npz')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    t=(np.arange(n_on+n_off)+1)*dt*1000
    for name,c in cases.items():
        for ax,key in zip(axes,('whole_network_count_per_step','selected_sensory_count_per_step','unique_MN_count_per_step')):
            ax.plot(t,c[key],label=name,alpha=.8)
    for ax,title in zip(axes,('Whole network','Stimulated sensory population','Unique MNs in 22 candidate muscle groups')):
        ax.axvspan(0,50,color='gray',alpha=.12);ax.set_ylabel('Spikes / 0.5 ms');ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (ms); fixed .05 nA pulse ends at 50 ms')
    fig.suptitle('ISOLATED STIMULUS PATHWAY TEST — NOT AUTONOMOUS\nNo body, motor commands, direction inference or weight tuning')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'pathways.png',dpi=130)
    return 0
if __name__=='__main__':raise SystemExit(main())
