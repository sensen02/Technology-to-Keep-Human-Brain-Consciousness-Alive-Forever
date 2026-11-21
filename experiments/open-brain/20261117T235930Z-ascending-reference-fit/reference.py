"""Source-supported AN identity and an explicitly NEW engineering teacher."""
import argparse
import time
import zipfile
import pyarrow.feather as feather
import pyarrow.parquet as parquet
from shared import *
from sources import flywire_annotations, verify_upstream
from circuit import Circuit


def binding():
    verify_upstream()
    catalog=read(BIDIR/'CATALOG.json')
    ports={r:p for r,p in catalog['ports'].items() if p['group']=='front_touch'}
    metadata={r['banc_888_id']:r for r in feather.read_table(ROOT/'data/banc888/banc_888_meta.feather').to_pylist()
              if r['banc_888_id'] in ports or r['banc_888_id'] in {x for p in ports.values() for x,_ in p['incoming_receptors']}}
    cls=flywire_annotations('classification');types=flywire_annotations('consolidated_cell_types')
    index,_=eon.get_hash_tables(eon.path_comp)
    data=parquet.read_table(eon.path_con,columns=['Postsynaptic_Index','Presynaptic_Index','Excitatory x Connectivity'])
    post,pre,weight=[c.to_numpy() for c in data.columns]
    source_files={ROOT/'data/banc888/banc_888_meta.feather',BIDIR/'CATALOG.json',eon.path_comp,eon.path_con,
                  ROOT/'data/flywire/fafb_classification.csv.gz',ROOT/'data/flywire/fafb_consolidated_cell_types.csv.gz'}
    side_text=(HERE/'sources/flywire_side_README.md').read_text()
    assert 'nerve-entry side for sensory/ascending neurons' in side_text
    source_files.update((HERE/'sources').iterdir())
    root_records=[]
    with zipfile.ZipFile(ROOT/'data/flywire/fafb_skeletons_swc.zip') as archive:
        for root,port in ports.items():
            m=metadata[root];wire=m['fafb_match']
            # The richer source metadata omitted by the earlier catalog supplies
            # a publisher-reviewed type and identity, not a guessed L/R swap.
            assert 'FAFB_MATCH_MANUALLY_CHECKED' in m['status'].split(',')
            assert m['fafb_cell_type']==m['fafb_alignment_cell_type']==types[wire]['primary_type']=='AN_GNG_178'
            assert m['fafb_nblast_match']==wire==port['fafb_match']
            assert cls[wire]['super_class']=='ascending' and cls[wire]['nerve']=='CV'
            assert m['root_region'].endswith('ProNM-T1') and m['neuromere']=='T1'
            assert m['neurotransmitter_verified']==port['transmitter']=='gaba'
            receptor_evidence=[]
            for receptor,count in port['incoming_receptors']:
                r=metadata[receptor]
                assert r['side']==port['side'] and r['cell_sub_class']=='front_leg_bristle_neuron'
                receptor_evidence.append(dict(root=receptor,synapses=count,body_side=r['side'],sub_class=r['cell_sub_class']))
            name=wire+'.swc';raw=archive.read(name);path=HERE/'sources'/name
            path.write_bytes(raw);source_files.add(path)
            geometry=BIDIR/'geometry'/port['object']['filename']
            assert sha(geometry)==port['object']['sha256'];source_files.add(geometry)
            hits=np.flatnonzero(pre==index[int(wire)])
            assert len(hits)>0 and (weight[hits]<0).all()
            boundary_feedback=np.isin(post[hits],[index[int(w)] for w in (p['fafb_match'] for p in ports.values())])
            root_records.append(dict(banc_root=root,body_input_side=port['side'],
                 banc_soma_side=m['side'],banc_soma_region=m['root_region'],banc_soma_position_nm=m['root_position_nm'],
                 flywire_root=wire,brain_entry_side=cls[wire]['side'],flywire_index=index[int(wire)],
                 type_pair=[m['cell_type'],m['fafb_cell_type']],publisher_status=m['status'],
                 neurotransmitter='gaba',signed_native_edges=len(hits),
                 edges_to_boundary_sources=int(boundary_feedback.sum()),receptors=receptor_evidence,
                 neck_geometry=port['geometry'],swc=str(path),swc_sha256=sha(path),
                 identity_scope='publisher manually checked paired identity; source-level engineering binding',
                 unresolved=['publisher ALT_MATCH/REVIEW flags retained; not a new biological homolog validation',
                             'no independent registration of complete brain morphology across donors']))
    assert len({r['flywire_root'] for r in root_records})==2
    pins=sources(source_files)
    write(HERE/'BINDING.json',dict(project='P1_open_source_brain',status='SOURCE_CURATED_ENGINEERING_BINDING',
          records=root_records,source_sha256=pins,
          side_resolution='BANC VNC soma/body-source side and FlyWire CV brain-entry side are distinct fields. '
                          'Keep publisher identity unchanged; do not swap roots or brain output sides.',
          boundary='prescribed AN output-event sources: inject once into original outgoing edges; native AN source '
                   'membrane excluded. Original inputs back onto the two externally owned ANs are not simulated; '
                   'this is a declared input-boundary intervention, not an unmodified full-brain/neck feedback loop.',
          teacher_kind='NEW_BANC_RECEPTOR_GRAPH_WITH_ORIGINAL_EON_LIF_ENGINEERING_REFERENCE',
          biological_fidelity='UNKNOWN beyond source identity, annotated synapses and skeletons'))
    verify(pins)
    print('source-curated front-touch identities bound; explicit body-side/entry-side fields',flush=True)


@torch.inference_mode()
def dataset(case):
    started=time.monotonic();bound=read(HERE/'BINDING.json');verify(bound['source_sha256'])
    folder=PREP/'DATA'/case;meta=read(folder/'RESULT.json');verify(meta['source_sha256'])
    assert sha(folder/'INPUTS.npz')==meta['inputs_sha256']
    with np.load(folder/'INPUTS.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    catalog=read(BIDIR/'CATALOG.json');ports={r:catalog['ports'][r] for r in meta['port_roots']}
    assert meta['cut_side']=='none' and list(ports)==[r['banc_root'] for r in bound['records']]
    native=Circuit(dict(catalog,ports=ports));cells=native._cells;state=cells.state_init()
    inputs=[native.validate_inputs([r for r,_ in p['incoming_receptors']],'body') for p in ports.values()]
    drive=np.zeros((1,len(native.roots)),dtype=np.float32)
    n=len(a['time_s']);events=np.zeros((n,2),dtype=bool)
    for tick in range(n):
        drive.fill(0)
        for c in range(2):
            if a['receptor_stimuli'][tick,c]:drive[0,inputs[c]]=68.75
        local=(native.local@state[2].numpy().T).T
        state=cells(torch.from_numpy(local*eon.MODEL_PARAMS['wScale']),torch.from_numpy(drive),*state)
        events[tick]=state[2][0,native.port_columns].numpy().astype(bool)
        if time.monotonic()-started>120:raise TimeoutError('reference generation budget')
    arriving=np.zeros_like(events)
    for c,d in enumerate(meta['port_delay_ticks']):arriving[d:,c]=events[:-d,c]
    # Causal 20 ms canonical rate is a declared engineering representation;
    # it preserves transit, rather than recovering the future undelayed input.
    target=np.zeros((n,2));rate=np.zeros(2);alpha=-np.expm1(-DT/.020)
    for tick in range(n):
        rate+=alpha*(arriving[tick]/DT-rate);target[tick]=rate
    assert np.isfinite(target).all() and target.max()<CAP_HZ
    out=HERE/'DATA'/case;out.mkdir(parents=True)
    pins=sources({HERE/'BINDING.json',folder/'INPUTS.npz',folder/'RESULT.json'})
    np.savez_compressed(out/'PAIRED.npz',features_hz=a['features_hz'],teacher_rate_hz=target,
         native_reference_emitted=events,native_reference_arrivals=arriving,
         receptor_stimuli=a['receptor_stimuli'],time_s=a['time_s'],student_arrivals=a['arrivals'])
    verify(pins)
    write(out/'RESULT.json',dict(case=case,split='train' if case in TRAIN_CASES else 'heldout',
        physical_episode_id=meta['source'],original_body_source_sha256=meta['source_sha256'][str(Path(meta['source'])/'TRACE.npz')],
        source_sha256=pins,trace_sha256=sha(out/'PAIRED.npz'),reference_ticks=n,new_neck_ticks=0,
        reused_real_fixed_neck_ticks=n,new_physical_steps=0,
        kind='NEW_ENGINEERING_REFERENCE_NOT_ARCHIVED_ORIGINAL_BODY_INTERFACE',
        source_port_order=list(ports),original_lif_events=events.sum(0).tolist(),target_peak_hz=target.max(0).tolist(),
        original_state_equations='unchanged Eon AlphaLIF on source BANC local receptor graph; no virtual neck in teacher soma',
        causal_transit_ticks=meta['port_delay_ticks'],canonical_rate_tau_s=.020,wall_seconds=time.monotonic()-started))
    print(case,'reference events',events.sum(0).tolist(),'target peak',target.max(0).round(2).tolist(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('binding',*TRAIN_CASES,*HELD_CASES));a=p.parse_args()
    torch.set_num_threads(1)
    binding() if a.stage=='binding' else dataset(a.stage)
