"""Aggregate causal counterfactuals without changing any models or old results."""
from scope_diag import *

BASE_CASES=('native','reference','translated')
MACRO=('refadvance_studentturn','studentadvance_refturn')
DN_CASES=('swap_all_steer','swap_DNa02','swap_DNa01','swap_DNg13','swap_propulsion')


def result(seed,name,mode):
    folder=HERE/'BODY'/f'seed{seed}'/(name+'_'+mode)
    r=read(folder/'RESULT.json');verify(r['source_sha256']);assert sha(folder/'TRACE.npz')==r['trace_sha256']
    assert not r['warning_or_timeout']
    return r


def stats(c):
    return dict(integrated_advance=float(c.mean(1).sum()*DT),
                integrated_turn=float(((c[:,1]-c[:,0])/2).sum()*DT))


def main():
    saved=read(HERE/'SAVED_DIAGNOSTIC.json');catalog=read(BIDIR/'CATALOG.json')['ports']
    rows=[];safety=[];replay={};native_corrected={}
    for seed in SEEDS:
        r={name:result(seed,name,'ann') for name in BASE_CASES+MACRO+DN_CASES}
        teacher={name:result(seed,name,'teacher') for name in BASE_CASES}
        for x in [*r.values(),*teacher.values()]:safety.append(x['metrics']['safety_pass'])
        assert all(all(r[name]['live_trace_equal'].values()) for name in BASE_CASES)
        ya={n:r[n]['metrics']['yaw_degrees'] for n in r};yt={n:teacher[n]['metrics']['yaw_degrees'] for n in BASE_CASES}
        decomposition={}
        for ref in ('native','reference'):
            ann_delta=ya['translated']-ya[ref];teacher_delta=yt['translated']-yt[ref]
            residual=(ya['translated']-yt['translated'])-(ya[ref]-yt[ref])
            assert abs(ann_delta-teacher_delta-residual)<1e-10
            decomposition[ref]=dict(ann_yaw_delta=ann_delta,teacher_yaw_delta=teacher_delta,
                differential_ann_teacher_residual=residual,
                ann_reference_yaw=ya[ref],ann_candidate_yaw=ya['translated'],
                teacher_reference_yaw=yt[ref],teacher_candidate_yaw=yt['translated'])
        neck=read(HERE/'NECK'/f'seed{seed}'/'RESULT.json');verify(neck['source_sha256'])
        assert all(neck['canonical_command_exact'].values())
        with np.load(HERE/'NECK'/f'seed{seed}'/'TRACE.npz',allow_pickle=False) as a:
            commands=a['command'].copy();spikes=a['source_events'].copy()
        counter={}
        for name in DN_CASES:
            i=neck['names'].index(name)
            counter[name]=dict(yaw=ya[name],change_from_candidate=ya[name]-ya['translated'],
                difference_from_reference=ya[name]-ya['reference'],command=stats(commands[:,i]))
        macro=dict(reference_yaw=ya['reference'],candidate_yaw=ya['translated'],
             refadvance_studentturn_yaw=ya['refadvance_studentturn'],
             studentadvance_refturn_yaw=ya['studentadvance_refturn'],
             factorial_interaction_degrees=ya['translated']-ya['refadvance_studentturn']-ya['studentadvance_refturn']+ya['reference'])
        context_checks=dict(reference_DN_with_candidate_sense_command_exact=bool(np.array_equal(commands[:,3],commands[:,1])),
             candidate_DN_with_reference_sense_command_exact=bool(np.array_equal(commands[:,4],commands[:,2])))
        old_folder=PREVIOUS/'FORMAL'/f'seed{seed}'
        with np.load(old_folder/'NEURAL.npz',allow_pickle=False) as a:
            original={k:a[k].copy() for k in a.files}
        # Correct the old native_full AN-slot diagnostic: lane0 uses endogenous
        # AN spikes, not the all-zero external-event array. Its volley consumes
        # previous native spikes, so compare the effective current volley input.
        effective=np.zeros_like(original['full_an_spikes']);effective[1:]=original['full_an_spikes'][:-1]
        difference=np.flatnonzero(np.any(effective!=original['brain_an_events'][:,2],axis=1))
        native_corrected[str(seed)]=dict(effective_native_AN_first_difference_tick=int(difference[0]) if len(difference) else None,
             correction='native_full external source slot is intentionally zero; use prior native AN spikes for recurrent volley',
             body_and_DN_ordering_from_original_diagnostic_unchanged=True)
        d=np.flatnonzero(np.any(original['dn_emitted'][:,1]!=original['dn_emitted'][:,2],axis=1))[0]
        root_order=read(old_folder/'RESULT.json')['dn_roots']
        first=[dict(root=root_order[i],group=catalog[root_order[i]]['group'],side=catalog[root_order[i]]['side'],
                    reference=int(original['dn_emitted'][d,1,i]),candidate=int(original['dn_emitted'][d,2,i]))
                for i in np.flatnonzero(original['dn_emitted'][d,1]!=original['dn_emitted'][d,2])]
        rows.append(dict(seed=seed,ann_yaw=ya,teacher_yaw=yt,decomposition=decomposition,
            macro_command_interventions=macro,dn_interventions=counter,downstream_sense_context_checks=context_checks,
            first_reference_candidate_DN_difference=dict(tick=int(d),ports=first),
            canonical_replay_exact={n:all(r[n]['live_trace_equal'].values()) for n in BASE_CASES},
            projection_nodes=[neck['full_graph_nodes'],neck['projected_nodes']]))
    for name in ('translated','swap_DNg13'):
        folder=HERE/'REPLAY'/(name+'_ann');m=read(folder/'RESULT.json');verify(m['source_sha256'])
        assert sha(folder/'TRACE.npz')==m['trace_sha256']
        with np.load(folder/'TRACE.npz',allow_pickle=False) as a,np.load(HERE/'BODY/seed10901'/(name+'_ann')/'TRACE.npz',allow_pickle=False) as b:
            replay[name]={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
        assert all(replay[name].values());safety.append(m['metrics']['safety_pass'])
    write(HERE/'SAVED_DIAGNOSTIC_CORRECTION.json',dict(native_AN_input_corrected=native_corrected,
         previous_file_unchanged=True,scope='correct event interpretation only; no weights or simulation changed'))
    result_all=dict(project='P1_open_source_brain',status='COMMAND_BODY_LOCALIZATION_COMPLETED',rows=rows,
        canonical_body_replays_exact=True,independent_replay=replay,safety_cases=sum(safety),body_cases=len(safety),
        actual_body_steps=560000,actual_neck_batch_ticks=40000,actual_neck_lane_ticks=400000,
        actual_brain_ticks=0,optimizer_updates=0,
        interpretation='Command histories are causally upstream of physical divergence; teacher decompositions are '
        'relative-controller evidence, not a unique biological percentage. DN interventions are conditional and nonadditive.')
    write(HERE/'FINAL_LOCALIZATION.json',result_all)
    print('teacher decomposition',[(x['seed'],x['decomposition']) for x in rows],flush=True)
    print('DN swaps',[(x['seed'],{k:round(v['yaw'],4) for k,v in x['dn_interventions'].items()}) for x in rows],flush=True)
    print('all body safety',sum(safety),'/',len(safety),'replays exact',flush=True)


if __name__=='__main__':main()
