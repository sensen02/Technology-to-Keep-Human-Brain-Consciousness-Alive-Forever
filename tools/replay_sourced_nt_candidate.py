"""Explicit offline S4 sourced-NT replay. No body/tier/training/production admission.
Run once, then verify in a fresh pure-array process, then seal once.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
import numpy as np
from scipy import sparse

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
B = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
G = Path('/home/sensen/Desktop/engine修复验收/s3-sourced-nt-candidate-graph/20261102T004326Z')
O = Path('/home/sensen/Desktop/engine修复验收/s4-sourced-nt-dynamic-replay/20261102T011240Z')
A = Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
FIELDS = ['threshold_offset_mV', '_ring', 'v', 'ge', 'gi', 'adaptation', 'refractory_until_ms', 'spike_count']
LABELS = ['BASELINE', 'CANDIDATE', 'SELECTED_CURRENT_CUT']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def matrix_exact(a, b):
    return (same(a.indptr, b.indptr) and same(a.indices, b.indices) and same(a.data, b.data))


def first(a, b, dt, offset=0.):
    ix = np.flatnonzero(np.any(a != b, axis=tuple(range(1, a.ndim))))
    return None if not len(ix) else float(offset + (int(ix[0]) + 1) * dt)


def checked_sources():
    pr = load(O / 'PREREGISTRATION.json')
    for p, h in pr['source_sha256'].items():
        assert sha(p) == h, p
    return pr


def context():
    ids = arrays(B / 'IDENTITIES.npz')
    selected = arrays(G / 'IDENTITIES_AND_SELECTED.npz')
    assert all(same(ids[k], selected[k]) for k in ids)
    roots = ids['root_ids']; sel = selected['selected_local']
    mapping = load(B / 'RUN/MAPPING.json')
    motor = np.array(sorted({i for row in mapping['rows'] for i in row['local_indices']}))
    assert len(roots) == len(set(roots)) == 12000 and len(sel) == len(set(sel)) == 963 and len(motor) == 204
    assert not set(sel) & set(motor)
    for row in mapping['rows']:
        assert list(roots[row['local_indices']]) == row['tier_root_ids']
    boundary = arrays(B / 'RUN/NEURAL_BOUNDARY.npz')
    warm = arrays(B / 'PREWARM_FINAL_STATE.npz')
    assert all(same(v, warm[k]) for k, v in boundary.items())
    clock = load(B / 'PREWARM_CLOCK.json')
    assert clock['net_scalars'] == dict(n=12000, dt=.5, delay_steps=4, _ring_i=0, t_ms=1100., steps=2200)
    assert clock['decoder_elapsed_s'] == 1.1 and clock['decoder_interval'] == 1099
    assert int(warm['steps']) == 2200 and int(warm['ring_pos']) == 0 and float(warm['time_s']) == 1.1
    weights = {k: sparse.load_npz(B / ('ACTUAL_' + k + '.npz')).tocsr() for k in ['We', 'Wi']}
    added = {k: sparse.load_npz(G / ('ADDED_' + k + '.npz')).tocsr() for k in weights}
    candidate = {k: sparse.load_npz(G / ('CANDIDATE_' + k + '.npz')).tocsr() for k in weights}
    for k in weights:
        assert matrix_exact(candidate[k] - added[k], weights[k]), k
        assert weights[k].multiply(added[k]).nnz == 0
        assert np.all(candidate[k].data > 0) and np.isfinite(candidate[k].data).all()
    assert added['We'].nnz == 17264 and added['Wi'].nnz == 0
    coo = added['We'].tocoo()
    assert set(coo.col) <= set(sel)
    assert np.all((weights['We'] + weights['Wi']).getnnz(axis=0)[sel] == 0)
    records = load(B / 'RUN/NEURAL_RECORDS.json')[:50]
    neural = arrays(B / 'RUN/NEURAL.npz')
    drive = neural['sensory_current_nA'][:50]
    assert same(drive, np.array([r['sensory_current_nA'] for r in records]))
    groups = load(B / 'SENSORY_GROUPS.json')
    lookup = load(A / 'SENSORY_LOOKUP_SNAPSHOT888.json')['leg_channel_by_root']
    rebuilt = np.zeros_like(drive)
    for leg, channels in groups.items():
        for channel, indices in channels.items():
            for i in indices:
                assert lookup[str(int(roots[i]))] == [leg, channel]
            for w, rec in enumerate(records):
                item = rec['channels']['per_leg'][leg][channel]
                rebuilt[w, indices] = rec['channels']['scale_nA_per_unit'] * float(item['value']) * item['weight']
    assert same(drive, rebuilt)
    expected = arrays(B / 'NEURAL_RASTER.npz')['spikes'][2200:2300]
    assert same(expected.reshape(50, 2, 12000).sum(1), neural['spike_counts'][:50])
    return roots, sel, motor, mapping, boundary, clock, weights, added, candidate, drive, records, expected


def prepare():
    assert not O.exists(), 'Refuse overwrite of an existing evidence package'
    c = context()
    source = [Path(__file__), B / 'ACTUAL_We.npz', B / 'ACTUAL_Wi.npz', B / 'IDENTITIES.npz',
        B / 'PREWARM_FINAL_STATE.npz', B / 'PREWARM_CLOCK.json', B / 'RUN/NEURAL_BOUNDARY.npz',
        B / 'RUN/NEURAL.npz', B / 'RUN/NEURAL_RECORDS.json', B / 'NEURAL_RASTER.npz',
        B / 'RUN/MAPPING.json', B / 'SENSORY_GROUPS.json', B / 'NEURAL_IDENTITY.json',
        B / 'MANIFEST_SHA256.json', G / 'MANIFEST_SHA256.json', A / 'SENSORY_LOOKUP_SNAPSHOT888.json',
        A / 'MN_POOLS_SNAPSHOT888.json', A / 'VERSION_CONTRACT.json']
    source += [G / name for name in ['CANDIDATE_We.npz','CANDIDATE_Wi.npz','ADDED_We.npz','ADDED_Wi.npz',
        'IDENTITIES_AND_SELECTED.npz','RESULT.json','INDEPENDENT_VERIFICATION.json','SELECTED_SOURCE_PROVENANCE.json',
        'ADDED_RAW_ROWS.csv','SOURCE_TO_MN_REACHABILITY.json','EXECUTED_build_sourced_nt_candidate_graph.py']]
    source += [R / name for name in ['engine/neural_cond.py','engine/embodied/motor_output.py',
        'engine/embodied/continuous_handover.py','engine/embodied/sensory_channels.py','tools/build_sourced_nt_candidate_graph.py']]
    graph = load(G / 'RESULT.json')
    for p, h in graph['source_sha256'].items():
        assert sha(p) == h, p
    source += [Path(p) for p in graph['source_sha256']]
    assert load(G / 'INDEPENDENT_VERIFICATION.json')['passed']
    O.mkdir(parents=True)
    dump(O / 'PREREGISTRATION.json', dict(stage='S1/S3/S4 fixed saved sensory replay, explicit offline receptor hypothesis',
        source_sha256={str(p):sha(p) for p in dict.fromkeys(source)},
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        arms=LABELS, gate='Actual BASELINE100 ticks must match original round13 same-package raster, counts, decoder rates, decoded/applied commands exactly. Stop before interventions on any failure.',
        added_cut='candidate minus ADDED exactly reconstructs old CSR support and values; reuse BASELINE without extra integration',
        selected_current_cut='candidate matrix, only963 selected sensory currents zeroed; other current and all initial states identical. Inherited boundary ring spikes still propagate; NOT a full source or edge cut.',
        schedule=dict(neural_dt_ms=.5,command_window_ms=1.,ticks_per_arm=100,windows_per_arm=50,total_ticks=300,decoder_tau_s=.005,rate_full_hz=100.,max_command=1.,per_actuator_gain=None),
        parameters='Original neutral ConductanceParams and zero threshold offsets; no learned gains, no training, no score/selection changes',
        arrival_trace='Per tick exact consumed delay-ring arrivals, positive added-edge event count by postsynaptic target and added excitatory conductance increment vector. Also full v/ge/gi/adaptation traces; changes can have multiple recurrent pathways.',
        verification='New process, pure arrays only: roots/manifest/input reconstruction, raw positive edge masks, delay-ring chronology, counts, decoder filter/commands, first differences, windows5/15/50ms, all source hashes.',
        limits='No physical body/model/forward/step, no new tier, no production guard bypass or default change. Physiological calibration, support improvement and walking NOT_ESTABLISHED.',
        planned_calls=dict(network_instances=3,neural_steps=300,body_model=0,body_forward=0,body_steps=0,tier_build=0,training=0)))
    Path(O / 'EXECUTED_replay_sourced_nt_candidate.py').write_bytes(Path(__file__).read_bytes())
    roots, sel, motor, mapping, boundary, clock, weights, added, candidate, drive, records, expected = c
    dump(O / 'MAPPING.json', mapping); dump(O / 'INITIAL_CLOCK.json', clock)
    np.savez_compressed(O / 'INITIAL_BOUNDARY.npz', **boundary)
    cut = drive.copy(); cut[:, sel] = 0
    np.savez_compressed(O / 'INPUTS.npz', root_ids=roots, selected_local=sel, motor_local=motor,
        saved_current_nA=drive, cut_current_nA=cut, selected_mask=np.isin(np.arange(12000),sel),
        added_post=added['We'].tocoo().row, added_pre=added['We'].tocoo().col, added_weight_uS=added['We'].tocoo().data)
    dump(O / 'IDENTITY_AND_INTERFACE.json', dict(n=12000,selected=963,MN=204,
        candidate_status=graph['status'],runtime_allowed=False, known_labels_overridden=0,
        source_rule='Frozen upstream verified-empty in both sources; previously nonzero-driven zero modeled-outdegree non-MN; official v2 pure NT score>=.9 and count>=20; unchanged selection.',
        selected_NT=graph['counts']['selected_NT'],added_edges=17264,added_inhibitory_edges=0,
        GABA_no_raw_output=True,collective_reachable_MN=204,sources_reaching_any_MN=961,
        encoder_saved_groups_and_root888_lookup_exact=True,added_cut_CSR_exact=True,
        all204MN_same_saved_decoder_boundary=True,decoder_newMN_cold_reset=False))
    return c


def snapshot(net, dec, pending):
    return dict(**{k:getattr(net,k).copy() for k in FIELDS}, decoder_rates=dec.rates.copy(),pending=pending.copy(),
        steps=np.array(net.steps),t_ms=np.array(net.t_ms),ring_i=np.array(net._ring_i),
        decoder_elapsed_s=np.array(dec.elapsed_s),decoder_interval=np.array(dec.interval))


def simulate(label, c, calls):
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from engine.embodied.motor_output import MotorOutput
    roots, sel, motor, mapping, boundary, clock, weights, added, candidate, drive, records, expected = c
    net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=.5, threshold_offset_mV=boundary['threshold_offset_mV'])
    calls['network_instances'] += 1
    for k in FIELDS:
        if k != 'threshold_offset_mV': setattr(net, k, boundary[k].copy())
    for k in ['steps','t_ms','_ring_i']: setattr(net,k,clock['net_scalars'][k])
    mats = weights if label == 'BASELINE' else candidate
    net.set_connectivity(mats['We'], mats['Wi'])
    dec = MotorOutput(mapping); dec.rates = boundary['decoder_rates'].copy()
    dec.elapsed_s = clock['decoder_elapsed_s']; dec.interval = clock['decoder_interval']
    pending = boundary['pending'].copy()
    np.savez_compressed(O / (label + '_INITIAL.npz'), **snapshot(net,dec,pending))
    assert all(same(snapshot(net,dec,pending)[k], boundary[k]) for k in boundary)
    I = drive.copy()
    if label == 'SELECTED_CURRENT_CUT': I[:,sel] = 0
    ev = np.zeros((100,12000),bool); arrivals = np.zeros_like(ev)
    traces = {k:np.empty((100,12000)) for k in ['v','ge','gi','adaptation']}
    increment = np.empty((100,12000)); event_counts = np.empty((100,12000),np.int32)
    half_rates = np.empty((100,90)); applied = np.empty((50,90)); decoded = np.empty_like(applied)
    poolcounts = np.empty((50,90),np.int64)
    positive = added['We'].astype(bool).astype(np.int32)
    decay = math.exp(-.0005/.005)
    for w in range(50):
        applied[w] = pending
        rate = dec.rates.copy()
        for sub in range(2):
            t = 2*w+sub; arrivals[t] = net._ring[net._ring_i]
            increment[t] = added['We'] @ arrivals[t]
            event_counts[t] = positive @ arrivals[t].astype(np.int32)
            ev[t] = net.step(I[w]); calls['neural_steps'] += 1
            for k in traces:
                traces[k][t] = getattr(net,k)
                assert np.isfinite(traces[k][t]).all()
            inst = np.array([ev[t,g].mean()/.0005 if g.size else 0. for g in dec.groups])
            rate = decay*rate + (1-decay)*inst; half_rates[t] = rate
        pending, trace = dec.consume(ev[2*w:2*w+2],spike_dt_s=.0005,start_s=1.1+w*.001,end_s=1.1+(w+1)*.001)
        assert same(rate,dec.rates)
        decoded[w] = pending; poolcounts[w] = trace['pool_spike_counts']
    np.savez_compressed(O / (label + '_FINAL.npz'), **snapshot(net,dec,pending))
    np.savez_compressed(O / (label + '_NEURAL.npz'),spikes_packed=np.packbits(ev,axis=1),counts=ev.reshape(50,2,12000).sum(1),
        motor_events=ev[:,motor],selected_events=ev[:,sel],arrivals_packed=np.packbits(arrivals,axis=1),
        added_arrival_increment_uS=increment,added_arrival_event_counts=event_counts,
        added_arrivals_applied=np.array(label!='BASELINE'),**traces)
    np.savez_compressed(O / (label + '_DECODE.npz'),rates_halfstep_hz=half_rates,rates_window_hz=half_rates[1::2],
        applied=applied,decoded_next=decoded,pool_counts=poolcounts,
        applied_source_interval=np.arange(1099,1149),decoded_source_interval=np.arange(1100,1150))
    return ev


def gate():
    ev = np.unpackbits(arrays(O / 'BASELINE_NEURAL.npz')['spikes_packed'],axis=1,count=12000).astype(bool)
    d = arrays(O / 'BASELINE_DECODE.npz'); n = arrays(O / 'BASELINE_NEURAL.npz')
    original = arrays(B / 'NEURAL_RASTER.npz')['spikes'][2200:2300]
    records = load(B / 'RUN/NEURAL_RECORDS.json')[:50]
    checks = dict(raster=same(ev,original),counts=same(n['counts'],arrays(B / 'RUN/NEURAL.npz')['spike_counts'][:50]))
    for name, field, decoder in [('applied','applied_command',False),('decoded_next','decoded_next_command',False),
            ('rates_window_hz','rates_hz',True),('pool_counts','pool_spike_counts',True)]:
        ref = np.array([r['decoder'][field] if decoder else r[field] for r in records])
        checks[name] = same(d[name],ref)
    checks['saved_neural_commands'] = same(d['applied'],arrays(B / 'RUN/NEURAL.npz')['commands'][:50])
    dump(O / 'BASELINE_GATE.json',dict(passed=all(checks.values()),checks=checks,
        first_raster_difference_ms=first(ev,original,.5),actual_neural_steps=100))
    assert all(checks.values()), checks


def metrics():
    inp = arrays(O / 'INPUTS.npz'); sel=inp['selected_local']; motor=inp['motor_local']
    m = load(O / 'MAPPING.json'); unique = list(dict.fromkeys(tuple(r['local_indices']) for r in m['rows'] if r['local_indices']))
    indices = [next(i for i,r in enumerate(m['rows']) if tuple(r['local_indices']) == group) for group in unique]
    ns = {l:arrays(O / (l+'_NEURAL.npz')) for l in LABELS}
    ds = {l:arrays(O / (l+'_DECODE.npz')) for l in LABELS}
    es = {l:np.unpackbits(ns[l]['spikes_packed'],axis=1,count=12000).astype(bool) for l in LABELS}
    answer = dict(windows={},differences={},arrival={},unique_present_pools=len(unique),
        added_cut=dict(exact_matrix_restoration=True,reused_arm='BASELINE',extra_steps=0),
        limits='Saved input, no physical feedback. Source current zeroing retains initial ring/ge/gi and recurrent drive; not complete cut. Raster differences cannot establish a unique causal path. No physiological or support improvement claim.')
    for ms in [5,15,50]:
        out={}
        for l in LABELS:
            e=es[l][:2*ms]; d=ds[l]
            out[l]=dict(network_spikes=int(e.sum()),selected_spikes=int(e[:,sel].sum()),MN204_spikes=int(e[:,motor].sum()),
                active_neurons=int(e.any(0).sum()),active_MN=int(e[:,motor].any(0).sum()),
                peak_network_spikes_per_tick=int(e.sum(1).max()),mean_network_rate_hz=float(e.sum()/12000/(ms*.001)),
                muscle_applied_nonzero=int(np.count_nonzero(d['applied'][:ms])),muscle_applied_saturated=int((d['applied'][:ms]>=1).sum()),
                muscle_next_nonzero=int(np.count_nonzero(d['decoded_next'][:ms])),muscle_next_saturated=int((d['decoded_next'][:ms]>=1).sum()),
                unique_pool_applied_nonzero=int(np.count_nonzero(d['applied'][:ms,indices])),
                unique_pool_applied_saturated=int((d['applied'][:ms,indices]>=1).sum()),
                max_pool_rate_hz=float(d['rates_halfstep_hz'][:2*ms].max()))
        answer['windows'][str(ms)+'ms']=out
    for l, ref in [('CANDIDATE','BASELINE'),('SELECTED_CURRENT_CUT','CANDIDATE')]:
        n=ns[l]; b=ns[ref]; d=ds[l]; bd=ds[ref]
        answer['differences'][l+'_vs_'+ref]=dict(
            first_network_spike_difference_end_ms=first(es[l],es[ref],.5),
            first_selected_spike_difference_end_ms=first(es[l][:,sel],es[ref][:,sel],.5),
            first_MN_spike_difference_end_ms=first(es[l][:,motor],es[ref][:,motor],.5),
            first_voltage_difference_end_ms=first(n['v'],b['v'],.5),
            first_MN_voltage_difference_end_ms=first(n['v'][:,motor],b['v'][:,motor],.5),
            first_ge_difference_end_ms=first(n['ge'],b['ge'],.5),
            first_MN_ge_difference_end_ms=first(n['ge'][:,motor],b['ge'][:,motor],.5),
            first_input_difference_start_ms=first(inp['cut_current_nA'],inp['saved_current_nA'],1.,-1.) if l=='SELECTED_CURRENT_CUT' else None,
            first_decoded_difference_available_ms=first(d['decoded_next'],bd['decoded_next'],1.),
            first_applied_difference_window_start_ms=first(d['applied'],bd['applied'],1.,-1.),
            MN_different_bits=int(np.count_nonzero(es[l][:,motor]!=es[ref][:,motor])),
            applied_different_elements=int(np.count_nonzero(d['applied']!=bd['applied'])),
            max_applied_abs_difference=float(np.max(np.abs(d['applied']-bd['applied']))))
    for l in LABELS:
        n=ns[l]; counts=n['added_arrival_event_counts']; inc=n['added_arrival_increment_uS']
        answer['arrival'][l]=dict(applied=l!='BASELINE',
            first_added_event_start_ms=first(counts,np.zeros_like(counts),.5,-.5),
            first_direct_MN_added_event_start_ms=first(counts[:,motor],np.zeros_like(counts[:,motor]),.5,-.5),
            added_edge_event_count=int(counts.sum()),direct_MN_edge_event_count=int(counts[:,motor].sum()),
            nonMN_edge_event_count=int(counts.sum()-counts[:,motor].sum()),
            summed_added_conductance_increment_uS=float(inc.sum()),
            inherited_boundary_ring_events_first5ticks=int(counts[:5].sum()))
    return answer


def run():
    calls=dict(network_instances=0,neural_steps=0,body_model=0,body_forward=0,body_steps=0,tier_build=0,training=0)
    c=prepare()
    try:
        simulate('BASELINE',c,calls); gate()
        simulate('CANDIDATE',c,calls); simulate('SELECTED_CURRENT_CUT',c,calls)
        result=metrics();result['calls']=calls;result['baseline_gate_passed']=True
        checked_sources();dump(O/'RESULT.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException:
        dump(O/'FAILURE.json',dict(calls=calls,traceback=traceback.format_exc()))
        raise


def verify():
    # No neural/body/decoder runtime import or call: pure saved-array arithmetic.
    checked_sources();c=context()
    roots,sel,motor,mapping,boundary,clock,weights,added,candidate,drive,records,expected=c
    inp=arrays(O/'INPUTS.npz')
    assert same(inp['saved_current_nA'],drive) and same(inp['root_ids'],roots)
    assert same(inp['selected_local'],sel) and same(inp['motor_local'],motor)
    cut=drive.copy();cut[:,sel]=0
    assert same(inp['cut_current_nA'],cut)
    assert same(inp['selected_mask'],np.isin(np.arange(12000),sel))
    assert same(cut[:,~inp['selected_mask']],drive[:,~inp['selected_mask']])
    # Independently reconstruct positive added support/weights from frozen raw row witnesses.
    rows=list(csv.DictReader((G/'ADDED_RAW_ROWS.csv').open()))
    post=[];pre=[];val=[]
    for row in rows:
        i=int(row['pre_local']);j=int(row['post_local']);w=min(float(row['synapses']),8)*.0005*.2
        assert int(row['pre_root888'])==roots[i] and int(row['post_root888'])==roots[j]
        assert i in sel and row['predicted_NT']=='acetylcholine' and row['receptor_assumption']=='EXC'
        assert float(row['conductance_increment_uS'])==w and w>0
        post.append(j);pre.append(i);val.append(w)
    reconstructed=sparse.coo_matrix((val,(post,pre)),shape=(12000,12000)).tocsr()
    assert matrix_exact(reconstructed,added['We']) and len(rows)==17264
    coo=reconstructed.tocoo()
    assert same(inp['added_post'],coo.row) and same(inp['added_pre'],coo.col) and same(inp['added_weight_uS'],coo.data)
    maxerr=0.; verified_ticks=0
    for l in LABELS:
        n=arrays(O/(l+'_NEURAL.npz'));d=arrays(O/(l+'_DECODE.npz'))
        init=arrays(O/(l+'_INITIAL.npz'));end=arrays(O/(l+'_FINAL.npz'))
        assert all(same(init[k],boundary[k]) for k in boundary)
        assert (int(init['steps']),float(init['t_ms']),int(init['ring_i']),float(init['decoder_elapsed_s']),int(init['decoder_interval']))==(2200,1100.,0,1.1,1099)
        e=np.unpackbits(n['spikes_packed'],axis=1,count=12000).astype(bool)
        arrival=np.unpackbits(n['arrivals_packed'],axis=1,count=12000).astype(bool)
        assert same(n['counts'],e.reshape(50,2,12000).sum(1))
        assert same(n['motor_events'],e[:,motor]) and same(n['selected_events'],e[:,sel])
        ring=boundary['_ring'].copy();pos=0
        rate=boundary['decoder_rates'].copy();decay=math.exp(-.0005/.005)
        for t in range(100):
            assert same(arrival[t],ring[pos])
            # bincount by raw positive edges is independent of CSR dot implementation.
            active=arrival[t,coo.col]
            counts=np.bincount(coo.row[active],minlength=12000)
            increments=np.bincount(coo.row[active],weights=coo.data[active],minlength=12000)
            assert same(counts,n['added_arrival_event_counts'][t])
            assert same(increments,n['added_arrival_increment_uS'][t])
            ring[pos]=e[t];pos=(pos+1)%5
            instant=np.array([sum(int(e[t,i]) for i in r['local_indices'])/len(r['local_indices'])/.0005 if r['local_indices'] else 0. for r in mapping['rows']])
            rate=decay*rate+(1-decay)*instant
            err=float(np.max(np.abs(rate-d['rates_halfstep_hz'][t])))
            assert err<=1e-12;maxerr=max(maxerr,err)
            if t%2==1:
                w=t//2
                assert same(d['pool_counts'][w],np.array([e[2*w:2*w+2,r['local_indices']].sum() for r in mapping['rows']]))
                assert np.max(np.abs(d['decoded_next'][w]-np.clip(rate/100,0,1)))<=1e-14
            verified_ticks+=1
        assert same(d['rates_window_hz'],d['rates_halfstep_hz'][1::2])
        assert same(d['applied'][0],boundary['pending']) and same(d['applied'][1:],d['decoded_next'][:-1])
        assert same(end['_ring'],ring) and int(end['ring_i'])==pos
        assert same(end['spike_count'],boundary['spike_count']+e.sum(0))
        assert (int(end['steps']),float(end['t_ms']),float(end['decoder_elapsed_s']),int(end['decoder_interval']))==(2300,1150.,1.1500000000000001,1149)
        for k in ['v','ge','gi','adaptation']:assert same(end[k],n[k][-1])
        assert same(end['decoder_rates'],d['rates_halfstep_hz'][-1]) and same(end['pending'],d['decoded_next'][-1])
        if l=='BASELINE':assert same(e,expected)
    gate();assert load(O/'BASELINE_GATE.json')['passed']
    actual=load(O/'RESULT.json');recomputed=metrics()
    for k,v in recomputed.items():assert v==actual[k],k
    assert actual['calls']['neural_steps']==300 and actual['calls']['network_instances']==3
    checked_sources()
    dump(O/'INDEPENDENT_VERIFICATION.json',dict(passed=True,process_pid=os.getpid(),pure_arrays=True,
        neural_steps=0,body_steps=0,network_instances=0,verified_saved_ticks=verified_ticks,
        max_decoder_filter_abs_error=maxerr,raw_positive_edges=17264,identity_and_input_exact=True,
        initial_states_identical=True,delay_arrivals_exact=True,added_cut_exact=True,
        baseline_original_raster_exact=True,metrics_first_differences_recomputed=True,source_hashes_unchanged=True))
    print('Independent pure-array verification PASS',maxerr)


def seal():
    assert not (O/'MANIFEST_SHA256.json').exists(), 'Already sealed: no post-seal writes'
    checked_sources();assert load(O/'INDEPENDENT_VERIFICATION.json')['passed']
    result=load(O/'RESULT.json')
    lines=['# 固定感觉输入下的来源NT候选神经重放','',
        '本次为S1/S3/S4离线显式研究。第13轮root888原包完整动态边界恢复门通过；全部204MN沿用相同90池滤波状态和pending，无冷置。未训练、未混入上轮三个gain。',
        '实际3个网络、300神经tick（每臂50ms/100tick）；0身体模型/forward/step、0新tier、0训练。新增边切断逐CSR精确还原旧矩阵，直接复用BASELINE，无额外积分。',
        '候选仍为未准入的预测递质/受体假设；无生产guard或默认运行修改。963来源中962ACh/1GABA，17264新增ACh边；GABA在当前tier无rawout。集合布尔可达204MN，仅961来源可达任意MN，不解释为每源到全部MN。','',
        '## 前5/15/50ms（脉冲与命令计数）','',
        '|窗|臂|全网spike|selected spike|204MN spike|肌肉applied非零|肌肉applied饱和|峰值每tick全网spike|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for w,arms in result['windows'].items():
        for l,v in arms.items():
            lines.append('|'+ '|'.join(map(str,[w,l,v['network_spikes'],v['selected_spikes'],v['MN204_spikes'],v['muscle_applied_nonzero'],v['muscle_applied_saturated'],v['peak_network_spikes_per_tick']]))+'|')
    lines+=['','## 首次差异与到达时间','', '全部时间为释放边界后ms。spike/状态时间为tick末；新增突触到达为tick首；decoded在窗口末可用、applied在下一窗首生效。','',
        '```json',json.dumps(result['differences'],ensure_ascii=False,indent=2),'```','',
        '```json',json.dumps(result['arrival'],ensure_ascii=False,indent=2),'```','',
        '## 解释范围','',
        '候选对基准是否改变MN事件、cut对候选是否敏感，由上述MN差异和命令首差明确报告。全网活动率、峰值每tick发放、活跃神经元及池/肌肉饱和分别在RESULT中；脉冲或命令增加不等于协调输出，更不等于支撑改善。',
        'SELECTED_CURRENT_CUT仅置零963来源保存感觉电流，其余电流及神经初态不变；边界ring/ge/gi与后续回返网络仍传播，不能称完全source cut。',
        '逐tick记录新增边事件到达计数和conductance增量，直接到MN与非MN分别统计；BASELINE中该trace是未实际施加的反事实新增事件。非MN接收不自动证明间接抵达MN，网络raster变化不能完全归于一条路径。',
        '本任务没有新的实时身体感觉闭环或物理结果；不称生理验证、承重改善或自主行走。参数/NT符号及受体均未获生理标定。',
        '独立新进程只用数组重算完整300tick的counts、delayring、新增事件到达/正边mask、90池滤波、命令窗口、first differences、input cut与identity，未调用神经积分。','',
        '## 清理与重要程度','',
        '本次无确认无用的任务产物，实际删除0；完整事件/状态/输入、源快照、原包hash与独立验证均保留。原封包只读，manifest最后生成，封存后不追加。',
        '核心必需：仓库工具、PREREGISTRATION、EXECUTED源、INPUTS、INITIAL边界/clock、MAPPING、三臂INITIAL/FINAL/NEURAL/DECODE、RESULT、BASELINE_GATE与INDEPENDENT_VERIFICATION。重要辅助：本报告、JOB_LEDGER、FILE_INVENTORY、MANIFEST。完整路径和用途见FILE_INVENTORY。']
    (O/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    inventory={}
    for p in sorted(O.iterdir()):
        if p.is_file():
            aux=p.name in ['REPORT.zh-CN.md','JOB_LEDGER.json']
            inventory[str(p)]=dict(importance='重要辅助' if aux else '核心必需',purpose='报告/执行记录' if aux else '重放、验证及来源复核所需')
    for name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:
        inventory[str(O/name)]=dict(importance='重要辅助',purpose='文件用途与最终只读hash清单')
    inventory[str(Path(__file__).resolve())]=dict(importance='核心必需',purpose='显式离线重放与纯数组独立验证入口')
    dump(O/'FILE_INVENTORY.json',dict(files=inventory,cleanup_deleted=0,cleanup_reason='No confirmed disposable outputs; scientific traces and failure sources retained'))
    dump(O/'MANIFEST_SHA256.json',dict(files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.iterdir()) if p.is_file()},sealed_once=True))
    print('Sealed',O)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify','seal']);args=parser.parse_args()
    {'run':run,'verify':verify,'seal':seal}[args.mode]()
