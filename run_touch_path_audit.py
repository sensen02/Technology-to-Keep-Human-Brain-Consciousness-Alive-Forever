"""Bounded artifact-driven neural-only diagnostic; no physics, gain or wiring changes."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import signal
import time
from collections import deque
import numpy as np
from scipy import sparse
from engine.embodied.closure_bundle import validate_bundle
from engine.embodied.adapters import TierConfig, build_neural_tier
from engine.embodied.sensory_channels import ChannelConfig, ChannelQuantities, ChannelSensoryEncoder, channel_index_map, LEGS, CHANNEL_SPEC
from engine.embodied.motor_output import build_motor_manifest
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
ARMS = ('normal', 'no_external_touch')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def directed_hops(we, wi, seeds):
    """Matrices multiply presynaptic vectors: row=post, column=pre."""
    graph = ((sparse.csr_matrix(we) != 0) + (sparse.csr_matrix(wi) != 0)).astype(bool).tocsc()
    distance = np.full(graph.shape[0], -1, dtype=np.int64)
    queue = deque(int(i) for i in seeds)
    distance[list(queue)] = 0
    while queue:
        pre = queue.popleft()
        for post in graph.indices[graph.indptr[pre]:graph.indptr[pre+1]]:
            if distance[post] < 0:
                distance[post] = distance[pre] + 1
                queue.append(int(post))
    return distance


def reconstruct(record, cfg):
    if record.get('detail') != 'full' or 'channel_record' not in record:
        raise ValueError('Missing full channel record; absent quantities must not be invented')
    channel = record['channel_record']
    if channel['scale_nA_per_unit'] != cfg.current_nA_per_unit:
        raise ValueError('Sensory scale mismatch')
    result = {}
    for leg in LEGS:
        rows = channel['per_leg'][leg]
        if set(rows) != set(CHANNEL_SPEC):
            raise ValueError('Incomplete channel set')
        v = {k: float(rows[k]['value']) for k in CHANNEL_SPEC}
        if not np.isfinite(list(v.values())).all():
            raise ValueError('Nonfinite channel quantity')
        if v['club_vibration'] != 0 or v['multidendritic_noxious'] != 0:
            raise ValueError('Unexpected unmodelled nonzero channel')
        result[leg] = ChannelQuantities(v['claw_position']*cfg.reference_angle_rad,
            v['hook_direction']*cfg.reference_velocity_rad_s,
            v['hairplate_angle']*cfg.reference_angle_rad,
            v['campaniform_strain']*cfg.reference_force, v['bristle_touch'])
    return result


def aggregate(spikes):
    if spikes.ndim != 2 or spikes.shape[0] != 100:
        raise ValueError('Expected 100 half-ms samples')
    return spikes.reshape(10, 10, spikes.shape[1]).sum(axis=1)


def first(mask, end=True):
    at = np.flatnonzero(np.any(mask, axis=1))
    return float((int(at[0]) + int(end))*.5) if at.size else None


def hist(d):
    return {str(int(k)): int(np.sum(d == k)) for k in np.unique(d)}


def run(report):
    started = time.monotonic()
    bundle = validate_bundle(OUT)
    root = Path(bundle['path'])
    results = json.loads((root/'closure_results.json').read_text())
    saved = np.load(root/'closure_traces.npz', allow_pickle=False)
    manifest = json.loads((root/'motor_manifest.json').read_text())['measured']
    prov = results['provenance']
    # Closure driver can legitimately drift after publication; all imported replay
    # dependencies/data must still match their saved hashes, or interpretation stops.
    hashes = dict(prov['code_hashes'])
    hashes.update({str(HERE/'data/flywire/banc_connectome.npz'): prov['connectome_sha256'],
                   str(HERE/'data/banc888/banc_888_meta.feather'): prov['source_feather_sha256'],
                   str(HERE/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'): prov['pool_result_sha256'],
                   str(HERE/'outputs/neural_rebuild/sensory_channel_lookup.json'): prov['sensory_lookup_sha256']})
    source = {p: {'saved_sha256': h, 'current_sha256': sha(p)} for p,h in hashes.items()}
    drift = [p for p,row in source.items() if row['saved_sha256'] != row['current_sha256']]
    if any(Path(p).name != 'run_neural_muscle_closure.py' for p in drift):
        raise ValueError('Replay dependency/data source drift: '+repr(drift))
    assert results['measured']['duration_s'] == .05
    assert prov['network_weights_scale'] == .2 and prov['background_current_nA'] == 0
    cfg = ChannelConfig(**prov['sensory_translation'])
    records = {a: results['measured']['episodes'][a]['records'] for a in ARMS}
    assert all(len(r) == 10 for r in records.values())
    quantities = {a: [reconstruct(r,cfg) for r in records[a]] for a in ARMS}
    tier_cfg = TierConfig(background_mean_nA=0, background_sd_nA=0)
    built = build_neural_tier(tier_cfg)
    n = int(built['n'])
    names = [r['actuator'] for r in manifest['rows']]
    rebuilt = build_motor_manifest(json.loads((HERE/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text()), built['tier']['root_ids'], names)
    assert rebuilt == manifest, 'Built root-id/local-index manifest mapping mismatch'
    groups, stats = channel_index_map(n, built['tier']['root_ids'])
    seeds = np.asarray(groups['lf']['bristle_touch'],dtype=int)
    assert len(seeds) == 450 and len(np.unique(seeds)) == 450
    motors = np.array(sorted({i for r in manifest['rows'] for i in r['local_indices']}),dtype=int)
    we, wi = built['edges']['We']*.2, built['edges']['Wi']*.2
    hops = directed_hops(we, wi, seeds)
    arrays = {'root_ids': np.asarray(built['tier']['root_ids']), 'lf_bristle_indices':seeds,
              'motor_indices': motors, 'hop_distance': hops}
    matches = {}
    states = {}
    for arm in ARMS:  # strictly serial, reset exact rest state per arm
        net = ConductanceNetwork(n, ConductanceParams(), dt_ms=.5)
        net.set_connectivity(we,wi)
        enc = ChannelSensoryEncoder(groups,n,cfg)
        spikes = np.zeros((100,n),dtype=bool)
        states[arm] = np.zeros((100,3,n))
        currents = np.zeros((10,n))
        incoming = np.zeros((100,len(motors),2))
        active = np.zeros((100,len(motors)),dtype=bool)
        candidate = np.zeros((100,len(motors)))
        for k,q in enumerate(quantities[arm]):
            current, channel = enc.encode(q)
            for leg in LEGS:
                for ch in CHANNEL_SPEC:
                    assert channel['per_leg'][leg][ch]['n_cells'] == records[arm][k]['channel_record']['per_leg'][leg][ch]['n_cells']
            inject = arm == 'normal' and 1 <= k < 3
            assert records[arm][k]['external_lf_touch_stimulus'] == inject
            assert np.isclose(records[arm][k]['time_start_s'], k*.005, atol=1e-12,rtol=0)
            if inject:
                current[seeds] += .3
            currents[k] = current
            for j in range(10):
                t = k*10+j
                arrivals = net._ring[net._ring_i].copy()
                e, i = net.We@arrivals, net.Wi@arrivals
                incoming[t,:,0], incoming[t,:,1] = e[motors],i[motors]
                active[t] = (net.t_ms+1e-10 >= net.refractory_until_ms[motors])
                p = net.p
                ge,gi = net.ge+e,net.gi+i
                v = (p.capacitance_nF/net.dt*net.v+p.leak_uS*p.rest_mV+ge*p.exc_reversal_mV+gi*p.inh_reversal_mV+current-net.adaptation)/(p.capacitance_nF/net.dt+p.leak_uS+ge+gi)
                candidate[t] = v[motors]
                spikes[t] = net.step(current_nA=current)
                states[arm][t] = np.array([net.v,net.ge,net.gi])
                assert all(np.isfinite(x).all() for x in (net.v, net.ge, net.gi, net.adaptation, v))
        matches[arm] = bool(np.array_equal(aggregate(spikes),saved[arm+'_spike_counts']))
        if not matches[arm]:
            raise ValueError('Saved per-neuron per-5ms spike aggregates mismatch: '+arm)
        arrays[arm+'_spikes'] = spikes
        arrays[arm+'_currents_nA'] = currents
        arrays[arm+'_motor_incoming_uS'] = incoming
        arrays[arm+'_motor_active'] = active
        arrays[arm+'_motor_candidate_mV'] = candidate
    # Interpret only after BOTH full saved spike-count matrices matched exactly.
    diff = arrays['normal_spikes'] != arrays['no_external_touch_spikes']
    curdiff = arrays['normal_currents_nA']-arrays['no_external_touch_currents_nA']
    expected = np.zeros_like(curdiff); expected[1:3,seeds] = .3
    assert np.allclose(curdiff,expected,rtol=0,atol=1e-14), 'Other sensory feedback currents differ'
    motor_diff = diff[:,motors]
    arrival_diff = np.any(arrays['normal_motor_incoming_uS'] != arrays['no_external_touch_motor_incoming_uS'],axis=2)
    evidence = []
    for h in np.unique(hops):
        idx = np.flatnonzero(hops == h)
        sub = diff[:,idx]
        evidence.append({'hops': int(h), 'cells':len(idx), 'cells_with_spike_timing_difference':int(np.any(sub,axis=0).sum()),
                         'differing_binary_samples':int(sub.sum()), 'first_difference_end_ms':first(sub)})
    probe = {}
    for arm in ARMS:
        a = arrays[arm+'_motor_active']; v = arrays[arm+'_motor_candidate_mV']
        probe[arm] = {'differing_arrival_samples':int(arrival_diff.sum()),
            'differing_arrival_samples_refractory':int((arrival_diff & ~a).sum()),
            'differing_arrival_samples_active_below_threshold':int((arrival_diff & a & (v < net.p.threshold_mV)).sum()),
            'differing_arrival_samples_active_at_or_above_threshold':int((arrival_diff & a & (v >= net.p.threshold_mV)).sum())}
    npz = OUT/'touch_path_audit_spikes.npz'
    np.savez_compressed(npz, **arrays)
    guard = dict(report('completed'))
    if not np.isfinite(guard.get('previous_soft_limit_mb',0)):
        guard['previous_soft_limit_mb'] = None
        guard['previous_soft_limit_note'] = 'unlimited before guard (not a physical measurement)'
    payload = {'status':'PASS_DIAGNOSTIC_NOT_BIOLOGICAL_PROOF','source_bundle':bundle,
        'source_checksums':source, 'source_drift_metadata_only':drift,
        'audit_source_sha256':sha(__file__), 'spike_npz':str(npz),'spike_npz_sha256':sha(npz),
        'configuration':{'n_neurons':n,'dt_ms':.5,'duration_ms':50,'weight_scale':.2,'background_nA':0,
            'model_compiled':False,'actuator_names_source':'validated manifest; exact rebuilt manifest equality',
            'threads':1,'serial_arms':True,'wall_limit_s':60},
        'validation':{'manifest_mapping_exact':True,'all_10_full_records_each_arm':True,
            'saved_per_5ms_per_neuron_spike_aggregate_exact':matches,'finite':True},
        'graph':{'orientation':'postsynaptic rows / presynaptic columns, union nonzero We/Wi',
            'lf_bristle_cells':len(seeds),'motor_cells':len(motors),'reachable_cells':int((hops>=0).sum()),
            'lf_bristle_outgoing_We_nonzero':int(sparse.csc_matrix(we)[:,seeds].count_nonzero()),
            'lf_bristle_outgoing_Wi_nonzero':int(sparse.csc_matrix(wi)[:,seeds].count_nonzero()),
            'lf_bristle_incoming_We_nonzero':int(sparse.csr_matrix(we)[seeds,:].count_nonzero()),
            'lf_bristle_incoming_Wi_nonzero':int(sparse.csr_matrix(wi)[seeds,:].count_nonzero()),
            'reachable_motor_cells':int((hops[motors]>=0).sum()),'all_hop_histogram':hist(hops),
            'motor_hop_histogram':hist(hops[motors]),
            'minimum_motor_hops':int(hops[motors][hops[motors]>=0].min()) if np.any(hops[motors]>=0) else None},
        'current_difference':{'cells_each_interval':np.count_nonzero(curdiff,axis=1).tolist(),
            'only_450_lf_bristles_plus_0_3nA_intervals_1_2':True,'maximum_nA':float(curdiff.max())},
        'spikes':{'first_network_difference_end_ms':first(diff),'first_motor_difference_end_ms':first(motor_diff),
            'first_motor_incoming_difference_start_ms':first(arrival_diff,end=False),
            'network_differing_samples':int(diff.sum()),'motor_differing_samples':int(motor_diff.sum()),
            'network_total_each_arm':{a:int(arrays[a+'_spikes'].sum()) for a in ARMS},
            'motor_total_each_arm':{a:int(arrays[a+'_spikes'][:,motors].sum()) for a in ARMS},
            'network_cells_equal_total_but_different_timing':int(((arrays['normal_spikes'].sum(axis=0)==arrays['no_external_touch_spikes'].sum(axis=0)) & np.any(diff,axis=0)).sum()),
            'motor_cells_equal_total_but_different_timing':int(((arrays['normal_spikes'][:,motors].sum(axis=0)==arrays['no_external_touch_spikes'][:,motors].sum(axis=0)) & np.any(motor_diff,axis=0)).sum()),
            'propagation_by_shortest_hop':evidence},
        'state_difference_each_half_ms':{'sample_time':'end of step after reset/decay',
            'variables':['v_mV','ge_uS','gi_uS'],
            'all_cells_max_abs':np.max(np.abs(states['normal']-states['no_external_touch']),axis=2).tolist(),
            'motor_cells_max_abs':np.max(np.abs(states['normal'][:,:,motors]-states['no_external_touch'][:,:,motors]),axis=2).tolist()},
        'motor_arrival_threshold_refractory_probe':probe,
        'blockers':['Graph reachability alone is not physiological transmission proof.',
            'No motor spike difference in this window cannot distinguish a general missing behavior from this specific operating point.',
            'Post-arrival refractory and candidate-voltage classifications are model diagnostics, not validated biological mechanisms.'],
        'next_useful_scope':('No directed LF-bristle-to-motor path exists in this built tier. Audit these exact root IDs against source connectome outgoing edges and tier selection/edge construction to distinguish source coverage absence from tier truncation or conversion defects. Do not tune thresholds, gains or wiring to hide the gap.' if not np.any(hops[motors]>=0) else 'If arrivals differ but motor spikes do not, inspect exact affected motor-cell voltage, refractory phase and E/I increments across those times before proposing changes; no gains or wiring tuned.'),
        'concrete_blocker':('No motor cell reachable from LF bristles in exact built We/Wi graph; threshold/refractory motor explanations are not supported for this disconnected path.' if not np.any(hops[motors]>=0) else None),
        'memory_guard':guard,'wall_s':time.monotonic()-started}
    for p,row in source.items():
        assert sha(p) == row['current_sha256'], 'Source changed during audit: '+p
    (OUT/'touch_path_audit.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:payload[k] for k in ('validation','graph','current_difference','spikes','motor_arrival_threshold_refractory_probe','wall_s')},indent=2),flush=True)


if __name__ == '__main__':
    def deadline(*_):
        raise TimeoutError('Audit exceeded hard 60-second wall bound')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(60)
    with mg.guarded_run('touch_path_audit', **mg.BUDGETS['neural_muscle_closure']) as report:
        run(report)
    signal.alarm(0)
