"""Read-only normal-current replay and motor state observations, not causal attribution.

No body is built. Existing neural step is called unchanged; pre-update drives use
arrivals at interval start, before decay. End voltages explicitly include resets.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import run_neural_muscle_closure as clo
from run_connected_feedback_audit import EXPECTED_RUN, PROPRIO, sha, json_safe_guard
from engine.embodied.closure_bundle import validate_bundle

OUT = clo.OUT


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()))


def current_stats(x):
    x = np.asarray(x)
    return dict(L1_nA=float(np.abs(x).sum()), n_nonzero=int(np.count_nonzero(x)),
                positive_sum_nA=float(x[x > 0].sum()), negative_sum_nA=float(x[x < 0].sum()))


def reconstruct_current(record, groups, n):
    """Original leg/channel indexed addition order; external touch follows encoding."""
    current = np.zeros(n)
    parts = {ch: np.zeros(n) for ch in next(iter(groups.values()))}
    cr = record['channel_record']
    for leg, channels in cr['per_leg'].items():
        for ch, row in channels.items():
            idx = np.asarray(groups[leg][ch], dtype=int)
            assert len(idx) == row['n_cells']
            value = cr['scale_nA_per_unit'] * float(row['value'])
            current[idx] += value
            parts[ch][idx] += value
    external = np.zeros(n)
    if record['external_lf_touch_stimulus']:
        for i in groups['lf']['bristle_touch']:
            current[int(i)] += .3
            external[int(i)] += .3
    grouped = dict(proprio=sum((parts[ch] for ch in PROPRIO), np.zeros(n)),
                   camp=parts['campaniform_strain'], bristle=parts['bristle_touch']+external)
    return current, dict(per_channel={ch:current_stats(x) for ch,x in parts.items()},
                        groups={ch:current_stats(x) for ch,x in grouped.items()},
                        external_touch=current_stats(external), total=current_stats(current))


def observed_step(net, current, motor):
    """Observe, never replace the neural integrator. All currents signed inward-positive."""
    p = net.p
    v = net.v[motor].copy()
    # Same sparse dot as step; slices only after computing exact arriving conductance.
    arrivals = net._ring[net._ring_i]
    ge = (net.ge + net.We @ arrivals)[motor]
    gi = (net.gi + net.Wi @ arrivals)[motor]
    adaptation = net.adaptation[motor].copy()
    exc = ge*(p.exc_reversal_mV-v)
    inh = gi*(p.inh_reversal_mV-v)
    leak = p.leak_uS*(p.rest_mV-v)
    direct = current[motor].copy()
    total = exc+inh+leak+direct-adaptation
    active = net.t_ms+1e-10 >= net.refractory_until_ms[motor]
    # Algebraic BE increment (NOT a second integrator). Only valid for active cells,
    # before threshold/reset; captures attenuation versus instantaneous total drive.
    dv = total/(p.capacitance_nF/net.dt+p.leak_uS+ge+gi)
    start = net.t_ms
    spikes = net.step(current_nA=current)
    row = dict(time_start_ms=start, time_end_ms=net.t_ms, actual_spikes=int(spikes[motor].sum()),
        v_pre_update_mV=stats(v), v_postreset_mV=stats(net.v[motor]),
        adaptation_pre_nA=stats(adaptation), adaptation_post_nA=stats(net.adaptation[motor]),
        ge_pre_update_uS=stats(ge), gi_pre_update_uS=stats(gi),
        ge_postdecay_uS=stats(net.ge[motor]), gi_postdecay_uS=stats(net.gi[motor]),
        exc_drive_pre_nA=stats(exc), inh_drive_pre_nA=stats(inh),
        exc_drive_abs_pre_nA=stats(abs(exc)), inh_drive_abs_pre_nA=stats(abs(inh)),
        leak_drive_pre_nA=stats(leak), adaptation_drive_pre_nA=stats(-adaptation),
        direct_current_nA=stats(direct), total_drive_pre_nA=stats(total),
        refractory_start_count=int((~active).sum()),
        refractory_end_count=int(np.sum(net.t_ms+1e-10 < net.refractory_until_ms[motor])),
        active_prethreshold_BE_delta_mV=stats(dv[active]) if active.any() else None)
    arrays=dict(spikes=spikes[motor].copy(), v_pre_mV=v, v_postreset_mV=net.v[motor].copy(),
                adaptation_pre_nA=adaptation, adaptation_post_nA=net.adaptation[motor].copy(),
                ge_pre_uS=ge, gi_pre_uS=gi, exc_drive_nA=exc, inh_drive_nA=inh,
                leak_drive_nA=leak, total_drive_nA=total, active_start=active,
                direct_current_nA=direct)
    return spikes, row, arrays


def interval_summary(rows, channels, k):
    block = rows[k*10:(k+1)*10]
    keys = ('v_postreset_mV','adaptation_pre_nA','exc_drive_pre_nA','inh_drive_pre_nA',
            'exc_drive_abs_pre_nA','inh_drive_abs_pre_nA','leak_drive_pre_nA',
            'total_drive_pre_nA','ge_pre_update_uS','gi_pre_update_uS')
    return dict(interval_start_ms=k*5, interval_end_ms=(k+1)*5,
        motor_spikes=sum(r['actual_spikes'] for r in block),
        time_population_means={key:float(np.mean([r[key]['mean'] for r in block])) for key in keys},
        channel_groups=channels[k]['groups'])


def main(guard_report=None):
    started = time.monotonic()
    bundle = validate_bundle(OUT)
    assert bundle['run_id'] == EXPECTED_RUN
    source = Path(bundle['path'])
    result = json.loads((source/'closure_results.json').read_text())
    identity = json.loads((source/'neural_identity.json').read_text())
    manifest = json.loads((source/'motor_manifest.json').read_text())['measured']
    normal = result['measured']['episodes']['normal']
    records = normal['records']
    with np.load(source/'closure_traces.npz', allow_pickle=False) as z:
        expected = z['normal_spike_counts']
    assert expected.shape == (10,12000) and len(records) == 10
    hashes = {p:dict(recorded_sha256=h,current_sha256=sha(p))
              for p,h in result['provenance']['code_hashes'].items()}
    for name in ('neural_cond.py','sensory_channels.py','adapters.py'):
        selected = [v for p,v in hashes.items() if Path(p).name == name]
        assert selected and all(v['recorded_sha256']==v['current_sha256'] for v in selected)
    inputs = {str(p):sha(p) for p in (clo.POOLS,clo.LOOKUP,clo.HERE/'data/flywire/banc_connectome.npz')}
    assert sha(clo.POOLS)==result['provenance']['pool_result_sha256']
    assert sha(clo.LOOKUP)==result['provenance']['sensory_lookup_sha256']
    assert inputs[str(clo.HERE/'data/flywire/banc_connectome.npz')]==result['provenance']['connectome_sha256']
    built = clo.build_neural_tier(clo.TierConfig(background_mean_nA=0, background_sd_nA=0))
    assert list(map(int,built['tier']['root_ids'])) == identity['tier_root_ids']
    groups = identity['sensory_groups']
    assert clo.channel_index_map(int(built['n']),built['tier']['root_ids'])[0] == groups
    motor = np.asarray(sorted({i for row in manifest['rows'] for i in row['local_indices']}))
    assert len(motor) == 202
    sensory = {int(i) for g in groups.values() for ids in g.values() for i in ids}
    assert not sensory.intersection(motor)
    net = clo.ConductanceNetwork(int(built['n']), clo.ConductanceParams(), dt_ms=.5)
    net.set_connectivity(built['edges']['We']*.2,built['edges']['Wi']*.2)
    probes = normal['probe_local_indices']
    rows, channels, detailed, all_counts = [], [], [], []
    for k, record in enumerate(records):
        assert abs(record['time_start_s']-k*.005)<1e-10
        assert abs(record['time_end_s']-(k+1)*.005)<1e-10
        current, ch = reconstruct_current(record, groups, net.n)
        assert current.min()==record['sensory_current_min_nA']
        assert current.max()==record['sensory_current_max_nA']
        assert np.count_nonzero(current[motor])==0
        ch.update(interval_start_ms=k*5, interval_end_ms=(k+1)*5)
        channels.append(ch)
        sp = []
        for j in range(10):
            s,row,detail = observed_step(net,current,motor)
            sp.append(s); rows.append(row); detailed.append(detail)
        sp = np.asarray(sp)
        counts = sp.sum(axis=0)
        # Gate all downstream reporting on ALL neurons in every interval, not totals.
        np.testing.assert_array_equal(counts,expected[k])
        assert np.argwhere(sp[:,probes]).tolist()==record['probe_raw_spike_events_step_column']
        all_counts.append(counts)
    assert time.monotonic()-started < 60
    assert validate_bundle(OUT)==bundle
    assert all(sha(p)==v['current_sha256'] for p,v in hashes.items())
    assert all(sha(p)==h for p,h in inputs.items())
    traces = OUT/'motor_state_audit.npz'
    arrays = {key:np.asarray([d[key] for d in detailed]) for key in detailed[0]}
    np.savez_compressed(traces, **arrays, motor_local_indices=motor,
        motor_root_ids=np.asarray(identity['tier_root_ids'],dtype=np.int64)[motor],
        time_end_ms=np.arange(1,101)*.5, interval_spike_counts=np.asarray(all_counts))
    summaries = [interval_summary(rows,channels,k) for k in range(10)]
    audit = dict(schema_version=1,status='PASS',status_meaning='Exact source replay and state observation; not a causal or standing acceptance',
        source_bundle=dict(run_id=bundle['run_id'],path=str(source),manifest_sha256=sha(source/'manifest.json'),files=bundle['manifest']['files']),
        provenance=dict(code=str(Path(__file__).resolve()),code_sha256=sha(__file__),source_code_hashes=hashes,
                        inputs=inputs,original_translation=result['provenance']),
        replay_contract=dict(arm='normal',body_constructed=False,source_counts_shape=list(expected.shape),
            all_120000_counts_exact=True,all_saved_probe_events_exact=True,dt_ms=.5,steps=100,
            weights_scale=.2,parameters=asdict(net.p),voltage_method=net.voltage_method,
            current_reconstruction='channel_records original indexed summand order then original external LF touch',
            motor_count=202,motor_sensory_overlap=[],motor_current_nonzero_count=0,
            spike_timestamp='interval_start + (step_index+1)*0.5ms',
            state_timing='v_postreset and adaptation_post/ge_postdecay/gi_postdecay at END; pre fields at START after arrivals, before voltage update/decay',
            drive_units='uS*mV=nA; signed inward-positive: ge*(Ee-v), gi*(Ei-v), leak*(rest-v), direct I, -adaptation; absolute drive separately',
            refractory_semantics='start count excluded from integration; end count pending next step',
            interpretation='Pre-update drives are instantaneous diagnostics, not actual BE step-average currents; refractory cells clamped, spike voltage reset'),
        motor_snapshots=rows, channel_current_intervals=channels, interval_summaries=summaries,
        peak20ms_vs40ms=dict(window_15_20_ms=summaries[3],window_35_40_ms=summaries[7],
                             endpoint_20_ms=rows[39],endpoint_40_ms=rows[79]),
        observations=dict(motor_spikes_per_5ms=[r['motor_spikes'] for r in summaries],
            causal_attribution='not_identified_by_observational_replay',
            interpretation='Compare signed input and E/I drive trends jointly with adaptation; population means mix spiking, silent and refractory cells. No unique cause or corrective parameter is inferred.'),
        limitations=['Only normal saved-current replay, not closed-loop intervention.',
            'Current L1 is not effective excitatory drive: signed proprio input can hyperpolarize sensory cells.',
            'Adaptation, E/I balance and input changes co-vary; disabling adaptation or changing weights was not tested.',
            'Illustrative conductance parameters and channel polarity remain uncalibrated.'],
        traces=dict(path=str(traces),sha256=sha(traces)),
        memory_guard=json_safe_guard(guard_report('completed')) if guard_report else None,
        wall_s=time.monotonic()-started)
    (OUT/'motor_state_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(wall_s=audit['wall_s'],observations=audit['observations'],comparison={k:v for k,v in audit['peak20ms_vs40ms'].items() if k.startswith('window')}),indent=2))
    return audit


if __name__=='__main__':
    with clo.mg.guarded_run('motor_state_audit',**clo.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
