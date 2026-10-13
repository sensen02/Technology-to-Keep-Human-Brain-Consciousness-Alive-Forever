"""Read-only current-bundle command audit. No physics, network or tier construction."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from engine.embodied.closure_bundle import validate_bundle

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
OFFLINE = HERE / 'outputs/workspace_recheck/dynamic_command_search.json'
ATOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def distribution(values, maximum=1.0):
    a = np.asarray(values, float)
    require(a.size > 0 and np.isfinite(a).all(), 'nonfinite/empty distribution')
    return dict(n=int(a.size), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), median=float(np.median(a)),
                zero_count=int(np.sum(a == 0)), active_count=int(np.count_nonzero(a)),
                saturated_count=int(np.sum(a >= maximum)),
                zero_fraction=float(np.mean(a == 0)), saturated_fraction=float(np.mean(a >= maximum)))


def validate_arrays(manifest, result, arrays):
    """Pure validation: identities are manifest ordering and arm-key identities, not NPZ labels."""
    n, nu = manifest['n_neurons'], manifest['n_actuators']
    rows = manifest['rows']
    require(len(rows) == nu, 'manifest row count')
    require([r['actuator_index'] for r in rows] == list(range(nu)), 'actuator ordering')
    require(len({r['actuator'] for r in rows}) == nu, 'duplicate actuator identity')
    for r in rows:
        ids = r['local_indices']
        require(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < n for i in ids), 'invalid local indices')
        require(len(r['tier_root_ids']) == len(ids), 'root/local identity count')
        require(set(r['tier_root_ids']).issubset(r['pool_root_ids']), 'tier/pool identity')
    root_to_index, index_to_root = {}, {}
    for r in rows:
        for root, i in zip(r['tier_root_ids'], r['local_indices']):
            require(root_to_index.setdefault(root, i) == i and index_to_root.setdefault(i, root) == root, 'inconsistent tier identity')
    episodes = result['measured']['episodes']
    expected = {'seated_qpos'} | {arm+'_'+suffix for arm in episodes for suffix in ('controls','spike_counts','trajectory')}
    require(set(arrays) == expected, 'NPZ arm/key identities')
    nq = len(arrays['seated_qpos'])
    require(np.isfinite(arrays['seated_qpos']).all(), 'seated pose nonfinite')
    shapes = {}
    for arm, ep in episodes.items():
        k = len(ep['records'])
        require(ep['arm'] == arm and k > 0, 'episode arm identity')
        for suffix, shape in [('controls',(k,nu)), ('spike_counts',(k,n)), ('trajectory',(k,2*nq-1+nu))]:
            a = arrays[arm+'_'+suffix]
            require(a.shape == shape and np.isfinite(a).all(), 'shape/finite '+arm+'_'+suffix)
            shapes[arm+'_'+suffix] = list(a.shape)
        c, s = arrays[arm+'_controls'], arrays[arm+'_spike_counts']
        require(np.all((c >= 0) & (c <= 1)), 'control range')
        require(np.all((s >= 0) & (s <= 10) & (s == np.floor(s))), 'spike count range')
    return shapes


def check_lag(records, controls, config):
    """Compare APPLIED[k] with clip(saved end-of-interval rate[k-1]); no raster reconstruction."""
    errors = []
    for k, record in enumerate(records):
        require(record['applied_source_interval'] == k-1, 'source interval lag')
        require(abs(record['time_start_s'] - k*.005) <= ATOL, '5ms interval start')
        require(abs(record['time_end_s'] - (k+1)*.005) <= ATOL, '5ms interval end')
        require(abs(record['source_completed_by_s'] - record['time_start_s']) <= ATOL, 'source completion stamp')
        trace = record['decoder']
        require(trace['source_interval'] == k, 'decoder source identity')
        require(abs(trace['source_start_s']-record['time_start_s']) <= ATOL and abs(trace['source_end_s']-record['time_end_s']) <= ATOL, 'decoder interval timing')
        require(abs(trace['valid_from_s']-record['time_end_s']) <= ATOL, 'decoder validity')
        rates = np.asarray(trace['rates_hz'], float)
        require(rates.shape == controls[k].shape and np.isfinite(rates).all(), 'decoder rate shape/finite')
        expected = np.zeros_like(controls[k]) if k == 0 else np.clip(np.asarray(records[k-1]['decoder']['rates_hz']) / config['rate_for_full_control_hz'], 0, config['max_control'])
        err = float(np.max(np.abs(controls[k]-expected)))
        require(err <= ATOL, 'previous decoder output mismatch')
        errors.append(err)
    return dict(atol=ATOL, rtol=0, maximum_absolute_error=max(errors), per_interval_max_error=errors,
                exact_all=all(e == 0 for e in errors),
                explanation='Causal sample-and-hold: spikes collected in [k*5,(k+1)*5] ms decode at the end and drive the NEXT body interval; initial pending command is zero. This is a 5ms interval-index lag, not a measured biological conduction delay. A spike waits 0–5ms to the next boundary plus the declared rate-filter dynamics; last decoded output is not applied within the saved run.',
                limit='Counts aggregate ten 0.5ms steps; cannot independently recompute exponential filtering or sub-ms spike timing from counts. Equality checks saved rate-to-command translation, not the full spike filter.')


def audit(out=OUT, offline_path=OFFLINE):
    started = time.monotonic()
    bundle = validate_bundle(out)
    path = Path(bundle['path'])
    result = json.loads((path/'closure_results.json').read_text())
    wrapped_manifest = json.loads((path/'motor_manifest.json').read_text())
    manifest = wrapped_manifest['measured']
    require(result['provenance'] == wrapped_manifest['provenance'], 'companion provenance mismatch')
    with np.load(path/'closure_traces.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    shapes = validate_arrays(manifest, result, arrays)
    rows = manifest['rows']; episodes = result['measured']['episodes']
    config = result['provenance']['motor_translation']
    source_hashes = {p:dict(recorded_sha256=h, current_sha256=sha(p), matches=sha(p)==h)
                     for p,h in result['provenance']['code_hashes'].items()}
    for name in ('engine/embodied/motor_output.py', 'engine/embodied/sensory_channels.py', 'engine/neural_cond.py'):
        require(source_hashes[str(HERE/name)]['matches'], 'decoder/encoder/network source changed; interpretation unanchored')
    motor = sorted({i for row in rows for i in row['local_indices']})
    covered = [r['actuator_index'] for r in rows if r['local_indices']]
    unresolved = [r['actuator_index'] for r in rows if not r['local_indices']]
    clusters = {}
    for row in rows:
        if row['local_indices']:
            clusters.setdefault(tuple(sorted(row['local_indices'])), []).append(row['actuator_index'])
    matrix = np.zeros((len(rows), len(motor)))
    local_column = {i:j for j,i in enumerate(motor)}
    for row in rows:
        ids = row['local_indices']
        for i in ids:
            matrix[row['actuator_index'],local_column[i]] = 1/len(ids)
    # Matrix restricted to participating cells, not a dense 90x12000 SVD.
    rank = int(np.linalg.matrix_rank(matrix))
    per_arm = {}
    for arm, episode in episodes.items():
        c, s = arrays[arm+'_controls'], arrays[arm+'_spike_counts']
        require(not np.any(c[:,unresolved]), 'unresolved commands nonzero')
        for group in clusters.values():
            require(all(np.array_equal(c[:,group[0]],c[:,i]) for i in group), 'identical pool commands differ')
        for k, rec in enumerate(episode['records']):
            expected_counts = [int(s[k,r['local_indices']].sum()) for r in rows]
            require(expected_counts == rec['decoder']['pool_spike_counts'], 'pool spike record mismatch')
            require(int(s[k,motor].sum()) == rec['motor_spikes'], 'motor subset record mismatch')
        per_arm[arm] = dict(distribution=distribution(c), intervals=[dict(interval=k,
            start_s=rec['time_start_s'], end_s=rec['time_end_s'], controls=distribution(c[k]),
            covered_controls=distribution(c[k,covered]), all_neuron_spikes=int(s[k].sum()),
            all_neuron_active_cells=int(np.count_nonzero(s[k])), motor_subset_spikes=int(s[k,motor].sum()),
            motor_subset_active_cells=int(np.count_nonzero(s[k,motor]))) for k,rec in enumerate(episode['records'])])
    lag = check_lag(episodes['normal']['records'], arrays['normal_controls'], config)
    require(not np.any(arrays['motor_cut_controls'][3:]), 'motor cut nonzero at/after 15ms')
    require(not np.any(arrays['spike_output_silence_controls']), 'silenced commands nonzero')
    require(np.array_equal(arrays['normal_controls'], arrays['replay_controls']) and np.array_equal(arrays['normal_controls'], arrays['normal_repeat_controls']), 'repeat/replay controls')
    a,b = arrays['normal_spike_counts'], arrays['no_external_touch_spike_counts']
    touch_intervals = []
    for k,rec in enumerate(episodes['normal']['records']):
        ch = rec.get('channel_record',{}).get('per_leg',{}).get('lf',{}).get('bristle_touch')
        other = episodes['no_external_touch']['records'][k]
        touch_intervals.append(dict(interval=k, stimulus_on=rec['external_lf_touch_stimulus'],
            lf_bristle_record=ch, added_current_nA_per_target=.3 if rec['external_lf_touch_stimulus'] else 0,
            normal_current_extrema_nA=[rec['sensory_current_min_nA'],rec['sensory_current_max_nA']],
            no_touch_current_extrema_nA=[other['sensory_current_min_nA'],other['sensory_current_max_nA']],
            differing_all_neuron_count_cells=int(np.count_nonzero(a[k]-b[k])),
            signed_all_neuron_spike_difference=int(a[k].sum()-b[k].sum()),
            absolute_count_difference=int(np.abs(a[k].astype(float)-b[k]).sum()),
            differing_motor_count_cells=int(np.count_nonzero(a[k,motor]-b[k,motor]))))
    offline = json.loads(Path(offline_path).read_text())
    winner = next(c for c in offline['candidates'] if c['label'] == offline['selection']['winner_label'])
    require(offline['actuator_order'] == [r['actuator'] for r in rows], 'offline actuator ordering differs')
    oc = np.asarray(winner['activation'],float)
    require(oc.shape == (manifest['n_actuators'],), 'offline command shape')
    mismatches = [dict(actuators=[rows[i]['actuator'] for i in group], spread=float(np.ptp(oc[group]))) for group in clusters.values() if len(group)>1 and np.ptp(oc[group])>ATOL]
    normal = arrays['normal_controls']
    report = dict(schema_version=1, status='PASS_ARTIFACT_AUDIT_NOT_AUTONOMY',
        question='Are CURRENT neural commands traceable, bounded, delayed causally and correctly scoped?',
        snapshot=dict(run_id=bundle['run_id'], path=str(path), manifest_sha256=sha(path/'manifest.json'),
            manifest=bundle['manifest'], source_code_hashes=source_hashes,
            source_snapshot_limit='Recorded run hashes and present audit-time hashes retained separately. Any mismatch is source drift, not a bundle integrity failure; original source text is not embedded in bundle. No claim current runner bytes reproduce saved run.',
            audit_source_sha256=sha(__file__), test_source_sha256=sha(HERE/'test_neural_command_audit.py'),
            offline_path=str(offline_path), offline_sha256=sha(offline_path)),
        validation=dict(all_bundle_files_hash_verified=True, shapes=shapes, finite_controls=True,
            actuator_order=[r['actuator'] for r in rows], no_unresolved_commands=True,
            identical_present_pool_commands_equal=True, motor_cut_zero_from_s=.015, spike_output_silence_zero=True,
            identity_limit='NPZ has no embedded run UUID or actuator/root labels; identities are anchored by verified bundle manifest, identical JSON provenance, dimensions, manifest ordering and pool-count crosschecks, not independent NPZ self-identification.'),
        causal_lag=lag, arms=per_arm,
        pool_geometry=dict(n_neurons=manifest['n_neurons'], motor_subset_neurons=len(motor), n_outputs=len(rows),
            usable_outputs=len(covered), empty_outputs=len(unresolved), fully_covered_outputs=manifest['fully_covered_actuators'],
            present_pool_unique_count=len(clusters), linear_pool_mean_matrix_shape=list(matrix.shape), linear_pool_mean_matrix_rank=rank,
            observed_normal_command_time_matrix_rank=int(np.linalg.matrix_rank(normal)),
            identical_present_pool_clusters=[[rows[i]['actuator'] for i in g] for g in clusters.values() if len(g)>1],
            limit='Rank is algebraic rank of this declared present-cell averaging map, NOT biological impossibility, network dynamical controllability or reachability. Clipping/filtering constrain this implemented decoder; ten observed samples do not establish reachable dimension.'),
        external_touch=dict(intervals=touch_intervals, all_neuron_spikes_normal=int(a.sum()), all_neuron_spikes_no_touch=int(b.sum()),
            differing_interval_neuron_count_cells=int(np.count_nonzero(a-b)), motor_subset_counts_identical=bool(np.array_equal(a[:,motor],b[:,motor])),
            commands_identical=bool(np.array_equal(normal, arrays['no_external_touch_controls'])),
            encoder_clips_added_current=False, clipping_evidence='Hash-matched sensory_channels.encode adds scaled values without clipping; closure episode adds +0.3nA AFTER encode; neural_cond.step validates current and integrates it without clipping. Recorded global current extrema cannot identify a specific cell current.',
            tier_mapping='Saved full channel records establish LF bristle present-cell count. Full tier root order / sensory local index groups are not saved in this bundle; no build_neural_tier invoked. Count differences cannot be definitively assigned to LF bristle identities.',
            conclusion='External touch DOES change time-binned whole-network spike counts despite identical grand total; no observed motor-count or command change. Unchanged commands do not imply no sensory effect or encoder clipping.',
            unresolved_hypotheses=['Changed sensory activity may not propagate into the mapped motor subset in this window.', 'Refractoriness/adaptation/network path effects may redistribute timing; exact causal attribution needs saved targeted raster/state, not parameter tuning.']),
        offline_comparison=dict(role='DESCRIPTIVE ONLY; NOT target or acceptance criterion', winner_label=winner['label'],
            selection=offline['selection'], command_distribution=distribution(oc), covered_command_distribution=distribution(oc[covered]),
            normal_covered_distribution=distribution(normal[:,covered]),
            offline_initialization=offline['method']['command'], neural_initialization=result['provenance']['initialization'],
            same_initial_pose=bool(np.array_equal(np.asarray(offline['initial_pose']['qpos']),arrays['seated_qpos'])),
            initial_qpos_max_abs_difference=float(np.max(np.abs(np.asarray(offline['initial_pose']['qpos'])-arrays['seated_qpos']))),
            constant_command_from_s=0, neural_first_nonzero_applied_s=next((k*.005 for k,c in enumerate(normal) if np.any(c)),None),
            shared_pool_winner_inequalities=mismatches,
            limit='74 independently optimized offline muscle outputs relax shared-pool constraints, not 74 independently reachable neural channels. Different pose and initial activation (offline act=ctrl, neural act=0) prohibit interpreting distribution differences as required neural targets. Offline has no neuron spike counts. Zero neural command is not zero native physical activation because XML floors remain.'),
        next_step=dict(concrete_defect='Observability/provenance gap: bundle omits full tier root IDs, sensory local groups, per-target post-injection current and sub-ms spikes; cannot localize touch propagation or independently reconstruct filtered rates.',
            action='In a future separately authorized bounded instrumentation change, save tier identity/group mapping and targeted sensory-to-motor spike/current records with matched arms. Do not tune parameters, substitute offline winner, rebuild tier or run physics in this audit.',
            not_a_demonstrated_defect='No decoder lag/shared-pool/cut violation found; lack of motor touch response is a measured null result, not yet a proven connectivity or gain defect.'),
        labels=dict(units=result['units'], illustrative_vs_measured=result['illustrative_vs_measured'],
            limitations=result['what_this_does_not_support'], motor_manifest_assumptions={k:manifest[k] for k in ('provenance','normalization','shared_pool_caveat')},
            motor_translation=config, sensory_translation=result['provenance']['sensory_translation']),
        execution=dict(no_model_load=True, no_physics=True, no_parameter_tuning=True, wall_s=time.monotonic()-started))
    require(validate_bundle(out)['run_id'] == bundle['run_id'], 'CURRENT changed during audit')
    return report


def main():
    report = audit()
    destination = OUT/'neural_command_audit.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,destination)
    print(json.dumps(dict(status=report['status'], output=str(destination), run_id=report['snapshot']['run_id'],
                         wall_s=report['execution']['wall_s'], schema_version=1)))


if __name__ == '__main__':
    main()
