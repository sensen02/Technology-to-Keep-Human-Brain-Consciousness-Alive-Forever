"""Additive selective-current experiment; frozen native XML, no gain tuning.

Four live 50ms episodes: baseline, two selective cuts, one repeated cut.
Baseline also repeats the verified source normal and source normal_repeat arrays.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import run_neural_muscle_closure as clo
from engine.embodied.closure_bundle import validate_bundle
from engine.embodied.channel_connectivity import channel_connectivity
from run_neural_command_audit import validate_arrays

HERE = Path(__file__).resolve().parent
OUT = clo.OUT
EXPECTED_RUN = 'f4a2983e-e90a-4dec-9239-ff8296fab5f6'
PROPRIO = ('claw_position', 'hook_direction', 'hairplate_angle')
CUTS = {'normal': (), 'cut_proprio': PROPRIO,
        'cut_campaniform': ('campaniform_strain',), 'cut_proprio_repeat': PROPRIO}
SUFFIXES = ('trajectory', 'controls', 'spike_counts')


def sha(path):
    return clo.cp.sha256_file(Path(path))


def subtract_channels(current, record, groups, selected):
    """Remove each original summand, never zero an index shared by other channels."""
    out = current.copy()
    removed = []
    for leg, channels in record['per_leg'].items():
        for channel in selected:
            value = float(record['scale_nA_per_unit']) * channels[channel]['value']
            idx = np.asarray(groups[leg][channel], dtype=int)
            # Same indexed update semantics as original encode, even for duplicate ids.
            if value != 0 and idx.size:
                out[idx] -= value
            removed.append(dict(leg=leg, channel=channel, current_per_cell_nA=value,
                                n_cells=int(idx.size)))
    descriptor = dict(operation='subtract_original_channel_current_summands',
        selected_channels=list(selected), removed_summands=removed,
        original_physiological_record_preserved=True,
        external_touch='unchanged; original episode adds LF touch after encode',
        no_rerouting_or_additional_injection=True)
    return out, dict(record, intervention=descriptor)


@contextmanager
def intervention(selected):
    original = clo.ChannelSensoryEncoder
    class SelectiveEncoder(original):
        def encode(self, quantities):
            current, record = super().encode(quantities)
            return subtract_channels(current, record, self.groups, selected)
    clo.ChannelSensoryEncoder = SelectiveEncoder
    try:
        yield
    finally:
        clo.ChannelSensoryEncoder = original


def overlap_audit(groups):
    owners = {}
    for leg, channels in groups.items():
        for ch, ids in channels.items():
            for i in set(ids):
                owners.setdefault(int(i), []).append(leg+'/'+ch)
    shared = {str(i): names for i, names in owners.items() if len(names)>1}
    return dict(unique_indices=len(owners), shared_indices=shared,
                disjoint=not shared, removal_method='subtract per-channel summands regardless of overlap')


def first_difference(a, b, start=False):
    changed = np.flatnonzero(np.any(a != b, axis=1))
    if not changed.size:
        return None
    k = int(changed[0])
    return dict(interval=k, interval_start_s=k*clo.DT_COMMAND,
                interval_end_s=(k+1)*clo.DT_COMMAND,
                first_observable_saved_time_s=(k if start else k+1)*clo.DT_COMMAND)


def compare(base, cut, motor):
    return dict(motor_counts_exact=np.array_equal(base[2][:,motor],cut[2][:,motor]),
        controls_exact=np.array_equal(base[1],cut[1]), trajectory_exact=np.array_equal(base[0],cut[0]),
        motor_counts_per_interval=cut[2][:,motor].sum(axis=1).tolist(),
        first_motor_count_difference=first_difference(base[2][:,motor],cut[2][:,motor]),
        first_applied_control_difference=first_difference(base[1],cut[1],True),
        first_saved_trajectory_difference=first_difference(base[0],cut[0]),
        control_max_abs_difference=float(np.max(np.abs(base[1]-cut[1]))),
        trajectory_max_abs_difference=float(np.max(np.abs(base[0]-cut[0]))))


def heights(model, seated, trajectory):
    import mujoco
    scratch = mujoco.MjData(model)
    rows = []
    for k, q in enumerate([seated] + [row[:model.nq] for row in trajectory]):
        scratch.qpos[:] = q
        mujoco.mj_forward(model, scratch)
        com = np.average(scratch.xipos, axis=0, weights=model.body_mass)
        rows.append(dict(time_s=k*clo.DT_COMMAND, root_z_model=float(q[2]),
                         whole_body_com_z_model=float(com[2]),
                         thorax_z_model=float(scratch.xpos[model.body('Thorax').id,2])))
    return rows


def validate_experiment_arrays(arrays, episodes, n, model):
    expected = {arm+'_'+s for arm in CUTS for s in SUFFIXES} | {'seated_qpos'}
    assert set(arrays) == expected
    assert arrays['seated_qpos'].shape == (model.nq,)
    shapes = {}
    for arm in CUTS:
        assert len(episodes[arm]['records']) == 10
        for suffix, width in [('trajectory',model.nq+model.nv+model.na),('controls',model.nu),('spike_counts',n)]:
            key=arm+'_'+suffix; a=arrays[key]
            assert a.shape == (10,width) and np.isfinite(a).all(), key
            shapes[key]=list(a.shape)
        assert np.all((arrays[arm+'_controls']>=0)&(arrays[arm+'_controls']<=1))
        spikes=arrays[arm+'_spike_counts']
        assert np.all((spikes>=0)&(spikes<=10)&(spikes==np.floor(spikes)))
        for k,r in enumerate(episodes[arm]['records']):
            assert r['applied_source_interval']==k-1 and not any(r['mujoco_warning_counts'])
            assert abs(r['time_end_s']-(k+1)*.005)<1e-10
            assert r['channel_record']['intervention']['selected_channels']==list(CUTS[arm])
    return shapes


def json_safe_guard(value):
    # RLIMIT_AS previous limit may legitimately be infinity, not a measurement.
    if isinstance(value, dict):
        return {k:json_safe_guard(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_guard(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main(guard_report=None):
    import mujoco
    started=time.monotonic()
    bundle=validate_bundle(OUT)
    assert bundle['run_id']==EXPECTED_RUN, 'source bundle changed: review before rerun'
    source=Path(bundle['path'])
    result=json.loads((source/'closure_results.json').read_text())
    saved_manifest=json.loads((source/'motor_manifest.json').read_text())['measured']
    identity=json.loads((source/'neural_identity.json').read_text())
    with np.load(source/'closure_traces.npz',allow_pickle=False) as z:
        source_arrays={k:z[k] for k in z.files}
    source_shapes=validate_arrays(saved_manifest,result,source_arrays)
    assert result['measured']['duration_s']==.05 and result['measured']['external_perturbation'] is None
    source_code={p:dict(recorded_sha256=h,current_sha256=sha(p),matches=sha(p)==h)
                 for p,h in result['provenance']['code_hashes'].items()}
    for name in ('motor_output.py','sensory_channels.py','neural_cond.py','adapters.py','canonical_physics.py'):
        assert all(v['matches'] for p,v in source_code.items() if Path(p).name==name)
    model,seated,gate,joints,geom_leg=clo.prepare_body()
    built=clo.build_neural_tier(clo.TierConfig(background_mean_nA=0,background_sd_nA=0))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    manifest=clo.build_motor_manifest(json.loads(clo.POOLS.read_text()),built['tier']['root_ids'],names)
    groups,stats=clo.channel_index_map(int(built['n']),built['tier']['root_ids'])
    assert manifest==saved_manifest
    assert np.array_equal(seated,source_arrays['seated_qpos'])
    assert [int(i) for i in built['tier']['root_ids']]==identity['tier_root_ids']
    assert groups==identity['sensory_groups']
    motor=sorted({i for row in manifest['rows'] for i in row['local_indices']})
    coverage=channel_connectivity(built['edges']['We'],built['edges']['Wi'],groups,motor)
    for leg in clo.LEGS:
        assert coverage['channels'][leg]['bristle_touch']['reachable_motor_cells']==0
        for channel in PROPRIO:
            assert coverage['channels'][leg][channel]['reachable_motor_cells']==202
    runs={}; episodes={}; arrays={'seated_qpos':seated}; height_rows={}
    for arm,selected in CUTS.items():
        with intervention(selected):
            ep,*values=clo.episode(model,seated,joints,geom_leg,built,manifest,groups,'normal',.05)
        ep['arm']=arm
        episodes[arm]=ep; runs[arm]=values
        arrays.update({arm+'_'+s:a for s,a in zip(SUFFIXES,values)})
        height_rows[arm]=heights(model,seated,values[0])
        print(arm, 'motor counts',values[2][:,motor].sum(axis=1).tolist(),flush=True)
    shapes=validate_experiment_arrays(arrays,episodes,int(built['n']),model)
    exact={s:bool(np.array_equal(runs['normal'][j],source_arrays['normal_'+s])) for j,s in enumerate(SUFFIXES)}
    repeat={s:bool(np.array_equal(runs['cut_proprio'][j],runs['cut_proprio_repeat'][j])) for j,s in enumerate(SUFFIXES)}
    baseline_repeat={s:bool(np.array_equal(runs['normal'][j],source_arrays['normal_repeat_'+s])) for j,s in enumerate(SUFFIXES)}
    assert all(exact.values()) and all(repeat.values()) and all(baseline_repeat.values())
    touch={arm:[r['external_lf_touch_stimulus'] for r in ep['records']] for arm,ep in episodes.items()}
    assert all(flags==touch['normal'] for flags in touch.values())
    assert all(sha(p)==v['current_sha256'] for p,v in source_code.items()), 'source changed during experiment'
    assert sha(clo.cp.CANONICAL_XML)==clo.cp.CANONICAL_XML_SHA256
    assert validate_bundle(OUT)==bundle
    traces=OUT/'connected_feedback_traces.npz'
    np.savez_compressed(traces,**arrays)
    audit=dict(schema_version=1,status='PASS',status_meaning='Selective-current and repeat invariants, not standing acceptance',
        source_bundle=dict(run_id=bundle['run_id'],path=str(source),manifest_sha256=sha(source/'manifest.json'),files=bundle['manifest']['files']),
        provenance=dict(code=str(Path(__file__).resolve()),code_sha256=sha(__file__),
            source_code_hashes=source_code,canonical_xml_sha256=sha(clo.cp.CANONICAL_XML),
            inputs={str(p):sha(p) for p in (clo.POOLS,clo.LOOKUP,HERE/'data/flywire/banc_connectome.npz',HERE/'data/banc888/banc_888_meta.feather',clo.r39.ROUND37_POSE)},
            original_translation=result['provenance'],intervention_only='scoped clo.ChannelSensoryEncoder subclass; original encode then selected summand subtraction'),
        duration_s=.05,live_episode_count=4,baseline_exact_source=exact,baseline_exact_source_repeat=baseline_repeat,
        cut_repeat_exact=repeat,external_lf_touch_flags=touch,source_array_shapes=source_shapes,array_shapes=shapes,
        traces=dict(path=str(traces),sha256=sha(traces)),channel_index_overlap=overlap_audit(groups),
        source_channel_stats=stats,structural_coverage=coverage,initial_stance_gate=gate,
        comparisons={arm:compare(runs['normal'],runs[arm],motor) for arm in CUTS},
        contextual_cut_all=dict(source_arm='sensory_cut',not_a_substitute_for_selective_cuts=True,
            comparison=compare(runs['normal'],[source_arrays['sensory_cut_'+s] for s in SUFFIXES],motor)),
        height_samples=height_rows,height_method='scratch MjData, saved qpos, mj_forward; COM=sum(body_mass*xipos)/sum(body_mass); root is separately qpos[2]',
        height_units='native model length (declared mm); no root==COM assumption',episodes=episodes,
        limitations=['Structural paths do not prove functional transmission.','Selective differences establish modeled input dependence, not necessity or benefit for standing.',
            'Native XML positive control floor is frozen; strict neural-only standing is not established.',
            'No missing bristle transmitter signs invented; no rerouting or extra injection.',
            'Channel polarity, gains and within-subtype tuning remain unvalidated.',
            'Motor count first difference is a 5ms aggregation endpoint, not exact spike latency.'],
        memory_guard=json_safe_guard(guard_report('completed')) if guard_report else None,wall_s=time.monotonic()-started)
    (OUT/'connected_feedback_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:audit[k] for k in ('baseline_exact_source','cut_repeat_exact','comparisons','wall_s')},indent=2))


if __name__=='__main__':
    with clo.mg.guarded_run('connected_feedback_audit',**clo.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
