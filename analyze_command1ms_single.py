"""Saved-only S4 command-cadence analysis. Never imports runners or a simulator.

Run ONLY after the producer confirms all artifacts are ready. Historical all-node
spikes are window counts, not a tick raster. No invented historical event times.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z')
OLD = Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal')
REACTION = Path('/home/sensen/Desktop/engine修复验收/s2-postclip-half-command/20261027T181000Z/scale1_LAST50_GROUND.npz')
CLASSIFICATION = Path('/home/sensen/Desktop/engine修复验收/s2-early-support-region/20261027T200000Z/PROJECTION_AND_CLASSIFICATION.json')
DESKTOP = Path('/home/sensen/Desktop')
LEGS = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh']
BODY_DT = .0000125
NEURAL_DT = .0005
FLOOR = .0001 + 1e-12
SATURATION = 1 - 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_archive(path):
    """Verify direct archived file when its sibling manifest exists."""
    path = Path(path)
    for root in [path.parent, path.parent.parent]:
        manifest = root / 'MANIFEST_SHA256.json'
        if manifest.exists():
            entries = read_json(manifest)
            key = str(path.relative_to(root))
            if key in entries:
                assert sha(path) == entries[key], f'archive hash mismatch: {path}'
                return {'manifest': str(manifest), 'key': key, 'sha256': entries[key]}
    raise AssertionError(f'No archived hash entry for {path}')


def first(mask, times):
    ids = np.flatnonzero(mask)
    return None if not len(ids) else float(times[ids[0]])


def series_stats(a):
    a = np.asarray(a)
    return dict(mean=a.mean(axis=0), minimum=a.min(axis=0), maximum=a.max(axis=0), final=a[-1])


def bounded_window_first(counts, starts, ends):
    ids = np.flatnonzero(np.asarray(counts).sum(axis=1) > 0)
    if not len(ids):
        return dict(observed=False, exact_s=None)
    i = int(ids[0])
    return dict(observed=True, exact_s=None, lower_exclusive_s=float(starts[i]),
                upper_inclusive_s=float(ends[i]), earliest_possible_tick_s=float(starts[i] + NEURAL_DT),
                reason='Only all-network window counts saved historically; probe first is not all-network first.')


def saturation_stats(values, durations):
    saturated = values >= SATURATION
    exposure = (saturated * durations[:, None]).sum(axis=0)
    total = float(durations.sum())
    return dict(threshold=SATURATION, raw_actuator_window_count=int(saturated.sum()),
                windows=len(values), observed_duration_s=total,
                actuator_seconds=float(exposure.sum()), per_actuator_seconds=exposure,
                actuator_time_fraction=float(exposure.sum() / (total * values.shape[1])),
                any_actuator_seconds=float(durations[saturated.any(axis=1)].sum()),
                note='Time-weighted endpoint/held-window occupancy, not raw unequal-window counts or hidden within-window clipping.')


def contact_ledger(contacts, sample_ids, geomleg, planes):
    counts = np.zeros((len(sample_ids), 7), int)
    normal = np.zeros_like(counts, dtype=float)
    for j, idx in enumerate(sample_ids):
        for co in contacts[int(idx)]:
            a, b = int(co[1]), int(co[2])
            g = b if a in planes else a if b in planes else None
            if g is None:
                continue
            leg = geomleg.get(g, 'NONFOOT')
            col = LEGS.index(leg) if leg in LEGS else 6
            counts[j, col] += 1
            normal[j, col] += max(float(co[7]), 0.)
    return counts, normal


def summarize_arm(label, body, iv, rows, ticks, records, contacts, meta, old_ground):
    t = body['time']
    bt = np.rint(t / BODY_DT).astype(np.int64)
    np.testing.assert_array_equal(bt, np.arange(20001))
    np.testing.assert_allclose(t, bt * BODY_DT, atol=1e-10, rtol=0)
    # Explicitly exclude tick16000 (.2s), include tick20000 (.25s).
    last_ids = np.flatnonzero((bt > 16000) & (bt <= 20000))
    assert len(last_ids) == 4000
    early_body = (bt <= 1200)
    new = ticks is not None
    starts = iv['motor_start_s'] if new else iv['interval_start_s']
    ends = iv['motor_end_s'] if new else iv['interval_end_s']
    decoded = iv['motor_decoded_next_controls'] if new else iv['decoded_next_controls']
    commands, counts = iv['commands'], iv['spikes']
    np.testing.assert_allclose(ends - starts, .001 if new else .005, atol=1e-12, rtol=0)
    assert counts.shape == (250 if new else 50, 12000)
    assert commands.shape == decoded.shape == (len(starts), 90)
    motor = sorted({int(i) for r in rows for i in r['local_indices']})
    events = []
    if new:
        raster = ticks['spikes']
        assert raster.dtype == np.bool_ and raster.shape == (500, 12000)
        np.testing.assert_array_equal(raster.reshape(250, 2, 12000).sum(1), counts)
        tt = ticks['tick_complete_s']
        np.testing.assert_allclose(tt, np.arange(1, 501) * NEURAL_DT, atol=1e-12, rtol=0)
        assert ticks['rates_hz'].shape == ticks['decoded_controls'].shape == (500, 90)
        np.testing.assert_array_equal(ticks['decoded_controls'][1::2], decoded)
        rr, cc = np.nonzero(raster[:, motor])
        events = [(float(tt[r]), int(motor[c])) for r, c in zip(rr, cc)]
        network_first = dict(exact_s=first(raster.any(1), tt), source='saved full-network tick raster')
        decode_times, decode_values = tt, ticks['decoded_controls']
        np.testing.assert_array_equal(iv['motor_applied_controls'], commands)
        np.testing.assert_array_equal(commands[1:], decoded[:-1])
        np.testing.assert_array_equal(iv['motor_source_end_tick'][1:], np.rint(starts[1:] / NEURAL_DT).astype(int))
    else:
        probe = records['probe_local_indices']
        assert set(motor).issubset(set(probe)), 'Historical probe must cover every motor node'
        for k, record in enumerate(records['records']):
            for step, col in record['probe_emitted_spike_events_step_column']:
                node = int(probe[col])
                if node in motor:
                    events.append((float(starts[k] + (step + 1) * NEURAL_DT), node))
        network_first = bounded_window_first(counts, starts, ends)
        decode_times, decode_values = ends, decoded
    # Actual BODY samples are post-step; boundary values are from saved scheduler arrays.
    body_source = np.searchsorted(np.rint(starts / BODY_DT).astype(int), bt[1:] - 1, side='right') - 1
    np.testing.assert_array_equal(body['command'][1:], commands[body_source])
    reconstructed = np.zeros(12000, int)
    for _, node in events:
        reconstructed[node] += 1
    np.testing.assert_array_equal(reconstructed[motor], counts[:, motor].sum(0))
    perleg = {}
    for leg in LEGS:
        ai = [int(r['actuator_index']) for r in rows if r['actuator'].lower().startswith(leg)]
        pool = sorted({int(i) for r in rows if int(r['actuator_index']) in ai for i in r['local_indices']})
        ev = sorted((s, node) for s, node in events if node in pool)
        ev15 = [(s, node) for s, node in ev if int(round(s / NEURAL_DT)) <= 30]
        d = first((decode_values[:, ai] > 0).any(1), decode_times)
        w = first((commands[:, ai] > 0).any(1), starts)
        b = first((body['command'][:, ai] > 0).any(1), t)
        a = first((body['activation'][:, ai] > FLOOR).any(1), t)
        e = None if not ev else ev[0][0]
        de = np.rint(decode_times / NEURAL_DT).astype(int) <= 30
        we = np.rint(starts / NEURAL_DT).astype(int) <= 30
        perleg[leg] = dict(actuator_indices=ai, unique_pool_nodes=pool,
            first_motor_event_s=e, first_saved_decode_s=d,
            first_effective_motor_readout_s=first((decoded[:,ai]>0).any(1), ends), first_write_boundary_s=w,
            first_body_command_sample_s=b, first_activation_above_native_floor_s=a,
            event_to_first_saved_decode_s=None if e is None or d is None else d-e,
            event_to_write_boundary_s=None if e is None or w is None else w-e,
            write_boundary_to_body_sample_s=None if b is None or w is None else b-w,
            write_boundary_to_activation_s=None if a is None or w is None else a-w,
            first15ms=dict(event_count=len(ev15), events=[dict(time_s=s, local_node=n) for s,n in ev15],
                decoded_times_s=decode_times[de], decoded_controls=decode_values[de][:,ai],
                write_boundary_times_s=starts[we], written_controls=commands[we][:,ai],
                first_actual_body_command_s=b if b is not None and b <= .015+1e-12 else None,
                activation_above_floor_sample_count=int((body['activation'][early_body][:,ai] > FLOOR).sum()),
                first_activation_above_floor_s=a if a is not None and a <= .015+1e-12 else None,
                activation_above_floor_samples_by_actuator=(body['activation'][early_body][:,ai]>FLOOR).sum(0),
                first_activation_above_floor_by_actuator_s=[first(body['activation'][early_body,j]>FLOOR,t[early_body]) for j in ai],
                activation_peak_by_actuator=body['activation'][early_body][:,ai].max(0),
                activation_at15ms=body['activation'][1200,ai]))
    geomleg = {int(k): v['leg'] for k,v in meta['foot_geoms'].items()}
    planes = set(meta.get('plane_geom_ids', [0]))
    contact_counts, normal = contact_ledger(contacts, last_ids, geomleg, planes)
    if new:
        assert body['ground_force_foot_world'].shape == (20001,6,3)
        assert body['ground_force_nonfoot_world'].shape == (20001,3)
        world = np.concatenate((body['ground_force_foot_world'][last_ids], body['ground_force_nonfoot_world'][last_ids,None,:]), axis=1)
        force_source = 'same-run BODY saved perfoot/NONFOOT world forces'
    else:
        gt = np.rint(old_ground['time'] / BODY_DT).astype(int)
        loc = np.searchsorted(gt, bt[last_ids])
        np.testing.assert_array_equal(gt[loc], bt[last_ids])
        world = old_ground['world'][loc]
        np.testing.assert_array_equal(old_ground['geometric_ground_contact_counts'][loc], contact_counts)
        force_source = str(REACTION) + ' (hash-verified historical saved-state recovery; no new recovery)'
    residual = float(abs(world.sum(1)-body['ground_force_world'][last_ids]).max())
    assert residual < 1e-9, f'World reaction decomposition mismatch {label}: {residual}'
    np.testing.assert_array_equal(contact_counts[:,6], body['nonfoot_floor_count'][last_ids])
    last = dict(interval='(0.2, 0.25] seconds, rounded body ticks (16000,20000]', samples=4000,
        categories=LEGS+['NONFOOT'], geometric_contact_samples=(contact_counts>0).sum(0),
        geometric_contact_fraction=(contact_counts>0).mean(0), contact_count_stats=series_stats(contact_counts),
        true_floor_foot_legs_minmax=[int((contact_counts[:,:6]>0).sum(1).min()),int((contact_counts[:,:6]>0).sum(1).max())],
        local_normal_force_stats=series_stats(normal), world_reaction_xyz_stats=series_stats(world),
        upward_reaction_samples=(world[:,:,2]>1e-9).sum(0),
        world_reaction_source=force_source, total_world_reaction_max_residual=residual,
        force_unit='MODEL_UNITS; absolute physical force NOT_ESTABLISHED',
        thorax_height=series_stats(body['thorax_position'][last_ids,2]),
        com_height=series_stats(body['com_position'][last_ids,2]),
        com_vz=series_stats(body['com_velocity'][last_ids,2]))
    return dict(all_network_first_event=network_first, perleg=perleg,
        first_saved_decode_s=first((decode_values>0).any(1),decode_times),
        first_write_boundary_s=first((commands>0).any(1), starts),
        first_actual_body_command_sample_s=first((body['command']>0).any(1), t),
        first_activation_above_native_floor_s=first((body['activation']>FLOOR).any(1), t),
        first_nonfoot_s=first(body['nonfoot_floor_count']>0,t),
        total_network_events=int(counts.sum()), unique_motor_node_events=int(counts[:,motor].sum()),
        first15ms_network_events=int(counts[np.rint(ends/NEURAL_DT).astype(int)<=30].sum()),
        first15ms_unique_motor_events=int(counts[np.rint(ends/NEURAL_DT).astype(int)<=30][:,motor].sum()),
        first_effective_motor_readout_s=first((decoded>0).any(1), ends),
        motor_nodes_that_emitted=int(np.count_nonzero(counts[:,motor].sum(0))),
        motor_pool_unique_node_count=len(motor),
        saturation_applied=saturation_stats(commands, ends-starts),
        saturation_decoded_at_window_end=saturation_stats(decoded, ends-starts),
        final_thorax_position=body['thorax_position'][-1], final_com_position=body['com_position'][-1],
        final_com_velocity=body['com_velocity'][-1], last50ms=last,
        nonfinite={k:int((~np.isfinite(body[k])).sum()) for k in body if np.issubdtype(body[k].dtype,np.number)},
        warning_max=np.max(body['warnings']), activation_min=float(body['activation'].min()),
        activation_max=float(body['activation'].max()), activation_over1=int((body['activation']>1).sum()),
        stable_walking='NOT_ESTABLISHED', formal_stability='NOT_JUDGED'), world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--new-arm', type=Path, default=None, help='default: root/normal if present, otherwise root')
    args = parser.parse_args()
    root = args.root
    newpath = args.new_arm or (root/'normal' if (root/'normal/BODY.npz').exists() else root)
    sources = [root/'MAPPING.json', newpath/'BODY.npz', newpath/'INTERVALS.npz', newpath/'TICKS.npz',
               newpath/'CONTACTS.json', OLD/'BODY.npz', OLD/'INTERVALS.npz', OLD/'RECORDS.json', OLD/'CONTACTS.json',
               REACTION, CLASSIFICATION]
    assert all(p.is_file() for p in sources), 'Artifacts not ready; no analysis outputs written'
    archive_checks = {str(p):verify_archive(p) for p in [OLD/'BODY.npz', OLD/'INTERVALS.npz', OLD/'RECORDS.json', OLD/'CONTACTS.json',REACTION,CLASSIFICATION]}
    hashes = {str(p):sha(p) for p in sources}
    rows = read_json(root/'MAPPING.json')['rows']
    assert [r['actuator_index'] for r in rows] == list(range(90))
    meta, old_ground = read_json(CLASSIFICATION), load_npz(REACTION)
    arms, plotdata = {}, {}
    for name,path in [('old5ms',OLD),('new1ms',newpath)]:
        body, iv = load_npz(path/'BODY.npz'),load_npz(path/'INTERVALS.npz')
        ticks = load_npz(path/'TICKS.npz') if name=='new1ms' else None
        records = read_json(path/'RECORDS.json') if name=='old5ms' else None
        contacts = read_json(path/'CONTACTS.json')
        assert len(contacts)==len(body['time'])
        arms[name], world = summarize_arm(name,body,iv,rows,ticks,records,contacts,meta,old_ground)
        plotdata[name] = (body,iv,world)
    oldbody, oldiv,_ = plotdata['old5ms']; newbody,newiv,_ = plotdata['new1ms']
    initial = {k:bool(np.array_equal(oldbody[k][0],newbody[k][0])) for k in ['qpos','qvel','activation','command']}
    assert all(initial.values())
    np.testing.assert_array_equal(oldbody['time'],newbody['time'])
    np.testing.assert_allclose(newiv['sensory_sample_start_s'],oldiv['interval_start_s'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(newiv['sensory_current_nA'][0],oldiv['sensory_current_nA'][0])
    diff = newiv['sensory_current_nA'] != oldiv['sensory_current_nA']
    binned = newiv['spikes'].reshape(50,5,12000).sum(1)
    feedback = dict(initial_state_exact=initial, first_sensory_current_exact=True,
        first_saved_sensory_current_divergence_s=first(diff.any(1),oldiv['interval_start_s']),
        differing_sensory_entries=int(diff.sum()), first_5ms_count_divergence_start_s=first((binned!=oldiv['spikes']).any(1),oldiv['interval_start_s']),
        differing_neuron_5ms_counts=int(np.count_nonzero(binned!=oldiv['spikes'])),
        caveat='Later closed-loop trajectories and sensory currents can diverge; not pure fixed neural-delay causal attribution. Historical full-network tick timing unavailable.')
    report = dict(schema_version=1, scope='S4 saved-only single command1ms vs historical5ms analysis',
        execution=dict(model_loads=0, forward_calls=0, body_steps=0, neural_steps=0, decoder_calls=0),
        sources_sha256=hashes, archive_checks=archive_checks, activation_threshold=FLOOR,
        arms=arms, feedback_divergence=feedback,
        limitations=['Historical pools unresolved; index coverage is not biological identity certification.',
                    'Biological calibration, physical/absolute force units, stable standing and walking NOT_ESTABLISHED.',
                    'Saved plane-tarsal contact geometry is not itself load-bearing or stability certification.',
                    'first_saved_decode_s is a logical filter-sample timestamp, NOT actual decoder-call completion. New logical6.5ms sample is produced during consume at7ms; actual completed-block decode/readout7ms versus historical10ms. Old subwindow logical samples were not saved.',
                    'Last50 reaction statistics are equal-step saved samples over (0.2,.25], not new simulation or exact continuous impulse.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,2,figsize=(16,17))
    for name,(body,iv,world) in plotdata.items():
        t=body['time']*1000
        data=[(t,body['command'].max(1)),(t,body['activation'].max(1)),
              (t,body['thorax_position'][:,2]),(t,body['com_position'][:,2]),
              (t,body['com_velocity'][:,2]),(t,body['nonfoot_floor_count']),
              (t,np.count_nonzero(body['foot_contact'],axis=1)),
              (t[-4000:],world[:,6,2])]
        for ax,(x,y) in zip(axs.flat,data): ax.plot(x,y,label=name)
        axs[3,1].plot(t[-4000:],world[:,:6,2].sum(1),ls='--',label=name+' all feet')
    titles=['Maximum actual saved body command','Maximum native muscle activation',
            'Thorax height / model units','Full-body COM height / model units','Full-body COM vz / model units',
            'NONFOOT plane contact count','Saved foot-contact legs (see true plane ledger in JSON)',
            'Last50ms world vertical reaction / model units']
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title);ax.set_xlabel('physical time / ms');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Command1ms vs historical5ms / saved closed-loop evidence / NOT standing or biological certification')
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(DESKTOP/'S4_COMMAND1MS.png',dpi=140);plt.close(fig)
    text=['# S4 command1ms — saved-only analysis','',
          'No simulation, model load, forward, neural step or decoder replay was executed.','',
          '| Metric | Historical5ms | New1ms |','|---|---:|---:|']
    for key in ['total_network_events','unique_motor_node_events','first15ms_network_events','first15ms_unique_motor_events','first_saved_decode_s','first_effective_motor_readout_s','first_write_boundary_s','first_actual_body_command_sample_s','first_activation_above_native_floor_s','first_nonfoot_s']:
        label = 'logical_filter_sample_time_s (NOT actual decoder call)' if key=='first_saved_decode_s' else 'actual_completed_block_decode_readout_s' if key=='first_effective_motor_readout_s' else key
        text.append('| '+label+' | '+' | '.join(str(arms[a][key]) for a in ['old5ms','new1ms'])+' |')
    text += ['', '## First network event',json.dumps({a:arms[a]['all_network_first_event'] for a in arms},indent=2),
             '', '## Per-leg chain (seconds)', '| Arm/leg | Motor event | Logical filter sample | Actual completed-block decode/readout | Write boundary | Actual BODY sample | Activation above floor |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        for leg,r in arms[arm]['perleg'].items():
            text.append('| '+arm+'/'+leg+' | '+' | '.join(str(r[k]) for k in ['first_motor_event_s','first_saved_decode_s','first_effective_motor_readout_s','first_write_boundary_s','first_body_command_sample_s','first_activation_above_native_floor_s'])+' |')
    text += ['', 'Activation threshold: native floor .0001 + 1e-12, not any increase from initial zero.',
             'First15ms per-leg deduplicated events (time/node), full decoded and written muscle vectors, and activation counts/onsets/peaks are in ANALYSIS.json.',
             'IMPORTANT: first_saved_decode_s denotes logical_filter_sample_time, not actual decoder-call completion. New logical6.5ms sample is produced by consume at7ms. Actual completed-block decode/readout occurs at7ms versus historical10ms; BODY first write7.0125ms versus10.0125ms. Old subwindow logical filter samples were not saved.',
             '', '## First15ms per-leg events and activation',
             '| Arm/leg | Unique pool events | Event times / ms | First above-floor activation / ms | Peak activation |',
             '|---|---:|---|---:|---:|']
    for arm in arms:
        for leg,r in arms[arm]['perleg'].items():
            early=r['first15ms']; onset=early['first_activation_above_floor_s']
            times=[round(e['time_s']*1000,6) for e in early['events']]
            text.append(f"| {arm}/{leg} | {early['event_count']} | {times} | {None if onset is None else onset*1000} | {max(early['activation_peak_by_actuator'])} |")
    text += [
             '', '## Saturation: compare time, not unequal windows']
    for arm in arms:
        s=arms[arm]['saturation_applied']
        text.append(f"- {arm}: raw actuator-window count {s['raw_actuator_window_count']} / {s['windows']} windows; actuator-seconds {s['actuator_seconds']}; actuator-time fraction {s['actuator_time_fraction']}.")
    text += ['', '## Last50ms true plane contact and world reaction',
             'Exactly rounded body ticks (16000,20000], i.e. (0.2,.25], 4000 samples per arm; .2 is excluded explicitly.',
             'Reaction xyz means below are MODEL_UNITS, not SI force. Contact presence and positive upward load are separate.']
    for arm in arms:
        last=arms[arm]['last50ms']
        text += ['',f'### {arm}','| Category | True contact samples | Mean world x,y,z reaction |','|---|---:|---|']
        for j,leg in enumerate(LEGS+['NONFOOT']):
            text.append(f"| {leg} | {last['geometric_contact_samples'][j]} | {last['world_reaction_xyz_stats']['mean'][j].tolist()} |")
        text.append('Reaction source: '+last['world_reaction_source'])
        text.append('Last50 height/vz statistics: '+json.dumps({k:last[k] for k in ['thorax_height','com_height','com_vz']},default=jsonable))
    text += ['', '## Interpretation boundaries',feedback['caveat']]+['- '+x for x in report['limitations']]
    text += ['', '## Reproducibility, actual calls and failures',
             'Unique dynamic run bash267: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_command1ms_single.py ; exit0. Actual: 1 model load, 20004 forward calls, 20000 body steps, 500 full-network neural steps, 250 decoder consumes, 50 sensory samples.',
             'Independent verifier: same environment with /run/media/sensen/Data2/cell_wound_prototype/verify_command1ms_single.py ; exit0, zero physics/model calls, 750 decoder consumes.',
             'Five deterministic tests plus one supplementary test passed. Initial deterministic trace-indent shape test failed exit1 and was corrected before dynamics; see '+str(root/'TEST_FAILURE.log')+'. No failed simulation is concealed.',
             'Saved-only analysis bash268/269 exit0 and later report/plot rerenders execute zero model loads, forward calls, body/neural steps and decoder consumes.',
             'Source/artifact final hashes: '+str(root/'MANIFEST_SHA256.json')+' ; source change ledger: '+str(root/'CHANGE_LEDGER.json')+' (final freeze handled after all reports/plots).',
             'Runner: /run/media/sensen/Data2/cell_wound_prototype/run_command1ms_single.py',
             'Tests: /run/media/sensen/Data2/cell_wound_prototype/test_split_write_clock.py',
             'Verifier: /run/media/sensen/Data2/cell_wound_prototype/verify_command1ms_single.py',
             'Analyzer: /run/media/sensen/Data2/cell_wound_prototype/analyze_command1ms_single.py',
             'Early-chain plot script: /run/media/sensen/Data2/cell_wound_prototype/plot_command1ms_early_chain.py',
             '', '## Outputs',str(DESKTOP/'S4_COMMAND1MS.png'),str(DESKTOP/'S4_COMMAND1MS_EARLY_CHAIN.png'),str(DESKTOP/'S4_COMMAND1MS.json'),str(root/'ANALYSIS.json')]
    assert all(sha(p)==hashes[str(p)] for p in sources), 'Input changed during analysis'
    encoded=json.dumps(report,indent=2,default=jsonable,allow_nan=False)+'\n'
    (root/'ANALYSIS.json').write_text(encoded)
    (DESKTOP/'S4_COMMAND1MS.json').write_text(encoded)
    (DESKTOP/'S4_COMMAND1MS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(outputs=[str(root/'ANALYSIS.json'),str(DESKTOP/'S4_COMMAND1MS.json'),str(DESKTOP/'S4_COMMAND1MS.md'),str(DESKTOP/'S4_COMMAND1MS.png')],execution=report['execution'])))


if __name__ == '__main__':
    main()
