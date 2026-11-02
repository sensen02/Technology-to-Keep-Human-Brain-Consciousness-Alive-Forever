"""S3/S4 fixed positive added-edge gain engineering calibration, explicitly offline.
No body, new tier, optimizer, training, production admission or default changes.
run -> fresh-process verify -> parent review -> explicit seal (never automatic).
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
import numpy as np
from scipy import sparse

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'tools'))
import replay_sourced_nt_candidate as old
sys.path.insert(0, str(R))
B, G, REF = old.B, old.G, old.O
O = Path('/home/sensen/Desktop/engine修复验收/s4-sourced-nt-gain-calibration/20261102T020329Z')
GRID = (.03125, .0625, .125, .25, .5)
sha, load, arrays, dump, same, matrix_exact, first = old.sha, old.load, old.arrays, old.dump, old.same, old.matrix_exact, old.first
LIMITS = ('Engineering calibration only; 2% is an engineering ceiling, not physiology or goal progress. '
          'Saved sensory replay has no physical feedback. Inherited base warm history, ring, ge/gi and '
          'recurrent inputs remain. Current cut is not a complete source cut. Raster changes cannot '
          'identify a unique direct/indirect pathway. Predicted NT/receptor signs and SNN rates are '
          'not biological truth. Physiological calibration, body support improvement and walking '
          'NOT_ESTABLISHED. No production admission or default graph change.')


def label(g):
    return 'GAIN_' + str(g).replace('.', 'p')


def calls_zero():
    return dict(network_instances=0, neural_steps=0, body_model=0, body_forward=0,
                body_steps=0, tier_build=0, training=0, optimizer=0)


def check_manifest(folder):
    manifest = folder / 'MANIFEST_SHA256.json'
    data = load(manifest)
    entries = data.get('files', data)
    hashes = {}
    for p, item in entries.items():
        h = item if isinstance(item, str) else item['sha256']
        assert sha(p) == h, p
        if isinstance(item, dict) and 'bytes' in item:
            assert Path(p).stat().st_size == item['bytes'], p
        hashes[p] = h
    return {str(manifest): sha(manifest), **hashes}


def source_context():
    c = old.context()
    roots, sel, motor, mapping, boundary, clock, weights, added, candidate, drive, records, expected = c
    full = arrays(B / 'RUN/NEURAL.npz')
    rec150 = load(B / 'RUN/NEURAL_RECORDS.json')[:150]
    drive150 = full['sensory_current_nA'][:150]
    raster150 = arrays(B / 'NEURAL_RASTER.npz')['spikes'][2200:2500]
    assert len(rec150) == 150 and drive150.shape == (150, 12000) and raster150.shape == (300, 12000)
    assert same(drive150, np.array([r['sensory_current_nA'] for r in rec150]))
    assert same(drive150[:50], drive)
    assert same(raster150.reshape(150, 2, 12000).sum(1), full['spike_counts'][:150])
    assert matrix_exact(weights['Wi'], candidate['Wi'])
    return c, drive150, rec150, raster150


def unpack(n):
    return np.unpackbits(n['spikes_packed'], axis=1, count=12000).astype(bool)


def audit_arm(folder, arm, c, gain, ticks, raw, drive=None):
    """Saved-array arithmetic only; no neural integrator or decoder instance."""
    roots, sel, motor, mapping, boundary, clock, weights, added, candidate, _, _, _ = c
    n = arrays(folder / (arm + '_NEURAL.npz'))
    d = arrays(folder / (arm + '_DECODE.npz'))
    init = arrays(folder / (arm + '_INITIAL.npz'))
    end = arrays(folder / (arm + '_FINAL.npz'))
    assert all(same(init[k], boundary[k]) for k in boundary)
    assert (int(init['steps']), float(init['t_ms']), int(init['ring_i']),
            float(init['decoder_elapsed_s']), int(init['decoder_interval'])) == (2200, 1100., 0, 1.1, 1099)
    e = unpack(n)
    arrival = np.unpackbits(n['arrivals_packed'], axis=1, count=12000).astype(bool)
    assert e.shape == (ticks, 12000) and arrival.shape == e.shape
    assert same(n['counts'], e.reshape(ticks // 2, 2, 12000).sum(1))
    assert same(n['motor_events'], e[:, motor]) and same(n['selected_events'], e[:, sel])
    ring = boundary['_ring'].copy(); pos = 0
    rate = boundary['decoder_rates'].copy(); decay = math.exp(-.0005 / .005)
    ref_until = boundary['refractory_until_ms'].copy()
    maxerr = 0.
    # Old baseline trace is explicitly counterfactual gain1, although actual gain0 matrix has no added mask.
    trace_gain = 1. if folder == REF else gain
    edge_weights = raw.data * trace_gain
    for t in range(ticks):
        assert same(arrival[t], ring[pos])
        active = arrival[t, raw.col]
        counts = np.bincount(raw.row[active], minlength=12000)
        increments = np.bincount(raw.row[active], weights=edge_weights[active], minlength=12000)
        assert same(counts, n['added_arrival_event_counts'][t])
        assert same(increments, n['added_arrival_increment_uS'][t])
        if 'arrival_selected' in n:
            assert same(n['arrival_selected'][t], arrival[t, sel])
        ring[pos] = e[t]; pos = (pos + 1) % 5
        ref_until[e[t]] = (2201 + t) * .5 + 2.
        instant = np.array([e[t, row['local_indices']].mean() / .0005 if row['local_indices'] else 0.
                            for row in mapping['rows']])
        rate = decay * rate + (1 - decay) * instant
        err = float(np.max(np.abs(rate - d['rates_halfstep_hz'][t])))
        assert err <= 1e-12; maxerr = max(maxerr, err)
        if t % 2:
            w = t // 2
            assert same(d['pool_counts'][w], np.array([e[2*w:2*w+2, row['local_indices']].sum() for row in mapping['rows']]))
            assert same(d['decoded_next'][w], np.clip(rate / 100., 0., 1.))
    assert same(d['rates_window_hz'], d['rates_halfstep_hz'][1::2])
    assert same(d['applied'][0], boundary['pending']) and same(d['applied'][1:], d['decoded_next'][:-1])
    assert same(d['applied_source_interval'], np.arange(1099, 1099 + ticks // 2))
    assert same(d['decoded_source_interval'], np.arange(1100, 1100 + ticks // 2))
    assert same(end['_ring'], ring) and int(end['ring_i']) == pos
    assert same(end['spike_count'], boundary['spike_count'] + e.sum(0))
    assert same(end['refractory_until_ms'], ref_until)
    assert same(end['threshold_offset_mV'], boundary['threshold_offset_mV'])
    assert int(end['steps']) == 2200 + ticks and float(end['t_ms']) == 1100 + ticks * .5
    assert float(end['decoder_elapsed_s']) == 1.1 + ticks // 2 * .001
    assert int(end['decoder_interval']) == 1099 + ticks // 2
    for k in ['v', 'ge', 'gi', 'adaptation']:
        assert n[k].shape == e.shape and np.isfinite(n[k]).all()
        assert same(end[k], n[k][-1])
    assert same(end['decoder_rates'], rate) and same(end['pending'], d['decoded_next'][-1])
    assert bool(n['added_arrivals_applied']) == (gain > 0)
    return dict(ticks=ticks, max_decoder_error=maxerr, pure_arrays=True, initial_final_exact=True,
                delay_arrivals_exact=True, decoder_command_causality_exact=True)


def raw_graph(c):
    roots, sel = c[:2]
    rows = list(csv.DictReader((G / 'ADDED_RAW_ROWS.csv').open()))
    post, pre, val = [], [], []
    for r in rows:
        i, j = int(r['pre_local']), int(r['post_local'])
        w = min(float(r['synapses']), 8) * .0005 * .2
        assert int(r['pre_root888']) == roots[i] and int(r['post_root888']) == roots[j]
        assert i in sel and r['predicted_NT'] == 'acetylcholine' and r['receptor_assumption'] == 'EXC'
        assert w > 0 and float(r['conductance_increment_uS']) == w
        post.append(j); pre.append(i); val.append(w)
    mat = sparse.coo_matrix((val, (post, pre)), shape=(12000, 12000)).tocsr()
    assert len(rows) == 17264 and matrix_exact(mat, c[7]['We'])
    return mat.tocoo()


def baseline_arrays(c, records, raster):
    """Rebuild original 50/150ms decoder from the true round13 root888 raster."""
    boundary, mapping, motor = c[4], c[3], c[2]
    neural = arrays(B / 'RUN/NEURAL.npz')
    windows = len(records)
    rate = boundary['decoder_rates'].copy(); half = []
    for sp in raster:
        inst = np.array([sp[r['local_indices']].mean() / .0005 if r['local_indices'] else 0. for r in mapping['rows']])
        rate = math.exp(-.0005/.005)*rate + (1-math.exp(-.0005/.005))*inst
        half.append(rate.copy())
    half = np.array(half); nxt = np.clip(half[1::2] / 100, 0, 1)
    app = np.vstack([boundary['pending'], nxt[:-1]])
    counts = np.array([[raster[2*w:2*w+2, r['local_indices']].sum() for r in mapping['rows']] for w in range(windows)])
    for key, val, field, dec in [('applied', app, 'applied_command', False),
            ('decoded_next', nxt, 'decoded_next_command', False),
            ('rates_window_hz', half[1::2], 'rates_hz', True), ('pool_counts', counts, 'pool_spike_counts', True)]:
        ref = np.array([r['decoder'][field] if dec else r[field] for r in records])
        assert same(val, ref), key
    assert same(app, neural['commands'][:windows])
    assert same(raster.reshape(windows, 2, 12000).sum(1), neural['spike_counts'][:windows])
    return dict(spikes_packed=np.packbits(raster, axis=1), motor_events=raster[:, motor],
                rates_halfstep_hz=half, rates_window_hz=half[1::2], applied=app, decoded_next=nxt, pool_counts=counts)


def reference_gate(c, rec150, raster150, raw):
    old.checked_sources()
    audits = {l: audit_arm(REF, l, c, g, 100, raw) for l, g in
              [('BASELINE', 0), ('CANDIDATE', 1), ('SELECTED_CURRENT_CUT', 1)]}
    base = baseline_arrays(c, rec150, raster150)
    n = arrays(REF / 'BASELINE_NEURAL.npz'); d = arrays(REF / 'BASELINE_DECODE.npz')
    assert same(unpack(n), raster150[:100])
    for key in ['applied', 'decoded_next', 'rates_halfstep_hz', 'rates_window_hz', 'pool_counts']:
        assert same(d[key], base[key][:100 if key == 'rates_halfstep_hz' else 50]), key
    assert int(n['motor_events'].sum()) == 54
    inp = arrays(REF / 'INPUTS.npz'); cut = c[9].copy(); cut[:, c[1]] = 0
    assert same(inp['saved_current_nA'], c[9]) and same(inp['cut_current_nA'], cut)
    assert same(inp['selected_mask'], np.isin(np.arange(12000), c[1]))
    assert same(cut[:, ~inp['selected_mask']], c[9][:, ~inp['selected_mask']])
    return dict(passed=True, reused_gain0_gain1_cut1=True, original_root888_150ms_exact=True,
                source_raster_slice=[2200, 2500], record_slice=[0, 150], source_release_ms=[0, 150],
                base_history='Original unmodified matrix prewarm; no candidate prewarm', audits=audits,
                baseline_actual_added_edges=0, baseline_trace='Old saved counterfactual gain1 arrival trace; actual applied=0',
                extra_networks=0, extra_ticks=0)


def checked_sources():
    for p, h in load(O / 'PREREGISTRATION.json')['source_sha256'].items():
        assert sha(p) == h, p


def prepare():
    assert not O.exists(), 'Refuse overwrite'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip().startswith('10ab1b4')
    source = {}
    for folder in [B, G, REF]:
        source.update(check_manifest(folder))
    c, drive150, rec150, raster150 = source_context()
    raw = raw_graph(c)
    gate = reference_gate(c, rec150, raster150, raw)
    for p in [Path(__file__), Path(old.__file__), R/'engine/neural_cond.py', R/'engine/embodied/motor_output.py']:
        source[str(p)] = sha(p)
    O.mkdir(parents=True)
    dump(O/'PREREGISTRATION.json', dict(stage='S3/S4', purpose='Engineering added-edge positive-gain calibration; not biological learning or behavior improvement',
        grid=GRID, matrix='baseWe + gain * ADDED_We; baseWi unchanged; gain0 original matrix without added mask',
        fixed='Original roots/masks/old weights/204MN/90pool decoder/state/rates/pending/base warm history; neutral parameters',
        selection=dict(window_ms=50, actual_applied_elements=4500, max_saturated=90, engineering_ceiling=.02,
                       min_MN_spikes=54, max_MN_spikes=162, MN_event_difference_required=True,
                       priority='highest passing gain', no_pass='NO_CANDIDATE; no additional sweep or changed thresholds'),
        cut='Only selected963 saved current zeroed, same chosen gain and initial state; observational and cannot reselect',
        extension='Only selected gain, original saved150ms current, 300ticks, first100 exact; no reselect based on150ms',
        baseline150='Original round13 root888 raster[2200:2500]/records[:150], independent array decoder; no baseline integration',
        planned_counts_if_selected=dict(network_instances=7, neural_steps=900),
        planned_counts_if_no_candidate=dict(network_instances=5, neural_steps=500),
        zero_calls=['body_model', 'body_forward', 'body_steps', 'tier_build', 'training', 'optimizer'],
        source_sha256=source, limits=LIMITS, repository_head='10ab1b4de91b5f499bbb70856b834f5701f8bbd4'))
    dump(O/'OLD_REFERENCE_GATE.json', gate)
    dump(O/'MAPPING.json', c[3]); dump(O/'INITIAL_CLOCK.json', c[5])
    np.savez_compressed(O/'INITIAL_BOUNDARY.npz', **c[4])
    cut = c[9].copy(); cut[:, c[1]] = 0
    np.savez_compressed(O/'INPUTS.npz', root_ids=c[0], selected_local=c[1], motor_local=c[2],
                        selected_mask=np.isin(np.arange(12000), c[1]), saved_current_nA=c[9],
                        cut_current_nA=cut, saved_current150_nA=drive150)
    np.savez_compressed(O/'BASELINE150_REUSED.npz', **baseline_arrays(c, rec150, raster150))
    (O/'EXECUTED_calibrate_sourced_nt_increment.py').write_bytes(Path(__file__).read_bytes())
    return c, drive150


def matrix(gain, c):
    return dict(We=(c[6]['We'] + gain*c[7]['We']).tocsr(), Wi=c[6]['Wi'].copy())


def simulate(arm, gain, c, current, calls):
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from engine.embodied.motor_output import MotorOutput
    roots, sel, motor, mapping, boundary, clock = c[:6]
    net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=.5, threshold_offset_mV=boundary['threshold_offset_mV'])
    calls['network_instances'] += 1
    for k in old.FIELDS:
        if k != 'threshold_offset_mV': setattr(net, k, boundary[k].copy())
    for k in ['steps', 't_ms', '_ring_i']: setattr(net, k, clock['net_scalars'][k])
    mats = matrix(gain, c); net.set_connectivity(mats['We'], mats['Wi'])
    for k in mats:
        assert matrix_exact(mats[k], getattr(net, k))
    sparse.save_npz(O/(arm+'_ACTUAL_We.npz'), net.We)
    sparse.save_npz(O/(arm+'_ACTUAL_Wi.npz'), net.Wi)
    sparse.save_npz(O/(arm+'_MATRIX_DELTA.npz'), gain*c[7]['We'])
    dec = MotorOutput(mapping); dec.rates = boundary['decoder_rates'].copy()
    dec.elapsed_s = clock['decoder_elapsed_s']; dec.interval = clock['decoder_interval']
    pending = boundary['pending'].copy()
    np.savez_compressed(O/(arm+'_INITIAL.npz'), **old.snapshot(net, dec, pending))
    assert all(same(old.snapshot(net,dec,pending)[k], boundary[k]) for k in boundary)
    windows = len(current); ticks = windows*2
    ev = np.zeros((ticks,12000),bool); arrivals = np.zeros_like(ev)
    traces = {k:np.empty((ticks,12000)) for k in ['v','ge','gi','adaptation']}
    increment = np.empty((ticks,12000)); counts = np.empty((ticks,12000),np.int32)
    half = np.empty((ticks,90)); app = np.empty((windows,90)); nxt = np.empty_like(app)
    pool = np.empty((windows,90),np.int64)
    delta = gain*c[7]['We']; positive = c[7]['We'].astype(bool).astype(np.int32)
    decay = math.exp(-.0005/.005)
    for w in range(windows):
        app[w] = pending; rate = dec.rates.copy()
        for sub in range(2):
            t=2*w+sub; arrivals[t]=net._ring[net._ring_i]
            increment[t]=delta@arrivals[t]; counts[t]=positive@arrivals[t].astype(np.int32)
            ev[t]=net.step(current[w]); calls['neural_steps']+=1
            for k in traces:
                traces[k][t]=getattr(net,k); assert np.isfinite(traces[k][t]).all()
            inst=np.array([ev[t,g].mean()/.0005 if g.size else 0. for g in dec.groups])
            rate=decay*rate+(1-decay)*inst; half[t]=rate
        pending, trace=dec.consume(ev[2*w:2*w+2],spike_dt_s=.0005,start_s=1.1+w*.001,end_s=1.1+(w+1)*.001)
        assert same(rate,dec.rates)
        nxt[w]=pending;pool[w]=trace['pool_spike_counts']
    np.savez_compressed(O/(arm+'_FINAL.npz'),**old.snapshot(net,dec,pending))
    np.savez_compressed(O/(arm+'_NEURAL.npz'),spikes_packed=np.packbits(ev,axis=1), counts=ev.reshape(windows,2,12000).sum(1),
        motor_events=ev[:,motor],selected_events=ev[:,sel],arrivals_packed=np.packbits(arrivals,axis=1),
        arrival_selected=arrivals[:,sel],added_arrival_increment_uS=increment,added_arrival_event_counts=counts,
        added_arrivals_applied=np.array(True),**traces)
    np.savez_compressed(O/(arm+'_DECODE.npz'),rates_halfstep_hz=half,rates_window_hz=half[1::2],applied=app,
        decoded_next=nxt,pool_counts=pool,applied_source_interval=np.arange(1099,1099+windows),decoded_source_interval=np.arange(1100,1100+windows))
    dump(O/(arm+'_MATRIX_IDENTITY.json'),dict(gain=gain,formula='baseWe+gain*ADDED_We',Wi_unchanged=True,
        added_positive_edges=int(delta.nnz), old_weights_preserved=True, roots_unchanged=True,
        actual_We_sha256=sha(O/(arm+'_ACTUAL_We.npz')),actual_Wi_sha256=sha(O/(arm+'_ACTUAL_Wi.npz')),
        delta_sha256=sha(O/(arm+'_MATRIX_DELTA.npz')),baseWe_sha256=sha(B/'ACTUAL_We.npz'),baseWi_sha256=sha(B/'ACTUAL_Wi.npz')))
    dump(O/'ACTUAL_CALLS.json',calls)
    print(arm, 'completed',ticks,'ticks',flush=True)


def selection():
    base=arrays(REF/'BASELINE_NEURAL.npz')['motor_events']; rows=[]
    for gain in GRID:
        l=label(gain);n=arrays(O/(l+'_NEURAL.npz'));d=arrays(O/(l+'_DECODE.npz'))
        saturated=int((d['applied']>=1).sum()); mn=int(n['motor_events'].sum())
        bits=int(np.count_nonzero(n['motor_events']!=base))
        rows.append(dict(gain=gain,arm=l,applied_saturated=saturated,applied_elements=4500,
            MN_spikes=mn,MN_different_bits=bits,pass_saturation=saturated<=90,pass_MN_range=54<=mn<=162,
            pass_MN_event_difference=bits>0,passed=saturated<=90 and 54<=mn<=162 and bits>0))
    eligible=[r['gain'] for r in rows if r['passed']]
    return dict(rows=rows,selected_gain=max(eligible) if eligible else None,
                status='SELECTED' if eligible else 'NO_CANDIDATE',rule='Highest gain passing fixed50ms gates; cut/150ms cannot change selection')


def arm_summary(folder, arm, c, ms):
    n=arrays(folder/(arm+'_NEURAL.npz'));d=arrays(folder/(arm+'_DECODE.npz'));e=unpack(n)[:2*ms]
    return dict(window_ms=ms,network_spikes=int(e.sum()),MN_spikes=int(e[:,c[2]].sum()),
        selected_spikes=int(e[:,c[1]].sum()),active_MN=int(e[:,c[2]].any(0).sum()),
        applied_saturated=int((d['applied'][:ms]>=1).sum()),applied_elements=ms*90,
        next_saturated=int((d['decoded_next'][:ms]>=1).sum()),max_rate_hz=float(d['rates_halfstep_hz'][:2*ms].max()))


def difference(folder, arm, ref_folder, ref_arm, c, ms):
    n=arrays(folder/(arm+'_NEURAL.npz'));b=arrays(ref_folder/(ref_arm+'_NEURAL.npz'))
    d=arrays(folder/(arm+'_DECODE.npz'));bd=arrays(ref_folder/(ref_arm+'_DECODE.npz'))
    motor=c[2];a=unpack(n)[:2*ms];be=unpack(b)[:2*ms]
    out=dict(MN_different_bits=int(np.count_nonzero(a[:,motor]!=be[:,motor])),
        first_MN_event_end_ms=first(a[:,motor],be[:,motor],.5),first_network_event_end_ms=first(a,be,.5),
        first_decoded_available_ms=first(d['decoded_next'][:ms],bd['decoded_next'][:ms],1.),
        first_applied_window_start_ms=first(d['applied'][:ms],bd['applied'][:ms],1.,-1.),
        applied_different_elements=int(np.count_nonzero(d['applied'][:ms]!=bd['applied'][:ms])))
    for k in ['v','ge','gi','adaptation']:
        out['first_'+k+'_end_ms']=first(n[k][:2*ms],b[k][:2*ms],.5)
    return out


def extension_prefix(chosen):
    for suffix in ['NEURAL','DECODE']:
        short=arrays(O/(chosen+'_'+suffix+'.npz'));long=arrays(O/('VALIDATE150_'+suffix+'.npz'))
        for k,v in short.items():
            assert same(v,long[k][:len(v)]) if v.ndim else same(v,long[k]), (suffix,k)
    for suffix in ['INITIAL']:
        a=arrays(O/(chosen+'_'+suffix+'.npz'));b=arrays(O/('VALIDATE150_'+suffix+'.npz'))
        assert all(same(v,b[k]) for k,v in a.items())
    shortend=arrays(O/(chosen+'_FINAL.npz'));long=arrays(O/'VALIDATE150_NEURAL.npz')
    e=unpack(long)[:100];init=arrays(O/'INITIAL_BOUNDARY.npz');ring=init['_ring'].copy();pos=0
    for sp in e:ring[pos]=sp;pos=(pos+1)%5
    assert same(shortend['_ring'],ring) and same(shortend['spike_count'],init['spike_count']+e.sum(0))
    for k in ['v','ge','gi','adaptation']:assert same(shortend[k],long[k][99])
    return True


def metrics(c, chosen):
    summaries={l:{str(ms):arm_summary(REF,l,c,ms) for ms in [5,15,50]} for l in old.LABELS}
    for gain in GRID:
        l=label(gain);summaries[l]={str(ms):arm_summary(O,l,c,ms) for ms in [5,15,50]}
    diffs={label(g):difference(O,label(g),REF,'BASELINE',c,50) for g in GRID}
    out=dict(summaries=summaries,differences_vs_baseline50=diffs,limits=LIMITS,
             old_reference_window='Round13 root888 release0..50ms; raster2200:2300; gain0/1/cut1 reused, no rerun')
    if chosen is not None:
        arm=label(chosen)
        out['cut_summary']=arm_summary(O,'SELECTED_CURRENT_CUT',c,50)
        out['cut_vs_same_gain']=difference(O,'SELECTED_CURRENT_CUT',O,arm,c,50)
        out['validate150_summary']=arm_summary(O,'VALIDATE150',c,150)
        base=arrays(O/'BASELINE150_REUSED.npz');n=arrays(O/'VALIDATE150_NEURAL.npz');d=arrays(O/'VALIDATE150_DECODE.npz')
        out['baseline150_summary']=dict(MN_spikes=int(base['motor_events'].sum()),applied_saturated=int((base['applied']>=1).sum()),applied_elements=13500)
        out['validate150_vs_synchronous_baseline']=dict(MN_different_bits=int(np.count_nonzero(n['motor_events']!=base['motor_events'])),
            first_MN_event_end_ms=first(n['motor_events'],base['motor_events'],.5),
            first_applied_window_start_ms=first(d['applied'],base['applied'],1.,-1.))
        out['extension_first50_exact']=extension_prefix(arm)
        out['extension_does_not_reselect']=True
    return out


def run():
    calls=calls_zero(); c,drive150=prepare()
    try:
        for gain in GRID:simulate(label(gain),gain,c,c[9],calls)
        selected=selection();dump(O/'SELECTION_TABLE.json',selected)
        chosen=selected['selected_gain']
        if chosen is not None:
            cut=c[9].copy();cut[:,c[1]]=0
            simulate('SELECTED_CURRENT_CUT',chosen,c,cut,calls)
            simulate('VALIDATE150',chosen,c,drive150,calls)
        result=metrics(c,chosen);result.update(selection=selected,calls=calls)
        checked_sources();dump(O/'RESULT.json',result)
        print(json.dumps(dict(selection=selected,calls=calls),indent=2),flush=True)
    except BaseException:
        dump(O/'FAILURE.json',dict(calls=calls,traceback=traceback.format_exc()))
        raise


def verify():
    # This process imports no engine module, creates no networks and performs no integration.
    checked_sources();c,drive150,rec150,raster150=source_context();raw=raw_graph(c)
    gate=reference_gate(c,rec150,raster150,raw)
    assert gate==load(O/'OLD_REFERENCE_GATE.json')
    inp=arrays(O/'INPUTS.npz');cut=c[9].copy();cut[:,c[1]]=0
    for k,v in dict(root_ids=c[0],selected_local=c[1],motor_local=c[2],selected_mask=np.isin(np.arange(12000),c[1]),
                    saved_current_nA=c[9],cut_current_nA=cut,saved_current150_nA=drive150).items():assert same(inp[k],v),k
    assert same(cut[:,~inp['selected_mask']],c[9][:,~inp['selected_mask']])
    rebuilt=baseline_arrays(c,rec150,raster150);saved=arrays(O/'BASELINE150_REUSED.npz')
    assert set(rebuilt)==set(saved) and all(same(v,saved[k]) for k,v in rebuilt.items())
    select=selection();assert select==load(O/'SELECTION_TABLE.json')
    chosen=select['selected_gain'];arms=[(label(g),g,100) for g in GRID]
    if chosen is not None:arms += [('SELECTED_CURRENT_CUT',chosen,100),('VALIDATE150',chosen,300)]
    audits={}; identities={}
    for arm,gain,ticks in arms:
        audits[arm]=audit_arm(O,arm,c,gain,ticks,raw)
        actual={k:sparse.load_npz(O/(arm+'_ACTUAL_'+k+'.npz')).tocsr() for k in ['We','Wi']}
        delta=sparse.load_npz(O/(arm+'_MATRIX_DELTA.npz')).tocsr()
        assert matrix_exact(delta,gain*c[7]['We']) and np.all(delta.data>0)
        assert same(delta.indptr,c[7]['We'].indptr) and same(delta.indices,c[7]['We'].indices)
        assert matrix_exact(actual['We'],c[6]['We']+gain*c[7]['We'])
        assert matrix_exact(actual['Wi'],c[6]['Wi'])
        assert matrix_exact(actual['We']-delta,c[6]['We'])
        # Exact old support AND weight preservation, independent of subtraction.
        coo=c[6]['We'].tocoo();assert same(np.asarray(actual['We'][coo.row,coo.col]).ravel(),coo.data)
        assert actual['We'].nnz==c[6]['We'].nnz+delta.nnz
        ident=load(O/(arm+'_MATRIX_IDENTITY.json'))
        assert ident['gain']==gain and ident['added_positive_edges']==17264
        assert ident['actual_We_sha256']==sha(O/(arm+'_ACTUAL_We.npz')) and ident['actual_Wi_sha256']==sha(O/(arm+'_ACTUAL_Wi.npz'))
        assert ident['delta_sha256']==sha(O/(arm+'_MATRIX_DELTA.npz'))
        identities[arm]=dict(positive_same_mask=True,old_values_exact=True,Wi_exact=True,root_mask_fixed=True)
    if chosen is not None:extension_prefix(label(chosen))
    result=load(O/'RESULT.json');recomputed=metrics(c,chosen)
    for k,v in recomputed.items():assert result[k]==v,k
    assert result['selection']==select
    expected=calls_zero();expected.update(network_instances=len(arms),neural_steps=sum(t for _,_,t in arms))
    assert result['calls']==expected and load(O/'ACTUAL_CALLS.json')==expected
    assert 'engine.neural_cond' not in sys.modules and 'engine.embodied.motor_output' not in sys.modules
    checked_sources()
    dump(O/'INDEPENDENT_VERIFICATION.json',dict(passed=True,pid=os.getpid(),pure_arrays=True,calls=calls_zero(),
        verified_old_ticks=300,verified_new_ticks=expected['neural_steps'],original_baseline150_ticks=300,
        old_reference_gate=gate,arms=audits,matrix_identities=identities,grid_selection_exact=True,
        cut_others_unchanged=True,extension_first50_exact=True if chosen is not None else None,
        baseline150_synchronous_exact=True,source_hashes_unchanged=True))
    report()
    print('Fresh-process pure-array verification PASS; saved new ticks',expected['neural_steps'])


def report():
    result=load(O/'RESULT.json');sel=result['selection'];chosen=sel['selected_gain']
    lines=['# S3/S4 来源新增边正尺度工程校准（待父review，未封存）','',
        '固定5档网格；0身体模型/forward/step、0训练/optimizer、0新tier。不是生物参数学习、动作改善或生产准入。',
        '预热历史继承原base矩阵；各臂从同原checkpoint state/rates/pending开始。旧gain0/1/cut1只读hash及实际数组门复用，0额外baseline积分。',
        '硬门：50ms实际applied饱和≤90/4500（工程上限2%，不是生理标准或目标进度）；MN事件异于baseline；54≤MN总spikes≤162；合格者取最高gain。','',
        '|gain|MNspikes|MN不同事件位|实际applied饱和/4500|通过|','|---|---:|---:|---:|---|']
    for r in sel['rows']:lines.append(f"|{r['gain']}|{r['MN_spikes']}|{r['MN_different_bits']}|{r['applied_saturated']}|{r['passed']}|")
    lines += ['',f"选择：{sel['status']}；gain={chosen}。实际调用：`{json.dumps(result['calls'])}`。",'',
        '旧参考窗口为第13轮root888释放后0–50ms，raster[2200:2300]/records[:50]；延长基准为0–150ms，raster[2200:2500]/records[:150]。没有重积分baseline。',
        'gain0仅原矩阵，无新增边mask；旧baseline保存arrival为gain1反事实，实际新增边施加0，不能混为实际输入。']
    if chosen is not None:
        lines += ['', '同gain只将963selected保存current置0（其余current/state不变）；结果不改变选择：',
            '```json',json.dumps(result['cut_vs_same_gain'],ensure_ascii=False,indent=2),'```',
            '150ms只验证唯一已选档，前50ms全部已存逐tick状态/raster/arrival/decoder/命令exact，不以延长结果重选：',
            '```json',json.dumps({k:result[k] for k in ['validate150_summary','baseline150_summary','validate150_vs_synchronous_baseline']},ensure_ascii=False,indent=2),'```']
    else:
        lines += ['', 'NO_CANDIDATE：按预登记停止，未运行cut或延长；不继续扫值或改阈值。下一策略仅可另行预登记并引入独立受体/突触尺度约束或真实生理资料，不调肌肉/decoder或补假连接。']
    lines += ['',LIMITS,'', '独立新进程只用数组：旧参考3臂重构、raw正边及mask/旧值、完整delayring/204MN/counts/90rates/因果命令/初末状态、选择表、current cut非目标不变、150ms同期基准与前50ms复算。无额外积分。',
        '完整状态含初末v/ge/gi/adaptation/refractory/count/ring/threshold/clock、decoder rates/pending；逐tick存v/ge/gi/adaptation、packed raster/arrivals、selected arrivals、added events/increments、204MN events及90rates/applied/next。',
        '', '清理：实际删除0；无确认可弃文件，失败与完整证据保留。核心必需为脚本、预登记、输入/状态/权重/事件、选择表、结果和验证；报告/执行台账/文件清单为重要辅助。完整路径及重要程度见FILE_INVENTORY.json。',
        '封存必须在父级review完成后显式执行seal；当前不生成manifest。']
    (O/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    inventory={str(p):dict(importance='重要辅助' if p.name in ['REPORT.zh-CN.md','JOB_LEDGER.json'] else '核心必需',purpose='复算、审查及执行来源') for p in sorted(O.iterdir()) if p.is_file()}
    inventory[str(Path(__file__))]=dict(importance='核心必需',purpose='固定网格运行及独立数组验证入口')
    inventory[str(O/'FILE_INVENTORY.json')]=dict(importance='重要辅助',purpose='完整文件用途与保留级别')
    dump(O/'FILE_INVENTORY.json',dict(files=inventory,cleanup_deleted=0,sealed=False))


def seal():
    assert not (O/'MANIFEST_SHA256.json').exists()
    review=load(O/'PARENT_REVIEW.json');assert review['passed'] and review['reviewer']=='parent'
    checked_sources();assert load(O/'INDEPENDENT_VERIFICATION.json')['passed']
    dump(O/'MANIFEST_SHA256.json',dict(files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.iterdir()) if p.is_file()},sealed_once=True,parent_review_before_seal=True))
    print('Sealed once after parent review',O)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify','seal']);args=parser.parse_args()
    {'run':run,'verify':verify,'seal':seal}[args.mode]()
