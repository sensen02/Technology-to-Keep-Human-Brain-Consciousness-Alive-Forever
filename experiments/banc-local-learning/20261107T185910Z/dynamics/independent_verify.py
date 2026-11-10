"""Independent S3/S4 verification. Run with venv_body/bin/python -B.
Only writes INDEPENDENT_VERIFICATION.json. No physics, training or synthetic edges.
Reference uses original-cache edge counts, numpy bincount, absolute-time queues,
and an algebraically separate voltage increment; never calls sparse matmul.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from dataclasses import asdict
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'
OUT = HERE / 'INDEPENDENT_VERIFICATION.json'
N = 20000
REPORT = dict(schema='banc-independent-verification/1', status='RUNNING',
    scope='S3/S4 independent numerical/identity engineering verification; not physiology or behavior validation',
    counters=dict(actual_network_step_calls=0, completed_network_steps=0,
                  manual_full_vector_iterations=0, manual_scalar_row_iterations=0,
                  physical_steps=0, training_updates=0),
    checks={}, errors_max={}, runs=[], failures=[], source_hashes={})
ATOL = 2e-11
REPORT['tolerances'] = dict(absolute_state_tolerance=ATOL, relative_state_tolerance=0,
    spike_count_spikes_queue_identity='exact', weight_absolute_tolerance=2e-17)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def check(name, condition):
    REPORT['checks'][name] = bool(condition)
    if not condition:
        raise AssertionError(name)

def close(name, a, b, tol=ATOL):
    a, b = np.asarray(a), np.asarray(b)
    error = float(np.max(np.abs(a-b), initial=0))
    REPORT['errors_max'][name] = max(error, REPORT['errors_max'].get(name, 0.))
    if a.shape != b.shape or not np.isfinite(error) or error > tol:
        raise AssertionError(f'{name}: shape {a.shape}/{b.shape}, maximum error {error} > {tol}')

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

class Reference:
    def __init__(self, dt, p, pre, post, raw, excitatory, scale):
        self.dt, self.p = dt, p
        self.pre, self.post = pre, post
        self.weight = raw * (scale / np.bincount(post, weights=raw, minlength=N)[post])
        self.exc = excitatory
        self.v = np.full(N, p['rest_mV'])
        self.ge = np.zeros(N)
        self.gi = np.zeros(N)
        self.adaptation = np.zeros(N)
        self.refractory_until_ms = np.zeros(N)
        self.spike_count = np.zeros(N, dtype=np.int64)
        self.rates = np.zeros(N)
        self.t = 0.
        self.iterations = 0
        self.pending = []  # (absolute arrival time, presynaptic identity vector)
        self.arrival_batches = 0
        self.arrival_spikes = 0
        self.arrival_weight_uS = 0.
        self.first_emission = None
        self.first_arrival = None
        self.history = []

    def step(self, current, scalar_rows=()):
        REPORT['counters']['manual_full_vector_iterations'] += 1
        p, h, t = self.p, self.dt, self.t
        arrival = np.zeros(N)
        future = []
        for when, spikes in self.pending:
            if when <= t + 1e-12:
                if abs(when-t) > 1e-12:
                    raise AssertionError('reference late arrival')
                arrival += spikes
                self.arrival_batches += 1
                self.arrival_spikes += int(spikes.sum())
                if spikes.any() and self.first_arrival is None:
                    self.first_arrival = t
            else:
                future.append((when, spikes))
        self.pending = future
        edge_value = self.weight * arrival[self.pre]
        added_e = np.bincount(self.post[self.exc], weights=edge_value[self.exc], minlength=N)
        added_i = np.bincount(self.post[~self.exc], weights=edge_value[~self.exc], minlength=N)
        self.arrival_weight_uS += float(added_e.sum()+added_i.sum())
        g_e, g_i = self.ge + added_e, self.gi + added_i
        active = self.refractory_until_ms <= t + 1e-10
        # Independent rearrangement of backward Euler: V_new=V_old+deltaV.
        force = (p['leak_uS']*(p['rest_mV']-self.v)
                 + g_e*(p['exc_reversal_mV']-self.v)
                 + g_i*(p['inh_reversal_mV']-self.v) + current - self.adaptation)
        proposed = self.v + h * force / (p['capacitance_nF'] + h*(p['leak_uS']+g_e+g_i))
        spikes = active & (proposed >= p['threshold_mV'])
        next_v = np.where(active & ~spikes, proposed, p['reset_mV'])
        for row in scalar_rows:
            REPORT['counters']['manual_scalar_row_iterations'] += 1
            e, i = float(self.ge[row]), float(self.gi[row])
            for edge in np.flatnonzero(self.post == row):
                increment = float(self.weight[edge]) * float(arrival[self.pre[edge]])
                if self.exc[edge]: e += increment
                else: i += increment
            candidate = ((p['capacitance_nF']/h)*self.v[row] + p['leak_uS']*p['rest_mV']
                         + e*p['exc_reversal_mV'] + i*p['inh_reversal_mV']
                         + current[row] - self.adaptation[row]) / (p['capacitance_nF']/h+p['leak_uS']+e+i)
            expected = candidate if active[row] and candidate < p['threshold_mV'] else p['reset_mV']
            close('scalar_voltage_vs_vector', [expected], [next_v[row]])
            close('scalar_scatter_vs_vector', [e, i], [g_e[row], g_i[row]])
        self.t += h
        self.v = next_v
        self.ge = g_e * math.exp(-h/p['exc_tau_ms'])
        self.gi = g_i * math.exp(-h/p['inh_tau_ms'])
        self.adaptation = self.adaptation * math.exp(-h/p['adaptation_tau_ms']) + spikes*p['adaptation_increment_nA']
        self.refractory_until_ms[spikes] = self.t + p['refractory_ms']
        self.spike_count += spikes
        self.rates = self.rates*math.exp(-h/20.) + 50.*spikes
        if spikes.any():
            self.pending.append((self.t+p['delay_ms'], spikes.copy()))
            if self.first_emission is None: self.first_emission = self.t
        self.history.append((self.t, spikes.copy()))
        self.iterations += 1
        return spikes

    def compare(self, circuit, actual_spikes):
        net = circuit.net
        for field in ('v', 'ge', 'gi', 'adaptation', 'refractory_until_ms'):
            close(field, getattr(net, field), getattr(self, field))
        close('rates', circuit.filtered_rates_hz, self.rates)
        check('exact_spike_counts', np.array_equal(net.spike_count, self.spike_count))
        check('exact_spikes', np.array_equal(actual_spikes, self.history[-1][1]))
        check('exact_absolute_time', net.t_ms == self.t)
        # Decode implementation queue for comparison only. Reference never uses it.
        for offset in range(len(net._ring)):
            expected = np.zeros(N, bool)
            due = self.t + offset*self.dt
            for when, spikes in self.pending:
                if when == due: expected |= spikes
            check('ring_matches_absolute_timestamp_queue',
                  np.array_equal(net._ring[(net._ring_i+offset) % len(net._ring)], expected))


def main():
    prereg = json.loads((HERE/'PREREGISTRATION.json').read_text())
    manifest = json.loads((SOURCE/'NEURAL_INTERFACE_MANIFEST.json').read_text())
    files = [HERE/'module.py', HERE/'PREREGISTRATION.json', PROJECT/'engine/neural_cond.py',
             Path(__file__), SOURCE/'NEURAL_INTERFACE_MANIFEST.json']
    names = ['LOCAL_MODULE_CANDIDATE.npz', 'GROUP_CACHE_INDICES.npz',
             'LOCAL_MOTOR_READOUT_PORTS.csv', 'SENSORY_CHANNEL_REVIEW.json']
    for name in names:
        path = SOURCE/name
        files.append(path)
        if name in manifest['artifacts']:
            check('manifest_hash_'+name, sha(path) == manifest['artifacts'][name]['sha256'])
    cache_path = Path(manifest['sources']['cache']['path'])
    files.append(cache_path)
    check('original_cache_manifest_hash', sha(cache_path) == manifest['sources']['cache']['sha256'])
    REPORT['source_hashes'] = {str(path):sha(path) for path in files}
    candidate = load(SOURCE/'LOCAL_MODULE_CANDIDATE.npz')
    cache = load(cache_path)
    groups = load(SOURCE/'GROUP_CACHE_INDICES.npz')
    index = candidate['cache_indices']
    root = candidate['root_ids']
    rows = candidate['source_edge_rows']
    local = np.full(len(cache['root_ids']), -1, dtype=np.int64)
    local[index] = np.arange(N)
    source_local_mask = (local[cache['pre']] >= 0) & (local[cache['post']] >= 0)
    check('complete_induced_original_edges', np.array_equal(rows, np.flatnonzero(source_local_mask)))
    check('all_20000_unique_identity', len(index)==N and len(np.unique(index))==N and len(np.unique(root))==N)
    check('all_roots_match_original_cache', np.array_equal(root, cache['root_ids'][index]))
    check('all_pre_local_match_original_cache', np.array_equal(candidate['pre_local'],local[cache['pre'][rows]]))
    check('all_post_local_match_original_cache', np.array_equal(candidate['post_local'],local[cache['post'][rows]]))
    check('all_synapse_counts_match_original_cache', np.array_equal(candidate['synapse_counts'],cache['syn'][rows]))
    nt = cache['ann_nt_verified'][cache['pre'][rows]]
    check('all_source_nt_labels_match_original_cache', np.array_equal(candidate['source_verified_nt'],nt))
    strict = np.isin(nt,['acetylcholine','gaba','glutamate'])
    check('strict_mask_exact', np.array_equal(strict,candidate['pure_verified_label_mask']))
    check('strict_rows_exact', np.array_equal(rows[strict],candidate['strict_source_edge_rows']))
    check('declared_edge_counts', len(rows)==705068 and int(strict.sum())==435126)
    pre, post = local[cache['pre'][rows[strict]]], local[cache['post'][rows[strict]]]
    raw = cache['syn'][rows[strict]].astype(float)
    exc = nt[strict]=='acetylcholine'
    check('original_counts_positive_finite', np.isfinite(raw).all() and (raw>0).all())
    check('strict_pairs_unique', len(np.unique(post*N+pre))==len(pre))
    review = json.loads((SOURCE/'SENSORY_CHANNEL_REVIEW.json').read_text())
    excluded = [int(x['cache_index_0based']) for x in review['ambiguous_records']]
    sensory = np.flatnonzero(np.isin(index,groups['leg_sensory_nerve_union']) & ~np.isin(index,excluded))
    descending = np.flatnonzero(np.isin(index,groups['descending_boundary']))
    all_mn = np.flatnonzero(np.isin(index,groups['leg_motor_class']))
    with (SOURCE/'LOCAL_MOTOR_READOUT_PORTS.csv').open(newline='') as f:
        motor_rows = [r for r in csv.DictReader(f) if r['readout_eligible_single_leg']=='True']
    motor = np.array([int(r['local_index_0based']) for r in motor_rows])
    check('all391MN_excluded_from_both_inputs', len(all_mn)==391 and not np.intersect1d(all_mn,np.union1d(sensory,descending)).size)
    check('sensory_descending_disjoint', not np.intersect1d(sensory,descending).size)
    REPORT['identity'] = dict(neurons=N,candidate_edges=len(rows),strict_edges=len(pre),
        excitatory_edges=int(exc.sum()),inhibitory_edges=int((~exc).sum()),
        allowed_sensory=len(sensory),descending=len(descending),all_leg_motor=len(all_mn),readout_motor=len(motor),
        raw_strict_synapses=int(raw.sum()))
    spec = importlib.util.spec_from_file_location('_independently_verified_local_module',HERE/'module.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_step = module.ConductanceNetwork.step
    def counted_step(net,*args,**kwargs):
        REPORT['counters']['actual_network_step_calls'] += 1
        result = original_step(net,*args,**kwargs)
        REPORT['counters']['completed_network_steps'] += 1
        return result
    module.ConductanceNetwork.step = counted_step
    p = prereg['node_parameters']
    check('node_parameters_match_preregistration',asdict(module.ConductanceParams())==p)
    check('python_bytecode_disabled',sys.dont_write_bytecode)
    REPORT['runtime'] = dict(executable=sys.executable,python=sys.version,numpy=np.__version__)
    for scale in (.005,.02,.08):
        circuit = module.LocalCircuit(total_conductance_uS=scale,dt_ms=.5)
        check('module_root_identity',np.array_equal(circuit.root_ids,root))
        check('module_cache_identity',np.array_equal(circuit.cache_indices,index))
        check('module_sensory_identity',np.array_equal(circuit.allowed_sensory_local,sensory))
        check('module_descending_identity',np.array_equal(circuit.descending_local,descending))
        check('module_motor_order',np.array_equal(circuit.mn_local,motor) and np.array_equal(root[motor],[int(r['root_id_string']) for r in motor_rows]))
        check('module_motor_leg_order',np.array_equal(circuit.mn_leg,[r['nerve_matched_legs'] for r in motor_rows]))
        check('module_strict_source_rows',np.array_equal(circuit.strict_source_edge_rows,rows[strict]))
        ref = Reference(.5,p,pre,post,raw,exc,scale)
        for receptor, mask in [('We',exc),('Wi',~exc)]:
            w = getattr(circuit.net,receptor).tocoo()
            keys = w.row*N+w.col
            expected_keys = post[mask]*N+pre[mask]
            order, expected_order = np.argsort(keys),np.argsort(expected_keys)
            check(f'{receptor}_all_entries_exact_support_{scale}',np.array_equal(keys[order],expected_keys[expected_order]))
            close(f'{receptor}_all_weights_{scale}',w.data[order],ref.weight[mask][expected_order],2e-17)
        incoming = np.bincount(post,weights=ref.weight,minlength=N)
        close(f'incoming_normalization_{scale}',incoming[incoming>0],np.full(np.count_nonzero(incoming),scale),2e-15)
    s,d=np.zeros(N),np.zeros(N)
    s[sensory]=.05
    d[descending]=.05
    zero=np.zeros(N)
    rng=np.random.default_rng(20261107185910)
    for dt in (.5,.25):
        circuit=module.LocalCircuit(total_conductance_uS=.02,dt_ms=dt)
        ref=Reference(dt,p,pre,post,raw,exc,.02)
        # Deliberately assigned diagnostic state: no claim of an anatomical stimulus history.
        net=circuit.net
        net.v[:]=rng.uniform(-76.,-44.,N)
        net.ge[:]=rng.uniform(0.,.004,N)
        net.gi[:]=rng.uniform(0.,.008,N)
        net.adaptation[:]=rng.uniform(0.,.02,N)
        net.refractory_until_ms[:]=rng.choice([10.-dt,10.,10.+dt,12.],N)
        net.spike_count[:]=rng.integers(0,5,N)
        net.steps=int(10./dt)
        net.t_ms=10.
        net._ring_i=2
        circuit.filtered_rates_hz[:]=rng.uniform(0,90,N)
        for field in ('v','ge','gi','adaptation','refractory_until_ms','spike_count'):
            setattr(ref,field,getattr(net,field).copy())
        ref.rates=circuit.filtered_rates_hz.copy()
        ref.t=10.
        for offset in range(len(net._ring)):
            seeded=rng.random(N)<.03
            net._ring[(net._ring_i+offset)%len(net._ring)]=seeded
            ref.pending.append((10.+offset*dt,seeded.copy()))
        selected=np.unique(np.concatenate([np.arange(16),rng.choice(N,48,replace=False)]))
        prior_count=net.spike_count.copy()
        ref.step(s+d,selected)
        circuit.advance(s,d,duration_ms=dt)
        actual=net.spike_count-prior_count
        ref.compare(circuit,actual)
        check(f'randomized_spikes_exercised_{dt}',actual.sum()>0)
        check(f'randomized_nonspikes_exercised_{dt}',actual.sum()<N)
        REPORT['runs'].append(dict(kind='controlled_randomized_state',dt_ms=dt,actual_steps=1,manual_iterations=1,
            scalar_rows=len(selected),spikes=int(actual.sum()),seed=20261107185910,
            diagnostic_state='arbitrary voltage/conductance/adaptation/refractory/rates/counts and queued true-ID spikes; original edges unchanged'))
        circuit.reset()
        ref=Reference(dt,p,pre,post,raw,exc,.02)
        history=[]
        steps=int(30/dt)
        for k in range(steps):
            # Single rectangular engineering pulse 0 <= t < 20ms; then 10ms washout.
            cs,cd=(s,d) if k*dt<20. else (zero,zero)
            prior_count=circuit.net.spike_count.copy()
            ref.step(cs+cd)
            output=circuit.advance(cs,cd,duration_ms=dt)
            actual=circuit.net.spike_count-prior_count
            ref.compare(circuit,actual)
            close('exported_motor_rate',output,ref.rates[motor])
            history.append(dict(time_ms=ref.t,spikes=int(actual.sum()),motor_spikes=int(actual[motor].sum())))
        direct_rate=np.zeros(N)
        for time, spikes in ref.history:
            direct_rate += 50.*np.exp(-(ref.t-time)/20.)*spikes
        close('closed_form_20ms_rate',circuit.filtered_rates_hz,direct_rate)
        check(f'legal_run_spikes_exercised_{dt}',int(ref.spike_count.sum())>0)
        check(f'legal_run_delayed_edges_exercised_{dt}',ref.arrival_weight_uS>0)
        check(f'first_absolute_delay_exact_2ms_{dt}',ref.first_arrival==ref.first_emission+2.)
        REPORT['runs'].append(dict(kind='legal_real_identity_stimulus',dt_ms=dt,total_conductance_uS=.02,
            duration_ms=30.,pulse_interval_ms=[0.,20.],amplitude_nA=.05,interpretation='declared broad engineering sensory and descending drive, not brain output or measured physiology',
            actual_steps=steps,manual_iterations=steps,total_spikes=int(ref.spike_count.sum()),
            motor_spikes=int(ref.spike_count[motor].sum()),recruited_motor=int(np.count_nonzero(ref.spike_count[motor])),
            first_emission_ms=ref.first_emission,first_arrival_ms=ref.first_arrival,
            arrival_batches=ref.arrival_batches,arrival_spikes=ref.arrival_spikes,arrival_weight_uS=ref.arrival_weight_uS,
            final_motor_rate_max_hz=float(ref.rates[motor].max()),trace=history))
        # Guard rejections must leave every neural state and absolute queue unchanged.
        guards=[]
        def rejected(label,ss,dd,duration=dt):
            before=circuit.net.steps
            try: circuit.advance(ss,dd,duration_ms=duration)
            except ValueError: guards.append(label)
            else: raise AssertionError('guard accepted '+label)
            check('guard_rejections_have_zero_network_steps',before==circuit.net.steps)
        bad=zero.copy();bad[motor[0]]=.05
        rejected('motor_in_sensory',bad,zero)
        rejected('motor_in_descending',zero,bad)
        rejected('sensory_descending_role_swap',d,s)
        for label,values in [('nan',np.nan),('infinity',np.inf)]:
            bad=zero.copy();bad[sensory[0]]=values
            rejected(label,bad,zero)
        rejected('wrong_shape',np.zeros(N-1),zero)
        for duration in (0.,-dt,1.5*dt,np.nan,np.inf): rejected('invalid_duration_'+str(duration),zero,zero,duration)
        for cache_id in excluded:
            at=np.flatnonzero(index==cache_id)
            if len(at):
                bad=zero.copy();bad[at]=.05
                rejected('excluded_ambiguous_'+str(cache_id),bad,zero)
        REPORT.setdefault('guard_tests',[]).append(dict(dt_ms=dt,rejected=guards))
        circuit.reset()
        ref=Reference(dt,p,pre,post,raw,exc,.02)
        quiet_steps=int(5/dt)
        for _ in range(quiet_steps):
            ref.step(zero)
            before=circuit.net.spike_count.copy()
            circuit.advance(zero,zero,duration_ms=dt)
            ref.compare(circuit,circuit.net.spike_count-before)
        check(f'quiet_exact_zero_spikes_rates_{dt}',not circuit.net.spike_count.any() and not circuit.filtered_rates_hz.any())
        REPORT['runs'].append(dict(kind='quiet_from_reset',dt_ms=dt,duration_ms=5.,actual_steps=quiet_steps,manual_iterations=quiet_steps,total_spikes=0))
    check('precise_network_step_accounting',REPORT['counters']['actual_network_step_calls']==212 and REPORT['counters']['completed_network_steps']==212)
    check('precise_reference_accounting',REPORT['counters']['manual_full_vector_iterations']==212 and REPORT['counters']['manual_scalar_row_iterations']==128)
    check('all_recorded_source_hashes_unchanged',all(sha(path)==digest for path,digest in REPORT['source_hashes'].items()))
    REPORT['limitations']=['Engineering equality for declared illustrative dynamics, not biological validation.',
        '30ms bounded runs are shorter and differently timed than the 200ms calibration; no calibration acceptance claim.',
        'Randomized initial states and queued spike histories are explicit numerical diagnostics, not measured biology.',
        'No source synapses were added; full support/weights checked at all three scales; dynamic reference tested at .02uS.',
        'No physical-feedback propagation, body simulation, training, alternate voltage method or disabled-cell API tested.']
    REPORT['status']='PASS'

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        REPORT['status']='FAIL'
        REPORT['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        OUT.write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=REPORT['status'],counters=REPORT['counters'],failures=REPORT['failures'],report=str(OUT)),indent=2))
    sys.exit(0 if REPORT['status']=='PASS' else 1)
