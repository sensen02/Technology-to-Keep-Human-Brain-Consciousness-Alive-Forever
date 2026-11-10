"""ROUND9 independent engineering verifier; only output is INDEPENDENT_VERIFICATION.json.
No physical execution, no training, no new edges. Use existing venv_body Python -B.
Reference derives weights from original cache and uses full-vector bincount edge scatter,
voltage-increment backward Euler, and an absolute timestamp queue (no sparse matmul).
"""
from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
import math
import sys
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD = PROJECT/'experiments/banc-local-learning/20261107T185910Z/dynamics'
SOURCE = PROJECT/'experiments/flybody-retarget/20261107T182940Z/neural-interface'
OUT = HERE/'INDEPENDENT_VERIFICATION.json'
N = 20000
TOL = 2e-11
R = dict(schema='conductance-adaptation-independent-verification/1', status='RUNNING',
    scope='S3/S4 engineering hypothesis verification, NOT physiology or behavior',
    counters=dict(actual_network_step_calls=0, completed_network_steps=0, rejected_network_step_calls=0,
                  manual_full_vector_iterations=0, manual_scalar_row_iterations=0, physical_steps=0, training_updates=0),
    checks={}, errors_max={}, failures=[], runs=[], guard_tests=[], source_hashes={},
    tolerances=dict(state_absolute=TOL, state_relative=0, weights_absolute=2e-17, voltage_bound_absolute=1e-10,
                    spikes_counts_queue_identity='exact'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def check(name, ok):
    R['checks'][name] = bool(ok)
    if not ok: raise AssertionError(name)


def close(name, actual, expected, tol=TOL):
    a, b = np.asarray(actual), np.asarray(expected)
    check(name+'_shape', a.shape == b.shape)
    e = float(np.max(np.abs(a-b), initial=0))
    R['errors_max'][name] = max(e, R['errors_max'].get(name, 0.))
    if not np.isfinite(e) or e > tol: raise AssertionError(f'{name}: error {e} > {tol}')


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


class Reference:
    def __init__(self, dt, p, pre, post, raw, exc):
        self.dt, self.p = dt, p
        self.pre, self.post, self.exc = pre, post, exc
        self.weight = raw * (.02/np.bincount(post, weights=raw, minlength=N)[post])
        self.v = np.full(N, -65.)
        self.ge, self.gi, self.g_adapt = (np.zeros(N) for _ in range(3))
        self.refractory_until_ms = np.zeros(N)
        self.spike_count = np.zeros(N, np.int64)
        self.rates = np.zeros(N)
        self.t = 0.
        self.pending = []
        self.first_emission = self.first_arrival = None
        self.arrival_spikes = 0
        self.arrival_weight = 0.
        self.history = []
        self.min_bound_residual = None
        self.max_bound_algebra_error = 0.

    def step(self, current, scalar_rows=(), legal=False):
        R['counters']['manual_full_vector_iterations'] += 1
        p, dt = self.p, self.dt
        arriving = np.zeros(N, dtype=bool)
        future = []
        for timestamp, spikes in self.pending:
            if timestamp <= self.t + 1e-12:
                check('absolute_arrivals_on_time', abs(timestamp-self.t) < 1e-12)
                arriving |= spikes
                self.arrival_spikes += int(spikes.sum())
                if spikes.any() and self.first_arrival is None: self.first_arrival = self.t
            else: future.append((timestamp, spikes))
        self.pending = future
        edge = self.weight * arriving[self.pre]
        add_e = np.bincount(self.post[self.exc], weights=edge[self.exc], minlength=N)
        add_i = np.bincount(self.post[~self.exc], weights=edge[~self.exc], minlength=N)
        self.arrival_weight += float(add_e.sum()+add_i.sum())
        ge, gi, ga = self.ge+add_e, self.gi+add_i, self.g_adapt
        # Independent voltage-increment form; adaptation is a conductance at -80mV.
        force = (p['leak_uS']*(p['rest_mV']-self.v) + ge*(p['exc_reversal_mV']-self.v)
                 + gi*(p['inh_reversal_mV']-self.v) + ga*(-80.-self.v) + current)
        denom = p['capacitance_nF'] + dt*(p['leak_uS']+ge+gi+ga)
        proposed = self.v + dt*force/denom
        active = self.t+1e-10 >= self.refractory_until_ms
        spikes = active & (proposed >= p['threshold_mV'])
        self.diagnostic_above_threshold = proposed >= p['threshold_mV']
        self.diagnostic_active = active.copy()
        voltage = np.where(active & ~spikes, proposed, p['reset_mV'])
        if legal:
            residual = (p['capacitance_nF']*(self.v+80.) + dt*(p['leak_uS']*(p['rest_mV']+80.)
                        + ge*(p['exc_reversal_mV']+80.) + gi*(p['inh_reversal_mV']+80.) + current))/denom
            err = float(np.max(np.abs((proposed+80.)-residual)))
            self.max_bound_algebra_error = max(self.max_bound_algebra_error, err)
            self.min_bound_residual = float(residual.min()) if self.min_bound_residual is None else min(self.min_bound_residual,float(residual.min()))
            check('legal_algebra_nonnegative_residual', residual.min() >= -1e-10)
            check('legal_reference_proposed_above_minus80', proposed.min() >= -80.-1e-10)
            close('lower_bound_algebra', proposed+80., residual, 1e-10)
        for row in scalar_rows:
            R['counters']['manual_scalar_row_iterations'] += 1
            e, i = float(self.ge[row]), float(self.gi[row])
            for k in np.flatnonzero(self.post == row):
                contribution = float(self.weight[k])*int(arriving[self.pre[k]])
                if self.exc[k]: e += contribution
                else: i += contribution
            candidate = ((p['capacitance_nF']/dt)*float(self.v[row]) + p['leak_uS']*p['rest_mV']
                         + e*p['exc_reversal_mV'] + i*p['inh_reversal_mV'] - 80.*float(ga[row])
                         + float(current[row]))/(p['capacitance_nF']/dt+p['leak_uS']+e+i+float(ga[row]))
            expected = candidate if active[row] and candidate < p['threshold_mV'] else p['reset_mV']
            close('scalar_voltage', expected, voltage[row])
            close('scalar_edge_scatter', [e,i], [ge[row],gi[row]])
        self.t += dt
        self.v = voltage
        self.ge, self.gi = ge*math.exp(-dt/p['exc_tau_ms']), gi*math.exp(-dt/p['inh_tau_ms'])
        self.g_adapt = ga*math.exp(-dt/100.) + spikes*(.002/15.)
        self.refractory_until_ms[spikes] = self.t+p['refractory_ms']
        self.spike_count += spikes
        self.rates = self.rates*math.exp(-dt/20.) + spikes*50.
        if spikes.any():
            self.pending.append((self.t+2., spikes.copy()))
            if self.first_emission is None: self.first_emission = self.t
        self.history.append((self.t,spikes.copy()))
        return spikes

    def compare(self, circuit, expected):
        net = circuit.net
        for field in ('v','ge','gi','g_adapt','refractory_until_ms'):
            close(field, getattr(net,field),getattr(self,field))
        close('filtered_rates', circuit.filtered_rates_hz,self.rates)
        check('exact_spike_counts', np.array_equal(net.spike_count,self.spike_count))
        check('exact_spike_vectors', np.array_equal(net.last_spikes,expected))
        check('exact_absolute_clock', net.t_ms == self.t)
        for offset in range(len(net._ring)):
            expected_queue = np.zeros(N,bool)
            for when, spikes in self.pending:
                if when == self.t+offset*self.dt: expected_queue |= spikes
            check('ring_equals_absolute_queue', np.array_equal(net._ring[(net._ring_i+offset)%len(net._ring)], expected_queue))


def snapshot(circuit):
    net = circuit.net
    return [getattr(net,f).copy() for f in ('v','ge','gi','g_adapt','refractory_until_ms','spike_count','last_spikes','_ring')]+[circuit.filtered_rates_hz.copy(),np.array([net.t_ms,net.steps,net._ring_i])]


def main():
    prereg = json.loads((HERE/'PREREGISTRATION.json').read_text())
    manifest = json.loads((SOURCE/'NEURAL_INTERFACE_MANIFEST.json').read_text())
    cachepath = Path(manifest['sources']['cache']['path'])
    samplepath = OLD/'RECORDED_SAMPLE_INPUT.npz'
    files = [Path(__file__),HERE/'module.py',HERE/'PREREGISTRATION.json', OLD/'module.py',OLD/'independent_verify.py',
             OLD/'PREREGISTRATION.json',PROJECT/'engine/neural_cond.py',SOURCE/'NEURAL_INTERFACE_MANIFEST.json',cachepath,samplepath]
    for name in ('LOCAL_MODULE_CANDIDATE.npz','GROUP_CACHE_INDICES.npz','LOCAL_MOTOR_READOUT_PORTS.csv','SENSORY_CHANNEL_REVIEW.json'):
        path = SOURCE/name
        files.append(path)
        if name in manifest['artifacts']: check('manifest_'+name,sha(path)==manifest['artifacts'][name]['sha256'])
    R['source_hashes'] = {str(path):sha(path) for path in files}
    check('original_cache_manifest_hash',sha(cachepath)==manifest['sources']['cache']['sha256'])
    check('old_module_frozen_hash',sha(OLD/'module.py')==prereg['old_module_sha256'])
    candidate,cache,groups,sample = (load(path) for path in (SOURCE/'LOCAL_MODULE_CANDIDATE.npz',cachepath,SOURCE/'GROUP_CACHE_INDICES.npz',samplepath))
    index, roots, rows = (candidate[k] for k in ('cache_indices','root_ids','source_edge_rows'))
    local = np.full(len(cache['root_ids']),-1,np.int64); local[index] = np.arange(N)
    check('unique_20000_identity',len(index)==N and len(np.unique(index))==N and len(np.unique(roots))==N)
    check('original_roots',np.array_equal(roots,cache['root_ids'][index]))
    induced = (local[cache['pre']]>=0)&(local[cache['post']]>=0)
    check('complete_original_induced_edges',np.array_equal(rows,np.flatnonzero(induced)))
    for key, val in [('pre_local',local[cache['pre'][rows]]),('post_local',local[cache['post'][rows]]),('synapse_counts',cache['syn'][rows])]:
        check('original_'+key,np.array_equal(candidate[key],val))
    nt = cache['ann_nt_verified'][cache['pre'][rows]]
    check('original_verified_transmitter',np.array_equal(candidate['source_verified_nt'],nt))
    strict = np.isin(nt,['acetylcholine','gaba','glutamate'])
    check('exact_strict_mask',np.array_equal(strict,candidate['pure_verified_label_mask']))
    check('exact_strict_rows',np.array_equal(rows[strict],candidate['strict_source_edge_rows']))
    check('original_edge_counts',len(rows)==705068 and strict.sum()==435126)
    pre,post = local[cache['pre'][rows[strict]]],local[cache['post'][rows[strict]]]
    raw,exc = cache['syn'][rows[strict]].astype(float),nt[strict]=='acetylcholine'
    check('unique_strict_support',len(np.unique(post*N+pre))==len(pre))
    check('finite_positive_raw_weights',np.isfinite(raw).all() and (raw>0).all())
    excluded = [int(x['cache_index_0based']) for x in json.loads((SOURCE/'SENSORY_CHANNEL_REVIEW.json').read_text())['ambiguous_records']]
    source_sensory = np.flatnonzero(np.isin(index,groups['leg_sensory_nerve_union']))
    sensory = source_sensory[~np.isin(index[source_sensory],excluded)]
    descending = np.flatnonzero(np.isin(index,groups['descending_boundary']))
    ascending = np.flatnonzero(np.isin(index,groups['ascending_boundary']))
    all_mn = np.flatnonzero(np.isin(index,groups['leg_motor_class']))
    with (SOURCE/'LOCAL_MOTOR_READOUT_PORTS.csv').open(newline='') as f:
        motor_rows = [x for x in csv.DictReader(f) if x['readout_eligible_single_leg']=='True']
    motor = np.array([int(x['local_index_0based']) for x in motor_rows])
    check('323_motor_identity_order',len(motor)==323 and np.array_equal(roots[motor],[int(x['root_id_string']) for x in motor_rows]))
    check('391_motor_excluded_from_inputs',len(all_mn)==391 and not np.intersect1d(all_mn,np.union1d(sensory,descending)).size)
    check('input_roles_disjoint',not np.intersect1d(sensory,descending).size)
    spec = importlib.util.spec_from_file_location('_round9_independent_target',HERE/'module.py')
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original_step = module.AdaptationConductanceNetwork.step
    def counted(net,*args,**kwargs):
        R['counters']['actual_network_step_calls'] += 1
        try: result = original_step(net,*args,**kwargs)
        except Exception:
            R['counters']['rejected_network_step_calls'] += 1
            raise
        R['counters']['completed_network_steps'] += 1
        return result
    module.AdaptationConductanceNetwork.step = counted
    p = asdict(module.ConductanceParams())
    check('unchanged_original_node_parameters',p==json.loads((OLD/'PREREGISTRATION.json').read_text())['node_parameters'])
    check('fixed_adaptation_contract',asdict(module.AdaptationConductanceParams())==dict(reversal_mV=-80.,tau_ms=100.,spike_increment_uS=.002/15.))
    check('preregistered_adaptation_contract',prereg['adaptation_contract']['reversal_mV']==-80. and prereg['adaptation_contract']['tau_ms']==100. and prereg['adaptation_contract']['increment_uS']==.002/15.)
    check('bytecode_disabled',sys.dont_write_bytecode)
    R['runtime'] = dict(executable=sys.executable,python=sys.version,numpy=np.__version__)
    R['identity'] = dict(neurons=N,candidate_edges=len(rows),strict_edges=len(pre),excitatory_edges=int(exc.sum()),inhibitory_edges=int((~exc).sum()),sensory=len(sensory),descending=len(descending),ascending=len(ascending),all_motor=len(all_mn),exported_motor=len(motor))
    s,zero = sample['sensory_current_nA'],np.zeros(N)
    check('saved_zero_descending',np.array_equal(sample['descending_current_nA'],zero))
    check('saved_sensory_finite_nonnegative',s.shape==(N,) and np.isfinite(s).all() and (s>=0).all())
    check('saved_556_qualified_ports',len(sample['local_indices'])==556 and np.isin(sample['local_indices'],sensory).all() and np.isin(np.flatnonzero(s),sample['local_indices']).all())
    R['saved_input'] = dict(path=str(samplepath),sha256=sha(samplepath),source_time_s=float(sample['source_time_s']),nonzero_currents=int(np.count_nonzero(s)),maximum_nA=float(s.max()),descending_current='exact zero')
    R['lower_bound_proof'] = dict(domain='Reset-origin finite V=-65; original nonnegative conductance increments; gE,gI,gA>=0; finite I>=0; fixed positive C,dt,gL; unmutated threshold/reset and parameters.',
        algebra='D=C/dt+gL+gE+gI+gA>0; Vnext-L=[C/dt*(V-L)+gL*(EL-L)+gE*(EE-L)+gI*(EI-L)+gA*(EA-L)+I]/D >=0 for L=-80.',
        induction='All reversal potentials >=-80; gA term is zero at L. Exact positive decays, nonnegative spike increment and source weights preserve g>=0. Reset/refractory=-65 preserves bound. Unspiked active V<-50 and spiked reset=-65 imply poststep V<=-50.',
        caveat='Real-arithmetic proof, numerical residuals checked separately. Arbitrary public state mutation and nonfinite arithmetic are outside domain. Bound is not physiological calibration.',tolerance_mV=1e-10)
    rng = np.random.default_rng(20261107192931)
    for dt in (.5,.25):
        c = module.LocalCircuit(dt_ms=dt)
        ref = Reference(dt,p,pre,post,raw,exc)
        for field, expected in [('root_ids',roots),('cache_indices',index),('allowed_sensory_local',sensory),('source_sensory_local',source_sensory),('excluded_ambiguous_sensory_local',source_sensory[np.isin(index[source_sensory],excluded)]),('descending_local',descending),('ascending_local',ascending),('mn_local',motor),('mn_leg',np.array([x['nerve_matched_legs'] for x in motor_rows])),('strict_source_edge_rows',rows[strict])]:
            check('module_identity_'+field,np.array_equal(getattr(c,field),expected))
        for name,ids in [('_sensory_mask',sensory),('_descending_mask',descending)]:
            expected=np.zeros(N,bool);expected[ids]=True
            check(name+'_exact',np.array_equal(getattr(c,name),expected))
        for receptor,mask in [('We',exc),('Wi',~exc)]:
            w=getattr(c.net,receptor).tocoo();keys=w.row*N+w.col;target=post[mask]*N+pre[mask]
            order,want=np.argsort(keys),np.argsort(target)
            check(receptor+'_no_new_synapses',np.array_equal(keys[order],target[want]))
            close(receptor+'_all_weights',w.data[order],ref.weight[mask][want],2e-17)
        incoming=np.bincount(post,weights=ref.weight,minlength=N)
        close('incoming_normalization',incoming[incoming>0],np.full(np.count_nonzero(incoming),.02),2e-15)
        check('initial_voltage_rest',np.all(c.net.v==-65.))
        check('initial_all_other_state_zero',all(not a.any() for a in snapshot(c)[1:]))
        check('adaptation_current_state_absent',not hasattr(c.net,'adaptation'))
        check('initial_zero_threshold_offsets',not c.net.threshold_offset_mV.any())
        # Separate arbitrary state diagnostic, not a legal reset-origin biological trace.
        net=c.net
        net.v[:]=rng.uniform(-80.,-45.,N)
        net.ge[:]=rng.uniform(0.,.004,N);net.gi[:]=rng.uniform(0.,.008,N)
        net.g_adapt[:]=rng.uniform(0.,.004,N)
        net.refractory_until_ms[:]=rng.choice([10.-dt,10.,10.+dt,12.],N)
        net.spike_count[:]=rng.integers(0,5,N);net.steps=int(10./dt);net.t_ms=10.;net._ring_i=2
        c.filtered_rates_hz[:]=rng.uniform(0.,90.,N)
        for field in ('v','ge','gi','g_adapt','refractory_until_ms','spike_count'): setattr(ref,field,getattr(net,field).copy())
        ref.rates=c.filtered_rates_hz.copy();ref.t=10.
        for offset in range(len(net._ring)):
            seeded=rng.random(N)<.03
            net._ring[(net._ring_i+offset)%len(net._ring)]=seeded
            ref.pending.append((10.+offset*dt,seeded.copy()))
        selected=rng.choice(N,128,replace=False)
        spikes=ref.step(s,selected);c.advance(s,zero,duration_ms=dt);ref.compare(c,spikes)
        check('diagnostic_spikes_and_nonspikes',spikes.any() and not spikes.all())
        suppressed = ref.diagnostic_above_threshold & ~ref.diagnostic_active
        check('diagnostic_refractory_suppressed',suppressed.any() and not net.last_spikes[suppressed].any())
        check('diagnostic_active_subthreshold_exercised',np.any(ref.diagnostic_active & ~ref.diagnostic_above_threshold))
        R['runs'].append(dict(kind='arbitrary_randomized_state_diagnostic',dt_ms=dt,actual_steps=1,manual_iterations=1,scalar_rows=128,spikes=int(spikes.sum()),initial_voltage_interval_mV=[-80,-45],seed=20261107192931,biological_interpretation=False))
        c.reset();ref=Reference(dt,p,pre,post,raw,exc)
        vmin,vmax,gamax=0.,-100.,0.
        history=[]
        for k in range(int(100./dt)):
            spikes=ref.step(s,legal=True);output=c.advance(s,zero,duration_ms=dt);ref.compare(c,spikes)
            close('exported_motor_rates',output,ref.rates[motor])
            net=c.net
            check('legal_all_state_finite',all(np.isfinite(x).all() for x in snapshot(c)))
            check('legal_nonnegative_conductances',all((x>=0).all() for x in (net.ge,net.gi,net.g_adapt)))
            vmin=min(vmin,float(net.v.min()));vmax=max(vmax,float(net.v.max()));gamax=max(gamax,float(net.g_adapt.max()))
            check('legal_actual_lower_bound',net.v.min()>=-80.-1e-10)
            check('legal_actual_upper_bound',net.v.max()<=-50.+1e-10)
            check('original_engineering_voltage_band',net.v.min()>=-90. and net.v.max()<=-49.)
            history.append(dict(time_ms=net.t_ms,spikes=int(spikes.sum()),motor_spikes=int(spikes[motor].sum()),vmin_mV=float(net.v.min()),vmax_mV=float(net.v.max()),g_adapt_max_uS=float(net.g_adapt.max())))
        direct=np.zeros(N)
        for t,spikes in ref.history: direct+=50.*np.exp(-(ref.t-t)/20.)*spikes
        close('closed_form_rate_filter',c.filtered_rates_hz,direct)
        check('legal_propagation_exercised',ref.arrival_weight>0 and ref.spike_count[motor].sum()>0)
        check('legal_adaptation_exercised',gamax>0)
        check('first_emission_to_arrival_2ms',ref.first_arrival==ref.first_emission+2.)
        R['runs'].append(dict(kind='saved_sample_reset_origin',dt_ms=dt,duration_ms=100.,actual_steps=int(100./dt),manual_iterations=int(100./dt),total_spikes=int(ref.spike_count.sum()),motor_spikes=int(ref.spike_count[motor].sum()),recruited_motor=int(np.count_nonzero(ref.spike_count[motor])),motor_recruited_by_leg={leg:int(np.count_nonzero(ref.spike_count[motor[c.mn_leg==leg]])) for leg in np.unique(c.mn_leg)},vmin_mV=vmin,vmax_mV=vmax,g_adapt_max_uS=gamax,minimum_bound_residual_mV=ref.min_bound_residual,maximum_bound_algebra_error_mV=ref.max_bound_algebra_error,first_emission_ms=ref.first_emission,first_arrival_ms=ref.first_arrival,arriving_spikes=ref.arrival_spikes,arrival_weight_uS=ref.arrival_weight,trace=history))
        def reject(label,fn):
            before=snapshot(c)
            try: fn()
            except ValueError: pass
            else: raise AssertionError('guard accepted '+label)
            check('guard_state_unchanged',all(np.array_equal(a,b,equal_nan=True) for a,b in zip(before,snapshot(c))))
            R['guard_tests'].append(dict(dt_ms=dt,name=label,rejected=True))
        for role,ids in [('sensory',sensory),('descending',descending)]:
            for label,value in [('negative',-1e-6),('nan',np.nan),('infinite',np.inf)]:
                bad=zero.copy();bad[ids[0]]=value
                reject(role+'_'+label,lambda bad=bad,role=role:c.advance(bad if role=='sensory' else zero,bad if role=='descending' else zero,duration_ms=dt))
        bad=zero.copy();bad[motor[0]]=.01
        reject('motor_in_sensory',lambda:c.advance(bad,zero,duration_ms=dt))
        reject('motor_in_descending',lambda:c.advance(zero,bad,duration_ms=dt))
        bad_role=zero.copy();bad_role[descending[0]]=.01
        reject('descending_in_sensory',lambda:c.advance(bad_role,zero,duration_ms=dt))
        reject('sensory_in_descending',lambda:c.advance(zero,s,duration_ms=dt))
        reject('wrong_shape',lambda:c.advance(np.zeros(N-1),zero,duration_ms=dt))
        for duration in (0.,-dt,1.5*dt,np.nan,np.inf): reject('invalid_duration_'+str(duration),lambda duration=duration:c.advance(zero,zero,duration_ms=duration))
        for at in c.excluded_ambiguous_sensory_local:
            bad=zero.copy();bad[at]=.01
            reject('excluded_sensory_'+str(at),lambda bad=bad:c.advance(bad,zero,duration_ms=dt))
        for label,value in [('negative',-1e-6),('nan',np.nan),('infinite',np.inf)]:
            reject('net_current_'+label,lambda value=value:c.net.step(value))
        for field in ('ge','gi','g_adapt'):
            for label,value in [('negative',-1e-6),('nan',np.nan),('infinite',np.inf)]:
                arr=getattr(c.net,field);previous=arr[0];arr[0]=value
                reject('net_'+field+'_'+label,lambda:c.net.step(0.))
                arr[0]=previous
        c.reset();ref=Reference(dt,p,pre,post,raw,exc)
        for _ in range(int(5./dt)):
            spikes=ref.step(zero,legal=True);c.advance(zero,zero,duration_ms=dt);ref.compare(c,spikes)
        check('quiet_exact_zero_spikes_rates',not c.net.spike_count.any() and not c.filtered_rates_hz.any())
        R['runs'].append(dict(kind='quiet_reset_origin',dt_ms=dt,duration_ms=5.,actual_steps=int(5./dt),manual_iterations=int(5./dt),spikes=0))
    for args in ({'dt_ms':.1},{'dt_ms':np.nan},{'total_conductance_uS':.005},{'total_conductance_uS':.08}):
        try: module.LocalCircuit(**args)
        except ValueError: pass
        else: raise AssertionError('constructor accepted '+str(args))
    check('exact_completed_step_accounting',R['counters']['completed_network_steps']==632)
    check('exact_rejected_net_calls',R['counters']['rejected_network_step_calls']==24)
    check('exact_actual_step_calls',R['counters']['actual_network_step_calls']==656)
    check('exact_manual_iterations',R['counters']['manual_full_vector_iterations']==632 and R['counters']['manual_scalar_row_iterations']==256)
    R['limitations']=['100ms independent saved-input traces do not replace parent 200ms acceptance or full1817bin trace.', 'No physical steps or training; hypothesis not calibrated physiology.', 'Randomized initial state/queue is a separate numerical threshold diagnostic, not measured biology.', 'Public arbitrary voltage/parameter mutation is outside proof domain; guard tests cover exposed current roles, current finite/sign and finite/nonnegative conductances.', 'Historical current-adaptation voltage failure below -90mV is retained and not reclassified.']
    R['status']='PASS'


if __name__=='__main__':
    try: main()
    except Exception as exc:
        R['status']='FAIL';R['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        unchanged={path:sha(path)==digest for path,digest in R['source_hashes'].items()}
        R['hashes_unchanged']=unchanged
        if not all(unchanged.values()):
            R['status']='FAIL';R['failures'].append(dict(type='SourceMutation',paths=[p for p,ok in unchanged.items() if not ok]))
        R['checks']['all_source_oldmodule_engine_hashes_unchanged']=bool(unchanged) and all(unchanged.values())
        OUT.write_text(json.dumps(R,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=R['status'],counters=R['counters'],checks=len(R['checks']),guards=len(R['guard_tests']),failures=R['failures'],report=str(OUT)),indent=2))
    sys.exit(0 if R['status']=='PASS' else 1)
