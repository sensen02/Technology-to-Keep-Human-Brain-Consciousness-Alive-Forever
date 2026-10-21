"""Saved actual-body input through actual BANC across neural clocks.
No body integration, motor injection or invented edges. Diagnostic replay only.
手工研究原型，不是成品.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import time
import sys
from pathlib import Path
import numpy as np
import run_neural_muscle_closure as loop
from engine.embodied.source_snapshot import build_source_snapshot,verify_source_snapshot

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'outputs/model_branches/projected_pose_convergence/20261013_072227'
IDENTITY=HERE/'outputs/model_branches/canonical_paired_baseline/20261012_235335/neural_identity.json'
OUT=HERE/'outputs/model_branches/fixed_body_input_clock_replay'


def reconstruct_input(record,groups,n):
    current=np.zeros(n)
    occupied=set()
    for leg,channels in groups.items():
        for channel,indices in channels.items():
            summary=record['actual_network_input_by_channel'][leg][channel]
            ids=np.asarray(indices,dtype=int)
            if ids.ndim!=1 or len(ids)!=len(set(ids.tolist())) or np.any(ids<0) or np.any(ids>=n):
                raise ValueError('invalid sensory indices')
            if occupied.intersection(ids.tolist()):raise ValueError('overlapping groups cannot be reconstructed')
            occupied.update(ids.tolist())
            if summary['count']!=len(ids):raise ValueError('group count mismatch')
            if not np.isfinite([summary['min_nA'],summary['max_nA'],summary['sum_nA']]).all():raise ValueError('nonfinite current')
            if summary['min_nA']!=summary['max_nA']:raise ValueError('nonuniform group requires full saved current')
            value=summary['min_nA']
            if not np.isclose(summary['sum_nA'],value*len(ids),rtol=1e-12,atol=1e-12):raise ValueError('current sum mismatch')
            current[ids]=value
    if current.min()!=record['sensory_current_min_nA'] or current.max()!=record['sensory_current_max_nA']:
        raise ValueError('reconstructed global extrema mismatch')
    return current


def first_event_difference(a,b):
    for x,y in zip(a,b):
        if x!=y:return dict(base_ms=float(x),other_ms=float(y),earliest_ms=float(min(x,y)))
    if len(a)!=len(b):
        t=(a if len(a)>len(b) else b)[min(len(a),len(b))]
        return dict(base_ms=float(t) if len(a)>len(b) else None,
                    other_ms=float(t) if len(b)>len(a) else None,earliest_ms=float(t))
    return None


def main():
    start=time.monotonic()
    dest=OUT/time.strftime('%Y%m%d_%H%M%S');dest.mkdir(parents=True,exist_ok=False)
    saved=json.loads((SOURCE/'convergence_results.json').read_text())
    verify_source_snapshot(SOURCE/'source_snapshot.zip',saved['source_hashes'])
    identity=json.loads(IDENTITY.read_text())
    groups=identity['sensory_groups'];manifest=identity['motor_manifest']
    records=saved['episodes']['base_normal']['records']
    built=loop.build_neural_tier(loop.TierConfig(background_mean_nA=0,background_sd_nA=0))
    n=int(built['n']);roots=list(map(str,built['tier']['root_ids']))
    if roots!=list(map(str,identity['tier_root_ids'])):raise ValueError('tier differs from frozen identity')
    with np.load(SOURCE/'convergence_traces.npz') as old:
        if roots!=list(map(str,old['tier_root_ids'])):raise ValueError('tier differs from saved body experiment')
        original_counts=old['base_normal_spike_counts'].copy()
        original_controls=old['base_normal_controls'].copy()
    currents=np.array([reconstruct_input(r,groups,n) for r in records])
    motor=np.array(sorted({i for row in manifest['rows'] for i in row['local_indices']}),int)
    sensory=np.array(sorted({i for channels in groups.values() for indices in channels.values() for i in indices}),int)
    arrays=dict(input_current_nA=currents,tier_root_ids=np.asarray(roots),motor_indices=motor)
    experiments={};events_by_clock={};checks={}
    for label,dt_ms in [('base',.5),('half',.25),('quarter',.125),('eighth',.0625)]:
        net=loop.ConductanceNetwork(n,loop.ConductanceParams(),dt_ms=dt_ms)
        net.set_connectivity(built['edges']['We']*.2,built['edges']['Wi']*.2)
        decoder=loop.MotorOutput(manifest,loop.MotorOutputConfig())
        steps=round(5/dt_ms)
        spike_counts=[];controls=[];motor_voltage=[];motor_counts=[];times=[];event_rows=[]
        root_events=[[] for _ in range(n)]
        pending=np.zeros(len(manifest['rows']))
        for k,current in enumerate(currents):
            spike_window=np.zeros((steps,n),bool)
            controls.append(pending.copy())
            for j in range(steps):
                sp=net.step(current_nA=current);spike_window[j]=sp
                for i in np.flatnonzero(sp):
                    t=float(net.t_ms);root_events[int(i)].append(t);event_rows.append((t,int(i)))
                motor_voltage.append(net.v[motor].copy());motor_counts.append(int(sp[motor].sum()));times.append(net.t_ms)
            spike_counts.append(spike_window.sum(axis=0))
            pending,trace=decoder.consume(spike_window,spike_dt_s=dt_ms/1000,start_s=k*.005,end_s=(k+1)*.005)
        spike_counts=np.asarray(spike_counts);controls=np.asarray(controls)
        arrays[label+'_spike_counts']=spike_counts;arrays[label+'_controls']=controls
        arrays[label+'_events_time_ms_index']=np.asarray(event_rows,float).reshape(-1,2)
        arrays[label+'_motor_voltage_mV']=np.asarray(motor_voltage)
        arrays[label+'_times_ms']=np.asarray(times)
        events_by_clock[label]=root_events
        experiments[label]=dict(dt_ms=dt_ms,total_spikes=int(spike_counts.sum()),
            motor_spikes=int(spike_counts[:,motor].sum()),sensory_spikes=int(spike_counts[:,sensory].sum()),
            motor_count_per_5ms=spike_counts[:,motor].sum(axis=1).tolist(),
            first_motor_spike_ms=min((ts[0] for i,ts in enumerate(root_events) if i in set(motor.tolist()) and ts),default=None))
        if label=='base':
            checks['base_all_neuron_counts_exact_saved']=bool(np.array_equal(spike_counts,original_counts))
            checks['base_applied_controls_exact_saved']=bool(np.array_equal(controls,original_controls))
        print(label,experiments[label],flush=True)
    comparisons={}
    motor_set=set(motor.tolist());sensory_set=set(sensory.tolist())
    for label in ('half','quarter','eighth'):
        differences=[]
        for i,(a,b) in enumerate(zip(events_by_clock['base'],events_by_clock[label])):
            d=first_event_difference(a,b)
            if d: differences.append(dict(root_id=roots[i],tier_index=i,group='motor' if i in motor_set else ('sensory' if i in sensory_set else 'other'),base_count=len(a),other_count=len(b),**d))
        differences.sort(key=lambda d:(d['earliest_ms'],d['tier_index']))
        changed_bins=np.flatnonzero(np.any(arrays['base_spike_counts'][:,motor]!=arrays[label+'_spike_counts'][:,motor],axis=1))
        comparisons[label]=dict(changed_root_count=len(differences),first_any=differences[0] if differences else None,
            first_motor=next((d for d in differences if d['group']=='motor'),None),
            first_common_5ms_motor_count_difference_interval_start_ms=float(changed_bins[0]*5) if len(changed_bins) else None,
            controls_max_abs_difference=float(np.max(abs(arrays['base_controls']-arrays[label+'_controls']))),per_root_event_differences=differences)
    checks['original_xml_hash']=loop.cp.sha256_file(loop.cp.CANONICAL_XML)==loop.cp.CANONICAL_XML_SHA256
    assert all(checks.values()),checks
    np.savez_compressed(dest/'clock_replay_traces.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for label,r in experiments.items():
        axes[0].plot(np.arange(1,len(records)+1)*5,r['motor_count_per_5ms'],label=label)
        axes[1].plot(np.arange(1,len(records)+1)*5,arrays[label+'_controls'].max(axis=1),label=label)
    axes[0].set_ylabel('Motor spikes / common 5ms');axes[0].legend()
    axes[1].set_ylabel('Max decoded applied control');axes[1].set_xlabel('Time (ms)')
    fig.suptitle('Manual diagnostic: identical actual-body input, actual BANC; NO body control')
    fig.tight_layout();fig.savefig(dest/'clock_replay.png',dpi=150);plt.close(fig)
    sources={}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(x in ('venv','venv_body','vendor') for x in Path(p).parts):sources[str(Path(p).resolve())]=loop.cp.sha256_file(p)
    sources[str(Path(__file__).resolve())]=loop.cp.sha256_file(__file__)
    build_source_snapshot(sources,dest/'source_snapshot.zip')
    result=dict(label='手工研究原型，不是成品',status='FIXED_ACTUAL_BODY_INPUT_NUMERICAL_DIAGNOSTIC_NOT_CLOSED_LOOP',
        source_dir=str(SOURCE),source_result_sha256=loop.cp.sha256_file(SOURCE/'convergence_results.json'),
        source_npz_sha256=loop.cp.sha256_file(SOURCE/'convergence_traces.npz'),identity_path=str(IDENTITY),identity_sha256=loop.cp.sha256_file(IDENTITY),
        source_hashes=sources,checks=checks,experiments=experiments,comparisons=comparisons,
        limits=['Input is saved base-body history held for identical 5ms windows; not a new physical feedback experiment.',
                'Comparison is exact timestamps; grid quantization alone counts as an event difference.',
                'All neuron events retained; motor voltage sampled after reset so not threshold overshoot.',
                'No gains, signs or network edges altered. Decoder output diagnostic only, no body actuation.'],wall_s=time.monotonic()-start)
    (dest/'clock_replay_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('OUTPUT',dest,flush=True)

if __name__=='__main__':
    with loop.mg.guarded_run('fixed_body_clock_replay',**loop.mg.BUDGETS['neural_muscle_closure']):main()
