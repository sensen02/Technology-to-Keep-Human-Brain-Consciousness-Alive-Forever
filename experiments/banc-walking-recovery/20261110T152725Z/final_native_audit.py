"""Read-only S2/S5 final evidence audit. Never integrates physics or edits inputs.
Python from the frozen experiment environment is required. Output uses exclusive create.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDER = ROOT / '20261110T131413Z'
FROZEN = ROOT / '20261110T054510Z/native_acceptance.py'
FREEZE = RECORDER / 'ACCEPTANCE_MEASUREMENT_FREEZE.json'
CAP = 2_000_000
PINNED = {
    FROZEN: '379707f72a231e5e2ff6c5c827971cb29572a65cf3cccc5c7cd63df707b83f8c',
    FREEZE: 'c0c5e361faf49f3ba5cf69201d1a95da0fcbcae8bcf929eaaf1230b81815a07a',
    RECORDER/'audited_eval.py': 'e25d66ac385d9d9ce5e9b143d0ead6ec17eff2bdcd7072f1ecf8b72b9d2687cf',
}
WINDOWS = {'auxiliary': (1., 5.8), 'autonomous': (5.8, 15.4)}
THRESHOLDS = (1e-6, .001, .01, .05, .1)
TAUS = (.00025, .0005, .001, .002)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_header(z, key):
    # Inspect the NPY header before allocating/decompressing an array.
    with z.zip.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        require(version in ((1,0),(2,0)), 'unsupported NPY header version')
        reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(f)
    require(math.prod(shape) <= CAP, 'scalar cap exceeded: ' + key)
    require(not dtype.hasobject and dtype.kind in 'biuf' and dtype.itemsize<=8, 'unsupported/object dtype: ' + key)
    return shape, dtype


def bounded(z, key):
    array_header(z,key)
    a = z[key]
    require(a.dtype.kind in 'biuf' and np.isfinite(a).all(), 'nonfinite/nonnumeric ' + key)
    return a


def compare_npz(left, right):
    rows = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), 'replay key mismatch')
        for key in a.files:
            x, y = bounded(a, key), bounded(b, key)
            exact = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
            rows[key] = {'exact': bool(exact), 'shape': list(x.shape), 'dtype': str(x.dtype)}
            del x, y
    return {'all_exact': all(v['exact'] for v in rows.values()), 'arrays': rows,
            'left': str(left), 'right': str(right), 'left_sha256': sha(left), 'right_sha256': sha(right)}


def load_frozen():
    spec = importlib.util.spec_from_file_location('_final_frozen_native_acceptance', FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_identity(identity):
    payload = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == identity['identity_sha256'], 'identity self digest mismatch')
    for name, entry in identity['files'].items():
        require(Path(name).is_absolute() and sha(name) == entry['sha256'], 'identity file drift: ' + name)
    require(identity['physics'] == {'nq':49, 'nv':48, 'nu':42, 'timestep':12.5e-6, 'gravity':[0.,0.,-9801.]},
            'physical dimensions/gravity/timestep mismatch')
    require(not identity['adhesion']['enabled'], 'adhesion enabled')


def runtime(identity, baseline_identity):
    # Constructor/forward only. This module never calls tick, step, or reset.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RECORDER))
    import physics
    import student_wave as sw
    from identity import verify_runtime
    import torch
    torch.set_num_threads(1)
    for module in (physics, sw, sys.modules['models']):
        p=Path(module.__file__).resolve()
        require(str(p) in identity['files'] and sha(p)==identity['files'][str(p)]['sha256'],
                'actual imported source not recorded: '+str(p))
    for p in sw.SOURCE_FILES:
        require(str(p) in identity['files'] and sha(p)==identity['files'][str(p)]['sha256'],
                'deployment source closure missing: '+str(p))
    observer=RECORDER/'audited_eval.py'
    require(str(observer) in identity['files'] and sha(observer)==identity['files'][str(observer)]['sha256'],
            'run does not identify frozen audited recorder')
    core = physics.Core(adhesion=False)
    verify_runtime(identity, core)
    verify_runtime(baseline_identity, core)
    require((core.plant.nq, core.plant.nv, core.plant.nu) == (49,48,42), 'full body dimensions')
    mesh = core.plant.geom_type == 7  # MuJoCo mjGEOM_MESH
    require(core.plant.nmesh == 71 and np.count_nonzero(mesh) == 71, 'expected all 71 mesh geometries')
    require(np.all(core.plant.geom_rgba[mesh,3] == 1), 'mesh opacity differs from 1')
    require(float(core.d.time) == 0., 'constructor unexpectedly advanced physical time')
    mass = float(core.plant.body_subtreemass[core.th])
    require(np.isclose(mass, .002494271478, rtol=0, atol=5e-13), 'whole body mass mismatch')
    checkpoint = Path(identity['config']['cli']['checkpoint']).resolve()
    require(checkpoint == Path(baseline_identity['config']['cli']['checkpoint']).resolve(), 'baseline checkpoint differs')
    require(str(checkpoint) in identity['files'] and str(checkpoint) in baseline_identity['files'], 'checkpoint missing from identities')
    model, _ = sw.load_student(checkpoint, core)
    return core, sw, torch, model, {'mass_native':mass, 'mesh_count':71, 'opaque_mesh_count':71,
        'nq':49,'nv':48,'nu':42,'physical_steps':0,'xml':str(physics.PAD_XML),
        'initial_heading_world_xy':core.initial_heading.tolist(),
        'xml_sha256':sha(physics.PAD_XML),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'compiled_runtime_verified_against_both_identities':True,
        'scope':'derived complete body / substitute ANN / finite incremental PD; SI and biological fidelity NOT_ESTABLISHED'}


def neural_check(run, result, model, torch):
    path = run / 'neural_decisions.npz'
    require(sha(path) == result['neural']['sha256'], 'neural archive hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        n = {key: bounded(z,key) for key in z.files}
    expected_shapes = {'time':(2880,), 'X238':(2880,238),'V255':(2880,255),
        'hidden_before':(2880,128),'hidden_after':(2880,128),'policy_action':(2880,42),
        'mean':(2880,42),'latent':(2880,42),'logstd':(2880,42), 'value':(2880,), 'log_prob':(2880,),
        'control_source':(2880,), 'actually_applied_action':(2880,42), 'actually_applied_action_is_policy':(2880,)}
    for key, shape in expected_shapes.items():
        require(n[key].shape == shape, 'neural shape mismatch: '+key)
    sources = np.r_[np.full(960,2), np.full(1920,3)]
    require(np.array_equal(n['control_source'],sources), 'neural sources differ')
    require(np.array_equal(n['hidden_before'][1:],n['hidden_after'][:-1]), 'hidden chain interrupted')
    require(np.array_equal(n['hidden_before'][0],np.zeros(128)), 'nonzero initial hidden')
    require(np.array_equal(n['actually_applied_action_is_policy'],sources==3), 'applied mask mismatch')
    require(np.array_equal(n['policy_action'][960:],n['actually_applied_action'][960:]), 'neural action not applied')
    with np.load(run/'decisions_5ms.npz',allow_pickle=False) as z:
        for key, actual in [('time','time'),('X238','X238'),('V255','V255'),('action','actually_applied_action'),('source','control_source')]:
            require(np.array_equal(bounded(z,key),n[actual]), 'neural/decision mismatch '+key)
    maxdiff = {k:0. for k in ('state','action','mean','latent','logstd','log_prob','value')}
    with torch.no_grad():
        for i in range(2880):
            out = model.act(n['X238'][i],n['V255'][i],state=torch.as_tensor(n['hidden_before'][i:i+1]), deterministic=True)
            for key, recorded in [('state','hidden_after'),('action','policy_action'),('mean','mean'),('latent','latent'),
                                  ('logstd','logstd'),('log_prob','log_prob'),('value','value')]:
                actual = out[key].detach().cpu().numpy()[0]
                require(np.array_equal(actual,n[recorded][i]), 'checkpoint forward differs from neural record: '+key+' row '+str(i))
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
        require(np.array_equal(bounded(z,'control_source'),np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]), 'trace source blocks')
        time, actions = bounded(z,'time'), bounded(z,'action')
        require(np.allclose(n['time'],time[999:-1:5],rtol=0,atol=1e-10), 'decision timestamps not trace aligned')
        require(np.array_equal(actions[1000:],np.repeat(n['actually_applied_action'],5,axis=0)), 'trace action differs from decisions')
        for key in ('external_body_force_max','external_generalized_force_max'):
            require(np.all(bounded(z,key)==0), 'trace extra force '+key)
    return {'status':'PASS','samples':2880,'priming':960,'actual_neural':1920,'hchain_exact':True,
            'all_checkpoint_outputs_bitexact':True,'actual_actions_synchronized':True,'physical_steps':0}


def median_hist(hist):
    n = int(hist.sum())
    require(n>0, 'empty histogram')
    return float(np.mean([np.searchsorted(hist.cumsum(),r,side='right') for r in ((n-1)//2,n//2)]))


def extra_native(manifest, weight, trace_path, windows=None):
    """Single bounded pass; exact quantiles retain only selected scalar residuals.
    scipy lfilter equals EMA mathematically; filter state persists across ALL chunks/windows.
    """
    from scipy.signal import lfilter
    windows = WINDOWS if windows is None else windows
    hist = {key:np.zeros((5,7),dtype=np.int64) for key in windows}
    residuals = {key:{tau:array('d') for tau in (0.,)+TAUS} for key in windows}
    accounted = {key:{tau:array('d') for tau in (0.,)+TAUS} for key in windows}
    state = {tau:None for tau in TAUS}
    names = ('external_body_wrench_native','external_generalized_force_native','adhesion_body_wrench_native',
             'adhesion_enabled','adhesion_application_body_leg_point_force_native','external_total_force_native')
    peaks = dict.fromkeys(names,0.)
    events = samples = matched = 0
    with np.load(trace_path,allow_pickle=False) as z:
        trace = {k:bounded(z,k) for k in ('time','qpos','qvel','com','action','target','actualctrl')}
    trace_i = 0
    for entry in manifest['chunks']:
        p = Path(entry['path'])
        require(sha(p)==entry['sha256'], 'native chunk changed before second pass')
        with np.load(p,allow_pickle=False) as z:
            keys = ('time_before_s','time_after_s','com_velocity_before','com_velocity_after','ground_force_native',
                    'loads_native','nonfoot_force_native','contact_offsets','contact_geom_leg_ground')
            a = {k:bounded(z,k) for k in keys}
            before, after = a['time_before_s'],a['time_after_s']
            samples += len(before)
            ext = bounded(z,'external_total_force_native')[:,2]
            for key in names:
                v=bounded(z,key)
                if v.size: peaks[key]=max(peaks[key],float(np.abs(v).max()))
                del v
            offsets=a['contact_offsets']
            require(offsets[0]==0 and len(offsets)==len(before)+1 and np.all(np.diff(offsets)>=0)
                    and offsets[-1]==len(a['contact_geom_leg_ground']), 'contact offsets invalid')
            events += int(offsets[-1])
            raw = ((a['com_velocity_after'][:,2]-a['com_velocity_before'][:,2]) / (after-before)) / 9801. - (a['ground_force_native'][:,2]/weight-1.)
            extras = ext/weight
            values = {0.:raw}; acc={0.:raw-extras}
            for tau in TAUS:
                alpha=float(-np.expm1(-12.5e-6/tau))
                zi=state[tau]
                if zi is None: zi=(1-alpha)*np.array([raw[0],extras[0]])
                filt, final=lfilter([alpha],[1.,-(1-alpha)],np.stack([raw,extras]),axis=1,zi=zi[:,None])
                state[tau]=final[:,0]
                values[tau]=filt[0];acc[tau]=filt[0]-filt[1]
            for key,(start,end) in windows.items():
                keep=(before>=start-1e-10)&(after<=end+1e-10)
                loads=a['loads_native'][keep]
                for j,threshold in enumerate(THRESHOLDS):
                    hist[key][j]+=np.bincount(np.count_nonzero(loads>=threshold*weight,axis=1),minlength=7)
                for tau in values:
                    require(len(residuals[key][tau])+int(keep.sum())<=CAP,'residual cap exceeded')
                    residuals[key][tau].extend(values[tau][keep]);accounted[key][tau].extend(acc[tau][keep])
            # Native endpoints prove same physical run as the hashed logical trace.
            ti=[];ni=[]
            while trace_i<len(trace['time']) and trace['time'][trace_i]<=after[-1]+1e-10:
                idx=int(np.searchsorted(after,trace['time'][trace_i]-1e-10))
                require(idx<len(after) and abs(after[idx]-trace['time'][trace_i])<1e-10,'native/trace endpoint missing')
                ti.append(trace_i);ni.append(idx);trace_i+=1
            if ti:
                for native_key,trace_key in [('qpos_after','qpos'),('qvel_after','qvel'),('com_after','com'),
                                             ('action','action'),('target_rad','target'),('commanded_torque_native','actualctrl')]:
                    v=bounded(z,native_key)
                    require(np.array_equal(v[ni],trace[trace_key][ti]),'native/trace mismatch '+trace_key)
                    del v
                matched+=len(ti)
        require(sha(p)==entry['sha256'],'native chunk changed during second pass')
    require(matched==len(trace['time']) and samples==manifest['native_samples'],'native/trace sample count mismatch')
    output={'samples':samples,'contact_event_count':events,'native_trace_endpoints_bitexact':matched,
            'actual_force_absolute_peaks':peaks,'all_actual_extra_forces_zero':all(v==0 for v in peaks.values()),'windows':{}}
    for key in windows:
        d7=[{'threshold_fraction_W':threshold,'histogram':h.tolist(),'median':median_hist(h)} for threshold,h in zip(THRESHOLDS,hist[key])]
        d6=[]
        for tau in residuals[key]:
            r=np.frombuffer(residuals[key][tau],dtype=np.float64);a=np.frombuffer(accounted[key][tau],dtype=np.float64)
            d6.append({'tau_ms':tau*1000,'samples':len(r),'signed_p95_W':float(np.quantile(r,.95)),
                'absolute_p95_W':float(np.quantile(np.abs(r),.95)),'external_accounted_signed_p95_W':float(np.quantile(a,.95)),
                'external_accounted_absolute_p95_W':float(np.quantile(np.abs(a),.95)),
                'original_signed_criterion_pass':bool(np.quantile(r,.95)<=.15)})
        output['windows'][key]={'D7_sensitivity':d7,'D6_sensitivity':d6}
    return output


def recorded_windows(run):
    # Use actual integration boundary timestamps, preserving every declared step.
    # Accumulated MuJoCo time can differ from nominal seconds by a few 1e-10s.
    with np.load(run/'decisions_5ms.npz',allow_pickle=False) as z:decisions=bounded(z,'time')
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:trace=bounded(z,'time')
    require(decisions.shape==(2880,) and trace.shape==(15400,), 'nominal full timeline length required')
    require(decisions[0]==trace[999] and decisions[960]==trace[5799], 'decision/trace boundary mismatch')
    actual={'auxiliary':(float(decisions[0]),float(decisions[960])),
            'autonomous':(float(decisions[960]),float(trace[-1]))}
    require(all(np.allclose(actual[key],WINDOWS[key],rtol=0,atol=1e-8) for key in WINDOWS),
            'recorded source windows differ from predeclared nominal schedule')
    return actual


def replay_cycles(run, result, sw):
    recorded=result['autonomous_cycles']
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
        fields={key:bounded(z,key) for key in ('time','loads','padz','relativefoot')}
    observer=sw.CycleTracker(sw.CycleConfig(**recorded['config'])).reset(start_time=float(fields['time'][5799]))
    events=[]
    for i in range(5800,15400):
        events.extend(observer.update({key:value[i] for key,value in fields.items()})['events'])
    replay=sw.cycle_report(observer,events)
    require(replay['events']==recorded['events'] and replay['complete_per_leg']==recorded['complete_per_leg'],
            'sensed cycle replay differs from committed events')
    return {'status':'PASS','all_events_exact':True,'complete_per_leg':replay['complete_per_leg'],
            'sample_count':9600,'scope':'replayed original 1ms sensed cycle diagnostic, no action generation'}


def dependency_evidence(run, result, model, torch):
    zero = ROOT/'20261110T111240Z/EVAL_ZERO_MATCHED_RETRY'
    zr = read_json(zero/'RESULT.json'); zi=read_json(zero/'IDENTITY.json')
    verify_identity(zi)
    require(zr['control']=='zero' and zr['sampled'] is False, 'zero control identity mismatch')
    for key, name in [('trace','trace_1ms.npz'),('decisions','decisions_5ms.npz')]:
        require(sha(zero/name)==zr[key]['sha256'], 'zero evidence archive hash mismatch')
    rows={}
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as x, np.load(zero/'trace_1ms.npz',allow_pickle=False) as y:
        for key in sorted(set(x.files)&set(y.files)):
            left,right=bounded(x,key),bounded(y,key)
            exact=left.dtype==right.dtype and np.array_equal(left[:5800],right[:5800])
            rows[key]={'exact':bool(exact),'prefix_samples':5800}
            del left,right
    require(rows and all(v['exact'] for v in rows.values()), 'zero/candidate physical prefix differs')
    groups={'joint_position':(0,42),'joint_velocity':(84,126),'foot_load':(168,174),
        'gravity':(174,177),'root_velocity':(177,180),'root_omega':(180,183),
        'pad_height':(183,189),'pad_vertical_velocity':(189,195)}
    cuts=[]
    with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:
        x,v,h=bounded(z,'X238'),bounded(z,'V255'),bounded(z,'hidden_before')
        with torch.no_grad():
            for i in (960,1200,1600,2200,2879):
                state=torch.as_tensor(h[i:i+1])
                baseline=model.act(x[i],v[i],state=state,deterministic=True)['mean'][0].detach().cpu().numpy()
                for name,(start,end) in groups.items():
                    modified=x[i].copy();modified[start:end]=0.
                    candidate=model.act(modified,v[i],state=state,deterministic=True)['mean'][0].detach().cpu().numpy()
                    delta=np.abs(candidate-baseline)
                    cuts.append({'row':i,'group':name,'columns':[start,end],'max_abs_mean_change':float(delta.max()),
                        'L2_mean_change':float(np.linalg.norm(delta)),'any_change':bool(np.any(delta!=0))})
    return {'matched_zero_control':{'path':str(zero),'identity_verified':True,'result_sha256':sha(zero/'RESULT.json'),
        'prefix_arrays':rows,'prefix_all_exact':True,'prefix_physical_samples':5800,
        'zero_autonomous':zr['autonomous'],'zero_cycles':zr['autonomous_cycles'],
        'candidate_forward':result['autonomous']['forward_signed_native_per_s'],
        'candidate_cycles':result['autonomous_cycles']['complete_per_leg'],
        'interpretation':'same actual physical teacher prefix; zero-output actuator holding alone does not establish neural locomotion'},
        'offline_saved_hidden_sensory_cuts':{'rows':cuts,'any_command_dependence':any(r['any_change'] for r in cuts),
        'scope':'offline counterfactual actor-input zero cuts at actual saved hidden; critic unchanged; no physics/outcome claim and no all-sensors-must-fall gate'}}


def audit(run,baseline):
    for path,digest in PINNED.items():require(sha(path)==digest,'frozen dependency changed: '+str(path))
    input_files=[run/name for name in ('RESULT.json','IDENTITY.json','trace_1ms.npz','decisions_5ms.npz','neural_decisions.npz')]
    input_files += [baseline/name for name in ('RESULT.json','IDENTITY.json','trace_1ms.npz','decisions_5ms.npz')]
    barrier_hashes={str(path):sha(path) for path in input_files}
    result=read_json(run/'RESULT.json'); original=read_json(baseline/'RESULT.json')
    identity=read_json(run/'IDENTITY.json'); baseline_identity=read_json(baseline/'IDENTITY.json')
    verify_identity(identity);verify_identity(baseline_identity)
    require(result['control']=='student' and result['sampled'] is False and result['native']['pending_rows']==0,'not complete deterministic native student evidence')
    require(result['native']['native_samples']==1_232_000,'full native interval required')
    require(result['transition']['all_exact'] and original['transition']['all_exact'],'handover not exact')
    for key in ('trace','decisions','neural'):
        path=run/({'trace':'trace_1ms.npz','decisions':'decisions_5ms.npz','neural':'neural_decisions.npz'}[key])
        require(sha(path)==result[key]['sha256'],key+' hash differs')
    for key,name in [('trace','trace_1ms.npz'),('decisions','decisions_5ms.npz')]:
        require(sha(baseline/name)==original[key]['sha256'],'baseline '+key+' hash differs')
    replay={name:compare_npz(run/name,baseline/name) for name in ('trace_1ms.npz','decisions_5ms.npz')}
    require(all(v['all_exact'] for v in replay.values()),'fast logical replay not exact')
    core,sw,torch,model,physical=runtime(identity,baseline_identity)
    neural=neural_check(run,result,model,torch)
    dependency=dependency_evidence(run,result,model,torch)
    cycle_replay=replay_cycles(run,result,sw)
    require(core.d.time==0.,'physical time advanced')
    # audited_eval's completion report omits teacher metadata. Exact logical replay
    # above authorizes the explicit baseline provenance link, not invented fields.
    reference=original['algorithm_prefix']['teacher_metadata']['reference']
    require(sha(reference['source_path'])==reference['source_sha256'],'measured reference changed')
    for path,digest in original['teacher_reference_evidence']['files'].items():
        require(sha(path)==digest,'teacher reference evidence changed: '+path)
    # Enforce the stricter wrapper scalar limit before frozen evaluator allocation.
    for entry in result['native']['chunks']:
        p=Path(entry['path']).resolve()
        require(p.parent==(run/'native').resolve(), 'native manifest path outside same run')
        with np.load(p,allow_pickle=False) as z:
            for key in z.files:array_header(z,key)
    actual_windows=recorded_windows(run)
    frozen=load_frozen()
    report=frozen.evaluate_directory(run/'native',autonomous_window=actual_windows['autonomous'],auxiliary_window=actual_windows['auxiliary'],
        mass_native=physical['mass_native'],source_manifest=result['native'],full_model=physical['xml'],
        heading_world_xy=physical['initial_heading_world_xy'],
        metadata={'run_id':str(run),'evidence':[{'path':str(run/'RESULT.json'),'sha256':sha(run/'RESULT.json'),
                  'claims':['auxiliary_control_valid','continuous_handover']}]},max_scalar_samples=CAP)
    require(report['status']=='OK','frozen evaluator error: '+str(report.get('errors')))
    extra=extra_native(result['native'],physical['mass_native']*9801.,run/'trace_1ms.npz',windows=actual_windows)
    counts_by_phase={'settle':80_000,'auxiliary':384_000,'autonomous':768_000}
    for key in WINDOWS:
        require(report['windows'][key]['samples']==counts_by_phase[key], 'complete native phase count required: '+key)
        require(extra['windows'][key]['D6_sensitivity'][0]['samples']==counts_by_phase[key], 'diagnostic omitted native interval: '+key)
    require(report['integrity']['recorded_window_s'][0]==0. and report['integrity']['samples']==sum(counts_by_phase.values()),
            'complete settle/AUX/ANN native timeline required')
    for key in WINDOWS:
        primary=next(row for row in extra['windows'][key]['D6_sensitivity'] if row['tau_ms']==1.)
        require(abs(primary['signed_p95_W']-report['windows'][key]['original_D6_signed_p95_W'])<1e-10,
                'primary D6 differs from frozen evaluator')
        require(extra['windows'][key]['D7_sensitivity'][-1]['median']==report['windows'][key]['loaded_leg_count_median'],
                'primary D7 median differs from frozen evaluator')
    sufficient=extra['windows']['autonomous']['D7_sensitivity'][-1]['median']>=3
    criteria={k:dict(v) for k,v in report['criteria'].items()}
    criteria['D7']={'status':'PASS' if sufficient else 'UNKNOWN','value':extra['windows']['autonomous']['D7_sensitivity'][-1]['median'],
        'threshold':3,'reason':'predeclared >=.1W sufficient evidence for any reported lower positive carrying threshold; failure does not establish original D7 failure'}
    engineering={k:v for k,v in report['engineering_only'].items() if isinstance(v,dict) and 'status' in v}
    cycles=result['autonomous_cycles'];counts=cycles['complete_per_leg']
    # Count actual observed complete events; preserve all failed/timeout diagnostics.
    for leg in ('LF','LH','LM','RF','RH','RM'):
        require(counts[leg]==sum(e['type']=='complete' and e['leg']==leg for e in cycles['events']),'cycle event/count mismatch '+leg)
    six=all(counts[leg]>=1 for leg in ('LF','LH','LM','RF','RH','RM'))
    metrics_pass=all(v['status']=='PASS' for v in criteria.values())
    gates_pass=all(v['status']=='PASS' for v in engineering.values())
    failures=[k for k,v in criteria.items() if v['status']!='PASS']+[f'engineering.{k}' for k,v in engineering.items() if v['status']!='PASS']
    if not six:failures.append('six_complete_cycles')
    if not extra['all_actual_extra_forces_zero']:failures.append('actual_extra_force')
    verify_identity(identity);verify_identity(baseline_identity)
    for path,digest in barrier_hashes.items():require(sha(path)==digest,'evidence changed during audit: '+path)
    for path,digest in PINNED.items():require(sha(path)==digest,'frozen source changed during audit: '+str(path))
    # Never conflate this data audit with required same-run video/dependency review.
    return {'schema':'final-native-neural-audit-v1','status':'PASS' if not failures else 'FAIL',
        'scope':'S2/S5 read-only final physical/neural data audit; full goal NOT_ESTABLISHED',
        'run':str(run),'baseline':str(baseline),'nominal_declared_windows':WINDOWS,'actual_recorded_windows':actual_windows,
        'native_phase_sample_counts':counts_by_phase,'time_discretization':'actual source-boundary timestamps within1e-8 of nominal; all native intervals retained',
        'frozen_legacy_report':report,
        'predeclared_measurement_criteria':criteria,'physical_metric_conjunction':metrics_pass,
        'engineering_native':engineering,'engineering_native_all_pass':gates_pass,
        'engineering_1ms_legacy':result['autonomous'],'strict_six_leg_cycles':cycles,'six_cycles_pass':six,
        'cycle_trace_replay':cycle_replay,
        'logical_replay':replay,'neural_synchronization':neural,'physical_identity':physical,'native_diagnostics':extra,
        'dependency_evidence':dependency,'reference_provenance':{'source':'baseline RESULT after exact physical/logical replay',
            'baseline_result':str(baseline/'RESULT.json'),'baseline_result_sha256':sha(baseline/'RESULT.json'),
            'measured_reference':{k:reference[k] for k in ('label','source_path','source_sha256','source_trajectory_id')},
            'teacher_reference_evidence':original['teacher_reference_evidence']},
        'failures_or_unknowns':failures,'full_goal_status':'NOT_ESTABLISHED',
        'remaining_external_review':['same-run complete video visual review','neural/sensory dependency evidence and scope review'],
        'provenance':{'frozen_evaluator':{'path':str(FROZEN),'sha256':sha(FROZEN)},
            'measurement_freeze':{'path':str(FREEZE),'sha256':sha(FREEZE)},'wrapper':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
            'result_sha256':sha(run/'RESULT.json'),'baseline_result_sha256':sha(baseline/'RESULT.json'),
            'all_input_barrier_hashes':barrier_hashes,'input_hashes_reverified_at_end':True,
            'source_closure':'both identity file closures and compiled runtime verified; recorder pinned to frozen audited_eval source; actual neural outputs recomputed from checkpoint'}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--baseline',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    for key in ('run','baseline','out'):
        require(Path(getattr(args,key)).is_absolute(),'absolute path required: '+key)
    run,baseline,out=(Path(getattr(args,k)).resolve() for k in ('run','baseline','out'))
    require(run.is_dir() and baseline.is_dir(),'input directories absent')
    require(not out.exists(),'refusing existing output')
    require(not out.is_relative_to(run) and not out.is_relative_to(baseline),'output must be outside immutable evidence directories')
    # Pin required recording implementation, not a caller-selected import.
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    try:
        report=audit(run,baseline)
    except Exception as exc:
        report={'schema':'final-native-neural-audit-v1','status':'ERROR','errors':[type(exc).__name__+': '+str(exc)],
                'run':str(run),'baseline':str(baseline),'full_goal_status':'NOT_ESTABLISHED'}
    with out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'out':str(out),'status':report['status'],'failures':report.get('failures_or_unknowns',report.get('errors',[])),
                      'full_goal_status':'NOT_ESTABLISHED'}),flush=True)
    return 2 if report['status']=='ERROR' else (1 if report['status']=='FAIL' else 0)

if __name__=='__main__':
    raise SystemExit(main())
