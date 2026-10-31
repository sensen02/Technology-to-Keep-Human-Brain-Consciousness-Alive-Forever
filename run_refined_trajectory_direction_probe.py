"""S135/P1: five genuine full-neural refined directional trajectory trials; import is inert.
Parent executes this producer, never this author. No optimization or promotion.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s135-refined-trajectory-direction')
S133 = ROOT/'run_margin_candidate_probe.py'
S121 = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')
S132 = Path('/home/sensen/Desktop/engine修复验收/s132-margin-contact-candidate/20261024T180846Z')
S134 = Path('/home/sensen/Desktop/engine修复验收/s134-trajectory-direction/20261024T182145Z')
S134_VERIFIER = ROOT/'verify_trajectory_direction_probe.py'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
ARM = 'init80ms'
TRIALS = [('baseline', 0.), ('plus_h', .000125), ('minus_h', -.000125), ('plus_half', .0000625), ('minus_half', -.0000625)]
EXPECTED = dict(model_load=1, forward=121, step=120, neural_ticks=166)
TOLERANCES = dict(qacc=[1e-3,1e-3], qvel=[1e-5,1e-3], qpos=[1e-5,1e-3], load_ratio=[1e-4,1e-3], limit_excess=[1e-6,1e-3])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()


def js(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def load_private(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def hashes_ok(hashes):
    for name, digest in hashes.items():
        if sha(name) != digest: raise ValueError('input changed: '+name)


def sources():
    wrapper = load_private(S133, '_s134_s133_validation_only')
    commitments, candidate, baseline = wrapper.validate_sources()
    chain = npz(S132/'OFFLINE_CANDIDATE.npz')
    assert wrapper.exact(chain['reference_weights'], baseline)
    assert wrapper.exact(chain['proposed_weights'], candidate)
    delta = chain['delta']; norm = float(np.linalg.norm(delta))
    assert norm > 0 and np.isfinite(norm) and wrapper.exact(baseline+delta, candidate)
    v = delta/norm
    # Freeze the complete S134 producer/verifier chain as preregistered inputs;
    # these hashes are checked before every trial and before completion.
    commitments.update({
        str(S133): sha(S133),
        str(Path(__file__).resolve()): sha(__file__),
        str(S134/'PREREG.json'): sha(S134/'PREREG.json'),
        str(S134/'INPUT_HASHES.json'): sha(S134/'INPUT_HASHES.json'),
        str(S134/'DIRECTION.npz'): sha(S134/'DIRECTION.npz'),
        str(S134/'PREREG.json'): sha(S134/'PREREG.json'),
        str(S134/'OUTPUT_HASHES.json'): sha(S134/'OUTPUT_HASHES.json'),
        str(S134/'TRAJECTORY_DIRECTION_INDEPENDENT_CHECK.json'): sha(S134/'TRAJECTORY_DIRECTION_INDEPENDENT_CHECK.json'),
        str(S134/'DERIVATIVE_REPORT.json'): sha(S134/'DERIVATIVE_REPORT.json'),
        str(S134/'PARENT_CONCLUSION.zh-CN.md'): sha(S134/'PARENT_CONCLUSION.zh-CN.md'),
        str(S134_VERIFIER): sha(S134_VERIFIER),
    })
    return commitments, baseline, delta, v, norm


def configure(original, out, weight_path, weights, baseline, commitments, amplitude):
    """S133-style passive hooks, but S115 original local_weights is retained exactly."""
    assert original.PRE == 160 and original.TICKS == 6
    assert original.TIMESTEPS == {ARM:.000025} and original.SUBSTEPS == {ARM:20}
    assert original.EXPECTED == EXPECTED
    assert np.array_equal(npz(original.WEIGHTS)['weights'], baseline)
    original.WEIGHTS = weight_path
    original.OUTPUT_PARENT = out
    recorder = load_private(ROOT/'run_constraint_row_recording.py', '_s134_passive_s121')
    capture_original, save_json, save_npz = original.com_contact_record, original.atomic_json, original.atomic_npz
    records, metadata = [], {}
    # Preloaded imported S115 function objects are not re-imported/replaced by a
    # candidate wrapper; only this private module's globals and passive hooks change.
    def capture(mj, m, d, floor_ids):
        result = capture_original(mj,m,d,floor_ids)
        row = recorder.constraint_record(mj,m,d,len(records))
        row['efc_state'] = np.asarray(d.efc_state).copy()
        records.append(row)
        if not metadata: metadata.update(recorder.constraint_metadata(mj,m))
        return result
    def json_hook(path, value):
        path = Path(path)
        if path.name == 'INPUT_HASHES.json': value = dict(value, **commitments)
        if path.name == 'PREREG.json':
            value = dict(value, source_hashes=dict(value['source_hashes'], **commitments),
                task='S135/P1', amplitude_absolute_W_L2=amplitude,
                training_constraints=js(FORMAL), paired_s121=str(S121),
                selection_rule='NONE: direction derivative diagnostic; all failed safety gates retained',
                all_weights_identical=True, promotion=False,
                sensory_scope='5ms sensory updates; tick0 initial observation only in physical3ms; no rich feedback adaptation')
        if path.name == 'COMPLETE.json':
            hashes_ok(commitments); assert len(records) == 121
            assert {k:value['ledger'][k] for k in EXPECTED} == EXPECTED
            value = dict(value, source_chain_hashes=dict(value['source_chain_hashes'], **commitments), constraint_samples=121)
        return save_json(path,value)
    def npz_hook(path, **arrays):
        path = Path(path)
        if 'weights' in arrays: assert np.array_equal(arrays['weights'],weights)
        if path.name == 'TRAJECTORY.npz':
            assert np.array_equal(arrays['time'],np.asarray([r['time'] for r in records]))
            save_npz(path.parent/'CONSTRAINT_ROWS.npz', **recorder.pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json',metadata)
        return save_npz(path, **arrays)
    original.com_contact_record = capture
    original.atomic_json = json_hook
    original.atomic_npz = npz_hook


def worker(config_path):
    cfg = js(config_path); out = Path(cfg['trial_output']); parent = Path(cfg['parent_output'])
    assert parent.parent == BASE and out.parent == parent and not out.exists()
    assert config_path == parent/(cfg['label']+'_CONFIG.json')
    pre = js(parent/'PREREG.json'); assert pre['configs'][cfg['label']] == sha(config_path)
    commitments, baseline, delta, v, norm = sources()
    assert commitments == pre['source_hashes']; hashes_ok(commitments)
    weight_path = Path(cfg['weights_path'])
    assert weight_path == parent/(cfg['label']+'_WEIGHTS.npz') and sha(weight_path) == cfg['weights_sha256']
    assert (cfg['label'],cfg['amplitude']) in TRIALS
    weights = npz(weight_path)['weights']
    assert np.array_equal(weights,baseline+cfg['amplitude']*v)
    original = load_private(ROOT/'run_extended_initialization_probe.py','_s134_pristine_s115')
    configure(original,out,weight_path,weights,baseline,commitments,cfg['amplitude'])
    sys.argv = [str(original.__file__),'--output',str(out)]
    original.main()


def difference(a,b,atol,rtol):
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != b.shape: return dict(passed=False,shape_a=list(a.shape),shape_b=list(b.shape))
    error = float(np.max(np.abs(a-b),initial=0)); scale = max(float(np.max(np.abs(a),initial=0)),float(np.max(np.abs(b),initial=0)),1e-30)
    rel = error/scale
    return dict(max_abs=error,scaled_relative=rel,scale=scale,absolute_tolerance=atol,relative_tolerance=rtol,passed=bool(error<=atol and rel<=rtol))


def baseline_check(out):
    actual = npz(out/'baseline'/ARM/'TRAJECTORY.npz'); ref = npz(S121/ARM/'TRAJECTORY.npz')
    checks = {}
    # Compare every common raw saved field, not merely the requested minimal set.
    for k in sorted(ref):
        if k not in actual: checks[k] = dict(passed=False,missing=True); continue
        if ref[k].dtype.kind in 'fc': checks[k] = difference(actual[k],ref[k],1e-8,1e-6)
        else: checks[k] = dict(passed=bool(np.array_equal(actual[k],ref[k])))
    report = dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                  definition='max absolute AND max absolute / max(abs(a),abs(b),1e-30); every common trajectory field')
    save(out/'BASELINE_REPRODUCTION.json',report)
    return report['passed']


def limit_arrays(a, meta):
    limited = np.flatnonzero(meta['jnt_limited']); assert len(limited)==42
    qa = np.asarray(meta['jnt_qposadr'])[limited]; bounds=np.asarray(meta['jnt_range'])[limited]
    e = np.maximum(np.maximum(bounds[:,0]-a['qpos'][:,qa],a['qpos'][:,qa]-bounds[:,1]),0)
    assert np.array_equal(e.max(1),a['limit_excess'])
    return e


def mapped(features,weights,mapping):
    pooled=np.clip((features*weights).sum(2),0,1); result=np.zeros((len(features),len(mapping)))
    mask=mapping>=0; result[:,mask]=pooled[:,mapping[mask]]; return result


def offline(out):
    arrays={label:npz(out/label/ARM/'TRAJECTORY.npz') for label,_ in TRIALS}
    rows={label:npz(out/label/ARM/'CONSTRAINT_ROWS.npz') for label,_ in TRIALS}
    warm={label:npz(out/label/ARM/'INITIALIZATION.npz') for label,_ in TRIALS}
    meta=js(out/'baseline'/ARM/'MOMENT_METADATA.json'); a0=arrays['baseline']
    packed={}; checks={}; trial_metrics={}; branches={}; command_checks={}
    # No command synthesized here is ever delivered to a runner or neural model.
    for label,amp in TRIALS:
        a=arrays[label]; w=warm[label]; lim=limit_arrays(a,meta)
        f=np.concatenate((w['features'],a['features'])); u=np.concatenate((w['decoded'],a['applied']))
        predicted=mapped(f,a['weights'],a['actuator_pool'])
        fixed=mapped(np.concatenate((warm['baseline']['features'],a0['features'])),a['weights'],a['actuator_pool'])
        packed[label+'_all166_actual_commands']=u; packed[label+'_all166_analytical_commands']=predicted
        packed[label+'_all166_fixed_reference_features_commands']=fixed
        command_checks[label]=difference(u,predicted,1e-14,1e-12)
        same_spikes=np.array_equal(w['spikes'],warm['baseline']['spikes']) and np.array_equal(a['spikes'],a0['spikes'])
        same_features=np.array_equal(f,np.concatenate((warm['baseline']['features'],a0['features'])))
        c=rows[label]; b=rows['baseline']; same_contact=True; same_state=True
        for i in range(121):
            p=f'sample{i:03d}_'
            for k in ('nefc','ncon','efc_type','efc_id','contact_geom','contact_efc_address','contact_dim','contact_exclude'):
                same_contact &= np.array_equal(c[p+k],b[p+k])
            same_state &= np.array_equal(c[p+'efc_state'],b[p+'efc_state'])
        # Include translator clipping and native all42 max active identities.
        raw=(f*a['weights']).sum(2); f0=np.concatenate((warm['baseline']['features'],a0['features']))
        raw0=(f0*a0['weights']).sum(2)
        same_clip=np.array_equal(np.where(raw<=0,-1,np.where(raw>=1,1,0)),np.where(raw0<=0,-1,np.where(raw0>=1,1,0)))
        l0=limit_arrays(a0,meta)
        same_limit=np.array_equal(lim>0,l0>0) and np.array_equal(lim.argmax(1),l0.argmax(1))
        branches[label]=dict(spikes_identical=bool(same_spikes),FIR_features_identical=bool(same_features),contact_rows_identical=bool(same_contact),efc_active_state_identical=bool(same_state),clip_branch_identical=bool(same_clip),limit_branch_identical=bool(same_limit))
        trial_metrics[label]=dict(peak_all42_limit_excess=float(a['limit_excess'].max()),peak_sample=int(a['limit_excess'].argmax()),peak_window_samples=list(range(110,118)),peak_window_all42_max=[float(x) for x in lim[110:118].max(1)],sample113_qroll=float(a['qpos'][113,9]),load_min=float(a['load_ratio'].min()),all_gates_support_s=float(np.trapezoid(a['all_six_gates'].astype(float),x=a['time'])),failed_gate_samples=(~a['six_gates']).sum(0).tolist())
        packed[label+'_all42_limit_excess']=lim
        packed[label+'_six_gates']=a['six_gates']; packed[label+'_all_six_gates']=a['all_six_gates']
        packed[label+'_raw_qpos']=a['qpos']; packed[label+'_raw_qvel']=a['qvel']; packed[label+'_raw_qacc']=a['qacc']
        # Informative all42 joint peak window; diagnostic only, not a new gate.
        packed[label+'_peak_window_limit_excess']=lim[110:118]
        packed[label+'_peak_window_qroll']=a['qpos'][110:118,9]
        packed[label+'_peak_window_qvel_roll']=a['qvel'][110:118,8]
    fields=['qacc','qvel','qpos','act','ctrl','load_ratio','actuator_force','qfrc_actuator','qfrc_constraint','qfrc_passive','qfrc_bias','limit_excess']
    for k in fields:
        dh=(arrays['plus_h'][k]-arrays['minus_h'][k])/(2*.000125)
        dhalf=(arrays['plus_half'][k]-arrays['minus_half'][k])/(2*.0000625)
        # Raw quaternion components are archived, but not interpreted as tangent DOFs.
        if k=='qpos': dh=dh[:,np.r_[0:3,7:49]]; dhalf=dhalf[:,np.r_[0:3,7:49]]
        packed[k+'_D_h']=dh; packed[k+'_D_half']=dhalf
        tol=TOLERANCES.get(k)
        if tol: checks[k]=difference(dh,dhalf,*tol)
    scalar={}
    for k in ('peak_all42_limit_excess','sample113_qroll','load_min','all_gates_support_s'):
        dh=(trial_metrics['plus_h'][k]-trial_metrics['minus_h'][k])/(2*.000125)
        half=(trial_metrics['plus_half'][k]-trial_metrics['minus_half'][k])/(2*.0000625)
        tol=TOLERANCES['limit_excess' if k=='peak_all42_limit_excess' else 'load_ratio' if k=='load_min' else 'qpos']
        scalar[k]=dict(D_h=dh,D_half=half,**difference(dh,half,*tol),interpretation='discrete support, no differentiability claim' if k=='all_gates_support_s' else 'scalar central difference')
    branch_gate_keys=('spikes_identical','contact_rows_identical','efc_active_state_identical','clip_branch_identical')
    nonsmooth=not all(all(v[k] for k in branch_gate_keys) for v in branches.values())
    verified=not nonsmooth and all(x['passed'] for x in checks.values()) and all(x['passed'] for x in command_checks.values()) and all(scalar[k]['passed'] for k in ('peak_all42_limit_excess','sample113_qroll','load_min'))
    np.savez_compressed(out/'DERIVATIVES.npz',**packed)
    save(out/'DERIVATIVE_REPORT.json',dict(status='NONSMOOTH' if nonsmooth else 'TWO_SCALE_PASS' if verified else 'TWO_SCALE_FAIL',verified_gradient=verified,branches=branches,two_scale_checks=checks,scalar_derivatives=scalar,trial_metrics=trial_metrics,all166_command_checks=command_checks,qpos_columns=[*range(3),*range(7,49)],quaternion_derivative='excluded, no quaternion tangent claim',safety_admission=False,promotion=False,sensory_scope='5ms update; none after tick0 within3ms, not rich feedback adaptation'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--trial',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.trial:
        assert args.output is None; worker(args.trial); return
    now=datetime.now(timezone.utc); out=args.output or BASE/now.strftime('%Y%m%dT%H%M%SZ')
    assert out.is_absolute() and out.parent==BASE and not out.exists()
    # Output timestamp must be actual current UTC, not a historical fixed name.
    stamp=datetime.strptime(out.name,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    assert abs((now-stamp).total_seconds())<120
    commitments,baseline,delta,v,norm=sources()
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'DIRECTION.npz',reference_weights=baseline,stored_S132_delta=delta,v=v,delta_norm=np.asarray(norm),h=np.asarray(.000125),half=np.asarray(.0000625),peak_window=np.asarray([110,111,112,113,114,115,116,117]))
    configs={}
    for label,amp in TRIALS:
        weights=baseline+amp*v
        assert np.isfinite(weights).all() and np.all((weights>=0)&(weights<=4))
        assert np.array_equal(weights==0,baseline==0) and np.array_equal(weights[36],baseline[36])
        path=out/(label+'_WEIGHTS.npz'); np.savez_compressed(path,weights=weights)
        cfg=out/(label+'_CONFIG.json')
        save(cfg,dict(parent_output=str(out),trial_output=str(out/label),label=label,amplitude=amp,weights_path=str(path),weights_sha256=sha(path)))
        configs[label]=sha(cfg)
    save(out/'INPUT_HASHES.json',commitments)
    save(out/'PREREG.json',dict(task='S135/P1 refined trajectory-direction convergence diagnostic',created_utc=now.isoformat(),source_hashes=commitments,configs=configs,trials=TRIALS,max_trials=5,retry=False,autoscan=False,h=.000125,half=.0000625,direction='stored exact S132 delta / L2 norm',delta_norm=norm,training_constraints=js(FORMAL),expected_per_trial=EXPECTED,expected_total={k:5*v for k,v in EXPECTED.items()},baseline_tolerances=dict(absolute=1e-8,scaled_relative=1e-6,conjunction='AND'),two_scale_tolerances=TOLERANCES,relative_definition='max abs error / max(global max abs D_h,global max abs D_half,1e-30); absolute AND relative',baseline_mismatch='FAIL_STOP before perturbation trials; retain raw failed gates',branch_change='NONSMOOTH; no verified gradient',PRE=160,initialization_ms=80,physical_ms=3,physical_steps=120,samples=121,neural_ticks_physical=6,sensory_hold_ms=5,sensory_scope='tick0 only in physical3ms, no rich feedback adaptation',checkpoint_replacement=False,promotion=False,weight_optimization=False))
    ledger=dict(planned_trials=5,started=[],finished=[],actual_totals={k:0 for k in EXPECTED},trial_ledgers={})
    save(out/'CALL_LEDGER.json',ledger)
    try:
        for label,_ in TRIALS:
            hashes_ok(commitments); ledger['started'].append(label); save(out/'CALL_LEDGER.json',ledger)
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--trial',str(out/(label+'_CONFIG.json'))]
            with (out/(label+'_STDOUT.log')).open('w') as stdout, (out/(label+'_STDERR.log')).open('w') as stderr:
                result=subprocess.run(command,cwd=str(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),stdout=stdout,stderr=stderr)
            child=out/label/'CALL_LEDGER.json'
            if child.exists(): ledger['trial_ledgers'][label]=js(child)
            ledger['actual_totals']={k:sum(x.get(k,0) for x in ledger['trial_ledgers'].values()) for k in EXPECTED}
            ledger['finished'].append(dict(label=label,exit_code=result.returncode,command=command)); save(out/'CALL_LEDGER.json',ledger)
            if result.returncode: raise RuntimeError('trial failed, no retry: '+label)
            assert {k:ledger['trial_ledgers'][label][k] for k in EXPECTED}==EXPECTED
            if label=='baseline' and not baseline_check(out): raise RuntimeError('baseline mismatch: fail-stop; four perturbations not run')
        offline(out); hashes_ok(commitments)
        save(out/'COMPLETE.json',dict(status='COMPLETE',ledger=ledger,verified_gradient=js(out/'DERIVATIVE_REPORT.json')['verified_gradient'],promotion=False))
    except BaseException:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,promotion=False)); raise
    finally:
        save(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_HASHES.json'})
    print(str(out))


if __name__=='__main__': main()
