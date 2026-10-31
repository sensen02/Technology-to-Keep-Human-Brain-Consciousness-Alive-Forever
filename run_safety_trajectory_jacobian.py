"""S136/P1 safety-only basis trajectory Jacobian: 61 genuine fresh-process trials.
Import is inert. Parent executes; no optimization, candidate or checkpoint promotion.
The new safety-output gate explicitly does not establish general qacc validity.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian')
S121 = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')
S135 = Path('/home/sensen/Desktop/engine修复验收/s135-refined-trajectory-direction/20261024T182947Z')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
ARM = 'init80ms'
H, HALF = .000125, .0000625
WEIGHT_INDICES = [(p,b) for p in (3,15,38) for b in range(5)]
SCALES = [('plus_h',H),('minus_h',-H),('plus_half',HALF),('minus_half',-HALF)]
TRIALS = [('baseline',None,0.)] + [(f'p{p:03d}_b{b}_{s}',(p,b),a) for p,b in WEIGHT_INDICES for s,a in SCALES]
EXPECTED = dict(model_load=1,forward=121,step=120,neural_ticks=166)
TOLERANCES = dict(qpos=[1e-5,1e-3],qvel=[1e-5,1e-3],load_ratio=[1e-4,1e-3],limit_excess=[1e-6,1e-3],signed_joint_position=[1e-5,1e-3],lower_clearance=[1e-5,1e-3],upper_clearance=[1e-5,1e-3],root_angle_rad=[1e-5,1e-3],com_vertical_velocity=[1e-5,1e-3],height=[1e-5,1e-3])
QPOS_COLUMNS = [*range(3),*range(7,49)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def js(path): return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)


def npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def load_private(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def hashes_ok(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError('input changed: '+path)


def sources():
    wrapper=load_private(ROOT/'run_margin_candidate_probe.py','_s136_s133_validate_original')
    commitments,_,baseline=wrapper.validate_sources()
    # S132 remains an audited historical ancestor only, never a perturbation direction.
    for path in [ROOT/'run_margin_candidate_probe.py',Path(__file__).resolve(),ROOT/'run_refined_trajectory_direction_probe.py',ROOT/'verify_refined_trajectory_direction_probe.py',S135/'PREREG.json',S135/'INPUT_HASHES.json',S135/'OUTPUT_HASHES.json',S135/'COMPLETE.json',S135/'DERIVATIVE_REPORT.json',S135/'REFINED_TRAJECTORY_DIRECTION_INDEPENDENT_CHECK.json',S135/'PARENT_CONCLUSION.zh-CN.md']:
        commitments[str(path)]=sha(path)
    hashes_ok(js(S135/'OUTPUT_HASHES.json'))
    hashes_ok(js(S135/'INPUT_HASHES.json'))
    assert exact(npz(S121/ARM/'TRAJECTORY.npz')['weights'],baseline)
    assert baseline.shape==(66,5)
    return commitments,baseline


def basis_weights(baseline,index,amplitude):
    weights=baseline.copy()
    if index is not None: weights[tuple(index)]+=amplitude
    allowed=np.zeros_like(weights,dtype=bool)
    if index is not None: allowed[tuple(index)]=True
    assert exact(weights[~allowed],baseline[~allowed]) and exact(weights[36],baseline[36])
    assert exact(weights==0,baseline==0) and np.isfinite(weights).all() and np.all((weights>=0)&(weights<=4))
    return weights


def configure(original,out,weight_path,weights,baseline,commitments,cfg):
    assert original.PRE==160 and original.TICKS==6
    assert original.TIMESTEPS=={ARM:.000025} and original.SUBSTEPS=={ARM:20}
    assert original.EXPECTED==EXPECTED
    assert exact(npz(original.WEIGHTS)['weights'],baseline)
    original.WEIGHTS=weight_path; original.OUTPUT_PARENT=out
    recorder=load_private(ROOT/'run_constraint_row_recording.py','_s136_passive_s121')
    capture_original,save_json,save_npz=original.com_contact_record,original.atomic_json,original.atomic_npz
    records,metadata=[],{}
    def capture(mj,m,d,floor_ids):
        result=capture_original(mj,m,d,floor_ids)
        row=recorder.constraint_record(mj,m,d,len(records)); row['efc_state']=np.asarray(d.efc_state).copy(); records.append(row)
        if not metadata: metadata.update(recorder.constraint_metadata(mj,m))
        return result
    def json_hook(path,value):
        path=Path(path)
        if path.name=='INPUT_HASHES.json': value=dict(value,**commitments)
        if path.name=='PREREG.json':
            value=dict(value,source_hashes=dict(value['source_hashes'],**commitments),task='S136/P1',basis_trial=cfg,amplitude_absolute_per_weight=cfg['amplitude'],paired_s121=str(S121),training_constraints=js(FORMAL),promotion=False,selection_rule='NONE: all preregistered basis trials, invalid columns frozen, qacc informative only')
        if path.name=='COMPLETE.json':
            hashes_ok(commitments); assert len(records)==121
            assert {k:value['ledger'][k] for k in EXPECTED}==EXPECTED
            value=dict(value,source_chain_hashes=dict(value['source_chain_hashes'],**commitments),constraint_samples=121)
        return save_json(path,value)
    def npz_hook(path,**arrays):
        path=Path(path)
        if 'weights' in arrays: assert exact(arrays['weights'],weights)
        if path.name=='TRAJECTORY.npz':
            assert np.array_equal(arrays['time'],np.asarray([r['time'] for r in records]))
            save_npz(path.parent/'CONSTRAINT_ROWS.npz',**recorder.pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json',metadata)
        return save_npz(path,**arrays)
    original.com_contact_record=capture; original.atomic_json=json_hook; original.atomic_npz=npz_hook


def worker(config_path):
    cfg=js(config_path); parent=Path(cfg['parent_output']); out=Path(cfg['trial_output'])
    assert parent.parent==BASE and out==parent/cfg['label'] and not out.exists()
    assert config_path==parent/(cfg['label']+'_CONFIG.json')
    pre=js(parent/'PREREG.json'); assert pre['configs'][cfg['label']]==sha(config_path)
    commitments,baseline=sources(); assert commitments==pre['source_hashes']; hashes_ok(commitments)
    expected=[t for t in TRIALS if t[0]==cfg['label']]; assert len(expected)==1
    _,index,amp=expected[0]
    assert cfg['weight_index']==(list(index) if index is not None else None) and cfg['amplitude']==amp
    weight_path=Path(cfg['weights_path']); assert weight_path==parent/(cfg['label']+'_WEIGHTS.npz')
    assert sha(weight_path)==cfg['weights_sha256']
    weights=npz(weight_path)['weights']; assert exact(weights,basis_weights(baseline,index,amp))
    original=load_private(ROOT/'run_extended_initialization_probe.py','_s136_pristine_s115')
    configure(original,out,weight_path,weights,baseline,commitments,cfg)
    sys.argv=[str(original.__file__),'--output',str(out)]; original.main()


def difference(a,b,atol,rtol):
    assert a.shape==b.shape
    error=float(np.max(np.abs(a-b),initial=0)); scale=max(float(np.max(np.abs(a),initial=0)),float(np.max(np.abs(b),initial=0)),1e-30)
    return dict(max_abs=error,scaled_relative=error/scale,scale=scale,absolute_tolerance=atol,relative_tolerance=rtol,passed=bool(np.isfinite(error) and error<=atol and error/scale<=rtol))


def baseline_check(out):
    a=npz(out/'baseline'/ARM/'TRAJECTORY.npz'); b=npz(S121/ARM/'TRAJECTORY.npz')
    checks={k:bool(k in a and exact(a[k],v)) for k,v in b.items()}
    assert len(checks)==85
    report=dict(passed=all(checks.values()),checks=checks,reference_field_count=len(checks),definition='shape/dtype/bytes identical for all85 historical fields; independent verifier must repeat')
    save(out/'BASELINE_REPRODUCTION.json',report); return report['passed']


def safety_arrays(a,meta):
    limited=np.flatnonzero(meta['jnt_limited']); assert len(limited)==42
    qa=np.asarray(meta['jnt_qposadr'])[limited]; bounds=np.asarray(meta['jnt_range'])[limited]
    q=a['qpos'][:,qa]; lower=q-bounds[:,0]; upper=bounds[:,1]-q
    excess=np.maximum(np.maximum(-lower,-upper),0)
    assert np.array_equal(excess.max(1),a['limit_excess'])
    return dict(qpos=a['qpos'][:,QPOS_COLUMNS],qvel=a['qvel'],load_ratio=a['load_ratio'],limit_excess=a['limit_excess'],signed_joint_position=q,lower_clearance=lower,upper_clearance=upper,root_angle_rad=a['root_angle_rad'],com_vertical_velocity=a['com_velocity_world'][:,2],height=a['qpos'][:,2]/a['qpos'][0,2],qacc=a['qacc'])


def branch_check(a,w,c,a0,w0,c0):
    f=np.concatenate((w['features'],a['features'])); f0=np.concatenate((w0['features'],a0['features']))
    raw=(f*a['weights']).sum(2); raw0=(f0*a0['weights']).sum(2)
    clip=lambda x:np.where(x<=0,-1,np.where(x>=1,1,0))
    result=dict(spikes_identical=bool(exact(w['spikes'],w0['spikes']) and exact(a['spikes'],a0['spikes'])),clip_branch_identical=bool(exact(clip(raw),clip(raw0))),contact_topology_identical=True,efc_state_identical=True)
    for i in range(121):
        prefix=f'sample{i:03d}_'
        for k in ('nefc','ncon','efc_type','efc_id','contact_geom','contact_efc_address','contact_dim','contact_exclude'):
            result['contact_topology_identical'] &= exact(c[prefix+k],c0[prefix+k])
        result['efc_state_identical'] &= exact(c[prefix+'efc_state'],c0[prefix+'efc_state'])
    pooled=np.clip(raw,0,1); mapping=a['actuator_pool']; predicted=np.zeros((166,len(mapping))); mask=mapping>=0
    predicted[:,mask]=pooled[:,mapping[mask]]
    commands=np.concatenate((w['decoded'],a['applied']))
    command_check=difference(commands,predicted,1e-14,1e-12)
    features=dict(identical=exact(f,f0),max_abs_change=float(np.max(np.abs(f-f0))),gating=False,interpretation='continuous features allowed; actual features used per trial')
    return result,features,command_check


def offline(out):
    a0=npz(out/'baseline'/ARM/'TRAJECTORY.npz'); w0=npz(out/'baseline'/ARM/'INITIALIZATION.npz'); c0=npz(out/'baseline'/ARM/'CONSTRAINT_ROWS.npz')
    meta=js(out/'baseline'/ARM/'MOMENT_METADATA.json'); baseline=safety_arrays(a0,meta)
    matrices={k+'_D_'+scale:[] for k in baseline for scale in ('h','half')}
    columns=[]; metrics={}; all_branches={}; features={}; commands={}
    for p,b in WEIGHT_INDICES:
        prefix=f'p{p:03d}_b{b}_'; values={}; branches=[]
        for suffix,amp in SCALES:
            label=prefix+suffix; a=npz(out/label/ARM/'TRAJECTORY.npz'); w=npz(out/label/ARM/'INITIALIZATION.npz'); c=npz(out/label/ARM/'CONSTRAINT_ROWS.npz')
            values[suffix]=safety_arrays(a,meta)
            branch,feature,command=branch_check(a,w,c,a0,w0,c0)
            all_branches[label]=branch; features[label]=feature; commands[label]=command
            branches.append(all(branch.values()) and command['passed'])
            metrics[label]=dict(peak_limit_excess=float(a['limit_excess'].max()),peak_sample=int(a['limit_excess'].argmax()),gating=False)
        checks={}; qacc=None
        for k in baseline:
            dh=(values['plus_h'][k]-values['minus_h'][k])/(2*H)
            half=(values['plus_half'][k]-values['minus_half'][k])/(2*HALF)
            matrices[k+'_D_h'].append(dh); matrices[k+'_D_half'].append(half)
            if k=='qacc': qacc=dict(**difference(dh,half,1e-3,1e-3),gating=False,interpretation='INFORMATIVE_ONLY; no qacc-valid claim; does not revise S135 general-gradient failure')
            else: checks[k]=difference(dh,half,*TOLERANCES[k])
        valid=all(branches) and all(x['passed'] for x in checks.values())
        columns.append(dict(weight_index=[p,b],valid=valid,status='SAFETY_OUTPUT_TWO_SCALE_PASS' if valid else 'INVALID_FREEZE',two_scale_checks=checks,qacc=qacc,branch_and_commands_passed=all(branches)))
    packed={k:np.stack(v,axis=-1) for k,v in matrices.items()}
    valid=np.asarray([c['valid'] for c in columns],dtype=bool)
    packed.update(weight_indices=np.asarray(WEIGHT_INDICES),valid=valid,reference_weights=a0['weights'],qpos_columns=np.asarray(QPOS_COLUMNS),time=a0['time'],h=np.asarray(H),half=np.asarray(HALF),actuator_pool=a0['actuator_pool'])
    limited=np.flatnonzero(meta['jnt_limited']); packed.update(limited_joint_indices=limited,limited_qpos_addresses=np.asarray(meta['jnt_qposadr'])[limited],joint_bounds=np.asarray(meta['jnt_range'])[limited])
    for k,v in baseline.items():
        packed['reference_'+k]=v
        packed[k]=packed[k+'_D_half']
        packed['masked_'+k]=np.where(valid,packed[k],0.)
    np.savez_compressed(out/'JACOBIAN.npz',**packed)
    with (out/'COLUMN_VALIDITY.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['column','pool','bin','valid','status','branch_and_commands_passed','failed_continuous_outputs'])
        for i,c in enumerate(columns):
            writer.writerow([i,*c['weight_index'],c['valid'],c['status'],c['branch_and_commands_passed'],';'.join(k for k,v in c['two_scale_checks'].items() if not v['passed'])])
    save(out/'DERIVATIVE_REPORT.json',dict(task='S136/P1',columns=columns,valid_columns=valid.tolist(),valid_count=int(valid.sum()),branches=all_branches,features=features,command_checks=commands,trial_peak_metrics=metrics,qacc_gate='INFORMATIVE_ONLY: distinct new safety-output task; old S135 general qacc failure unchanged',peak_scalar_gate='INFORMATIVE_ONLY; full121x42 signed lower/upper clearances gate; active max switches not gated',invalid_policy='freeze invalid coordinates via masked matrix, not evidence of no perturbation; all raw trials and raw derivatives retained',safety_admission=False,promotion=False,independent_verification_required=True,height_definition='dimensionless root height ratio qpos[:,2]/qpos[0,2], matches original gate',com_vertical_velocity_definition='saved com_velocity_world z',signed_joint_position_definition='unclipped actual limited-joint qpos radians; lower=q-lower_bound, upper=upper_bound-q'))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path); parser.add_argument('--trial',type=Path,help=argparse.SUPPRESS); args=parser.parse_args()
    if args.trial:
        assert args.output is None; worker(args.trial); return
    now=datetime.now(timezone.utc); out=args.output or BASE/now.strftime('%Y%m%dT%H%M%SZ')
    assert out.is_absolute() and out.parent==BASE and not out.exists()
    stamp=datetime.strptime(out.name,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc); assert abs((now-stamp).total_seconds())<120
    commitments,baseline=sources(); out.mkdir(parents=True,exist_ok=False); configs={}
    for label,index,amp in TRIALS:
        weights=basis_weights(baseline,index,amp); path=out/(label+'_WEIGHTS.npz'); np.savez_compressed(path,weights=weights)
        cfg=out/(label+'_CONFIG.json'); save(cfg,dict(parent_output=str(out),trial_output=str(out/label),label=label,weight_index=list(index) if index is not None else None,amplitude=amp,weights_path=str(path),weights_sha256=sha(path))); configs[label]=sha(cfg)
    save(out/'INPUT_HASHES.json',commitments)
    save(out/'PREREG.json',dict(task='S136/P1 safety-output basis trajectory Jacobian',created_utc=now.isoformat(),source_hashes=commitments,configs=configs,trials=TRIALS,weight_indices=WEIGHT_INDICES,max_trials=61,retry=False,autoscan=False,h=H,half=HALF,amplitude_definition='absolute per single weight, basis coordinate; not S132 direction',training_constraints=js(FORMAL),expected_per_trial=EXPECTED,expected_total={k:61*v for k,v in EXPECTED.items()},baseline_gate='all85 historical saved fields shape dtype bytes identical, FAIL_STOP before other60',two_scale_tolerances=TOLERANCES,relative_definition='max abs D_h-D_half / max(global max abs D_h,global max abs D_half,1e-30); absolute AND relative',discrete_gate=['spikes','translator_clip','contact_topology','efc_state'],features='continuous actual features permitted and reported',qacc='INFORMATIVE_ONLY; new safety task does not override S135 general-gradient failure',peak_scalar='INFORMATIVE_ONLY; full signed121x42 bounds are training constraints',invalid_columns='freeze masked matrix; retain all raw evidence',PRE=160,physical_ms=3,physical_steps=120,samples=121,neural_ticks_physical=6,sensory_hold_ms=5,sensory_scope='tick0 only within3ms; no rich-feedback adaptation claim',checkpoint_replacement=False,promotion=False,weight_optimization=False))
    ledger=dict(planned_trials=61,started=[],finished=[],actual_totals={k:0 for k in EXPECTED},trial_ledgers={}); save(out/'CALL_LEDGER.json',ledger)
    try:
        for label,_,_ in TRIALS:
            hashes_ok(commitments); ledger['started'].append(label); save(out/'CALL_LEDGER.json',ledger)
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--trial',str(out/(label+'_CONFIG.json'))]
            with (out/(label+'_STDOUT.log')).open('w') as stdout,(out/(label+'_STDERR.log')).open('w') as stderr:
                result=subprocess.run(command,cwd=str(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),stdout=stdout,stderr=stderr)
            child=out/label/'CALL_LEDGER.json'
            if child.exists(): ledger['trial_ledgers'][label]=js(child)
            ledger['actual_totals']={k:sum(x.get(k,0) for x in ledger['trial_ledgers'].values()) for k in EXPECTED}
            ledger['finished'].append(dict(label=label,exit_code=result.returncode,command=command)); save(out/'CALL_LEDGER.json',ledger)
            if result.returncode: raise RuntimeError('trial failed, no retry: '+label)
            assert {k:ledger['trial_ledgers'][label][k] for k in EXPECTED}==EXPECTED
            if label=='baseline' and not baseline_check(out): raise RuntimeError('baseline bit mismatch; other60 not run')
        offline(out); hashes_ok(commitments)
        save(out/'COMPLETE.json',dict(status='COMPLETE',ledger=ledger,valid_columns=js(out/'DERIVATIVE_REPORT.json')['valid_columns'],independent_verification='PENDING',promotion=False))
    except BaseException:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,promotion=False)); raise
    finally:
        save(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_HASHES.json'})
    print(str(out))


if __name__=='__main__': main()
