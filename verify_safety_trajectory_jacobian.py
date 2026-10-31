"""S136 independent OFFLINE safety-output trajectory Jacobian verifier.
Never imports a producer, simulator, torch or neural runtime. No new trials.
Evidence integrity is separate from per-column validity (zero valid is allowed).
"""
from pathlib import Path
import argparse
import ast
import csv
import json
import sys
import traceback
import numpy as np
from verify_refined_trajectory_direction_probe import (
    Checks, canonical, bound_hashes, committed, npz, js, sha, serial,
    identical, compare_saved, fd_gate, audit_source, trial_path, trial,
    branch_report, NPZS, JSONS, mapped)
from verify_joint_candidate_probe import reconstruct_rows

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'run_safety_trajectory_jacobian.py'
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
COUNTS = dict(model_load=1, forward=121, step=120, neural_ticks=166)
INDICES = np.array([(p,b) for p in (3,15,38) for b in range(5)])
AMPLITUDES = dict(plus_h=.000125,minus_h=-.000125,plus_half=.0000625,minus_half=-.0000625)
GATES = dict(qpos=(1e-5,.001),qvel=(1e-5,.001),load_ratio=(1e-4,.001),
    limit_excess=(1e-6,.001),root_angle_rad=(1e-5,.001),
    com_vertical_velocity=(1e-5,.001),height=(1e-5,.001),
    signed_joint_position=(1e-5,.001),lower_clearance=(1e-5,.001),upper_clearance=(1e-5,.001))


def plan():
    return [('baseline',None,0.)]+[(f'p{p:03d}_b{b}_{s}',(p,b),a)
        for p,b in INDICES for s,a in AMPLITUDES.items()]


def safety_curves(t):
    """All original 121 states, 45 nonquaternion qpos, all42 joint positions."""
    a=t['item']['arrays']; m=t['item']['moment']
    limited=np.flatnonzero(m['jnt_limited'])
    if len(limited)!=42: raise ValueError('must contain exactly 42 limited joints')
    qi=np.asarray(m['jnt_qposadr'])[limited]; bounds=np.asarray(m['jnt_range'])[limited]
    q=a['qpos'][:,qi]; lo=bounds[:,0]; hi=bounds[:,1]
    quat=a['qpos'][:,3:7]; initial=quat[0]
    angle=2*np.arccos(np.clip(np.abs(quat@initial)/(np.linalg.norm(quat,axis=1)*np.linalg.norm(initial)),0,1))
    return dict(qpos=a['qpos'][:,np.r_[0:3,7:49]],qvel=a['qvel'],qacc=a['qacc'],
        load_ratio=a['load_ratio'],signed_joint_position=q,lower_clearance=q-lo,
        upper_clearance=hi-q,limit_excess=np.maximum(np.maximum(lo-q,q-hi),0).max(1),
        root_angle_rad=angle,com_vertical_velocity=a['com_velocity_world'][:,2],
        height=a['qpos'][:,2]/a['qpos'][0,2])


def independent_branches(base,t):
    discrete=('nefc','ncon','efc_type','efc_id','contact_efc_address','contact_dim','contact_exclude','contact_geom')
    def same_rows(keys):
        return all(identical(base['rows'][f'sample{i:03d}_{k}'],t['rows'][f'sample{i:03d}_{k}']) for i in range(121) for k in keys)
    spikes=[]; features=[]; clips=[]
    for stage in ('init','runtime'):
        a=base['init'] if stage=='init' else base['item']['arrays']
        b=t['init'] if stage=='init' else t['item']['arrays']
        spikes.append(identical(a['spikes'],b['spikes'])); features.append(identical(a['features'],b['features']))
        def clip(x):
            raw=(x['features']*x['weights']).sum(2)
            return np.where(raw<=0,-1,np.where(raw>=1,1,0))
        clips.append(identical(clip(a),clip(b)))
    return dict(spikes_identical=all(spikes),FIR_features_identical=all(features),
        contact_rows_identical=same_rows(discrete),efc_active_state_identical=same_rows(('efc_state',)),
        clip_branch_identical=all(clips))


def independent_column(curves):
    raw={}; gates={}
    for key in curves['plus_h']:
        h=(curves['plus_h'][key]-curves['minus_h'][key])/.00025
        half=(curves['plus_half'][key]-curves['minus_half'][key])/.000125
        raw[key]=(h,half)
        if key in GATES: gates[key]=fd_gate(h,half,*GATES[key])
    # all42 signed-position/clearance validity is inherited from all45 qpos,
    # never from the nonsmooth maximum-over-joints or selected-time peak.
    return raw,gates


def verify_output_manifest(out,checks):
    manifest=js(out/'OUTPUT_HASHES.json'); canonical(manifest)
    bound_hashes(checks,manifest,'producer output manifest')
    # Producer cannot commit its own future manifest bytes or parent-written reports.
    # Require every original producer artifact, but allow later father/verifier reports.
    required=('PREREG.json','INPUT_HASHES.json','COMPLETE.json','CALL_LEDGER.json',
        'JACOBIAN.npz','DERIVATIVE_REPORT.json','COLUMN_VALIDITY.csv','BASELINE_REPRODUCTION.json')
    for name in required: committed(checks,manifest,out/name,'required producer output')
    for label,_,_ in plan():
        for suffix in ('_CONFIG.json','_WEIGHTS.npz'):
            committed(checks,manifest,out/(label+suffix),'required producer trial input')
        path=trial_path(out,label)
        for suffix in ('_STDOUT.log','_STDERR.log'):
            committed(checks,manifest,out/(label+suffix),'required producer process log')
        for name in (*NPZS,*JSONS,'RESULT.json','CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'):
            committed(checks,manifest,path/name,'complete producer child output')
        for name in ('CALL_LEDGER.json','COMPLETE.json','PREREG.json','INPUT_HASHES.json','RESULTS.json','MODEL_DEPENDENCIES.json'):
            committed(checks,manifest,out/label/name,'complete producer child ledger')
    original=set(canonical(manifest)); actual={str(p.resolve()) for p in out.rglob('*') if p.is_file()}
    extras=sorted(actual-original-{str((out/'OUTPUT_HASHES.json').resolve())})
    return dict(extra_files_not_in_original_manifest=extras,
        semantics='Additional parent/verifier reports do not invalidate the original manifest; all original producer artifacts remain hash-checked.')


def inspect_trial(out,label,index,amplitude,checks,reference,w,hashes,pools,mapping,xml,pre,ledger):
    weight=w.copy()
    if index is not None: weight[index]+=amplitude
    checks.add(label+' bounds/masks/frozen pool36',np.isfinite(weight).all() and
        np.all((weight>=0)&(weight<=4)) and identical(weight==0,w==0) and identical(weight[36],w[36]))
    cfgpath=out/(label+'_CONFIG.json'); cfg=js(cfgpath)
    checks.add(label+' frozen config',pre['configs'][label]==sha(cfgpath) and cfg['label']==label and cfg['weight_index']==(list(index) if index is not None else None) and cfg['amplitude']==amplitude and cfg['weights_sha256']==sha(cfg['weights_path']))
    checks.add(label+' all330 config weights',identical(npz(cfg['weights_path'])['weights'],weight))
    childledger=js(out/label/'CALL_LEDGER.json')
    checks.add(label+' child ledger',childledger==dict(COUNTS,active=None,completed=['init80ms'],per_arm={'init80ms':COUNTS}) and ledger['trial_ledgers'][label]==childledger)
    checks.add(label+' child complete',js(out/label/'COMPLETE.json')['status']=='COMPLETE' and not (out/label/'FAILED.json').exists())
    t=trial(checks,trial_path(out,label),label,reference,weight,hashes,pools,mapping,xml)
    a=t['item']['arrays']; features=np.concatenate((t['init']['features'],a['features']))
    commands=np.concatenate((t['init']['decoded'],a['applied']))
    checks.eq(label+' all166 analytical commands actual masks weights',commands,mapped(features,weight,a['actuator_pool']),1e-14)
    t['curves']=safety_curves(t)
    return t


def compare_gate(checks,label,claim,g):
    expected=dict(max_abs=g['max_abs'],scaled_relative=g['scaled_relative'],scale=g['scale'],
        absolute_tolerance=g['absolute_bound'],relative_tolerance=g['relative_bound'],passed=g['passed'])
    checks.add(label,all(claim[k]==v for k,v in expected.items()))


def verify(out,checks,report):
    checks.add('offline initial boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    pre=js(out/'PREREG.json'); hashes=js(out/'INPUT_HASHES.json'); canonical(hashes)
    checks.add('preregistered full hash inputs',pre['source_hashes']==hashes)
    bound_hashes(checks,hashes,'full literal input hashes')
    for p in (SOURCE,ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py',REFERENCE/'RESULT.json'):
        committed(checks,hashes,p,'canonical required input')
    report['source_audit']=audit_source(checks,SOURCE)
    expected_plan=[[label,list(index) if index is not None else None,amp] for label,index,amp in plan()]
    checks.add('exact new preregistration no relaxation',pre['trials']==expected_plan and
        pre['weight_indices']==INDICES.tolist() and pre['h']==.000125 and pre['half']==.0000625 and
        pre['PRE']==160 and pre['samples']==121 and pre['max_trials']==61 and
        pre['two_scale_tolerances']=={k:list(v) for k,v in GATES.items()} and
        pre['promotion'] is False and pre['weight_optimization'] is False and
        pre['qacc'].startswith('INFORMATIVE_ONLY') and pre['peak_scalar'].startswith('INFORMATIVE_ONLY'))
    ledger=js(out/'CALL_LEDGER.json'); labels=[x[0] for x in plan()]
    checks.add('61 genuine successful child calls ledger',ledger['planned_trials']==61 and ledger['started']==labels and
        [r['label'] for r in ledger['finished']]==labels and all(r['exit_code']==0 for r in ledger['finished']) and
        ledger['actual_totals']=={k:61*v for k,v in COUNTS.items()})
    checks.add('complete no failure',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    report['output_manifest']=verify_output_manifest(out,checks)
    bound_hashes(checks,js(REFERENCE/'RESULT.json')['hashes'],'original S121')
    ref=reconstruct_rows(checks,REFERENCE,'original S121 full native closure'); w=ref['arrays']['weights']
    manifests={str(Path(p).resolve()) for p in hashes if Path(p).name=='MANIFEST.json'}
    checks.add('unique actual manifest',len(manifests)==1)
    manifest=sorted(js(Path(next(iter(manifests))))['rows'],key=lambda r:r['actuator_index'])
    pools=sorted({tuple(sorted(r['local_indices'])) for r in manifest if r['local_indices']})
    lookup={p:i for i,p in enumerate(pools)}; mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in manifest])
    xmls={str(Path(p).resolve()) for p in hashes if Path(p).suffix=='.xml'}
    checks.add('unique full model XML',len(xmls)==1); xml=Path(next(iter(xmls)))
    baseline=inspect_trial(out,'baseline',None,0.,checks,ref,w,hashes,pools,mapping,xml,pre,ledger)
    for name in NPZS: compare_saved(checks,npz(REFERENCE/name),npz(baseline['path']/name),'baseline exact '+name)
    for name in JSONS: checks.add('baseline exact '+name,js(REFERENCE/name)==js(baseline['path']/name))
    compare_saved(checks,npz(REFERENCE/'CONSTRAINT_ROWS.npz'),baseline['rows'],'baseline original native rows')
    checks.add('baseline reproduction report',js(out/'BASELINE_REPRODUCTION.json')['passed'])
    produced=npz(out/'JACOBIAN.npz'); claimed=js(out/'DERIVATIVE_REPORT.json'); matrices={}; columns=[]
    for j,(p,b) in enumerate(INDICES):
        curves={}; branches=[]
        for suffix,amp in AMPLITUDES.items():
            label=f'p{p:03d}_b{b}_{suffix}'
            t=inspect_trial(out,label,(p,b),amp,checks,ref,w,hashes,pools,mapping,xml,pre,ledger)
            curves[suffix]=t['curves']; branch=independent_branches(baseline,t)
            br=dict(spikes_identical=branch['spikes_identical'],clip_branch_identical=branch['clip_branch_identical'],
                contact_topology_identical=branch['contact_rows_identical'],efc_state_identical=branch['efc_active_state_identical'])
            checks.add(label+' exact branch flags',claimed['branches'][label]==br)
            a=t['item']['arrays']; f=np.concatenate((t['init']['features'],a['features']))
            f0=np.concatenate((baseline['init']['features'],baseline['item']['arrays']['features']))
            fc=claimed['features'][label]
            checks.add(label+' actual features reported',fc['identical']==identical(f,f0) and fc['max_abs_change']==float(np.max(np.abs(f-f0))) and fc['gating'] is False)
            commands=np.concatenate((t['init']['decoded'],a['applied']))
            cg=fd_gate(commands,mapped(f,a['weights'],a['actuator_pool']),1e-14,1e-12)
            compare_gate(checks,label+' command gate',claimed['command_checks'][label],cg)
            branches.append(all(br.values()) and cg['passed'])
            peak=claimed['trial_peak_metrics'][label]
            checks.add(label+' peak informational only',peak==dict(peak_limit_excess=float(a['limit_excess'].max()),peak_sample=int(a['limit_excess'].argmax()),gating=False))
            checks.eq(label+' independently recomputed root angle',a['root_angle_rad'],curves[suffix]['root_angle_rad'],0.)
        raw,gates=independent_column(curves)
        for key,(dh,half) in raw.items():
            matrices.setdefault(key+'_D_h',[]).append(dh); matrices.setdefault(key+'_D_half',[]).append(half)
        c=claimed['columns'][j]
        for key,g in gates.items(): compare_gate(checks,f'column{j} {key} full121 gate',c['two_scale_checks'][key],g)
        qg=fd_gate(*raw['qacc'],1e-3,1e-3); compare_gate(checks,f'column{j} informational qacc',c['qacc'],qg)
        checks.add(f'column{j} qacc not gate',c['qacc']['gating'] is False)
        valid=all(branches) and all(g['passed'] for g in gates.values())
        checks.add(f'column{j} iff all full continuous and all4 branches',c['weight_index']==[int(p),int(b)] and c['valid']==valid and
            c['branch_and_commands_passed']==all(branches) and c['status']==('SAFETY_OUTPUT_TWO_SCALE_PASS' if valid else 'INVALID_FREEZE'))
        columns.append(dict(weight_index=[int(p),int(b)],valid=valid,gates=gates,branches_passed=all(branches),qacc_informational=qg))
    with (out/'COLUMN_VALIDITY.csv').open(newline='') as stream: savedcsv=list(csv.reader(stream))
    expectedcsv=[['column','pool','bin','valid','status','branch_and_commands_passed','failed_continuous_outputs']]
    for j,c in enumerate(columns):
        expectedcsv.append([str(j),*[str(x) for x in c['weight_index']],str(c['valid']),
            'SAFETY_OUTPUT_TWO_SCALE_PASS' if c['valid'] else 'INVALID_FREEZE',str(c['branches_passed']),
            ';'.join(k for k in claimed['columns'][j]['two_scale_checks'] if not c['gates'][k]['passed'])])
    checks.add('all column validity CSV fields independently matched',savedcsv==expectedcsv)
    valid=np.array([c['valid'] for c in columns],dtype=bool)
    for key,values in matrices.items():
        checks.add('independent complete matrix '+key,identical(produced[key],np.stack(values,axis=-1)))
    for key,curve in baseline['curves'].items():
        half=np.stack(matrices[key+'_D_half'],axis=-1)
        checks.add('raw half alias '+key,identical(produced[key],half))
        checks.add('invalid columns frozen only in masked '+key,identical(produced['masked_'+key],np.where(valid,half,0.)))
        checks.add('original reference curve '+key,identical(produced['reference_'+key],curve))
    m=ref['moment']; limited=np.flatnonzero(m['jnt_limited'])
    metadata=dict(valid=valid,weight_indices=INDICES,reference_weights=w,qpos_columns=np.r_[0:3,7:49],time=ref['arrays']['time'],
        h=np.asarray(.000125),half=np.asarray(.0000625),actuator_pool=mapping,limited_joint_indices=limited,
        limited_qpos_addresses=np.asarray(m['jnt_qposadr'])[limited],joint_bounds=np.asarray(m['jnt_range'])[limited])
    for key,value in metadata.items(): checks.add('exact Jacobian metadata '+key,identical(produced[key],value))
    checks.add('valid flags/count distinct from evidence',claimed['valid_columns']==valid.tolist() and claimed['valid_count']==int(valid.sum()) and js(out/'COMPLETE.json')['valid_columns']==valid.tolist())
    checks.add('no admission/promotion or global gradient claim',claimed['safety_admission'] is False and claimed['promotion'] is False and claimed['qacc_gate'].startswith('INFORMATIVE_ONLY') and claimed['peak_scalar_gate'].startswith('INFORMATIVE_ONLY'))
    report.update(columns=columns,valid_columns=valid.tolist(),valid_count=int(valid.sum()),
        safety_output_scope_only=True,general_gradient_verified=False,qacc_gating=False,peak_scalar_gating=False,
        invalid_policy='freeze masked Jacobian columns; retain genuine raw trials and raw half/h derivatives',
        baseline_promotion=False,behavior_acceptance=False,physical_units='NOT_ESTABLISHED',actual_calls=ledger['actual_totals'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv); out=args.output.resolve(); target=(args.report or out/'SAFETY_TRAJECTORY_JACOBIAN_INDEPENDENT_CHECK.json').resolve()
    if target.exists(): raise ValueError('never overwrite existing report: '+str(target))
    checks=Checks(); report=dict(passed=False,evidence_passed=False,valid_count=None,general_gradient_verified=False,
        verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0))
    try:
        verify(out,checks,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=report['evidence_passed']=checks.passed
    except Exception as exc: report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['checks']=checks.rows
    report['helper_source_hashes']={str(Path(m.__file__).resolve()):sha(m.__file__) for n,m in tuple(sys.modules.items()) if n.startswith('verify_') and getattr(m,'__file__',None)}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],evidence_passed=report['evidence_passed'],valid_count=report['valid_count'],report=str(target),error=report.get('error'))))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
