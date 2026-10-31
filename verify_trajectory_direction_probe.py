"""S134 independent OFFLINE trajectory directional-FD evidence verification.
No runner imports, model loads, simulator execution, neural execution or new trials.
Evidence integrity and derivative admissibility are deliberately separate.
"""
from pathlib import Path
import argparse
import ast
import json
import sys
import traceback
import numpy as np
from verify_joint_candidate_probe import (Checks, canonical, bound_hashes, committed,
    reconstruct_rows, exact, npz, js, sha, serial, require_integrity,
    reconstruct_fir, mapped, NPZS, JSONS, physical_metrics, all42_limits)

ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
S132=Path('/home/sensen/Desktop/engine修复验收/s132-margin-contact-candidate/20261024T180846Z')
LABELS=('baseline','plus_h','minus_h','plus_half','minus_half')
AMPLITUDES=dict(baseline=0.,plus_h=.0005,minus_h=-.0005,plus_half=.00025,minus_half=-.00025)
GATES=dict(qacc=(1e-3,1e-3),qvel=(1e-5,1e-3),qpos=(1e-5,1e-3),load_ratio=(1e-4,1e-3),limit_excess=(1e-6,1e-3))
COUNTS=dict(model_load=1,forward=121,step=120,neural_ticks=166)


def identical(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def compare_saved(checks,a,b,label):
    checks.add(label+' all original keys retained',set(a)<=set(b))
    for key in sorted(set(a)&set(b)):
        checks.add(label+' '+key,identical(a[key],b[key]))


def fd_gate(a,b,absolute,relative):
    error=float(np.max(np.abs(a-b)))
    scale=max(float(np.max(np.abs(a))),float(np.max(np.abs(b))),1e-30)
    return dict(passed=bool(error<=absolute and error/scale<=relative),max_abs=error,
        scale=scale,scaled_relative=error/scale,absolute_bound=absolute,
        relative_bound=relative,operator='AND')


def audit_source(checks,path):
    tree=ast.parse(path.read_text()); calls=[]; writes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=ast.unparse(node.func)
            if name.endswith(('mj_forward','mj_step','from_xml_path','mj_resetData')) or name=='net.step': calls.append(name)
        targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign)) else []
        for target in targets:
            text=ast.unparse(target)
            if text.startswith(('d.','m.','net.')) or 'qpos[' in text: writes.append(ast.unparse(node))
    checks.add('producer delegates physics and has no explicit added physics/state writes',not calls and not writes,calls=calls,writes=writes)
    return dict(path=str(path),sha256=sha(path),calls=calls,writes=writes,
        limitation='Static hash-bound audit and saved ledger are not proof against unrecorded runtime events.')


def trial_path(out,label):
    base=out/label
    return base/'init80ms' if (base/'init80ms/TRAJECTORY.npz').is_file() else base


def trial(checks,path,label,reference,weight,hashes,pools,mapping,xml):
    result=js(path/'RESULT.json')
    bound_hashes(checks,result['hashes'],label+' artifacts')
    for name in (*NPZS,*JSONS,'CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'):
        committed(checks,result['hashes'],path/name,label+' required')
    checks.add(label+' exact actual call count',result['calls']==COUNTS)
    item=reconstruct_rows(checks,path,label); a=item['arrays']; init=npz(path/'INITIALIZATION.npz')
    checks.add(label+' recognized complete row partition',item['counts']['unrecognized']==0 and item['counts']['other']==0)
    checks.add(label+' all121 physical clock',identical(a['time'],reference['arrays']['time']) and np.allclose(a['time'],np.arange(121)*.000025,rtol=0,atol=1e-12))
    checks.add(label+' same model joint arrays',item['moment']==reference['moment'])
    for stage,z in [('runtime',a),('PRE',init)]: checks.add(label+' '+stage+' all330 exact weights',identical(z['weights'],weight))
    checks.add(label+' PRE160+6 completeness',init['features'].shape==(160,66,5) and init['decoded'].shape==(160,90) and a['features'].shape==(6,66,5) and a['applied'].shape==(6,90))
    checks.add(label+' actual pools',identical(a['actuator_pool'],mapping))
    fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,weight)
    _,_,warmhist=reconstruct_fir(init['spikes'],pools,mapping,weight)
    for name,x,y in [('PRE features',init['features'],fs[:160]),('PRE decoded',init['decoded'],us[:160]),('release features',init['release_features'],fs[159]),('release decoded',init['release_decoded'],us[159]),('PRE FIR',init['history'],warmhist),('runtime features',a['features'],fs[160:]),('runtime decoded',a['applied'],us[160:]),('final FIR',npz(path/'NEURAL_FINAL.npz')['FIR_history'],hist)]: checks.eq(label+' independent FIR '+name,x,y,1e-14)
    checks.eq(label+' physical hold',a['ctrl'][1:],np.repeat(a['applied'],20,axis=0),1e-14)
    checks.eq(label+' fixed current',a['current'],np.repeat(init['current'][None],6,axis=0))
    warm=npz(REFERENCE/'INITIALIZATION.npz')
    for key in ('body_qpos','body_qvel','body_act','body_ctrl','current'): checks.add(label+' original release '+key,identical(init[key],warm[key]))
    for key in ('qpos','qvel','act','ctrl'): checks.add(label+' original physical initial '+key,identical(a[key][0],reference['arrays'][key][0]))
    checks.add(label+' no applied hidden force or warnings',not a['qfrc_applied'].any() and not a['xfrc_applied'].any() and not a['warnings'].any())
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'): compare_saved(checks,npz(REFERENCE/name),npz(path/name),label+' original '+name)
    for phase in ('before','after'): checks.add(label+' all numeric model hashes '+phase,js(path/'MODEL_PROVENANCE.json')[phase]==js(REFERENCE/'MODEL_PROVENANCE.json')[phase])
    for name,ticks in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',160),('NEURAL_FINAL_META.json',166)]:
        scalars=js(path/name)['neural_scalars']; checks.add(label+' '+name+' clock',scalars['steps']==ticks and abs(scalars['t_ms']-.5*ticks)<=1e-10)
    schedule=js(path/'SCHEDULE.json'); checks.add(label+' six consumed intervals',len(schedule)==6)
    for tick,row in enumerate(schedule):
        checks.add(label+' consume '+str(tick),row['tick']==tick and row['consume_id']==160+tick)
        for key,value in dict(source_start_s=(160+tick)*.0005,source_end_s=(161+tick)*.0005,physical_start_s=tick*.0005,physical_end_s=(tick+1)*.0005,source_offset_s=.08).items(): checks.eq(label+' schedule '+str(tick)+' '+key,row[key],value,1e-12)
    physical,rec,native=physical_metrics(path,xml,checks,label)
    j=item['moment']['joint_names'].index('joint_LFCoxa_roll'); qi=item['moment']['jnt_qposadr'][j]
    scalars=dict(peak_limit=float(a['limit_excess'].max()),sampled_qroll113=float(a['qpos'][113,qi]),loadmin=float(a['load_ratio'].min()),support_discrete=float(np.trapezoid(physical['gates'].all(1).astype(float),x=a['time'])))
    return dict(item=item,init=init,physical=physical,native=native,scalars=scalars,rows=npz(path/'CONSTRAINT_ROWS.npz'),path=path)


def branch_report(trials):
    base=trials['baseline']; result={}
    discrete=('nefc','ncon','efc_type','efc_id','contact_efc_address','contact_dim','contact_exclude','contact_geom')
    for label in LABELS:
        other=trials[label]; topology=[]; raw=[]
        for i in range(121):
            prefix=f'sample{i:03d}_'; keys=[k for k in base['rows'] if k.startswith(prefix)]
            topology.append(dict(sample=i,same=all(identical(base['rows'][prefix+k],other['rows'][prefix+k]) for k in discrete)))
            raw.append(dict(sample=i,same=all(k in other['rows'] and identical(base['rows'][k],other['rows'][k]) for k in keys),differing_keys=[k[len(prefix):] for k in keys if k not in other['rows'] or not identical(base['rows'][k],other['rows'][k])]))
        neural={}
        for stage,key in [('init','spikes'),('init','features'),('runtime','spikes'),('runtime','features')]:
            a=base['init'] if stage=='init' else base['item']['arrays']; b=other['init'] if stage=='init' else other['item']['arrays']
            neural[stage+'_'+key]=identical(a[key],b[key])
        clip={}
        for stage in ('init','runtime'):
            def branch(t):
                a=t['init'] if stage=='init' else t['item']['arrays']; x=(a['features']*a['weights']).sum(2); return np.where(x<=0,-1,np.where(x>=1,1,0))
            clip[stage]=identical(branch(base),branch(other))
        result[label]=dict(topology_per_sample=topology,raw_exact_per_sample=raw,neural_exact=neural,clip_branch_exact=clip,
            topology_all=all(r['same'] for r in topology),raw_all=all(r['same'] for r in raw))
    return result


def independently_differentiate(trials):
    curves={k:{key:t['item']['arrays'][key] for key in GATES} for k,t in trials.items()}
    fd={}; gates={}; scalar_fd={}
    for key,(absolute,relative) in GATES.items():
        dh=(curves['plus_h'][key]-curves['minus_h'][key])/.001
        df=(curves['plus_half'][key]-curves['minus_half'][key])/.0005
        fd[key+'_h']=dh; fd[key+'_half']=df
        cols=np.r_[0:3,7:49] if key=='qpos' else None
        gates[key]=fd_gate(dh[:,cols] if cols is not None else dh,df[:,cols] if cols is not None else df,absolute,relative)
    scalar_names=('peak_limit','sampled_qroll113','loadmin','support_discrete')
    for key in scalar_names:
        h=(trials['plus_h']['scalars'][key]-trials['minus_h']['scalars'][key])/.001
        half=(trials['plus_half']['scalars'][key]-trials['minus_half']['scalars'][key])/.0005
        scalar_fd[key]=dict(h=h,half=half)
    return fd,gates,scalar_fd


def verify(out,checks,report):
    checks.add('offline initial boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    pre=js(out/'PREREG.json'); hashes=js(out/'INPUT_HASHES.json'); canonical(hashes)
    checks.add('preregistered full inputs',pre['source_hashes']==hashes)
    report['input_hashes']=bound_hashes(checks,hashes,'input')
    for path in (ROOT/'run_trajectory_direction_probe.py',ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py',REFERENCE/'RESULT.json',S132/'OFFLINE_CANDIDATE.npz'):
        committed(checks,hashes,path,'required source')
    report['source_audit']=audit_source(checks,ROOT/'run_trajectory_direction_probe.py')
    bound_hashes(checks,js(REFERENCE/'RESULT.json')['hashes'],'original S121')
    reference=reconstruct_rows(checks,REFERENCE,'S121 independent original')
    chain=npz(S132/'OFFLINE_CANDIDATE.npz'); w=reference['arrays']['weights']; delta=chain['delta']; v=delta/np.linalg.norm(delta)
    checks.add('S132 exact reference and stored delta',identical(chain['reference_weights'],w) and identical(chain['proposed_weights'],w+delta))
    mask=np.zeros_like(w,dtype=bool); mask[[3,15,38],:]=True
    checks.add('normalized direction and frozen pools',abs(float(np.linalg.norm(v))-1)<1e-14 and not v[~mask].any() and not v[36].any())
    manifests={str(Path(p).resolve()) for p in hashes if Path(p).name=='MANIFEST.json'}; checks.add('unique actual manifest',len(manifests)==1)
    manifest=sorted(js(Path(next(iter(manifests))))['rows'],key=lambda r:r['actuator_index']); pools=sorted({tuple(sorted(r['local_indices'])) for r in manifest if r['local_indices']}); lookup={p:i for i,p in enumerate(pools)}; mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in manifest])
    xmls={str(Path(p).resolve()) for p in hashes if Path(p).suffix=='.xml'}; checks.add('unique actual XML',len(xmls)==1); xml=Path(next(iter(xmls)))
    trials={}
    for label in LABELS:
        weight=w+AMPLITUDES[label]*v
        checks.add(label+' finite bounds masks pools',np.isfinite(weight).all() and np.all((weight>=0)&(weight<=4)) and identical(weight==0,w==0) and identical(weight[~mask],w[~mask]) and identical(weight[36],w[36]))
        trials[label]=trial(checks,trial_path(out,label),label,reference,weight,hashes,pools,mapping,xml)
    baseline=trials['baseline']
    for name in NPZS: compare_saved(checks,npz(REFERENCE/name),npz(baseline['path']/name),'fresh reference full exact '+name)
    for name in JSONS: checks.add('fresh reference exact JSON '+name,js(REFERENCE/name)==js(baseline['path']/name))
    compare_saved(checks,npz(REFERENCE/'CONSTRAINT_ROWS.npz'),baseline['rows'],'fresh reference all row arrays')
    fd,gates,scalars=independently_differentiate(trials); branch=branch_report(trials)
    verify_producer_analysis(out,checks,pre,trials,fd,gates,scalars,branch,report)
    report.update(two_scale_gates=gates,scalar_FD=scalars,branch_stability=branch,
        per_trial={k:dict(path=str(t['path']),scalars=t['scalars'],native=t['native']) for k,t in trials.items()},
        continuous_two_scale_valid=all(r['passed'] for r in gates.values()),
        discrete_events_verified=report['producer_classification']['branch_stable'],
        raw_FD=fd,derivative_valid=report['producer_classification']['verified_gradient'],
        classification='Continuous two-scale evidence does not require every-time branch identity; discrete event differentiation not verified across branch changes.',
        baseline_promotion=False,behavior_acceptance=False,physical_units='NOT_ESTABLISHED')
    return trials,fd


def verify_producer_analysis(out,checks,pre,trials,fd,gates,scalars,branch,report):
    checks.add('exact preregistered thresholds no relaxation',pre['two_scale_tolerances']=={k:list(v) for k,v in GATES.items()} and pre['h']==.0005 and pre['half']==.00025 and pre['PRE']==160 and pre['samples']==121 and pre['max_trials']==5 and pre['trials']==[[k,AMPLITUDES[k]] for k in LABELS])
    ledger=js(out/'CALL_LEDGER.json')
    checks.add('five and only five successful actual trials',ledger['planned_trials']==5 and ledger['started']==list(LABELS) and [r['label'] for r in ledger['finished']]==list(LABELS) and all(r['exit_code']==0 for r in ledger['finished']) and ledger['actual_totals']=={k:5*v for k,v in COUNTS.items()})
    checks.add('complete no failed marker',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    bound_hashes(checks,js(out/'OUTPUT_HASHES.json'),'entire output manifest')
    direction=npz(out/'DIRECTION.npz'); chain=npz(S132/'OFFLINE_CANDIDATE.npz'); delta=chain['delta']
    for key,value in [('reference_weights',trials['baseline']['item']['arrays']['weights']),('stored_S132_delta',delta),('v',delta/np.linalg.norm(delta)),('delta_norm',np.asarray(np.linalg.norm(delta))),('h',np.asarray(.0005)),('half',np.asarray(.00025))]: checks.add('saved direction '+key,identical(direction[key],value))
    produced=npz(out/'DERIVATIVES.npz'); claimed=js(out/'DERIVATIVE_REPORT.json'); branches={}
    for label,t in trials.items():
        cfgpath=out/(label+'_CONFIG.json'); cfg=js(cfgpath)
        checks.add(label+' frozen config hash',pre['configs'][label]==sha(cfgpath) and cfg['amplitude']==AMPLITUDES[label] and cfg['label']==label and cfg['weights_sha256']==sha(cfg['weights_path']))
        compare_saved(checks,{'weights':t['item']['arrays']['weights']},npz(cfg['weights_path']),label+' config weights')
        childledger=js(out/label/'CALL_LEDGER.json')
        checks.add(label+' original child ledger',childledger==dict(COUNTS,active=None,completed=['init80ms'],per_arm={'init80ms':COUNTS}) and ledger['trial_ledgers'][label]==childledger)
        checks.add(label+' complete child',js(out/label/'COMPLETE.json')['status']=='COMPLETE' and not (out/label/'FAILED.json').exists())
        b=branch[label]; a=t['item']['arrays']; z=t['rows']; z0=trials['baseline']['rows']
        same_state=all(identical(z[f'sample{i:03d}_efc_state'],z0[f'sample{i:03d}_efc_state']) for i in range(121))
        limits=[]
        for j,limited in enumerate(t['item']['moment']['jnt_limited']):
            if limited:
                qi=t['item']['moment']['jnt_qposadr'][j]; lo,hi=t['item']['moment']['jnt_range'][j]; limits.append(np.maximum(np.maximum(lo-a['qpos'][:,qi],a['qpos'][:,qi]-hi),0))
        lim=np.stack(limits,1); checks.add(label+' saved all42 independent',identical(produced[label+'_all42_limit_excess'],lim))
        base_lims=produced['baseline_all42_limit_excess']
        br=dict(spikes_identical=b['neural_exact']['init_spikes'] and b['neural_exact']['runtime_spikes'],FIR_features_identical=b['neural_exact']['init_features'] and b['neural_exact']['runtime_features'],contact_rows_identical=b['topology_all'],efc_active_state_identical=same_state,clip_branch_identical=all(b['clip_branch_exact'].values()),limit_branch_identical=identical(lim>0,base_lims>0) and identical(lim.argmax(1),base_lims.argmax(1)))
        branches[label]=br; checks.add(label+' exact producer branch classification',claimed['branches'][label]==br)
        for key in ('qpos','qvel','qacc'): checks.add(label+' raw archival '+key,identical(produced[label+'_raw_'+key],a[key]))
        for key,expected in [('peak_all42_limit_excess',t['scalars']['peak_limit']),('sample113_qroll',t['scalars']['sampled_qroll113']),('load_min',t['scalars']['loadmin']),('all_gates_support_s',t['scalars']['support_discrete'])]: checks.add(label+' scalar independent '+key,claimed['trial_metrics'][label][key]==expected)
        commands=np.concatenate((t['init']['decoded'],a['applied'])); features=np.concatenate((t['init']['features'],a['features']))
        checks.add(label+' all166 actual commands',identical(produced[label+'_all166_actual_commands'],commands))
        analytical=mapped(features,a['weights'],a['actuator_pool'])
        checks.eq(label+' all166 analytical commands',produced[label+'_all166_analytical_commands'],analytical,1e-14)
        commandgate=fd_gate(commands,analytical,1e-14,1e-12)
        commandclaim=claimed['all166_command_checks'][label]
        checks.add(label+' independently gated all166 commands',commandgate['passed'] and commandclaim['passed']==commandgate['passed'])
        fixedfeatures=np.concatenate((trials['baseline']['init']['features'],trials['baseline']['item']['arrays']['features']))
        checks.eq(label+' fixed-reference feature map',produced[label+'_all166_fixed_reference_features_commands'],mapped(fixedfeatures,a['weights'],a['actuator_pool']),1e-14)
    fields=('qacc','qvel','qpos','act','ctrl','load_ratio','actuator_force','qfrc_actuator','qfrc_constraint','qfrc_passive','qfrc_bias','limit_excess')
    for key in fields:
        for suffix,plus,minus,den in [('D_h','plus_h','minus_h',.001),('D_half','plus_half','minus_half',.0005)]:
            expected=(trials[plus]['item']['arrays'][key]-trials[minus]['item']['arrays'][key])/den
            if key=='qpos': expected=expected[:,np.r_[0:3,7:49]]
            checks.add('independent raw central FD '+key+' '+suffix,identical(produced[key+'_'+suffix],expected))
        if key in gates:
            c=claimed['two_scale_checks'][key]; g=gates[key]
            checks.add('exact two-scale report '+key,c['passed']==g['passed'] and c['max_abs']==g['max_abs'] and c['scaled_relative']==g['scaled_relative'] and c['absolute_tolerance']==g['absolute_bound'] and c['relative_tolerance']==g['relative_bound'])
    scalar_valid=True
    for key,local,gatekey in [('peak_all42_limit_excess','peak_limit','limit_excess'),('sample113_qroll','sampled_qroll113','qpos'),('load_min','loadmin','load_ratio'),('all_gates_support_s','support_discrete','qpos')]:
        s=scalars[local]; g=fd_gate(s['h'],s['half'],*GATES[gatekey]); c=claimed['scalar_derivatives'][key]
        checks.add('exact scalar FD '+key,c['D_h']==s['h'] and c['D_half']==s['half'] and c['passed']==g['passed'] and c['max_abs']==g['max_abs'] and c['scaled_relative']==g['scaled_relative'])
        if local!='support_discrete': scalar_valid &= g['passed']
    stable=all(all(v[k] for k in ('spikes_identical','contact_rows_identical','efc_active_state_identical','clip_branch_identical')) for v in branches.values())
    valid=stable and all(v['passed'] for v in gates.values()) and scalar_valid and all(v['passed'] for v in claimed['all166_command_checks'].values())
    status='NONSMOOTH' if not stable else 'TWO_SCALE_PASS' if valid else 'TWO_SCALE_FAIL'
    checks.add('producer strict preregistered classification independently reproduced',claimed['status']==status and claimed['verified_gradient']==valid and js(out/'COMPLETE.json')['verified_gradient']==valid)
    report['producer_classification']=dict(status=status,verified_gradient=valid,branch_stable=stable,scalar_two_scale_pass=scalar_valid,continuous_two_scale_pass=all(v['passed'] for v in gates.values()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv); out=args.output.resolve(); target=(args.report or out/'TRAJECTORY_DIRECTION_INDEPENDENT_CHECK.json').resolve()
    if target.exists(): raise ValueError('never overwrite an existing verification report: '+str(target))
    checks=Checks(); report=dict(passed=False,evidence_passed=False,derivative_valid=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0))
    try:
        verify(out,checks,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=report['evidence_passed']=checks.passed
    except Exception as exc: report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['checks']=checks.rows
    report['helper_source_hashes']={str(Path(m.__file__).resolve()):sha(m.__file__) for n,m in tuple(sys.modules.items()) if n.startswith('verify_') and getattr(m,'__file__',None)}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],evidence_passed=report['evidence_passed'],derivative_valid=report['derivative_valid'],report=str(target),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
