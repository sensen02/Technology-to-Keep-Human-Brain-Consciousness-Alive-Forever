"""S135 independent OFFLINE refined trajectory directional-FD verification.
No producer imports, model loads, simulator execution, neural execution or trials.
Integrity, continuous FD, scalar-peak FD and branch identities remain separate.
"""
from pathlib import Path
import argparse
import ast
import json
import sys
import traceback
import numpy as np
from verify_trajectory_direction_probe import (Checks, canonical, bound_hashes, committed,
    npz, js, sha, serial, identical, compare_saved, fd_gate, audit_source,
    trial_path, trial, branch_report, NPZS, JSONS, mapped)

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/sensen/Desktop/engine修复验收/s135-refined-trajectory-direction')
SOURCE=ROOT/'run_refined_trajectory_direction_probe.py'
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
S132=Path('/home/sensen/Desktop/engine修复验收/s132-margin-contact-candidate/20261024T180846Z')
LABELS=('baseline','plus_h','minus_h','plus_half','minus_half')
AMPLITUDES=dict(baseline=0.,plus_h=.000125,minus_h=-.000125,plus_half=.0000625,minus_half=-.0000625)
GATES=dict(qacc=(1e-3,1e-3),qvel=(1e-5,1e-3),qpos=(1e-5,1e-3),load_ratio=(1e-4,1e-3),limit_excess=(1e-6,1e-3))
COUNTS=dict(model_load=1,forward=121,step=120,neural_ticks=166)
WINDOW=np.arange(110,118)


def independently_differentiate(trials):
    fd={}; gates={}; scalar_fd={}
    for key,(absolute,relative) in GATES.items():
        dh=(trials['plus_h']['item']['arrays'][key]-trials['minus_h']['item']['arrays'][key])/.00025
        df=(trials['plus_half']['item']['arrays'][key]-trials['minus_half']['item']['arrays'][key])/.000125
        fd[key+'_h']=dh; fd[key+'_half']=df
        cols=np.r_[0:3,7:49] if key=='qpos' else None
        gates[key]=fd_gate(dh[:,cols] if cols is not None else dh,df[:,cols] if cols is not None else df,absolute,relative)
    for key in ('peak_limit','sampled_qroll113','loadmin','support_discrete'):
        scalar_fd[key]=dict(h=(trials['plus_h']['scalars'][key]-trials['minus_h']['scalars'][key])/.00025,
            half=(trials['plus_half']['scalars'][key]-trials['minus_half']['scalars'][key])/.000125)
    return fd,gates,scalar_fd


def independent_limits(checks,trials,produced):
    """Recompute all 5x121x42 excesses from original qpos and joint metadata."""
    limits={}; identities={}; per_trial={}
    for label,t in trials.items():
        meta=t['item']['moment']; a=t['item']['arrays']; limited=np.flatnonzero(meta['jnt_limited'])
        checks.add(label+' exactly42 limited joints',len(limited)==42)
        qi=np.asarray(meta['jnt_qposadr'])[limited]; bounds=np.asarray(meta['jnt_range'])[limited]
        q=a['qpos'][:,qi]; lo=bounds[:,0]; hi=bounds[:,1]
        lim=np.maximum(np.maximum(lo-q,q-hi),0); limits[label]=lim
        checks.add(label+' saved all42 independently recomputed',identical(produced[label+'_all42_limit_excess'],lim))
        checks.add(label+' max all42 independently matches saved curve',identical(lim.max(1),a['limit_excess']))
        side=np.where(q<lo,-1,np.where(q>hi,1,0))
        winners=lim.argmax(1); global_index=np.unravel_index(lim.argmax(),lim.shape)
        identities[label]=dict(side=side,winner=winners,global_index=global_index)
        per_trial[label]=dict(samples=WINDOW.tolist(),all42_excess=lim[WINDOW],
            active_side=side[WINDOW],winning_limited_joint=winners[WINDOW],
            winning_joint_names=[meta['joint_names'][limited[j]] for j in winners[WINDOW]],
            global_peak_sample=int(global_index[0]),global_peak_limited_joint=int(global_index[1]),
            global_peak_joint=meta['joint_names'][limited[global_index[1]]],
            global_peak_value=float(lim[global_index]),
            global_peak_all_ties=np.argwhere(lim==lim.max()).tolist(),
            sample113_all42=lim[113],sample114_all42=lim[114])
    dh=(limits['plus_h']-limits['minus_h'])/.00025
    half=(limits['plus_half']-limits['minus_half'])/.000125
    ref=identities['baseline']; branches={}
    for label,x in identities.items():
        branches[label]=dict(window_same_active_sides=identical(x['side'][WINDOW],ref['side'][WINDOW]),
            window_same_winning_joints=identical(x['winner'][WINDOW],ref['winner'][WINDOW]),
            sample113_same_active_sides=identical(x['side'][113],ref['side'][113]),
            sample114_same_active_sides=identical(x['side'][114],ref['side'][114]),
            sample113_same_peak_joint=bool(x['winner'][113]==ref['winner'][113]),
            sample114_same_peak_joint=bool(x['winner'][114]==ref['winner'][114]),
            global_peak_identity_same=x['global_index']==ref['global_index'])
    return limits,dict(window_inclusive=[110,117],per_trial=per_trial,branch_identities=branches,
        all42_full_D_h=dh,all42_full_D_half=half,all42_window_D_h=dh[WINDOW],all42_window_D_half=half[WINDOW],
        full_all42_two_scale=fd_gate(dh,half,*GATES['limit_excess']),
        window_all42_two_scale=fd_gate(dh[WINDOW],half[WINDOW],*GATES['limit_excess']),
        sample113_all42_two_scale=fd_gate(dh[113],half[113],*GATES['limit_excess']),
        sample114_all42_two_scale=fd_gate(dh[114],half[114],*GATES['limit_excess']),
        informational_only=True,changes_frozen_S134_classification=False)


def verify_producer_analysis(out,checks,pre,trials,fd,gates,scalars,branch,report):
    checks.add('exact preregistered thresholds no relaxation',pre['two_scale_tolerances']=={k:list(v) for k,v in GATES.items()} and pre['h']==.000125 and pre['half']==.0000625 and pre['PRE']==160 and pre['samples']==121 and pre['max_trials']==5 and pre['trials']==[[k,AMPLITUDES[k]] for k in LABELS])
    ledger=js(out/'CALL_LEDGER.json')
    checks.add('five and only five successful actual trials',ledger['planned_trials']==5 and ledger['started']==list(LABELS) and [r['label'] for r in ledger['finished']]==list(LABELS) and all(r['exit_code']==0 for r in ledger['finished']) and ledger['actual_totals']=={k:5*v for k,v in COUNTS.items()})
    checks.add('complete no failed marker',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    bound_hashes(checks,js(out/'OUTPUT_HASHES.json'),'entire output manifest')
    direction=npz(out/'DIRECTION.npz'); delta=npz(S132/'OFFLINE_CANDIDATE.npz')['delta']
    for key,value in [('reference_weights',trials['baseline']['item']['arrays']['weights']),('stored_S132_delta',delta),('v',delta/np.linalg.norm(delta)),('delta_norm',np.asarray(np.linalg.norm(delta))),('h',np.asarray(.000125)),('half',np.asarray(.0000625))]: checks.add('saved direction '+key,identical(direction[key],value))
    produced=npz(out/'DERIVATIVES.npz'); claimed=js(out/'DERIVATIVE_REPORT.json'); branches={}
    limits,window=independent_limits(checks,trials,produced); report['critical_window']=window
    for label,t in trials.items():
        cfgpath=out/(label+'_CONFIG.json'); cfg=js(cfgpath)
        checks.add(label+' frozen config hash',pre['configs'][label]==sha(cfgpath) and cfg['amplitude']==AMPLITUDES[label] and cfg['label']==label and cfg['weights_sha256']==sha(cfg['weights_path']))
        compare_saved(checks,{'weights':t['item']['arrays']['weights']},npz(cfg['weights_path']),label+' config weights')
        childledger=js(out/label/'CALL_LEDGER.json')
        checks.add(label+' original child ledger',childledger==dict(COUNTS,active=None,completed=['init80ms'],per_arm={'init80ms':COUNTS}) and ledger['trial_ledgers'][label]==childledger)
        checks.add(label+' complete child',js(out/label/'COMPLETE.json')['status']=='COMPLETE' and not (out/label/'FAILED.json').exists())
        b=branch[label]; a=t['item']['arrays']; z=t['rows']; z0=trials['baseline']['rows']
        same_state=all(identical(z[f'sample{i:03d}_efc_state'],z0[f'sample{i:03d}_efc_state']) for i in range(121))
        lim=limits[label]; base_lims=limits['baseline']
        br=dict(spikes_identical=b['neural_exact']['init_spikes'] and b['neural_exact']['runtime_spikes'],FIR_features_identical=b['neural_exact']['init_features'] and b['neural_exact']['runtime_features'],contact_rows_identical=b['topology_all'],efc_active_state_identical=same_state,clip_branch_identical=all(b['clip_branch_exact'].values()),limit_branch_identical=identical(lim>0,base_lims>0) and identical(lim.argmax(1),base_lims.argmax(1)))
        branches[label]=br; checks.add(label+' exact producer branch classification',claimed['branches'][label]==br)
        for key in ('qpos','qvel','qacc'): checks.add(label+' raw archival '+key,identical(produced[label+'_raw_'+key],a[key]))
        for key,expected in [('peak_all42_limit_excess',t['scalars']['peak_limit']),('sample113_qroll',t['scalars']['sampled_qroll113']),('load_min',t['scalars']['loadmin']),('all_gates_support_s',t['scalars']['support_discrete'])]: checks.add(label+' scalar independent '+key,claimed['trial_metrics'][label][key]==expected)
        checks.add(label+' independently identified global peak sample',claimed['trial_metrics'][label]['peak_sample']==window['per_trial'][label]['global_peak_sample'])
        commands=np.concatenate((t['init']['decoded'],a['applied'])); features=np.concatenate((t['init']['features'],a['features']))
        checks.add(label+' all166 actual commands',identical(produced[label+'_all166_actual_commands'],commands))
        analytical=mapped(features,a['weights'],a['actuator_pool'])
        checks.eq(label+' all166 analytical commands',produced[label+'_all166_analytical_commands'],analytical,1e-14)
        commandgate=fd_gate(commands,analytical,1e-14,1e-12)
        checks.add(label+' independently gated all166 commands',commandgate['passed'] and claimed['all166_command_checks'][label]['passed']==commandgate['passed'])
        fixedfeatures=np.concatenate((trials['baseline']['init']['features'],trials['baseline']['item']['arrays']['features']))
        checks.eq(label+' fixed-reference feature map',produced[label+'_all166_fixed_reference_features_commands'],mapped(fixedfeatures,a['weights'],a['actuator_pool']),1e-14)
    fields=('qacc','qvel','qpos','act','ctrl','load_ratio','actuator_force','qfrc_actuator','qfrc_constraint','qfrc_passive','qfrc_bias','limit_excess')
    for key in fields:
        for suffix,plus,minus,den in [('D_h','plus_h','minus_h',.00025),('D_half','plus_half','minus_half',.000125)]:
            expected=(trials[plus]['item']['arrays'][key]-trials[minus]['item']['arrays'][key])/den
            if key=='qpos': expected=expected[:,np.r_[0:3,7:49]]
            checks.add('independent raw central FD '+key+' '+suffix,identical(produced[key+'_'+suffix],expected))
        if key in gates:
            c=claimed['two_scale_checks'][key]; g=gates[key]
            checks.add('exact two-scale report '+key,c['passed']==g['passed'] and c['max_abs']==g['max_abs'] and c['scaled_relative']==g['scaled_relative'] and c['absolute_tolerance']==g['absolute_bound'] and c['relative_tolerance']==g['relative_bound'])
    scalar_valid=True; scalar_gates={}
    for key,local,gatekey in [('peak_all42_limit_excess','peak_limit','limit_excess'),('sample113_qroll','sampled_qroll113','qpos'),('load_min','loadmin','load_ratio'),('all_gates_support_s','support_discrete','qpos')]:
        s=scalars[local]; g=fd_gate(s['h'],s['half'],*GATES[gatekey]); c=claimed['scalar_derivatives'][key]; scalar_gates[key]=g
        checks.add('exact scalar FD '+key,c['D_h']==s['h'] and c['D_half']==s['half'] and c['passed']==g['passed'] and c['max_abs']==g['max_abs'] and c['scaled_relative']==g['scaled_relative'])
        if local!='support_discrete': scalar_valid &= g['passed']
    stable=all(all(v[k] for k in ('spikes_identical','contact_rows_identical','efc_active_state_identical','clip_branch_identical')) for v in branches.values())
    valid=stable and all(v['passed'] for v in gates.values()) and scalar_valid and all(v['passed'] for v in claimed['all166_command_checks'].values())
    status='NONSMOOTH' if not stable else 'TWO_SCALE_PASS' if valid else 'TWO_SCALE_FAIL'
    checks.add('producer strict preregistered classification independently reproduced',claimed['status']==status and claimed['verified_gradient']==valid and js(out/'COMPLETE.json')['verified_gradient']==valid)
    report['producer_classification']=dict(status=status,verified_gradient=valid,branch_stable=stable,scalar_two_scale_pass=scalar_valid,continuous_two_scale_pass=all(v['passed'] for v in gates.values()))
    report['scalar_peak_FD']=scalar_gates['peak_all42_limit_excess']
    report['scalar_peak_identity_stable']=all(v['global_peak_identity_same'] for v in window['branch_identities'].values())
    verify_window_claims(checks,produced,claimed,window,limits,trials,direction)


def verify_window_claims(checks,produced,claimed,window,limits,trials,direction):
    """Producer's added window fields remain informational, never gate relaxation."""
    checks.add('saved direction exact critical window',identical(direction['peak_window'],WINDOW))
    for label,t in trials.items():
        a=t['item']['arrays']; meta=t['item']['moment']; j=meta['joint_names'].index('joint_LFCoxa_roll')
        qi=meta['jnt_qposadr'][j]; vi=meta['jnt_dofadr'][j]
        for suffix,expected in [('limit_excess',limits[label][WINDOW]),('qroll',a['qpos'][WINDOW,qi]),('qvel_roll',a['qvel'][WINDOW,vi])]:
            checks.add(label+' producer critical window '+suffix,identical(produced[label+'_peak_window_'+suffix],expected))
        c=claimed['trial_metrics'][label]
        checks.add(label+' producer critical window samples',c['peak_window_samples']==WINDOW.tolist())
        checks.add(label+' producer critical window max',identical(np.asarray(c['peak_window_all42_max']),limits[label][WINDOW].max(1)))
        native={}
        for i in WINDOW:
            prefix=f'sample{i:03d}_'; z=t['rows']; z0=trials['baseline']['rows']
            keys=('nefc','ncon','efc_type','efc_id','contact_efc_address','contact_dim','contact_exclude','contact_geom','efc_state')
            native[str(i)]={key:identical(z[prefix+key],z0[prefix+key]) for key in keys}
        window['branch_identities'][label]['native_per_sample']=native
        window['branch_identities'][label]['sample113_native_same']=all(native['113'].values())
        window['branch_identities'][label]['sample114_native_same']=all(native['114'].values())


def verify(out,checks,report):
    checks.add('offline initial boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    pre=js(out/'PREREG.json'); hashes=js(out/'INPUT_HASHES.json'); canonical(hashes)
    checks.add('preregistered full inputs',pre['source_hashes']==hashes)
    report['input_hashes']=bound_hashes(checks,hashes,'input')
    for path in (SOURCE,ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py',REFERENCE/'RESULT.json',S132/'OFFLINE_CANDIDATE.npz'): committed(checks,hashes,path,'required source')
    report['source_audit']=audit_source(checks,SOURCE)
    bound_hashes(checks,js(REFERENCE/'RESULT.json')['hashes'],'original S121')
    from verify_joint_candidate_probe import reconstruct_rows
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
        discrete_events_verified=report['producer_classification']['branch_stable'],raw_FD=fd,
        derivative_valid=report['producer_classification']['verified_gradient'],
        baseline_promotion=False,behavior_acceptance=False,physical_units='NOT_ESTABLISHED')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv); out=args.output.resolve(); target=(args.report or out/'REFINED_TRAJECTORY_DIRECTION_INDEPENDENT_CHECK.json').resolve()
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
    print(json.dumps(dict(passed=report['passed'],evidence_passed=report['evidence_passed'],derivative_valid=report['derivative_valid'],scalar_peak_FD_valid=report.get('scalar_peak_FD',{}).get('passed'),report=str(target),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
