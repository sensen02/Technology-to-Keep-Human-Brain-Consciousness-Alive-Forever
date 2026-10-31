"""S127/P1/S2 offline saved-array verifier; never import a runner or simulator.
Admission is ONLY S119's short-window conjunction, not baseline promotion.
The verifier binds its own SHA in its report, not in runner preregistration.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import traceback
import numpy as np
from verify_com_contact_probe import Checks, reconstruct_fir
from verify_constraint_row_recording import sha, js, npz, serial, exact, bound_hashes
from verify_secant_constraint_pair import canonical, committed, reconstruct_rows, events, components, NPZS, JSONS
from verify_secant_weight_probe import all42_limits
from verify_single_weight_attenuation import decision, mapped, S113, S115
from verify_com_timestep_probe import reconstruct, physical
from verify_initial_height_probe import raw_feet, balance

ROOT = Path(__file__).resolve().parent
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s127-joint-candidate-closed-loop/20261024T082500Z/init80ms')
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
CANDIDATE = Path('/home/sensen/Desktop/engine修复验收/s126-fixed-contraction/20261024T081500Z/CANDIDATE_WEIGHTS.npz')
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
FOCUS = (0,1,2,4,5)


def later_or_absent(actual, reference):
    """None means no failure in the observed window, not time zero."""
    return actual is None or (reference is not None and actual >= reference)


def admission_table(formal, metrics, effective):
    """Every formal inequality with exact comparisons; signed positive margin is safe.
    A null margin denotes an ordered censored event, never numerical infinity.
    """
    rows=[]
    def add(name, actual, ref, op, source):
        if op == 'not_earlier':
            passed=later_or_absent(actual,ref)
            margin=None if actual is None or ref is None else actual-ref
        else:
            passed=actual >= ref if op == '>=' else actual <= ref
            margin=actual-ref if op == '>=' else ref-actual
        rows.append(dict(criterion=name,actual=actual,reference=ref,operator=op,margin=margin,passed=bool(passed),reference_source=source))
    b=formal['benefit_exact_bounds']; s=formal['safety_exact_bounds']
    for key,op in [('initial_load_ratio','>='),('load_deficit_integral0_1p7_ratio_ms','<='),('all_gates_support_ms','>='),('longest_all_gates_true_endpoint_span_ms','>=')]:
        add(key,metrics[key],b[key],op,'S119/S115')
    add('first_load_failure_ms',metrics['first_failures_ms']['load'],b['first_failures_ms']['load'],'not_earlier','S119/S115')
    for key,side in [('limit_excess','max'),('angle_rad','max'),('nonfoot_depth','max'),('height_ratio','min')]:
        add(key+'_'+side,metrics['extrema'][key][side],s['extrema'][key][side],'>=' if side=='min' else '<=','S119/S113')
    for key in ('max_downward_root_velocity','max_downward_COM_velocity'):
        add(key,metrics[key],s[key],'<=','S119/S113')
    for gate in ('height','angle','legs','nonfoot','limit'):
        add('first_'+gate+'_failure_ms',metrics['first_failures_ms'][gate],s['first_failures_ms'][gate],'not_earlier','S119/S113')
    add('actual_output_changed',bool(effective),True,'>=','actual S121 comparison')
    original=decision(s,b,metrics,effective)
    if (not original['reject']) != all(r['passed'] for r in rows):
        raise ValueError('formal table and original decision helper disagree')
    return rows,original


def require_integrity(checks, stage):
    if not checks.passed:
        raise ValueError(stage+': evidence integrity/completeness/closure failure; admission STOPPED')


def physical_metrics(path, xml, checks, label):
    a=npz(path/'TRAJECTORY.npz'); meta=js(path/'COM_CONTACT_METADATA.json'); moment=js(path/'MOMENT_METADATA.json')
    for key,value in a.items():
        if value.dtype.kind in 'biufc': checks.add(label+' finite '+key,np.isfinite(value).all())
    checks.add(label+' shapes',a['qpos'].shape==(121,49) and a['qvel'].shape==(121,48) and a['actuator_force'].shape==(121,90) and a['actuator_moment'].shape==(121,90,48))
    feet=raw_feet(a,meta)
    for key in ('foot_force','contacts','nonfoot_depth'): checks.eq(label+' raw feet '+key,a[key],feet[key],2e-12)
    rec=reconstruct(a,meta,checks,label); native=balance(a,rec)
    for key in ('instantaneous','impulse','momentum_FD'): checks.add(label+' native COM '+key,native[key]['passed'],details=native[key])
    weight=rec['mass']*abs(meta['gravity_world'][2])
    checks.eq(label+' RESULT native weight',js(path/'RESULT.json')['weight_native'],weight,1e-12)
    p=physical(a,rec,checks,label,weight,xml)
    limits,curve,peak=all42_limits(a,moment)
    checks.add(label+' exactly all42 limited joints',sum(moment['jnt_limited'])==42 and len(limits)==42)
    checks.eq(label+' metadata all42 maximum vs XML',curve,a['limit_excess'],1e-14)
    # Use independent all42 curve for formal admission (not a cached scalar).
    p['extrema']['limit_excess']['max']=float(curve.max())
    ix=np.flatnonzero(a['time']<=.0017+1e-12)
    p.update(initial_load_ratio=float(p['load_ratio'][0]),load_deficit_integral0_1p7_ratio_ms=float(np.trapezoid(np.maximum(.8-p['load_ratio'][ix],0),x=a['time'][ix])*1000),max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),max_downward_COM_velocity=float(np.maximum(-a['com_velocity_world'][:,2],0).max()),all42_joint_limits=limits,all42_peak=peak)
    return p,rec,native


def validate_weights(checks, old, new, init, candidate):
    w=old['weights']; mask=np.zeros_like(w,dtype=bool); mask[[3,15,38],:]=True
    checks.add('all330 weights shape bounds finite',candidate.shape==(66,5) and np.isfinite(candidate).all() and np.all((candidate>=0)&(candidate<=4)))
    checks.add('original zero mask preserved',np.array_equal(candidate==0,w==0))
    checks.add('all non-target weights exact S121',np.array_equal(candidate[~mask],w[~mask]))
    checks.add('entire pool36 exact original',np.array_equal(candidate[36],w[36]))
    checks.add('runtime all330 candidate exact',np.array_equal(new['weights'],candidate))
    checks.add('all initialization candidate exact',np.array_equal(init['weights'],candidate))
    checks.add('actual pool mapping unchanged',np.array_equal(old['actuator_pool'],new['actuator_pool']))


def verify(out, reference, candidate_path, formal_path, checks, report):
    checks.add('offline module boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    pre=js(out.parent/'PREREG.json'); hashes=js(out.parent/'INPUT_HASHES.json')
    checks.add('source prereg exact',pre['source_hashes']==hashes)
    canonical(hashes) # reject conflicting symlink-alias commitments
    report['input_hashes']=bound_hashes(checks,hashes,'canonical input')
    for path in (candidate_path,candidate_path.parent/'LOCAL_CHAIN.npz',formal_path,reference/'RESULT.json',ROOT/'run_joint_candidate_probe.py',ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py'):
        committed(checks,hashes,path,'required input')
    counts=dict(model_load=1,forward=121,step=120,neural_ticks=166)
    ledger=js(out.parent/'CALL_LEDGER.json')
    checks.add('complete no failed marker',js(out.parent/'COMPLETE.json')['status']=='COMPLETE' and not (out.parent/'FAILED.json').exists())
    checks.add('single original call ledger',ledger==dict(counts,active=None,completed=['init80ms'],per_arm={'init80ms':counts}),observed=ledger)
    result=js(out/'RESULT.json'); checks.add('RESULT calls exact',result['calls']==counts)
    checks.add('single result exact',js(out.parent/'RESULTS.json')==[result])
    report['artifact_hashes']={}
    for label,path in [('S127',out),('S121',reference),('S115',S115),('S113',S113)]:
        res=js(path/'RESULT.json'); canonical(res['hashes'])
        report['artifact_hashes'][label]=bound_hashes(checks,res['hashes'],label+' artifacts')
        if label in ('S113','S115'): committed(checks,hashes,path/'RESULT.json',label+' reference')
        for name in (*NPZS,*JSONS): committed(checks,res['hashes'],path/name,label+' required artifact')
        if label in ('S127','S121'):
            for name in ('CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'): committed(checks,res['hashes'],path/name,label+' row artifact')
    require_integrity(checks,'input commitments and completion')
    old=reconstruct_rows(checks,reference,'S121'); new=reconstruct_rows(checks,out,'S127')
    checks.add('same moment metadata',old['moment']==new['moment'])
    checks.add('all121 matched physical instants',exact(old['arrays']['time'],new['arrays']['time'])['identical'])
    checks.add('0..3ms 25us clock',np.allclose(new['arrays']['time'],np.arange(121)*.000025,rtol=0,atol=1e-12))
    for label,item in [('S121',old),('S127',new)]:
        checks.add(label+' recognized row classification only',item['counts']['unrecognized']==0 and item['counts']['other']==0)
    a=old['arrays']; b=new['arrays']; init=npz(out/'INITIALIZATION.npz'); warm=npz(reference/'INITIALIZATION.npz'); candidate=npz(candidate_path)['weights']
    validate_weights(checks,a,b,init,candidate)
    checks.add('160 initialization ticks',init['features'].shape==(160,66,5) and init['decoded'].shape==(160,90))
    checks.add('runtime six ticks',b['features'].shape==(6,66,5) and b['applied'].shape==(6,90))
    for stage,features,actual in [('warm',init['features'],init['decoded']),('runtime',b['features'],b['applied'])]:
        checks.eq(stage+' actual features all candidate weights clipped map',actual,mapped(features,candidate,b['actuator_pool']),1e-14)
    checks.eq('20-substep command hold',b['ctrl'][1:],np.repeat(b['applied'],20,axis=0),1e-14)
    manifests=[Path(p) for p in hashes if Path(p).name=='MANIFEST.json']
    checks.add('unique canonical pool manifest',len({str(p.resolve()) for p in manifests})==1)
    require_integrity(checks,'manifest completeness')
    manifest=sorted(js(manifests[0])['rows'],key=lambda row:row['actuator_index'])
    pools=sorted({tuple(sorted(row['local_indices'])) for row in manifest if row['local_indices']})
    lookup={pool:i for i,pool in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(row['local_indices'])),-1) for row in manifest])
    checks.eq('independent actual manifest pool mapping',b['actuator_pool'],mapping)
    checks.add('manifest actual actuator identity',[r['actuator'] for r in manifest]==new['moment']['actuator_names'])
    fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],b['spikes'])),pools,mapping,candidate)
    _,_,warmhist=reconstruct_fir(init['spikes'],pools,mapping,candidate)
    for key,x,y in [('warm features',init['features'],fs[:160]),('warm decoded',init['decoded'],us[:160]),('release features',init['release_features'],fs[159]),('release decoded',init['release_decoded'],us[159]),('warm FIR',init['history'],warmhist),('runtime features',b['features'],fs[160:]),('runtime decoded',b['applied'],us[160:]),('final FIR',npz(out/'NEURAL_FINAL.npz')['FIR_history'],hist)]:
        checks.eq('independent FIR '+key,x,y,1e-14)
    checks.eq('fixed current source',b['current'],np.repeat(init['current'][None],6,axis=0))
    schedule=js(out/'SCHEDULE.json'); checks.add('six consumed ticks',len(schedule)==6)
    for tick,row in enumerate(schedule):
        checks.add('consume identity '+str(tick),row['tick']==tick and row['consume_id']==160+tick)
        for key,value in dict(source_start_s=(160+tick)*.0005,source_end_s=(161+tick)*.0005,physical_start_s=tick*.0005,physical_end_s=(tick+1)*.0005,source_offset_s=.08).items():
            checks.eq('schedule '+str(tick)+' '+key,row[key],value,1e-12)
    for name,ticks in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',160),('NEURAL_FINAL_META.json',166)]:
        scalars=js(out/name)['neural_scalars']; checks.add('clock '+name,scalars['steps']==ticks and abs(scalars['t_ms']-.5*ticks)<=1e-10)
    checks.add('preregistered direct formal values',pre['training_constraints']==js(formal_path))
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'):
        x=npz(out/name); y=npz(reference/name); checks.add('cold keys '+name,set(x)==set(y))
        for key in x: checks.add('original cold exact '+name+':'+key,exact(x[key],y[key])['identical'])
    for key in ('body_qpos','body_qvel','body_act','body_ctrl','current'):
        checks.add('original release '+key,np.array_equal(init[key],warm[key]))
    for key in ('qpos','qvel','act','ctrl'): checks.add('original first physical '+key,np.array_equal(b[key][0],a[key][0]))
    checks.add('no external or warnings',not b['qfrc_applied'].any() and not b['xfrc_applied'].any() and not b['warnings'].any())
    for key in ('before','after'): checks.add('unchanged model numeric '+key,js(out/'MODEL_PROVENANCE.json')[key]==js(reference/'MODEL_PROVENANCE.json')[key])
    require_integrity(checks,'all121/all48 row closure and candidate initialization')
    xmls=[Path(p) for p in hashes if Path(p).suffix=='.xml']; checks.add('unique canonical XML',len({str(p.resolve()) for p in xmls})==1)
    require_integrity(checks,'canonical XML')
    formal=js(formal_path); report['formal_constraints']=formal
    report['per_arm']={}; measured={}
    for label,path in [('S113',S113),('S115',S115),('S127',out)]:
        p,r,native=physical_metrics(path,xmls[0],checks,label); measured[label]=p
        report['per_arm'][label]=dict(path=str(path),physical=p,native=native)
    # Direct-read rules and values, and independently verify that reference artifacts
    # really realize every formal value used by the original helper.
    for label,key in [('S113','safety_exact_bounds'),('S115','benefit_exact_bounds')]:
        p=measured[label]
        for k,v in formal[key].items():
            if k=='extrema':
                for metric,sides in v.items():
                    for side,value in sides.items(): checks.add(label+' exact formal '+metric+' '+side,p[k][metric][side]==value)
            else: checks.add(label+' exact formal '+k,p[k]==v)
    require_integrity(checks,'physical/native COM and formal reference reconstruction')
    effective=not np.array_equal(a['applied'],b['applied'])
    rows,d=admission_table(formal,measured['S127'],effective)
    report.update(admission_table=rows,original_helper_decision=d,S119_short_window_pass=all(row['passed'] for row in rows),admission_evaluated=True,baseline_promotion=False,checkpoint_replacement=False,events={label:events(item) for label,item in [('S121',old),('S127',new)]},closure={label:item['states'] for label,item in [('S121',old),('S127',new)]})
    return old,new,npz(candidate_path.parent/'LOCAL_CHAIN.npz')


def save_csv(out,name,header,rows):
    path=out/name
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    return str(path)


def outputs(out,old,new,chain,report):
    a=old['arrays']; b=new['arrays']; t=b['time']; m=new['moment']; paths=[]
    table=report['admission_table']; header=['criterion','actual','reference','operator','margin','passed','reference_source']
    paths.append(save_csv(out,'ADMISSION_TABLE.csv',header,[[r[k] for k in header] for r in table]))
    rows=[]; predictions=[]; states=[]
    pred_force=chain['gain']*chain['candidate_act']+chain['bias']
    pred_torque=pred_force[:,:,None]*chain['moment678']
    if not np.array_equal(chain['time'],t): raise ValueError('LOCAL_CHAIN unmatched physical instants')
    c0=components(old); c1=components(new)
    for i,time in enumerate(t):
        for u in FOCUS:
            for di,dof in enumerate((6,7,8)):
                rows.append([i,time*1000,u,dof,a['actuator_force'][i,u],b['actuator_force'][i,u],a['actuator_force'][i,u]*a['actuator_moment'][i,u,dof],b['actuator_force'][i,u]*b['actuator_moment'][i,u,dof]])
                actual=b['actuator_force'][i,u]*b['actuator_moment'][i,u,dof]
                predictions.append([i,time*1000,u,dof,chain['candidate_act'][i,u],b['act'][i,u],chain['gain'][i,u],chain['bias'][i,u],pred_force[i,u],b['actuator_force'][i,u],pred_torque[i,u,di],actual,actual-pred_torque[i,u,di]])
        for dof in (6,7,8):
            j=int(m['dof_jntid'][dof]); qi=int(m['jnt_qposadr'][j]); row=[i,time*1000,dof,m['joint_names'][j]]
            for item,c in [(old,c0),(new,c1)]:
                x=item['arrays']; _,curve,_=all42_limits(x,item['moment'])
                row.extend([c['qfrcact'][i,dof],c['contact'][i,dof],c['limit'][i,dof],c['other'][i,dof],x['qpos'][i,qi],x['qvel'][i,dof],x['load_ratio'][i],curve[i]])
            states.append(row)
    paths.append(save_csv(out,'ACTUAL_FORCES_TORQUES.csv',['sample','physical_ms','actuator','DOF','S121_force','S127_force','S121_torque','S127_torque'],rows))
    paths.append(save_csv(out,'FIXED_STATE_PREDICTION_ERROR.csv',['sample','physical_ms','actuator','DOF','S126_pred_activation','S127_actual_activation','S126_fixed_gain','S126_fixed_bias','S126_pred_force','S127_actual_force','S126_pred_torque','S127_actual_torque','actual_minus_prediction'],predictions))
    keys=['muscle','contact','limit','other_constraint','q','v','load_ratio','all42_max_excess']
    paths.append(save_csv(out,'COUPLED_DOF678.csv',['sample','physical_ms','DOF','joint']+[label+'_'+key for label in ('S121','S127') for key in keys],states))
    paths.append(save_csv(out,'ACTUAL_ROW_CLASSIFICATION.csv',['arm','sample','physical_ms','row','enum_value','enum_name','id','category','contact','geom1','geom2','joint','force','projection6','projection7','projection8','J_all48','projection_all48'],old['mapping']+new['mapping']))
    limits=report['per_arm']['S127']['physical']['all42_joint_limits']
    if limits:
        h=list(limits[0]); paths.append(save_csv(out,'ALL42_LIMIT_EXTREMA.csv',h,[[r[k] for k in h] for r in limits]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,3,figsize=(17,12)); ms=t*1000
    for col,dof in enumerate((6,7,8)):
        j=int(m['dof_jntid'][dof]); qi=int(m['jnt_qposadr'][j])
        for label,item,c,ls in [('S121',old,c0,'--'),('S127',new,c1,'-')]:
            for key in ('qfrcact','contact','limit'): axs[0,col].plot(ms,c[key][:,dof],ls=ls,label=label+' '+key)
            axs[1,col].plot(ms,item['arrays']['qpos'][:,qi],ls=ls,label=label+' q')
        axs[1,col].axhline(m['jnt_range'][j][1],color='k',ls=':',label='upper')
        axs[2,col].plot(ms,pred_torque[:,:,col].sum(1),label='S126 fixed-state prediction')
        axs[2,col].plot(ms,b['qfrc_actuator'][:,dof],label='S127 actual all90 muscle')
        axs[0,col].set_title('DOF'+str(dof)+' '+m['joint_names'][j])
    for ax in axs.flat: ax.legend(fontsize=6); ax.grid(alpha=.25); ax.set_xlabel('physical ms')
    fig.suptitle('S127 same-time coupled actual / S126 fixed-state prediction; NOT closed-loop prediction')
    fig.tight_layout(); p=out/'COUPLED_PREDICTED_ACTUAL.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    for label in ('S113','S115','S127'):
        p=report['per_arm'][label]['physical']
        # Full-body per-sample curves are independently derived above; raw saved
        # arrays here only choose plotting coordinates, not admission values.
        path=Path(report['per_arm'][label]['path']); x=npz(path/'TRAJECTORY.npz'); _,curve,_=all42_limits(x,js(path/'MOMENT_METADATA.json'))
        for ax,y in zip(axs.flat,[p['load_ratio'],curve,x['qpos'][:,2]/x['qpos'][0,2],2*np.arccos(np.clip(np.abs(x['qpos'][:,3:7]@x['qpos'][0,3:7])/(np.linalg.norm(x['qpos'][:,3:7],axis=1)*np.linalg.norm(x['qpos'][0,3:7])),0,1)),x['nonfoot_depth'],x['com_velocity_world'][:,2]]): ax.plot(x['time']*1000,y,label=label)
    for ax,title in zip(axs.flat,['load ratio','all42 maximum limit excess','root height ratio','orientation angle','nonfoot depth','whole-body COM vertical velocity']): ax.set_title(title); ax.grid(alpha=.25); ax.legend(); ax.set_xlabel('physical ms')
    axs[0,0].axhline(.8,color='k',ls=':'); axs[0,1].axhline(1e-6,color='k',ls=':')
    fig.suptitle('S119 short-window benefit / full-body safety only; units NOT_ESTABLISHED; no baseline promotion')
    fig.tight_layout(); p=out/'LOAD_FULLBODY_SAFETY.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT); p.add_argument('--reference',type=Path,default=REFERENCE); p.add_argument('--candidate',type=Path,default=CANDIDATE); p.add_argument('--formal',type=Path,default=FORMAL); args=p.parse_args(argv)
    out=args.output.resolve(); checks=Checks()
    report=dict(passed=False,admission_evaluated=False,S119_short_window_pass=False,baseline_promotion=False,checkpoint_replacement=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),physical_units='NOT_ESTABLISHED',behavior_acceptance=False)
    try:
        old,new,chain=verify(out,args.reference.resolve(),args.candidate.resolve(),args.formal.resolve(),checks,report)
        report['outputs']=outputs(out,old,new,chain,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=checks.passed
        if not checks.passed: report.update(S119_short_window_pass=False,admission_evaluated=False)
    except Exception as exc:
        report.update(passed=False,S119_short_window_pass=False,admission_evaluated=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['checks']=checks.rows
    path=out/'JOINT_CANDIDATE_INDEPENDENT_CHECK.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],S119_short_window_pass=report['S119_short_window_pass'],admission_evaluated=report['admission_evaluated'],report=str(path),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
