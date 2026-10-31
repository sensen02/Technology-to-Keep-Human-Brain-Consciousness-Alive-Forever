"""S139/P1 offline saved-array verifier; inert import, never a simulator.
Original S119 exact 17-row conjunction only. S138 full121 local safety-output
predictions are diagnostics, not admission, causal attribution or autonomy.
"""
from pathlib import Path
import argparse
import json
import sys
import traceback
import numpy as np
from verify_joint_candidate_probe import (Checks, canonical, bound_hashes, committed,
    reconstruct_rows, exact, npz, js, sha, serial, require_integrity,
    validate_weights, reconstruct_fir, mapped, S113, S115, NPZS, JSONS,
    physical_metrics, admission_table, events, components, all42_limits, save_csv)

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收/s139-formal-benefit-candidate-closed-loop')
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
S138 = Path('/home/sensen/Desktop/engine修复验收/s138-formal-benefit-safety-proposal/20261024T192136Z')
S136 = Path('/home/sensen/Desktop/engine修复验收/s136-safety-trajectory-jacobian/20261024T183631Z')
CANDIDATE = S138/'CANDIDATE_WEIGHTS.npz'
PROOF = S138/'PARENT_FORMAL_BENEFIT_PROPOSAL_INDEPENDENT_CHECK.json'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
TARGET = 'FORMAL_BENEFIT_CANDIDATE_INDEPENDENT_CHECK.json'


def bind_manifest(checks, hashes, path):
    committed(checks, hashes, path, 'source manifest')
    data = js(path); canonical(data)
    bound_hashes(checks, data, str(path))
    for name in data: committed(checks, hashes, Path(name), 'transitive source')


def prediction_arrays(checks, candidate, baseline, moment):
    """Reconstruct S138 arithmetic from saved actual weight delta, not optimizer z."""
    j=npz(S136/'JACOBIAN.npz'); sol=npz(S138/'SOLUTION.npz'); qp=npz(S138/'QP.npz')
    indices=np.asarray([(p,b) for p in (3,15,38) for b in range(5)])
    checks.eq('S136 exact weight indices',j['weight_indices'],indices,0)
    checks.eq('QP exact weight indices',qp['weight_indices'],indices,0)
    checks.eq('S136 reference W',j['reference_weights'],baseline['weights'],0)
    checks.eq('S138 exact proposed W',sol['proposed_weights'],candidate,0)
    checks.eq('S138 saved delta exact',sol['delta'],candidate-baseline['weights'],0)
    scale=float(qp['scale']); checks.add('S138 fixed scale',scale==.001/np.sqrt(15))
    dz=scale*sol['actual_z'][:15]
    checks.eq('S138 actual rounded delta',dz,(candidate-baseline['weights'])[indices[:,0],indices[:,1]],1e-19)
    checks.add('S136 validity shape and 11 valid',j['valid'].shape==(15,) and j['valid'].dtype==bool and int(j['valid'].sum())==11)
    checks.eq('S138 QP validity',qp['valid'],j['valid'],0)
    checks.add('invalid actual coordinates frozen',np.all(sol['actual_z'][:15][~j['valid']]==0))
    allowed=np.zeros_like(candidate,dtype=bool); allowed[indices[:,0],indices[:,1]]=j['valid']
    checks.eq('all invalid and non-target W exact',candidate[~allowed],baseline['weights'][~allowed],0)
    checks.eq('S136 full121 exact time',j['time'],baseline['time'],0)
    limited=np.flatnonzero(moment['jnt_limited']); qa=np.asarray(moment['jnt_qposadr'])[limited]
    checks.eq('all42 joint identities',j['limited_joint_indices'],limited,0)
    checks.eq('all42 qpos addresses',j['limited_qpos_addresses'],qa,0)
    checks.eq('all42 signed bounds',j['joint_bounds'],np.asarray(moment['jnt_range'])[limited],0)
    sources=dict(qpos=baseline['qpos'][:,j['qpos_columns']],signed_joint_position=baseline['qpos'][:,qa],
        load_ratio=baseline['load_ratio'],height=baseline['qpos'][:,2]/baseline['qpos'][0,2],
        root_angle_rad=baseline['root_angle_rad'],com_vertical_velocity=baseline['com_velocity_world'][:,2],qvel=baseline['qvel'])
    pred={}
    for key,reference in sources.items():
        raw=j[key+'_D_half']; masked=np.where(j['valid'],raw,0.)
        checks.add('finite full121 derivative '+key,raw.shape==(*reference.shape,15) and np.isfinite(raw).all())
        checks.eq('S136 reference '+key,j['reference_'+key],reference,0)
        checks.eq('independent valid mask '+key,j['masked_'+key],masked,0)
        pred[key]=reference+np.einsum('...k,k->...',masked,dz)
    # Match producer's matrix product for load to preserve its floating-point order.
    pred['load_ratio']=j['reference_load_ratio']+j['masked_load_ratio']@dz
    checks.eq('S138 saved full121 joint prediction',sol['predicted_joint_position'],pred['signed_joint_position'],0)
    checks.eq('S138 saved full121 load prediction',sol['predicted_load_ratio'],pred['load_ratio'],0)
    checks.eq('S138 saved load delta',sol['delta_load_ratio'],pred['load_ratio']-j['reference_load_ratio'],0)
    bounds=j['joint_bounds']; q=pred['signed_joint_position']
    pred['joint_excess']=np.maximum(np.maximum(bounds[:,0]-q,q-bounds[:,1]),0.)
    pred.update(limit_excess=pred['joint_excess'].max(axis=1),time=j['time'],qpos_columns=j['qpos_columns'],limited=limited,qa=qa,bounds=bounds)
    return pred


def verify(out,reference,candidate_path,formal_path,checks,report):
    checks.add('offline module boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    checks.add('fixed S138 candidate and formal and reference',candidate_path==CANDIDATE.resolve() and formal_path==FORMAL.resolve() and reference==REFERENCE.resolve())
    checks.add('S139 output root',out.name=='init80ms' and out.parent.parent==BASE.resolve())
    pre=js(out.parent/'PREREG.json'); hashes=js(out.parent/'INPUT_HASHES.json')
    checks.add('source prereg exact',pre['source_hashes']==hashes); canonical(hashes)
    report['input_hashes']=bound_hashes(checks,hashes,'canonical input')
    checks.add('S138 prereg source',Path(pre['s138_proposal']).resolve()==S138.resolve() and Path(pre['s138_proposal_independent_check']).resolve()==PROOF.resolve())
    for path in (candidate_path,PROOF,S138/'SOLUTION.npz',S138/'QP.npz',S138/'PROPOSAL_REPORT.json',S138/'QP_CERTIFICATE.json',S138/'COMPLETE.json',S138/'PREREG.json',S136/'JACOBIAN.npz',formal_path,reference/'RESULT.json',ROOT/'run_formal_benefit_candidate_probe.py',ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py'):
        committed(checks,hashes,path,'required input')
    for path in (S138/'INPUT_HASHES.json',S138/'OUTPUT_HASHES.json',S136/'INPUT_HASHES.json',S136/'OUTPUT_HASHES.json'): bind_manifest(checks,hashes,path)
    cert=js(PROOF)
    checks.add('S138 independent evidence and proposal admitted',all(cert[k] is True for k in ('passed','evidence_passed','proposal_admitted','candidate_admitted')))
    checks.add('S138 verifier identity',Path(cert['verifier']).resolve()==ROOT/'verify_formal_benefit_safety_proposal.py' and sha(cert['verifier'])==cert['verifier_sha256'])
    for name,digest in cert['helper_source_hashes'].items():
        committed(checks,hashes,Path(name),'S138 verifier helper')
        checks.add('S138 certified helper unchanged '+name,sha(name)==digest)
    checks.add('S138 proposal accepted',js(S138/'PROPOSAL_REPORT.json')['proposal_accepted'] is True and js(S138/'COMPLETE.json')['proposal_accepted'] is True and not (S138/'FAILED.json').exists())
    counts=dict(model_load=1,forward=121,step=120,neural_ticks=166); ledger=js(out.parent/'CALL_LEDGER.json')
    checks.add('complete no failed marker',js(out.parent/'COMPLETE.json')['status']=='COMPLETE' and not (out.parent/'FAILED.json').exists())
    checks.add('single original call ledger',ledger==dict(counts,active=None,completed=['init80ms'],per_arm={'init80ms':counts}),observed=ledger)
    result=js(out/'RESULT.json'); checks.add('RESULT calls exact',result['calls']==counts)
    checks.add('single result exact',js(out.parent/'RESULTS.json')==[result])
    report['artifact_hashes']={}
    for label,path in [('S139',out),('S121',reference),('S115',S115),('S113',S113)]:
        res=js(path/'RESULT.json'); canonical(res['hashes'])
        report['artifact_hashes'][label]=bound_hashes(checks,res['hashes'],label+' artifacts')
        if label in ('S113','S115'): committed(checks,hashes,path/'RESULT.json',label+' reference')
        for name in (*NPZS,*JSONS): committed(checks,res['hashes'],path/name,label+' required artifact')
        if label in ('S139','S121'):
            for name in ('CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'): committed(checks,res['hashes'],path/name,label+' row artifact')
    require_integrity(checks,'input commitments and completion')
    old=reconstruct_rows(checks,reference,'S121'); new=reconstruct_rows(checks,out,'S139')
    checks.add('same moment metadata',old['moment']==new['moment'])
    checks.add('all121 matched physical instants',exact(old['arrays']['time'],new['arrays']['time'])['identical'])
    checks.add('0..3ms 25us clock',np.allclose(new['arrays']['time'],np.arange(121)*.000025,rtol=0,atol=1e-12))
    for label,item in [('S121',old),('S139',new)]: checks.add(label+' recognized row classification only',item['counts']['unrecognized']==0 and item['counts']['other']==0)
    a=old['arrays']; b=new['arrays']; init=npz(out/'INITIALIZATION.npz'); warm=npz(reference/'INITIALIZATION.npz'); candidate=npz(candidate_path)['weights']
    validate_weights(checks,a,b,init,candidate)
    checks.add('160 initialization ticks',init['features'].shape==(160,66,5) and init['decoded'].shape==(160,90))
    checks.add('runtime six ticks',b['features'].shape==(6,66,5) and b['applied'].shape==(6,90))
    for stage,features,actual in [('warm',init['features'],init['decoded']),('runtime',b['features'],b['applied'])]: checks.eq(stage+' actual candidate map',actual,mapped(features,candidate,b['actuator_pool']),1e-14)
    checks.eq('20-substep command hold',b['ctrl'][1:],np.repeat(b['applied'],20,axis=0),1e-14)
    manifests=[Path(p) for p in hashes if Path(p).name=='MANIFEST.json']
    checks.add('unique canonical pool manifest',len({str(p.resolve()) for p in manifests})==1)
    require_integrity(checks,'manifest completeness')
    manifest=sorted(js(manifests[0])['rows'],key=lambda row:row['actuator_index'])
    pools=sorted({tuple(sorted(row['local_indices'])) for row in manifest if row['local_indices']}); lookup={pool:i for i,pool in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(row['local_indices'])),-1) for row in manifest])
    checks.eq('independent actual manifest pool mapping',b['actuator_pool'],mapping)
    checks.add('manifest actual actuator identity',[r['actuator'] for r in manifest]==new['moment']['actuator_names'])
    fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],b['spikes'])),pools,mapping,candidate)
    _,_,warmhist=reconstruct_fir(init['spikes'],pools,mapping,candidate)
    for key,x,y in [('warm features',init['features'],fs[:160]),('warm decoded',init['decoded'],us[:160]),('release features',init['release_features'],fs[159]),('release decoded',init['release_decoded'],us[159]),('warm FIR',init['history'],warmhist),('runtime features',b['features'],fs[160:]),('runtime decoded',b['applied'],us[160:]),('final FIR',npz(out/'NEURAL_FINAL.npz')['FIR_history'],hist)]: checks.eq('independent FIR '+key,x,y,1e-14)
    checks.eq('fixed current source',b['current'],np.repeat(init['current'][None],6,axis=0))
    schedule=js(out/'SCHEDULE.json'); checks.add('six consumed ticks',len(schedule)==6)
    for tick,row in enumerate(schedule):
        checks.add('consume identity '+str(tick),row['tick']==tick and row['consume_id']==160+tick)
        for key,value in dict(source_start_s=(160+tick)*.0005,source_end_s=(161+tick)*.0005,physical_start_s=tick*.0005,physical_end_s=(tick+1)*.0005,source_offset_s=.08).items(): checks.eq('schedule '+str(tick)+' '+key,row[key],value,1e-12)
    for name,ticks in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',160),('NEURAL_FINAL_META.json',166)]:
        scalars=js(out/name)['neural_scalars']; checks.add('clock '+name,scalars['steps']==ticks and abs(scalars['t_ms']-.5*ticks)<=1e-10)
    checks.add('preregistered direct formal values',pre['training_constraints']==js(formal_path))
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'):
        x=npz(out/name); y=npz(reference/name); checks.add('cold keys '+name,set(x)==set(y))
        for key in x: checks.add('original cold exact '+name+':'+key,exact(x[key],y[key])['identical'])
    for key in ('body_qpos','body_qvel','body_act','body_ctrl','current'): checks.add('original release '+key,np.array_equal(init[key],warm[key]))
    for key in ('qpos','qvel','act','ctrl'): checks.add('original first physical '+key,np.array_equal(b[key][0],a[key][0]))
    checks.add('no external or warnings',not b['qfrc_applied'].any() and not b['xfrc_applied'].any() and not b['warnings'].any())
    for key in ('before','after'): checks.add('unchanged model numeric '+key,js(out/'MODEL_PROVENANCE.json')[key]==js(reference/'MODEL_PROVENANCE.json')[key])
    pred=prediction_arrays(checks,candidate,a,new['moment'])
    require_integrity(checks,'all121/all48 closure, initialization, full121 prediction reconstruction')
    xmls=[Path(p) for p in hashes if Path(p).suffix=='.xml']; checks.add('unique canonical XML',len({str(p.resolve()) for p in xmls})==1)
    require_integrity(checks,'canonical XML')
    formal=js(formal_path); report['formal_constraints']=formal; report['per_arm']={}; measured={}
    for label,path in [('S113',S113),('S115',S115),('S139',out)]:
        p,r,native=physical_metrics(path,xmls[0],checks,label); measured[label]=p
        report['per_arm'][label]=dict(path=str(path),physical=p,native=native)
    for label,key in [('S113','safety_exact_bounds'),('S115','benefit_exact_bounds')]:
        p=measured[label]
        for k,v in formal[key].items():
            if k=='extrema':
                for metric,sides in v.items():
                    for side,value in sides.items(): checks.add(label+' exact formal '+metric+' '+side,p[k][metric][side]==value)
            else: checks.add(label+' exact formal '+k,p[k]==v)
    require_integrity(checks,'physical/native COM and formal reference reconstruction')
    rows,d=admission_table(formal,measured['S139'],not np.array_equal(a['applied'],b['applied']))
    checks.add('exactly 17 formal admission rows',len(rows)==17)
    report.update(admission_table=rows,original_helper_decision=d,S119_short_window_pass=all(row['passed'] for row in rows),admission_evaluated=True,events={label:events(item) for label,item in [('S121',old),('S139',new)]},closure={label:item['states'] for label,item in [('S121',old),('S139',new)]})
    return old,new,pred


def outputs(out,old,new,pred,report):
    a=old['arrays']; b=new['arrays']; t=b['time']; m=new['moment']; paths=[]
    header=['criterion','actual','reference','operator','margin','passed','reference_source']
    paths.append(save_csv(out,'ADMISSION_TABLE.csv',header,[[r[k] for k in header] for r in report['admission_table']]))
    paths.append(save_csv(out,'ACTUAL_ROW_CLASSIFICATION.csv',['arm','sample','physical_ms','row','enum_value','enum_name','id','category','contact','geom1','geom2','joint','force','projection6','projection7','projection8','J_all48','projection_all48'],old['mapping']+new['mapping']))
    _,actual_limit,_=all42_limits(b,m); _,base_limit,_=all42_limits(a,m)
    q=b['qpos'][:,pred['qa']]; bounds=pred['bounds']; actual_excess=np.maximum(np.maximum(bounds[:,0]-q,q-bounds[:,1]),0.)
    rows=[]
    for i,time in enumerate(t):
        for k,joint in enumerate(pred['limited']):
            rows.append([i,time*1000,int(joint),m['joint_names'][joint],int(pred['qa'][k]),*bounds[k],a['qpos'][i,pred['qa'][k]],pred['signed_joint_position'][i,k],q[i,k],q[i,k]-pred['signed_joint_position'][i,k],pred['joint_excess'][i,k],actual_excess[i,k],actual_excess[i,k]-pred['joint_excess'][i,k]])
    paths.append(save_csv(out,'FULL121_ALL42_ACTUAL_VS_LINEAR.csv',['sample','physical_ms','joint_index','joint','qpos_address','lower_bound','upper_bound','S121_q','S138_linear_q','S139_actual_q','q_error','linear_excess','actual_excess','excess_error'],rows))
    rows=[]
    for i,time in enumerate(t):
        for k,col in enumerate(pred['qpos_columns']): rows.append([i,time*1000,int(col),a['qpos'][i,col],pred['qpos'][i,k],b['qpos'][i,col],b['qpos'][i,col]-pred['qpos'][i,k]])
    paths.append(save_csv(out,'FULL121_QPOS_ACTUAL_VS_LINEAR.csv',['sample','physical_ms','qpos_address','S121_reference','S138_linear','S139_actual','actual_minus_linear'],rows))
    rows=[]
    actual=dict(load_ratio=b['load_ratio'],limit_excess=actual_limit,height=b['qpos'][:,2]/b['qpos'][0,2],root_angle_rad=b['root_angle_rad'],com_vertical_velocity=b['com_velocity_world'][:,2],root_vertical_velocity=b['qvel'][:,2])
    predicted={key:pred[key] for key in actual if key!='root_vertical_velocity'}; predicted['root_vertical_velocity']=pred['qvel'][:,2]
    keys=list(actual)
    for i,time in enumerate(t): rows.append([i,time*1000]+[v for key in keys for v in (predicted[key][i],actual[key][i],actual[key][i]-predicted[key][i])])
    paths.append(save_csv(out,'FULL121_SAFETY_ACTUAL_VS_LINEAR.csv',['sample','physical_ms']+[key+'_'+suffix for key in keys for suffix in ('linear','actual','error')],rows))
    peaks=[]
    for label,curve in [('S121',base_limit),('S138_linear',pred['limit_excess']),('S139_actual',actual_limit)]:
        i=int(curve.argmax()); peaks.append(dict(label=label,peak_sample=i,physical_ms=float(t[i]*1000),peak=float(curve[i]),sample113=float(curve[113]),sample114=float(curve[114])))
    report.update(prediction_peaks=peaks,prediction_scope='Full121 local safety-output linear predictions from S136 valid masked half-scale Jacobian times S138 actual saved weight delta. S138 SOLUTION signed all42 joint/load predictions independently exact. No general qacc-gradient or causal-validity claim.',prediction_errors={key:dict(max_abs=float(np.max(np.abs(actual[key]-predicted[key]))),sample113=float(actual[key][113]-predicted[key][113]),sample114=float(actual[key][114]-predicted[key][114])) for key in keys})
    paths.append(save_csv(out,'LIMIT_PEAKS_113_114.csv',list(peaks[0]),[[row[k] for k in peaks[0]] for row in peaks]))
    limits=report['per_arm']['S139']['physical']['all42_joint_limits']; h=list(limits[0])
    paths.append(save_csv(out,'ALL42_LIMIT_EXTREMA.csv',h,[[r[k] for k in h] for r in limits]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(15,12)); ms=t*1000
    roll_joint=m['joint_names'].index('joint_LFCoxa_roll'); roll_qi=int(m['jnt_qposadr'][roll_joint]); ri=int(np.flatnonzero(pred['limited']==roll_joint)[0])
    curves=[('LFCoxa_roll q',a['qpos'][:,roll_qi],pred['signed_joint_position'][:,ri],b['qpos'][:,roll_qi]),('load ratio',a['load_ratio'],pred['load_ratio'],b['load_ratio']),('all42 max excess',base_limit,pred['limit_excess'],actual_limit),('height ratio',a['qpos'][:,2]/a['qpos'][0,2],pred['height'],actual['height']),('orientation angle',a['root_angle_rad'],pred['root_angle_rad'],actual['root_angle_rad']),('COM vertical velocity',a['com_velocity_world'][:,2],pred['com_vertical_velocity'],actual['com_vertical_velocity'])]
    for ax,(title,base,linear,observed) in zip(axs.flat,curves):
        for label,y,ls in [('S121',base,':'),('S138 linear',linear,'--'),('S139 actual',observed,'-')]: ax.plot(ms,y,ls=ls,label=label)
        for i in (113,114): ax.axvline(ms[i],color='grey',alpha=.35)
        ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_xlabel('physical ms')
    axs[1,0].axhline(report['formal_constraints']['safety_exact_bounds']['extrema']['limit_excess']['max'],color='k',ls=':')
    fig.suptitle('S139 actual vs S138 full121 local linear safety predictions; not autonomy')
    fig.tight_layout(); p=out/'SAFETY_ACTUAL_VS_PREDICTED.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    for label in ('S113','S115','S139'):
        p=report['per_arm'][label]['physical']; path=Path(report['per_arm'][label]['path']); x=npz(path/'TRAJECTORY.npz'); _,curve,_=all42_limits(x,js(path/'MOMENT_METADATA.json'))
        angle=2*np.arccos(np.clip(np.abs(x['qpos'][:,3:7]@x['qpos'][0,3:7])/(np.linalg.norm(x['qpos'][:,3:7],axis=1)*np.linalg.norm(x['qpos'][0,3:7])),0,1))
        for ax,y in zip(axs.flat,[p['load_ratio'],curve,x['qpos'][:,2]/x['qpos'][0,2],angle,x['nonfoot_depth'],x['com_velocity_world'][:,2]]): ax.plot(x['time']*1000,y,label=label)
    for ax,title in zip(axs.flat,['load ratio','all42 maximum limit excess','root height ratio','orientation angle','nonfoot depth','whole-body COM vertical velocity']): ax.set_title(title); ax.grid(alpha=.25); ax.legend(); ax.set_xlabel('physical ms')
    axs[0,0].axhline(.8,color='k',ls=':'); axs[0,1].axhline(1e-6,color='k',ls=':')
    fig.suptitle('Original S119 short-window full-body safety; units NOT_ESTABLISHED; no promotion')
    fig.tight_layout(); p=out/'LOAD_FULLBODY_SAFETY.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--reference',type=Path,default=REFERENCE); p.add_argument('--candidate',type=Path,default=CANDIDATE); p.add_argument('--formal',type=Path,default=FORMAL); args=p.parse_args(argv)
    out=args.output.resolve(); checks=Checks(); target=out/TARGET
    if target.exists(): raise ValueError('never overwrite previous verification report')
    report=dict(passed=False,admission_evaluated=False,S119_short_window_pass=False,baseline_promotion=False,promotion=False,checkpoint_replacement=False,autonomy=False,standing_or_walking_claim=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),physical_units='NOT_ESTABLISHED',behavior_acceptance=False)
    try:
        old,new,pred=verify(out,args.reference.resolve(),args.candidate.resolve(),args.formal.resolve(),checks,report)
        report['outputs']=outputs(out,old,new,pred,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=checks.passed
        if not checks.passed: report.update(S119_short_window_pass=False,admission_evaluated=False)
    except Exception as exc: report.update(passed=False,S119_short_window_pass=False,admission_evaluated=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['helper_source_hashes']={str(Path(module.__file__).resolve()):sha(module.__file__) for name,module in tuple(sys.modules.items()) if name.startswith('verify_') and getattr(module,'__file__',None)}
    report['checks']=checks.rows
    with target.open('x') as stream: stream.write(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],S119_short_window_pass=report['S119_short_window_pass'],admission_evaluated=report['admission_evaluated'],report=str(target),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
