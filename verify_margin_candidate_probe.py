"""S133 offline verifier derived from frozen S127 verification, never a simulator.
All121/all48 actual closure and exact S119 17-row conjunction. S132 predictions
exist at exactly samples 77/100/105, not a full121 closed-loop prediction.
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

ROOT=Path(__file__).resolve().parent
DEFAULT=Path('/home/sensen/Desktop/engine修复验收/s133-margin-candidate-closed-loop/20261024T181113Z/init80ms')
REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
S132=Path('/home/sensen/Desktop/engine修复验收/s132-margin-contact-candidate/20261024T180846Z')
CANDIDATE=S132/'CANDIDATE_WEIGHTS.npz'
FORMAL=Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
STATES=(77,100,105)


def verify(out,reference,candidate_path,formal_path,checks,report):
    checks.add('offline module boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    pre=js(out.parent/'PREREG.json'); hashes=js(out.parent/'INPUT_HASHES.json')
    checks.add('source prereg exact',pre['source_hashes']==hashes)
    canonical(hashes)
    report['input_hashes']=bound_hashes(checks,hashes,'canonical input')
    for path in (candidate_path,candidate_path.parent/'OFFLINE_CANDIDATE.npz',candidate_path.parent/'QP_PROBLEM.npz',candidate_path.parent/'RAW_PROBES.npz',candidate_path.parent/'PARENT_INDEPENDENT_CHECK_FINAL.json',formal_path,reference/'RESULT.json',ROOT/'run_margin_candidate_probe.py',ROOT/'run_extended_initialization_probe.py',ROOT/'run_constraint_row_recording.py'):
        committed(checks,hashes,path,'required input')
    cert=js(candidate_path.parent/'PARENT_INDEPENDENT_CHECK_FINAL.json')
    checks.add('S132 independent proposal and certificate',all(cert[k] is True for k in ('passed','evidence_passed','proposal_accepted','certificate_passed')))
    counts=dict(model_load=1,forward=121,step=120,neural_ticks=166)
    ledger=js(out.parent/'CALL_LEDGER.json')
    checks.add('complete no failed marker',js(out.parent/'COMPLETE.json')['status']=='COMPLETE' and not (out.parent/'FAILED.json').exists())
    checks.add('single original call ledger',ledger==dict(counts,active=None,completed=['init80ms'],per_arm={'init80ms':counts}),observed=ledger)
    result=js(out/'RESULT.json'); checks.add('RESULT calls exact',result['calls']==counts)
    checks.add('single result exact',js(out.parent/'RESULTS.json')==[result])
    report['artifact_hashes']={}
    for label,path in [('S133',out),('S121',reference),('S115',S115),('S113',S113)]:
        res=js(path/'RESULT.json'); canonical(res['hashes'])
        report['artifact_hashes'][label]=bound_hashes(checks,res['hashes'],label+' artifacts')
        if label in ('S113','S115'): committed(checks,hashes,path/'RESULT.json',label+' reference')
        for name in (*NPZS,*JSONS): committed(checks,res['hashes'],path/name,label+' required artifact')
        if label in ('S133','S121'):
            for name in ('CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'): committed(checks,res['hashes'],path/name,label+' row artifact')
    require_integrity(checks,'input commitments and completion')
    old=reconstruct_rows(checks,reference,'S121'); new=reconstruct_rows(checks,out,'S133')
    checks.add('same moment metadata',old['moment']==new['moment'])
    checks.add('all121 matched physical instants',exact(old['arrays']['time'],new['arrays']['time'])['identical'])
    checks.add('0..3ms 25us clock',np.allclose(new['arrays']['time'],np.arange(121)*.000025,rtol=0,atol=1e-12))
    for label,item in [('S121',old),('S133',new)]: checks.add(label+' recognized row classification only',item['counts']['unrecognized']==0 and item['counts']['other']==0)
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
    require_integrity(checks,'all121/all48 row closure and initialization')
    xmls=[Path(p) for p in hashes if Path(p).suffix=='.xml']; checks.add('unique canonical XML',len({str(p.resolve()) for p in xmls})==1)
    require_integrity(checks,'canonical XML')
    formal=js(formal_path); report['formal_constraints']=formal; report['per_arm']={}; measured={}
    for label,path in [('S113',S113),('S115',S115),('S133',out)]:
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
    rows,d=admission_table(formal,measured['S133'],not np.array_equal(a['applied'],b['applied']))
    checks.add('exactly 17 formal admission rows',len(rows)==17)
    report.update(admission_table=rows,original_helper_decision=d,S119_short_window_pass=all(row['passed'] for row in rows),admission_evaluated=True,baseline_promotion=False,checkpoint_replacement=False,events={label:events(item) for label,item in [('S121',old),('S133',new)]},closure={label:item['states'] for label,item in [('S121',old),('S133',new)]})
    chain=npz(candidate_path.parent/'OFFLINE_CANDIDATE.npz'); qp=npz(candidate_path.parent/'QP_PROBLEM.npz')
    checks.eq('S132 original reference weights',chain['reference_weights'],a['weights'],0)
    checks.eq('S132 exact proposed runtime weights',chain['proposed_weights'],candidate,0)
    checks.add('S132 exactly three qacc states',chain['predicted_delta_qacc'].shape==(3,48) and qp['Q'].shape==(3,48,15))
    delta=chain['delta'][qp['weight_indices'][:,0],qp['weight_indices'][:,1]]
    checks.eq('S132 three-state predicted delta independently',chain['predicted_delta_qacc'],np.einsum('sdj,j->sd',qp['Q'],delta),1e-12)
    require_integrity(checks,'prediction source arithmetic')
    return old,new,chain


def outputs(out,old,new,chain,report):
    a=old['arrays']; b=new['arrays']; t=b['time']; m=new['moment']; paths=[]
    header=['criterion','actual','reference','operator','margin','passed','reference_source']
    paths.append(save_csv(out,'ADMISSION_TABLE.csv',header,[[r[k] for k in header] for r in report['admission_table']]))
    c0=components(old); c1=components(new); states=[]; forces=[]
    for i,time in enumerate(t):
        for u in range(90):
            for dof in (6,7,8): forces.append([i,time*1000,u,dof,a['actuator_force'][i,u],b['actuator_force'][i,u],a['actuator_force'][i,u]*a['actuator_moment'][i,u,dof],b['actuator_force'][i,u]*b['actuator_moment'][i,u,dof]])
        for dof in (6,7,8):
            j=int(m['dof_jntid'][dof]); qi=int(m['jnt_qposadr'][j]); row=[i,time*1000,dof,m['joint_names'][j]]
            for item,c in [(old,c0),(new,c1)]:
                x=item['arrays']; _,curve,_=all42_limits(x,item['moment'])
                row.extend([c['qfrcact'][i,dof],c['contact'][i,dof],c['limit'][i,dof],c['other'][i,dof],x['qpos'][i,qi],x['qvel'][i,dof],x['qacc'][i,dof],x['load_ratio'][i],curve[i]])
            states.append(row)
    paths.append(save_csv(out,'ACTUAL_FORCES_TORQUES.csv',['sample','physical_ms','actuator','DOF','S121_force','S133_force','S121_torque','S133_torque'],forces))
    keys=['muscle','contact','limit','other_constraint','q','v','qacc','load_ratio','all42_max_excess']
    paths.append(save_csv(out,'COUPLED_DOF678.csv',['sample','physical_ms','DOF','joint']+[label+'_'+key for label in ('S121','S133') for key in keys],states))
    paths.append(save_csv(out,'ACTUAL_ROW_CLASSIFICATION.csv',['arm','sample','physical_ms','row','enum_value','enum_name','id','category','contact','geom1','geom2','joint','force','projection6','projection7','projection8','J_all48','projection_all48'],old['mapping']+new['mapping']))
    predictions=[]
    raw=npz(S132/'RAW_PROBES.npz')
    for k,i in enumerate(STATES):
        for dof in range(48):
            fixedbase=raw[f'sample{i:03d}_baseline_qacc'][dof]; fixedcandidate=raw[f'sample{i:03d}_candidate_qacc'][dof]
            pred=chain['predicted_delta_qacc'][k,dof]; actual=b['qacc'][i,dof]-a['qacc'][i,dof]
            predictions.append([i,t[i]*1000,dof,pred,fixedcandidate-fixedbase,a['qacc'][i,dof],b['qacc'][i,dof],actual,actual-pred])
    paths.append(save_csv(out,'THREE_STATE_QACC_PREDICTION_ERROR.csv',['sample','physical_ms','DOF','S132_linear_fixed_state_delta','S132_nonlinear_fixed_state_delta','S121_actual_qacc','S133_actual_qacc','coupled_actual_delta','coupled_minus_linear'],predictions))
    report['prediction_scope']='Only samples77/100/105, fixed S121 mechanical states. Coupled S133 states differ. No full121 prediction or causal error attribution.'
    limits=report['per_arm']['S133']['physical']['all42_joint_limits']
    if limits:
        h=list(limits[0]); paths.append(save_csv(out,'ALL42_LIMIT_EXTREMA.csv',h,[[r[k] for k in h] for r in limits]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,3,figsize=(17,12)); ms=t*1000
    for col,dof in enumerate((6,7,8)):
        j=int(m['dof_jntid'][dof]); qi=int(m['jnt_qposadr'][j])
        for label,item,c,ls in [('S121',old,c0,'--'),('S133',new,c1,'-')]:
            for key in ('qfrcact','contact','limit'): axs[0,col].plot(ms,c[key][:,dof],ls=ls,label=label+' '+key)
            axs[1,col].plot(ms,item['arrays']['qpos'][:,qi],ls=ls,label=label+' q')
        axs[1,col].axhline(m['jnt_range'][j][1],color='k',ls=':',label='upper')
        ti=ms[list(STATES)]
        axs[2,col].scatter(ti,chain['predicted_delta_qacc'][:,dof],label='S132 linear fixed-state delta',marker='x')
        nonlinear=[raw[f'sample{i:03d}_candidate_qacc'][dof]-raw[f'sample{i:03d}_baseline_qacc'][dof] for i in STATES]
        axs[2,col].scatter(ti,nonlinear,label='S132 nonlinear fixed-state delta',marker='+')
        axs[2,col].plot(ms,b['qacc'][:,dof]-a['qacc'][:,dof],label='S133-S121 actual coupled delta')
        axs[0,col].set_title('DOF'+str(dof)+' '+m['joint_names'][j])
    for ax in axs.flat: ax.legend(fontsize=6); ax.grid(alpha=.25); ax.set_xlabel('physical ms')
    fig.suptitle('S133 actual coupled states / S132 THREE fixed-state predictions; NOT closed-loop predictions')
    fig.tight_layout(); p=out/'COUPLED_PREDICTED_ACTUAL.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    fig,axs=plt.subplots(3,2,figsize=(14,12))
    for label in ('S113','S115','S133'):
        p=report['per_arm'][label]['physical']; path=Path(report['per_arm'][label]['path']); x=npz(path/'TRAJECTORY.npz'); _,curve,_=all42_limits(x,js(path/'MOMENT_METADATA.json'))
        angle=2*np.arccos(np.clip(np.abs(x['qpos'][:,3:7]@x['qpos'][0,3:7])/(np.linalg.norm(x['qpos'][:,3:7],axis=1)*np.linalg.norm(x['qpos'][0,3:7])),0,1))
        for ax,y in zip(axs.flat,[p['load_ratio'],curve,x['qpos'][:,2]/x['qpos'][0,2],angle,x['nonfoot_depth'],x['com_velocity_world'][:,2]]): ax.plot(x['time']*1000,y,label=label)
    for ax,title in zip(axs.flat,['load ratio','all42 maximum limit excess','root height ratio','orientation angle','nonfoot depth','whole-body COM vertical velocity']): ax.set_title(title); ax.grid(alpha=.25); ax.legend(); ax.set_xlabel('physical ms')
    axs[0,0].axhline(.8,color='k',ls=':'); axs[0,1].axhline(1e-6,color='k',ls=':')
    fig.suptitle('S119 short-window benefit / full-body safety; units NOT_ESTABLISHED; no promotion')
    fig.tight_layout(); p=out/'LOAD_FULLBODY_SAFETY.png'; fig.savefig(p,dpi=130); plt.close(fig); paths.append(str(p))
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT); p.add_argument('--reference',type=Path,default=REFERENCE); p.add_argument('--candidate',type=Path,default=CANDIDATE); p.add_argument('--formal',type=Path,default=FORMAL); args=p.parse_args(argv)
    out=args.output.resolve(); checks=Checks(); target=out/'MARGIN_CANDIDATE_INDEPENDENT_CHECK.json'
    if target.exists(): raise ValueError('never overwrite previous verification report')
    report=dict(passed=False,admission_evaluated=False,S119_short_window_pass=False,baseline_promotion=False,checkpoint_replacement=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),physical_units='NOT_ESTABLISHED',behavior_acceptance=False)
    try:
        old,new,chain=verify(out,args.reference.resolve(),args.candidate.resolve(),args.formal.resolve(),checks,report)
        report['outputs']=outputs(out,old,new,chain,report)
        checks.add('offline final boundary',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
        report['passed']=checks.passed
        if not checks.passed: report.update(S119_short_window_pass=False,admission_evaluated=False)
    except Exception as exc: report.update(passed=False,S119_short_window_pass=False,admission_evaluated=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['helper_source_hashes']={str(ROOT/name):sha(ROOT/name) for name in ('verify_joint_candidate_probe.py','verify_com_contact_probe.py','verify_constraint_row_recording.py','verify_secant_constraint_pair.py','verify_secant_weight_probe.py','verify_single_weight_attenuation.py','verify_com_timestep_probe.py','verify_initial_height_probe.py')}
    report['checks']=checks.rows
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],S119_short_window_pass=report['S119_short_window_pass'],admission_evaluated=report['admission_evaluated'],report=str(target),error=report.get('error')),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
