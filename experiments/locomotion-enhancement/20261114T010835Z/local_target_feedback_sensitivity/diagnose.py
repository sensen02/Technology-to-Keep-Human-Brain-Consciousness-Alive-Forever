"""Explicit-parent-only actual local FD sensitivity. No numerical imports on import."""
from pathlib import Path
import argparse,hashlib,inspect,json,os,resource,signal,sys,threading,time,traceback
import contracts as C
ROOT=C.ROOT


def execute(checkpoint,digest):
    C.require(Path(sys.executable).absolute()==C.PYTHON.absolute(),'Qualified125712 Python required')
    C.require(checkpoint==C.CHECKPOINT and digest==C.CHECKPOINT_SHA,'Only actual source-bound new600 identity admitted')
    C.require(not any(n in sys.modules for n in ('torch','numpy','mujoco','physics','flygym')),'Fresh process before numerical imports required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    # Pin before creating watchdog or importing Torch so all descendant threads
    # inherit this mask; sched_setaffinity(0) later would affect main only.
    single_cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{single_cpu})
    out=ROOT/('RUN_'+digest);out.mkdir(exist_ok=False)
    started=time.monotonic();rows=[];points=[];matrices={};ledgers={};checks={};counts=dict(public_step_details=0,explicit_old_step=0,internal_old_step_from_public=0)
    result=dict(kind='LOCAL_ACTUAL_TARGET_FEEDBACK_SENSITIVITY_V1',status='FAILED',diagnostic_integrity_pass=False,checkpoint=str(checkpoint),checkpoint_sha256=digest,training_updates=0,optimizer_operations=0,gradient_operations=0,Core_constructions=0,physical_steps=0,full138_map_performed=False,fullbody_stability_established=False,root_cause_established=False)
    stop=threading.Event()
    def gate(*_):
        if time.monotonic()-started>=300:raise RuntimeError('Diagnostic300s wall budget reached; no point/epsilon reduction')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>1800:raise RuntimeError('Diagnostic1800MiB RSS budget reached')
    def limit(*_):raise RuntimeError('Diagnostic resource watchdog limit reached')
    old_alarm=signal.signal(signal.SIGALRM,limit);old_usr=signal.signal(signal.SIGUSR1,limit);signal.alarm(300)
    def monitor():
        while not stop.wait(.2):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>1800:os.kill(os.getpid(),signal.SIGUSR1);return
    threading.Thread(target=monitor,daemon=True).start()
    try:
        support=C.load_support();cal_proof,tr=support.final_qualification(checkpoint,digest);body_proof,selected,saved_body_sources=C.body_qualification()
        batch_path=C.SCOPE/'final_batch_parity_audit'/('RUN_'+digest)/'VERIFY.json';batch=C.read(batch_path)
        C.require(batch['status']=='PASS' and batch['qualified_final600'] is True and batch['checkpoint_sha256']==digest and batch['checks'] and all(v is True for v in batch['checks'].values()),'Actual final batch gate must PASS first')
        ledgers['proof_before']={**cal_proof,**body_proof,str(batch_path):C.sha(batch_path)}
        critical=[ROOT/name for name in ('PREREG.json','contracts.py','diagnose.py','staticcheck.py','README.zh-CN.md')]+[C.CAL/'INPUT_BINDINGS.json',C.SUPPORT,C.BODY/'adapter.py',C.BODY/'runner.py',C.BODY/'audit.py',C.BODY/'run.py']
        extra_sources=C.merge_ledgers(tr['source_hashes'],C.ledger(critical));C.verify(extra_sources)
        source_before=C.merge_ledgers(saved_body_sources,extra_sources);ledgers['source_before']=source_before
        ledgers['saved_body_source_provenance']=saved_body_sources
        print('ALL_FINAL_BODY_AND_CALIBRATION_EVIDENCE_ADMITTED',flush=True);gate()
        with support.v4_context() as modules:
            learning=modules['learning'];learning.cpu1()
            import torch,numpy as np
            torch.set_num_interop_threads(1)
            result['environment']=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)),OMP=os.environ['OMP_NUM_THREADS'],OPENBLAS=os.environ['OPENBLAS_NUM_THREADS'],MKL=os.environ['MKL_NUM_THREADS'])
            C.require(torch.get_num_threads()==1 and torch.get_num_interop_threads()==1 and len(os.sched_getaffinity(0))==1,'Actual singleCPU/threads required')
            task_affinities={}
            for tid in os.listdir('/proc/self/task'):
                try:task_affinities[tid]=sorted(os.sched_getaffinity(int(tid)))
                except ProcessLookupError:continue
            C.require(bool(task_affinities) and all(mask==[single_cpu] for mask in task_affinities.values()),'Every live process thread must share the same CPU1 mask')
            result['environment']['all_live_thread_affinities']=task_affinities
            rng=learning.rng_snapshot();m,payload=modules['common'].load_checkpoint(checkpoint,digest,final=True)
            checks['strict_final600_and_state_ABI']=payload['step']==600 and m.identity()['state_dims']==[128,64,64,96]
            checks['strict_load_rng_preserved']=learning.rng_equal(rng,learning.rng_snapshot())
            weights=modules['model'].state_digest(m.state_dict());result['weights_digest_before']=weights
            result['public_source']={name:inspect.getsource(getattr(m,name)) for name in ('step_details','cached_features','legacy_projection')}
            result['old_source']=inspect.getsource(m.old.step)
            C.require('self.old.step(obs,self.legacy_projection(command),old_state)' in result['public_source']['cached_features'],'Actual frozen source API/order changed')
            C.require('oldmean,features,after=self.cached_features' in result['public_source']['step_details'],'Actual public step_details source changed')
            print('STRICT600_LOADED',round(time.monotonic()-started,3),flush=True);gate()
            binding=C.read(C.CAL/'INPUT_BINDINGS.json');mapping=binding['normalizer_mapping']
            q0=torch.tensor(mapping['q0_42'],dtype=torch.float64);half=torch.tensor(mapping['halfspan42'],dtype=torch.float64);qa=np.asarray(mapping['qpos_indices'],dtype=int)
            C.require(tuple(q0.shape)==tuple(half.shape)==(42,) and qa.shape==(42,) and bool(torch.isfinite(q0).all() and torch.isfinite(half).all()) and bool((half>=.1).all()),'Explicit actual normalizer42 required')
            C.require(np.array_equal(np.asarray(mapping['key_qpos49'],dtype=np.float64)[qa],q0.numpy()),'q0 actual compiled key_qpos mapping differs')
            C.require(torch.equal(m.old.base.initial_halfspan,half),'Frozen base initial normalizer differs')
            C.require(tuple(m.old.base.physical_mask.shape)==(238,) and not bool(m.old.base.physical_mask[42:84].any()) and not bool(m.old.base.physical_mask[126:168].any()),'Frozen physical mask must exclude both target fields')
            gain=m.old.base.gain().detach().clone();C.require(tuple(gain.shape)==(42,) and bool(torch.isfinite(gain).all()),'Source frozen gain42 required')
            result['frozen_gain42']=gain.tolist();result['normalizers_source']=dict(path=str(C.CAL/'INPUT_BINDINGS.json'),sha256=C.sha(C.CAL/'INPUT_BINDINGS.json'),q0_42=q0.tolist(),halfspan42=half.tolist(),qpos_indices=qa.tolist())
            def encode(x,target,qactual):
                xx=x.clone();xx[:,42:84]=((target-qactual)/half).clamp(-10.,10.).float().double();xx[:,126:168]=((target-q0)/half).clamp(-10.,10.).float().double();return xx
            def integrate(action,target,bounds,max_rate):
                rate=torch.clamp(action*2.,-2.,2.);limited=torch.minimum(torch.maximum(rate,-max_rate),max_rate)
                for _ in range(5):target=torch.minimum(torch.maximum(target+limited*.001,bounds[:,0]),bounds[:,1])
                return target
            def clone_state(state):return tuple(value.clone() for value in state)
            def call_new(xx,u,state):
                counts['public_step_details']+=1;counts['internal_old_step_from_public']+=1
                return m.step_details(xx,u,clone_state(state))
            def call_old(xx,u,state):
                counts['explicit_old_step']+=1
                return m.old.step(xx,m.legacy_projection(u),clone_state(state[:3]))
            def stats(matrix):
                C.require(matrix.ndim==2 and bool(np.isfinite(matrix).all()),'Finite FD matrix required')
                values=dict(frobenius_norm=float(np.linalg.norm(matrix)),operator_2_norm=float(np.linalg.norm(matrix,ord=2)),max_abs=float(np.abs(matrix).max()))
                if matrix.shape[0]==matrix.shape[1]:
                    eig=np.linalg.eigvals(matrix);values.update(spectral_radius=float(np.max(np.abs(eig))),eigenvalues_real=eig.real.tolist(),eigenvalues_imag=eig.imag.tolist(),positive_diagonal_count=int(np.sum(np.diag(matrix)>0.)),diagonal=np.diag(matrix).tolist())
                return values
            with torch.no_grad():
                for case in ('stop','walk'):
                    identity=selected[case]['identity'];compiled=identity['compiled_arrays'];normalizer_proof={}
                    for name,value in (('jnt_range',mapping['joint_ranges43']),('key_qpos',[mapping['key_qpos49']])):
                        spec=compiled[name];array=np.asarray(value,dtype=np.dtype(spec['dtype']));ok=list(array.shape)==spec['shape'] and hashlib.sha256(array.tobytes()).hexdigest()==spec['sha256'];C.require(ok,'Actual compiled normalizer SHA differs: '+case+'/'+name);normalizer_proof[name]=dict(dtype=spec['dtype'],shape=spec['shape'],sha256=spec['sha256'])
                    bounds=torch.tensor(identity['actuator']['target_bounds_rad'],dtype=torch.float64);max_rate=torch.tensor(identity['actuator']['max_rate_rad_s'],dtype=torch.float64)
                    native=np.asarray([mapping['joint_ranges43'][1+int(address-7)] for address in qa],dtype=np.float64)
                    C.require(tuple(bounds.shape)==(42,2) and tuple(max_rate.shape)==(42,) and np.array_equal(native,bounds.numpy()) and bool((max_rate>=0).all()),'Native actual bounds/rates42 required')
                    C.require(torch.equal(half,torch.tensor(np.maximum((native[:,1]-native[:,0])*.5,.1),dtype=torch.float64)),'Actual halfspan/ranges relation differs')
                    folder=C.BODY/'trials'/case;casepoints=[p for p in C.POINTS if p[0]==case];indices=[p[3] for p in casepoints];pre=[p[4] for p in casepoints];post=[p[2]+4 for p in casepoints]
                    with np.load(folder/'neural_decisions.npz',allow_pickle=False) as archive:
                        saved={key:archive[key][indices].copy() for key in ('time','physical_step_ms','X238','command','hidden_before','hidden_after','mean','residual','policy_action','actually_applied_action','control_source')}
                    with np.load(folder/'trace_1ms.npz',allow_pickle=False) as archive:
                        tpre=archive['target'][pre].copy();qactuals=archive['qpos'][pre][:,qa].copy();tpost=archive['target'][post].copy();times=archive['time'][pre].copy()
                    C.require(all(bool(np.isfinite(value).all()) for value in (*saved.values(),tpre,qactuals,tpost,times)),'Selected actual records finite')
                    for slot,(_,relative,ms,index,traceindex) in enumerate(casepoints):
                        gate();label=case+'_'+str(ms);x=torch.tensor(saved['X238'][slot:slot+1],dtype=torch.float64);u=torch.tensor(saved['command'][slot:slot+1],dtype=torch.float64);h=torch.tensor(saved['hidden_before'][slot:slot+1],dtype=torch.float64)
                        state=tuple(torch.split(h,(128,64,64,96),dim=-1));target=torch.tensor(tpre[slot],dtype=torch.float64);qactual=torch.tensor(qactuals[slot],dtype=torch.float64)
                        C.require(int(saved['physical_step_ms'][slot])==ms and abs(float(saved['time'][slot])-ms*.001)<=1e-8 and abs(float(times[slot])-ms*.001)<=1e-8 and int(saved['control_source'][slot])==3,'Exact preaction source/clock required')
                        C.require(torch.equal(u,u.new_tensor([[0.,0.] if case=='stop' else [1.,0.]])),'Actual case command differs')
                        xx=encode(x,target,qactual);C.require(torch.equal(xx,x),'Baseline actual target encoding must be bitexact before FD')
                        mean,after,residual=call_new(xx,u,state);oldmean,oldafter=call_old(xx,u,state);action=mean.tanh()[0];pred=integrate(action,target,bounds,max_rate)
                        baseline=dict(mean=float(np.abs(mean[0].numpy()-saved['mean'][slot]).max()),action=float(np.abs(action.numpy()-saved['policy_action'][slot]).max()),actually_applied_action=float(np.abs(action.numpy()-saved['actually_applied_action'][slot]).max()),residual=float(np.abs(residual[0].numpy()-saved['residual'][slot]).max()),after352=float(np.abs(torch.cat(after,-1)[0].numpy()-saved['hidden_after'][slot]).max()),posttarget=float(np.abs(pred.numpy()-tpost[slot]).max()),old_direct_vs_mean_minus_residual=float((oldmean-(mean-residual)).abs().max()))
                        exactold=all(torch.equal(a,b) for a,b in zip(oldafter,after[:3])) and np.array_equal(torch.cat(oldafter,-1)[0].numpy(),saved['hidden_after'][slot,:256])
                        C.require(max(baseline.values())<=1e-12 and exactold,'Actual baseline replay/actuator parity failed: '+label)
                        point=dict(case=case,relative_ANN_s=relative,physical_step_ms=ms,neural_row=index,preaction_trace_row=traceindex,postaction_trace_row=ms+4,baseline_errors=baseline,old256exact=exactold,normalizer_compiled_proof=normalizer_proof,body_task_pass=selected[case]['result']['candidate_metrics']['task_pass_candidate'])
                        points.append(point);checks[label+'_baseline']=True
                        matrices[label+'_target']=target.numpy().copy();matrices[label+'_qactual']=qactual.numpy().copy();matrices[label+'_incoming352']=h.numpy().copy();matrices[label+'_baseline_after96']=after[3][0].numpy().copy()
                        pointrows=[];postmat=[];meandiags=[]
                        for epsilon in C.EPSILONS:
                            arrays={key:np.empty(shape,dtype=np.float64) for key,shape in (('new_dmean_dT',(42,42)),('new_daction_dT',(42,42)),('new_dpostT_dT',(42,42)),('new_dafter96_dT',(96,42)),('old_dmean_dT',(42,42)),('old_dpostT_dT',(42,42)),('quantized_old_gain_expectation',(42,42)))};arrays['quantized_old_gain_expectation'].fill(0.);outside=[]
                            for j in range(42):
                                plus=target.clone();minus=target.clone();plus[j]+=epsilon;minus[j]-=epsilon
                                xp=encode(x,plus,qactual);xm=encode(x,minus,qactual)
                                mp,sp,_=call_new(xp,u,state);mm,sm,_=call_new(xm,u,state);op,osp=call_old(xp,u,state);om,osm=call_old(xm,u,state)
                                C.require(all(torch.equal(a,b) for oldstate in (sp[:3],sm[:3],osp,osm) for a,b in zip(oldstate,oldafter)),'Target perturbation changed frozen old256 unexpectedly')
                                ap=mp.tanh()[0];am=mm.tanh()[0]
                                arrays['new_dmean_dT'][:,j]=((mp-mm)[0]/(2*epsilon)).numpy();arrays['new_daction_dT'][:,j]=((ap-am)/(2*epsilon)).numpy();arrays['new_dafter96_dT'][:,j]=((sp[3]-sm[3])[0]/(2*epsilon)).numpy()
                                arrays['new_dpostT_dT'][:,j]=((integrate(ap,plus,bounds,max_rate)-integrate(am,minus,bounds,max_rate))/(2*epsilon)).numpy()
                                arrays['old_dmean_dT'][:,j]=((op-om)[0]/(2*epsilon)).numpy();arrays['old_dpostT_dT'][:,j]=((integrate(op.tanh()[0],plus,bounds,max_rate)-integrate(om.tanh()[0],minus,bounds,max_rate))/(2*epsilon)).numpy()
                                arrays['quantized_old_gain_expectation'][j,j]=float(-gain[j]*(xp[0,126+j]-xm[0,126+j])/(2*epsilon))
                                outside.append(dict(channel=j,plus_outside_bounds=bool(plus[j]<bounds[j,0] or plus[j]>bounds[j,1]),minus_outside_bounds=bool(minus[j]<bounds[j,0] or minus[j]>bounds[j,1])))
                                if j%14==0:gate()
                            olderror=float(np.abs(arrays['old_dmean_dT']-arrays['quantized_old_gain_expectation']).max());C.require(olderror<=1e-8,'Old fixed-state FD disagrees with source quantized negative prior')
                            token=label+'_eps_'+format(epsilon,'.4g')
                            for key,value in arrays.items():matrices[token+'_'+key]=value
                            row=dict(point=label,case=case,relative_ANN_s=relative,epsilon=epsilon,FD_mode='central_native_target_rad_with_float32_encoding',old_gain_quantization_max_error=olderror,symmetric_probes_outside_bounds=outside,statistics={key:stats(value) for key,value in arrays.items() if key!='quantized_old_gain_expectation'})
                            rows.append(row);pointrows.append(row);postmat.append(arrays['new_dpostT_dT']);meandiags.append(np.diag(arrays['new_dmean_dT']))
                            checks[token+'_oldFD_reference']=True;print('POINT_EPS_DONE',label,epsilon,round(time.monotonic()-started,3),flush=True);gate()
                        rhos=[row['statistics']['new_dpostT_dT']['spectral_radius'] for row in pointrows];reldiff=float(np.linalg.norm(postmat[0]-postmat[1])/max(np.linalg.norm(postmat[0]),np.linalg.norm(postmat[1]),1e-12));consistent=abs(rhos[0]-rhos[1])<=.001 and reldiff<=.05
                        point.update(epsilon_post_matrix_relative_Frobenius_difference=reldiff,epsilon_spectral_radius_abs_difference=abs(rhos[0]-rhos[1]),epsilon_consistent=bool(consistent),amplification_flag=bool(consistent and all(rho>=1.0001 for rho in rhos)),robust_positive_mean_diagonal_channels=np.flatnonzero((meandiags[0]>=1e-4)&(meandiags[1]>=1e-4)).tolist())
                        matrices[label+'_smooth_unclipped_old_dmean_diagonal']=(-gain/half).numpy().copy();point['baseline_normalized_offset']=((target-q0)/half).tolist();point['clipped_normalized_channels']=torch.nonzero(((target-q0)/half).abs()>=10).flatten().tolist()
            checks['all6points_2epsilon_complete']=len(points)==6 and len(rows)==12
            checks['exact_registered_forward_counts']=counts==dict(public_step_details=1014,explicit_old_step=1014,internal_old_step_from_public=1014)
            checks['all_weights_unchanged']=modules['model'].state_digest(m.state_dict())==weights;result['weights_digest_after']=modules['model'].state_digest(m.state_dict())
            checks['frozen_old_verified']=m.verify_old();checks['no_gradients_created']=all(p.grad is None for p in m.parameters());checks['entire_RNG_unchanged']=learning.rng_equal(rng,learning.rng_snapshot())
            checks['no_Core_or_physical_backend']=not any(n in sys.modules for n in ('mujoco','physics','flygym','static_teacher','student_wave'))
        checks['public_alias_context_restored']=True;gate()
        ledgers['source_after']={path:C.sha(path) for path in source_before};checks['source_SHA_unchanged']=ledgers['source_after']==source_before
        ledgers['proof_after']={path:C.sha(path) for path in ledgers['proof_before']};checks['body_cal_batch_proof_SHA_unchanged']=ledgers['proof_after']==ledgers['proof_before'];checks['checkpoint_SHA_unchanged']=C.sha(checkpoint)==digest
        np.savez_compressed(out/'FD_ARRAYS.npz',**matrices);result['FD_arrays']=dict(path=str(out/'FD_ARRAYS.npz'),sha256=C.sha(out/'FD_ARRAYS.npz'));gate()
        result['diagnostic_integrity_pass']=all(checks.values());result['status']='COMPLETED' if result['diagnostic_integrity_pass'] else 'FAIL'
        result['points']=points
    except BaseException as error:
        result.update(status='FAILED',diagnostic_integrity_pass=False,exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:
        final_success=False
        counts['top_level_neural_evaluations']=counts['public_step_details']+counts['explicit_old_step'];counts['model_step_invocations_including_nested']=counts['top_level_neural_evaluations']+counts['internal_old_step_from_public']
        result.update(checks=checks,NN_calls=counts,completed_points=points)
        try:
            C.save(out/'ROWS.json',rows);C.save(out/'SOURCE_LEDGER.json',ledgers)
            result['artifacts']={name:dict(path=str(out/name),sha256=C.sha(out/name)) for name in ('ROWS.json','SOURCE_LEDGER.json')}
            lines=['# 实际局部目标反馈敏感性诊断','',f"科学阶段状态：{result['status']}；最终资格还必须检查RESOURCE_COMPLETION.pass及脚本exit0。固定physical/incoming352局部有限差分不是完整身体稳定性证明。",'',f"训练/optimizer/Core/物理均0；NN调用：{json.dumps(counts)}；完整输出后的wall/RSS见RESOURCE_COMPLETION.json。",'']
            for point in points:lines.append(f"- {point['case']} ANN+{point['relative_ANN_s']}s：epsilon一致={point.get('epsilon_consistent')}，局部postT amplification flag={point.get('amplification_flag')}，两epsilon稳健正mean对角通道={point.get('robust_positive_mean_diagonal_channels')}；body原行为PASS={point['body_task_pass']}（不改旧结论）。")
            lines.extend(['','raw矩阵/特征值/baseline/身份见FD_ARRAYS.npz、ROWS.json、RESULT.json。未实施138维完整map，不推断fullbody稳定性。FINALIZATION_FAILURE.json存在或RESOURCE_COMPLETION缺失/不通过，整次审核都不通过。',result.get('traceback','')]);(out/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
            gate();result.update(wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            C.save(out/'RESULT.json',result);gate()
            names=['RESULT.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md']
            if (out/'FD_ARRAYS.npz').is_file():names.append('FD_ARRAYS.npz')
            artifacts=C.ledger(out/name for name in names);gate()
            completion=dict(kind='LOCAL_SENSITIVITY_FINAL_RESOURCE_COMMIT',wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,artifacts=artifacts,science_and_success_output_included=True,failure_bookkeeping_never_qualifies=True)
            completion['pass']=result['status']=='COMPLETED' and result['diagnostic_integrity_pass'] is True and completion['wall_s']<=300 and completion['RSS_MiB']<=1800
            C.save(out/'RESOURCE_COMPLETION.json',completion);gate()
            final_success=C.completion_success(result,completion)
        except BaseException as error:
            result.update(status='FAILED',diagnostic_integrity_pass=False,finalization_error=repr(error))
            failure=dict(kind='LOCAL_SENSITIVITY_FINALIZATION_FAILURE',status='FAILED',diagnostic_integrity_pass=False,error=repr(error),wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            # Failure bookkeeping can finish after a resource signal; it never
            # authorizes scientific completion or resumes any model operations.
            try:
                C.save(out/'FINALIZATION_FAILURE.json',failure)
                if not (out/'RESULT.json').exists():C.save(out/'RESULT.json',result)
                if not (out/'RESOURCE_COMPLETION.json').exists():C.save(out/'RESOURCE_COMPLETION.json',dict(failure,**{'pass':False}))
            except BaseException:pass
            print('FINALIZATION_FAILED',repr(error),flush=True)
        finally:
            stop.set();signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGUSR1,old_usr)
        print('LOCAL_TARGET_SENSITIVITY',result['status'],'resource_commit',final_success,str(out/'RESULT.json'),flush=True)
    return 0 if final_success and result['status']=='COMPLETED' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-local-target-sensitivity',action='store_true');a=p.parse_args()
    if not a.execute_parent_local_target_sensitivity:p.error('Explicit parent release required; this scope is ONLYCODE until released')
    if Path(a.checkpoint).resolve()!=C.CHECKPOINT or a.sha256!=C.CHECKPOINT_SHA:p.error('Exact actual source-bound final600 path and SHA required')
    return execute(Path(a.checkpoint).resolve(),a.sha256)
if __name__=='__main__':sys.exit(main())
