"""Parent-only final900 actualTF/conditional batch audit. Stdlib until execute flag."""
from pathlib import Path
import argparse,json,os,resource,signal,sys,threading,time,traceback
import contracts as C
ROOT=C.ROOT

def execute(checkpoint,digest):
    C.require(Path(sys.executable).absolute()==C.PYTHON.absolute(),'Use qualified125712 Python')
    C.require(not any(name in sys.modules for name in ('torch','numpy','mujoco','physics','flygym')),'Fresh neural-only process required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    single_cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{single_cpu})
    out=ROOT/('RUN_'+digest);out.mkdir(exist_ok=False)
    started=time.monotonic();rows=[];checks={};ledgers={};counts=dict(data_prepare=0,conditional_fullproduction=0,actual_forced_fullproduction=0,conditional_scan=0,actual_TF_scan=0)
    result=dict(kind='INDEPENDENT_STATE_ALIGNED_FINAL900_BATCH_AUDIT_V1',status='FAILED',qualified_final900=False,checkpoint=str(checkpoint),checkpoint_sha256=digest,format_version=C.FORMAT,final_step=C.FINAL_STEP,gate=C.GATE,training_updates=0,optimizer_operations=0,gradient_operations=0,Core_constructions=0,physical_steps=0,physical_behavior_qualified=False)
    stop=threading.Event()
    def gate(*_):
        if time.monotonic()-started>=C.WALL_S:raise RuntimeError('Predeclared final900 audit600s budget reached; no automatic extension')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>C.RSS_MIB:raise RuntimeError('Final900 audit1800MiB RSS limit reached')
    def limit(*_):raise RuntimeError('Final900 audit resource watchdog limit reached')
    previous_alarm=signal.signal(signal.SIGALRM,limit);previous_usr=signal.signal(signal.SIGUSR1,limit);signal.alarm(C.WALL_S)
    def monitor():
        while not stop.wait(.2):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>C.RSS_MIB:os.kill(os.getpid(),signal.SIGUSR1);return
    threading.Thread(target=monitor,daemon=True).start()
    try:
        frozen=C.read(ROOT/'FROZEN.json');C.verify(frozen['source_hashes'])
        proof_before,tr=C.final_qualification(checkpoint,digest);ledgers['final_qualification_before']=proof_before
        producer_source=dict(tr['source_hashes']);C.verify(producer_source)
        own_source=C.ledger([ROOT/name for name in ('audit.py','contracts.py','staticcheck.py','README.zh-CN.md','PREREG.json','FROZEN.json')]);source_before=C.merge(producer_source,own_source);ledgers['source_before']=source_before
        result['qualification_basis']='FINAL900_ALL3_STAGES_STORAGE600_750_900_EVAL_AND_REUSED_ACTUAL96_WALL9000'
        print('ACTUAL_FINAL900_COMPLETE_QUALIFICATION_ACCEPTED',flush=True);gate()
        with C.producer_context() as modules:
            learning=modules['learning'];learning.cpu1()
            import torch
            torch.set_num_interop_threads(1)
            affinities={}
            for tid in os.listdir('/proc/self/task'):
                try:affinities[tid]=sorted(os.sched_getaffinity(int(tid)))
                except ProcessLookupError:continue
            C.require(torch.get_num_threads()==1 and torch.get_num_interop_threads()==1 and bool(affinities) and all(mask==[single_cpu] for mask in affinities.values()),'All live threads must share pre-pinned CPU1')
            result['environment']=dict(python=sys.executable,torch=torch.__version__,intra_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),all_live_thread_affinities=affinities,OMP=os.environ['OMP_NUM_THREADS'],BLAS=os.environ['OPENBLAS_NUM_THREADS'],MKL=os.environ['MKL_NUM_THREADS'])
            rng=learning.rng_snapshot();m,payload=modules['common'].load_checkpoint(checkpoint,digest,final=True)
            checks['strict_actual_new900_FORMAT']=payload['step']==C.FINAL_STEP and payload['format_version']==C.FORMAT and m.identity()['format_version']==C.FORMAT and m.identity()['state_dims']==[128,64,64,96]
            C.require(checks['strict_actual_new900_FORMAT'],'Strict final900 format/state migration identity differs')
            checks['strict_loader_RNG_preserved']=learning.rng_equal(rng,learning.rng_snapshot())
            C.require(payload['config']['source_hashes']==producer_source and payload['config']['data_hashes']==tr['data_hashes'],'Actual loaded final900 provenance differs from final TRAIN/EVAL')
            result['weights_digest_before']=modules['model'].state_digest(m.state_dict());print('STRICT_NEW900_LOADED',round(time.monotonic()-started,3),flush=True);gate()
            with torch.no_grad():
                counts['data_prepare']+=1;data=modules['data'].prepare(m)
                C.require(len(data['raw'])==12 and tuple(data['physical'].shape)==(1920,12,112) and tuple(data['oldafter'].shape)==(1920,12,256),'All12 full1920 source/cache schema required')
                conditional=modules['scan'].make_scan(m);actual=modules['scan'].make_actual_scan(m);refs_cond=[];refs_tf=[]
                def errors(values,reference,names):
                    C.require(len(values)==len(reference)==len(names),'Real producer output arity mismatch')
                    C.require(all(a.shape==b.shape and a.shape[0]==1920 and bool(torch.isfinite(a).all() and torch.isfinite(b).all()) for a,b in zip(values,reference)),'All full1920 output shapes and values must be finite')
                    return {name:float((a-b).abs().max()) for name,a,b in zip(names,values,reference)}
                def outputs(part):
                    counts['conditional_scan']+=1;cond=learning.forward(conditional,part)
                    counts['actual_TF_scan']+=1;tf=learning.tf_forward(actual,part)
                    return cond,tf
                def record(i,mode,cond,tf,condref,tfref):
                    ce=errors(cond,condref,C.COND_NAMES);te=errors(tf,tfref,C.TF_NAMES)
                    rows[i]['conditional'][mode]=ce;rows[i]['actualX_TF'][mode]=te
                    checks[f'{i}_{rows[i]["id"]}_{mode}_conditional6']=rows[i]['old256_conditional_exact'] and max(ce.values())<=C.GATE
                    checks[f'{i}_{rows[i]["id"]}_{mode}_actualTF4']=rows[i]['old256_actual_exact'] and max(te.values())<=C.GATE
                for i,raw in enumerate(data['raw']):
                    counts['conditional_fullproduction']+=1;full,old=modules['utils'].production_feedback(m,raw,data,i)
                    counts['actual_forced_fullproduction']+=1;forced,old_actual=modules['utils'].production_feedback(m,raw,data,i,forced=True)
                    # The true TF API has four outputs. Forced actual targets
                    # reproduce actual recordedX, without inventing TF states.
                    tfref=(forced[0],forced[1],forced[2],forced[5]);refs_cond.append(full);refs_tf.append(tfref)
                    rows.append(dict(id=raw['entry']['id'],index=i,actual_rows=1920,old256_conditional_exact=torch.equal(old,data['oldafter'][:,i]),old256_actual_exact=torch.equal(old_actual,data['oldafter'][:,i]),conditional={},actualX_TF={}))
                    cond,tf=outputs(learning.slice_data(data,i,i+1));record(i,'B1',tuple(v[:,0] for v in cond),tuple(v[:,0] for v in tf),full,tfref)
                    del cond,tf,old,old_actual;gate()
                batch_cond,batch_tf=outputs(data)
                for i in range(12):record(i,'B12',tuple(v[:,i] for v in batch_cond),tuple(v[:,i] for v in batch_tf),refs_cond[i],refs_tf[i])
                del batch_cond,batch_tf;gate()
                for lo in (0,4,8):
                    micro_cond,micro_tf=outputs(learning.slice_data(data,lo,lo+4))
                    for j in range(4):
                        i=lo+j;record(i,'micro4',tuple(v[:,j] for v in micro_cond),tuple(v[:,j] for v in micro_tf),refs_cond[i],refs_tf[i]);rows[i]['micro_batch_start']=lo
                    del micro_cond,micro_tf;gate()
            checks['all72_conditional_and_actual_comparisons_complete']=len(rows)==12 and all(set(row[branch])=={'B1','B12','micro4'} for row in rows for branch in ('conditional','actualX_TF'))
            checks['oldweights_frozen_verified']=m.verify_old();result['weights_digest_after']=modules['model'].state_digest(m.state_dict());checks['all_weights_unchanged']=result['weights_digest_before']==result['weights_digest_after']
            checks['entire_RNG_unchanged']=learning.rng_equal(rng,learning.rng_snapshot());checks['no_gradients_created']=all(p.grad is None for p in m.parameters())
            checks['no_Core_or_physical_backend']=not any(name in sys.modules for name in ('mujoco','physics','flygym','static_teacher','student_wave'))
            result['conditional36_all6outputs_exact']=all(all(value==0. for value in row['conditional'][mode].values()) for row in rows for mode in ('B1','B12','micro4'))
            result['actualTF36_all4outputs_exact']=all(all(value==0. for value in row['actualX_TF'][mode].values()) for row in rows for mode in ('B1','B12','micro4'))
        checks['public_alias_and_path_restored']=True;gate()
        ledgers['source_after']={path:C.sha(path) for path in source_before};checks['source_SHA_unchanged']=ledgers['source_after']==source_before
        ledgers['final_qualification_after']={path:C.sha(path) for path in proof_before};checks['final_qualification_SHA_unchanged']=ledgers['final_qualification_after']==proof_before;checks['checkpoint_SHA_unchanged']=C.sha(checkpoint)==digest;gate()
        result['qualified_final900']=all(checks.values());result['status']='PASS' if result['qualified_final900'] else 'FAIL'
    except BaseException as error:
        result.update(status='FAILED',qualified_final900=False,exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:
        success=False;result.update(checks=checks,neural_execution_calls=counts,conditional_groups=36,actualTF_groups=36)
        try:
            C.save(out/'ROWS.json',rows);C.save(out/'SOURCE_LEDGER.json',ledgers)
            result['artifacts']={name:dict(path=str(out/name),sha256=C.sha(out/name)) for name in ('ROWS.json','SOURCE_LEDGER.json')}
            text=['# 最终900状态对齐动作模型独立batch审核','',f"科学阶段状态：{result['status']}；最终接纳仍需脚本exit0、RESOURCE_COMPLETION.pass、artifact SHA一致且无FINALIZATION_FAILURE。",'', '完整12×1920，conditional为真实六输出、actualTF为真实四输出；B1/B12/micro4逐项≤1e-12、old256exact/finite。训练、优化器、梯度、Core、物理均0。', '', '新900审核首次READY前预登记600秒/1800MiB/单CPU；不是已测速度，也不修改旧600审核或FD300秒。完整输出后的wall/RSS见RESOURCE_COMPLETION.json。','', '仅数值门，不是body行为或完整颈部通过。',result.get('traceback','')]
            (out/'REPORT.zh-CN.md').write_text('\n'.join(text)+'\n');gate()
            result.update(wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,wall_limit_s=C.WALL_S,RSS_limit_MiB=C.RSS_MIB);C.save(out/'VERIFY.json',result);gate()
            artifacts=C.ledger(out/name for name in ('VERIFY.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md'));gate()
            completion=dict(kind='FINAL900_BATCH_FINAL_RESOURCE_COMMIT',wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,wall_limit_s=C.WALL_S,RSS_limit_MiB=C.RSS_MIB,artifacts=artifacts,science_and_success_outputs_included=True,failure_bookkeeping_never_qualifies=True)
            completion['pass']=result['status']=='PASS' and result['qualified_final900'] is True and completion['wall_s']<=C.WALL_S and completion['RSS_MiB']<=C.RSS_MIB
            C.save(out/'RESOURCE_COMPLETION.json',completion);gate();success=C.accept_run(out);gate()
        except BaseException as error:
            success=False
            result.update(status='FAILED',qualified_final900=False,finalization_error=repr(error));failure=dict(kind='FINAL900_BATCH_FINALIZATION_FAILURE',status='FAILED',qualified_final900=False,error=repr(error),wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            try:
                C.save(out/'FINALIZATION_FAILURE.json',failure)
                result.update(wall_s=failure['wall_s'],RSS_MiB=failure['RSS_MiB'])
                C.replace_failed_output(out,'VERIFY.json',result)
                failed_artifacts=C.ledger(out/name for name in ('VERIFY.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md') if (out/name).is_file())
                C.replace_failed_output(out,'RESOURCE_COMPLETION.json',dict(failure,**{'pass':False},artifacts=failed_artifacts,wall_limit_s=C.WALL_S,RSS_limit_MiB=C.RSS_MIB))
            except BaseException:pass
            print('FINALIZATION_FAILED',repr(error),flush=True)
        finally:
            stop.set();signal.alarm(0);signal.signal(signal.SIGALRM,previous_alarm);signal.signal(signal.SIGUSR1,previous_usr)
        print('FINAL_STATE_ALIGNED900_BATCH_AUDIT',result['status'],'resource_commit',success,str(out/'VERIFY.json'),flush=True)
    return 0 if success and result['status']=='PASS' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-final-state-aligned-batch-audit',action='store_true');a=p.parse_args()
    if not a.execute_parent_final_state_aligned_batch_audit:p.error('Explicit parent release required; ONLYCODE until released')
    if not C.sha_argument(a.sha256):p.error('Explicit actual lowercase64 final900 SHA required; unknown values are not inferred')
    checkpoint=Path(a.checkpoint).resolve()
    if checkpoint!=C.PRODUCER/'TRAIN/checkpoint_000900.pt':p.error('Only exact new wall_replan_9000 final900 path; parent600/750 are not final900')
    return execute(checkpoint,a.sha256)
if __name__=='__main__':sys.exit(main())
