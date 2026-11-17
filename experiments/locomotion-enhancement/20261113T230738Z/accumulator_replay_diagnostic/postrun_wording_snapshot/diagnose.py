"""Parent savedarray-only 13actualtarget/encoding gates then9conditionalNNoutputs."""
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
from metadata import ROOT,CAL,SHA,read,save,sha,verify,qualify,identity_contract


def report(stat):
    lines=['# 增量执行器保存数组重放与条件性目标误差','','本次0NN、0Core、0物理、0训练。先以实际应用动作通过13条target与感觉编码强门，再读取已保存fcbd评估动作做9条源固定X条件性target积分。三条辅助混合轨迹不作未混合教师累计监督。','','|来源|实际target重放最大误差|targeterror／position float32精确|','|---|---:|---|']
    for r in stat['phaseA']:lines.append(f"|{r['id']}|{r['actual_replay']['max_abs_target_error']:.8g}|{r['encoding']['passed']}|")
    lines+=['','|9条真实监督源|动作RMSE|条件target RMS|峰值rad|最终RMS rad|','|---|---:|---:|---:|---:|']
    for r in stat['phaseB']:
        x=r['comparison'];lines.append(f"|{r['id']}|{x['global_action_error_RMS']:.8g}|{x['conditional_target_RMS']:.8g}|{x['conditional_target_absmax']:.8g}|{x['conditional_target_final_RMS']:.8g}|")
    lines+=['','逐通道有符号动作偏差、目标peak/final及rate/angle cap计数见STAT。输入normalizer由真实冻结XML重建并核实际IDENTITY的compiledjnt_range/key_qpos哈希；12来源bounds/rate取实际IDENTITY显式列表，唯一derived plane来源依据实际身份绑定的原执行器默认值、未改有限PD的派生加载源码与compiled哈希证明，逐case保留provenance，未从float32感觉倒推。','阶段B只积分保存在真实源X上的网络预测动作，没有将预测target反馈重新送入NN；qpos/载荷/oldhidden等仍来自源条件。目标误差不能证明新身体位移或停步，也不能判唯一根因。三条assisted真实target积了混合动作，未被拿作fullteacher累计监督。','已有fcbd四项身体FAIL保持，离线损失下降不等神经自主运动通过。任何可微累计loss、target-feedback训练或第四模型须另预登记，本诊断不创建候选或更新权重。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-accumulator-diagnostic',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_accumulator_diagnostic:p.error('Parentsavedarraydiagnosticrelease required')
    start=time.monotonic()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if any((ROOT/n).exists() for n in ('STAT.json','VERIFY.json','REPORT.zh-CN.md')):raise FileExistsError('No overwrite/repeateddiagnostic')
    meta=qualify();verify(meta['source_hashes'])
    import numpy as np
    from analysis import actual_replay,encoding_check,replay_target,compare_targets
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
        if rss>512 or time.monotonic()-start>60:raise RuntimeError('60s/512MiBcap exceeded,noextension')
        return rss
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def mappedidentity(identity):
        mapping=meta['mapping']
        for name,value in (('jnt_range',mapping['joint_ranges43']),('key_qpos',[mapping['key_qpos49']])):
            record=identity['compiled_arrays'][name];array=np.asarray(value,dtype=np.dtype(record['dtype']))
            if list(array.shape)!=record['shape'] or hashlib.sha256(array.tobytes()).hexdigest()!=record['sha256']:raise ValueError('XMLnormalizercompiledidentitymismatch: '+name)
        qa=np.asarray(mapping['qpos_indices']);joints=np.asarray(mapping['joint_ranges43'])[1:]
        # Frozenplant all42hinges are actuated; actual targetbounds are the nativejoint ranges.
        actual=np.asarray(identity['actuator']['target_bounds_rad']);xmlbounds=np.asarray([mapping['joint_ranges43'][1+int(i-7)] for i in qa])
        if not np.array_equal(actual,xmlbounds):raise ValueError('Actualactuatorbounds XMLmatchrequired')
        return actual,np.asarray(identity['actuator']['max_rate_rad_s'])
    phaseA=[]
    for e in meta['entries']:
        folder=Path(e['folder']);trace=arrays(folder/'trace_1ms.npz');s=arrays(e['sample_path']);identity=identity_contract(read(folder/'IDENTITY.json'));bounds,rate=mappedidentity(identity);startrow=e['start']
        sourcekey='Y42' if e['teacher'] and not e['collection'] else 'actually_applied_action';actions=s[sourcekey][startrow:];X=s['X238'][startrow:];ms=np.rint(s['time'][startrow:]*1000).astype(int)
        replay,_=actual_replay(trace,actions,ms,bounds,rate);encoding=encoding_check(trace,X,ms,meta['mapping']);phaseA.append(dict(id=e['id'],actual_new_body=e['actual_new_body'],actual_replay=replay,encoding=encoding,normalizer_compiled_hash_exact=True,actuator_contract_provenance=identity.get('actuator_contract_provenance','actualIDENTITY.actuator explicitnumericlists')))
        del trace,s;guard()
    checks={'phaseA_all13_actualtarget_encoding':all(r['actual_replay']['passed'] and r['encoding']['passed'] for r in phaseA)}
    if not checks['phaseA_all13_actualtarget_encoding']:
        verify(meta['source_hashes']);save(ROOT/'VERIFY.json',dict(integrity_all_pass=False,checks=checks,phaseA=phaseA,phaseB_executed=False,source_hashes=meta['source_hashes'],NN_forwards=0,physical_steps=0));raise ValueError('Actualtarget/encodinggatefailed; no conditionalanalysis')
    with np.load(meta['evaluation_path'],allow_pickle=False) as z:predictions=z['new_action'].copy();Y=z['actual_target'].copy()
    phaseB=[]
    for e in meta['entries']:
        if e['eval_index'] is None:continue
        folder=Path(e['folder']);trace=arrays(folder/'trace_1ms.npz');s=arrays(e['sample_path']);identity=identity_contract(read(folder/'IDENTITY.json'));bounds,rate=mappedidentity(identity);i=e['eval_index'];key='Y42' if e['teacher'] and not e['collection'] else 'actually_applied_action';actualactions=s[key][e['start']:]
        if not np.array_equal(actualactions,Y[:,i]):raise ValueError('RealizedsupervisionY differsfromactualappliedsource')
        predicted,caps=replay_target(predictions[:,i],trace['target'][5799],bounds,rate);comparison=compare_targets(predicted,trace['target'][5800:],predictions[:,i],actualactions)
        phaseB.append(dict(id=e['id'],evaluation_index=i,comparison=comparison,predicted_target_caps=caps,prediction_source='qualifiedsavedfcbdEVAL.new_action, noNNforward'))
        del trace,s;guard()
    checks['nine_realized_conditional_sources']=len(phaseB)==9
    checks['noTorch_Core_or_physicalimports']=not any(k in sys.modules for k in ('torch','physics','mujoco','flygym','static_teacher'))
    verify(meta['source_hashes']);rss=guard()
    stat=dict(kind='SAVED_ARRAY_INCREMENTAL_TARGET_REPLAY_DIAGNOSTIC_V1',checkpoint_sha256=SHA,checks=checks,phaseA=phaseA,phaseB=phaseB,source_mapping=meta['mapping'],source_hashes=meta['source_hashes'],wall_s=time.monotonic()-start,peak_RSS_MiB=rss,NN_forwards=0,Core_constructions=0,physical_steps=0,training_updates=0,conditional_notphysicalcounterfactual=True)
    save(ROOT/'STAT.json',stat)
    with (ROOT/'REPORT.zh-CN.md').open('x') as f:f.write(report(stat))
    verify(meta['source_hashes']);guard()
    save(ROOT/'VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,STAT_sha256=sha(ROOT/'STAT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),source_hashes=meta['source_hashes'],NN_forwards=0,Core_constructions=0,physical_steps=0))
    if not all(checks.values()):raise ValueError('Savedaccumulatorauditfailed')
    print('ACTUALTARGET_AND_ENCODING_REPLAY_CONDITIONAL_ACCUMULATOR_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':main()
