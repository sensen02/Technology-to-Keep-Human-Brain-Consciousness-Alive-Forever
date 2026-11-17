"""Parent report-only AST recovery + savedarray independent audit. No queries."""
from pathlib import Path
import argparse
import ast
import builtins
import dis
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'teacher_student_state_query'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for p,digest in hashes.items():
        if sha(p)!=digest:raise ValueError('Immutable source/artifact changed: '+p)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def report_function():
    """Compile ONLY pureoriginalreport function; explicit json binding, no starimports."""
    original=ORIGINAL/'query.py';tree=ast.parse(original.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='report_text']
    if len(nodes)!=1:raise ValueError('Exactlyoneoriginalreportfunctionrequired')
    selected=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(selected)
    namespace={'__builtins__':builtins.__dict__,'__name__':'_recovered_original_report_only','json':json}
    exec(compile(selected,str(original)+'::REPORT_TEXT_ONLY_EXPLICIT_JSON_GLOBAL','exec'),namespace)
    function=namespace['report_text'];needed=set()
    def inspect_code(code):
        for instruction in dis.get_instructions(code):
            if instruction.opname in ('LOAD_GLOBAL','LOAD_NAME'):needed.add(instruction.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):inspect_code(value)
    inspect_code(function.__code__)
    missing=[name for name in sorted(needed) if name not in namespace and name not in builtins.__dict__]
    if missing:raise ValueError('Unresolvedrecursive reportglobals: '+','.join(missing))
    resolved={name:('explicitjsonmodule' if name=='json' else 'explicitfunctionnamespace' if name in namespace else 'builtin') for name in sorted(needed)}
    return function,dict(original_source=str(original),source_sha256=sha(original),selected_AST_sha256=hashlib.sha256(ast.dump(selected,include_attributes=False).encode()).hexdigest(),recursive_LOAD_GLOBAL_or_NAME=resolved,missing_globals=missing,extraction='ONLYreport_text, originalquerymainnotimported/executed',json_binding_explicit=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-saved-report-recovery',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_report_recovery:p.error('Parent saved-only reportrecoveryrelease required')
    start=time.monotonic()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if any((ROOT/n).exists() for n in ('REPORT.zh-CN.md','AUDIT.json','VERIFY.json')):raise FileExistsError('Nooverwrite/repeatedrecovery')
    if (ORIGINAL/'VERIFY.json').exists():raise ValueError('OriginalVERIFYunexpected; preserveandreviewnewstate')
    stat=read(ORIGINAL/'STAT.json');failure=read(ORIGINAL/'FAILURE.json')
    if failure['error_type']!='NameError' or "json" not in failure['error'] or not all(stat['checks'].values()) or len(stat['qualification'])!=3 or len(stat['students'])!=6:raise ValueError('Expectedreport-export-onlyfailurewithcompletednumericalSTAT')
    if stat['physical_integration_steps']!=0 or stat['NN_forwards']!=0 or stat['Core_constructions']!=9:raise ValueError('Recordedoriginalnumericalscopeunexpected')
    oldreport=ORIGINAL/'REPORT.zh-CN.md';oldreport_size=oldreport.stat().st_size if oldreport.exists() else None
    if oldreport_size not in (0,None):raise ValueError('Originalreporthascontent, dosource-reviewbeforeanyrecovery')
    original_hashes=dict(stat['source_hashes'])
    # Explicitly omit uselesszero-byteREPORT, including if a future externalledger
    # accidentally mentionsit. It isnot a numericalinput or preservationdependency.
    original_hashes.pop(str(oldreport),None)
    paths=[ORIGINAL/'STAT.json',ORIGINAL/'FAILURE.json',ORIGINAL/'QUERY_EXECUTION.log',ORIGINAL/'PREFLIGHT.json',ORIGINAL/'PREFLIGHT_EXECUTION.log',ORIGINAL/'query.py',ROOT/'PREREG.json',Path(__file__),ROOT/'audit_arrays.py']
    for row in stat['qualification']+stat['students']:
        if sha(row['artifact']['path'])!=row['artifact']['sha256']:raise ValueError('ActualqueryartifactSHA')
        paths.extend((Path(row['artifact']['path']),ORIGINAL/'queries'/(row['id']+'.json')))
    original_hashes.update({str(path):sha(path) for path in paths});verify(original_hashes)
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
        if rss>1800 or time.monotonic()-start>120:raise RuntimeError('Recovery120s/1800MiB resourcecap exceeded, noextension')
        return rss
    function,globals_audit=report_function();text=function(stat)
    if not isinstance(text,str) or not text.startswith('# 阶段B'):raise ValueError('Recoveredpuretextunexpected')
    metadata=module('_recovery_original_metadata_only',ORIGINAL/'metadata.py');meta=metadata.qualify()
    guard()
    from audit_arrays import audit_all
    audit,checks=audit_all(stat,meta,guard=guard)
    checks.update(explicit_report_globals_resolved=not globals_audit['missing_globals'],all_original7_numeric_checks_recorded_true=all(stat['checks'].values()),original_run_exit1_preserved=True,original_report_placeholder_notrequired=True,no_Core_teacher_NN_physics_reexecution=not any(k in sys.modules for k in ('mujoco','physics','flygym','torch','static_teacher')))
    verify(original_hashes);rss=guard()
    audit.update(recovery_kind='ORIGINAL_RUN_EXIT1_REPORT_EXPORT_ONLY',checks=checks,original_failure=failure,original_empty_report_size_bytes=oldreport_size,empty_report_preservation_dependency=False,report_function_globals=globals_audit,original_recorded_numerics=dict(Core=stat['Core_constructions'],teacher_queries=sum(r['rows'] for r in stat['qualification']+stat['students']),physical_integration_steps=stat['physical_integration_steps'],NN_forwards=stat['NN_forwards'],optimizer_updates=stat['optimizer_updates'],verification_scope='Countsfrom originalrecordedtraps/STAT, notindependentlyre-enacted in this savedarrayaudit'),new_recovery_execution=dict(Core=0,teacher_queries=0,NN_forwards=0,physical_integration_steps=0,optimizer_updates=0),source_hashes=original_hashes,wall_s=time.monotonic()-start,peak_RSS_MiB=rss)
    save(ROOT/'AUDIT.json',audit)
    appendix=report_appendix(audit)
    with (ROOT/'REPORT.zh-CN.md').open('x') as stream:
        stream.write('# 报告导出恢复与保存数组独立审核\n\n原主任务数值查询已保存，但原程序退出码为1：report_text缺少json全局导致报告导出失败。本恢复没有重跑查询；保留原FAILURE、STAT、全部查询数组及日志。原0字节报告不作为需保留输入，父可按清理规则删除。\n\n以下正文只从原report_text AST抽取，并显式绑定json；不是原主任务exit0的证明。\n\n')
        stream.write(text);stream.write('\n\n'+appendix)
    verify(original_hashes);guard()
    save(ROOT/'VERIFY.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,original_main_exit_code=1,original_failure_kind='NameError_json_REPORT_EXPORT_ONLY',original_STAT_sha256=sha(ORIGINAL/'STAT.json'),original_FAILURE_sha256=sha(ORIGINAL/'FAILURE.json'),recovered_report_sha256=sha(ROOT/'REPORT.zh-CN.md'),independent_saved_array_audit_sha256=sha(ROOT/'AUDIT.json'),source_hashes=original_hashes,original_empty_report_required=False,Core_constructions=0,teacher_queries=0,NN_forwards=0,physical_steps=0,scope='Reportrecoveryandsavedarraysaudit, notoriginalmainexit0 orindependentphysics/NNre-enactment'))
    guard()
    if not all(checks.values()):raise ValueError('Savedarrayindependentauditfailed, recoveredartifactsretained')
    print('REPORT_RECOVERED_AND_SAVED_ARRAYS_INDEPENDENTLY_AUDITED_NO_QUERY_REPLAY',flush=True)


def report_appendix(audit):
    lines=['## 保存数组独立审核附录','','以下只重算已保存数组，不重新执行Core、教师、NN或积分。原积分／NN0依据原traps及STAT记录；本审核不冒称独立重演了该现场。','','|case|有效标签|有效比例|最长起始有效前缀行|最长连续有效段行|最常拒绝项|','|---|---:|---:|---:|---:|---|']
    for row in audit.get('cases',[]):
        # audit_arrays defines these explicitsummary fields; UNKNOWN is honest
        # fallback only for optional narrative, strictchecksverify requiredschema.
        lines.append('|'+str(row.get('id'))+'|'+str(row.get('qualified_count'))+'|'+str(row.get('qualified_fraction'))+'|'+str(row.get('longest_valid_prefix_rows'))+'|'+str(row.get('longest_valid_run_rows'))+'|'+str(row.get('most_common_rejection'))+'|')
    lines += ['','全部有效区间／每mask失败计数及目标寄存器范围RMS、与纠正动作的描述相关见AUDIT.json。失败项可重叠，不将各mask失败数相加冒充总无效数；invalid NaN和错误行保留原数组。','合格标签只对应预登记局部条件，不能掩盖原身体任务失败，也不证明施加后可恢复。若有效覆盖仅短前缀，下一步数据策略需父先评估；本恢复不授权或实施Dagger／teacher mixing训练、第三模型或新物理。','actualtarget／target-error／target-position offset与教师理想q_des访问层级不同，相关不构成因果证明；冻结oldmean含target-position prior，不能误称wholemean独立。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':main()
