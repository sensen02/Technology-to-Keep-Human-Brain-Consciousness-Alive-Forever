"""STD-only external binding checks and negative fixtures; never write BODY input."""
from pathlib import Path
import ast
import builtins
import dis
import importlib.util
import json
import sys
import types
ROOT=Path(__file__).resolve().parent
FILES=('prepare.py','staticcheck.py','PREREG.json','README.zh-CN.md')


def main():
    checks={};forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave','static_teacher'}
    original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('STDexternalbinding forbidsnumericalimports:'+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guard
    try:
        for name in FILES:
            p=ROOT/name
            if not p.is_file():raise FileNotFoundError(p)
            if name.endswith('.py'):compile(ast.parse(p.read_text()),str(p),'exec')
        spec=importlib.util.spec_from_file_location('_external_postrun_bind_STD',ROOT/'prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        body=m.check_body_unexecuted();checks['actual_frozen_BODY32checks_11files_exact']=len(body['checks'])==32 and len(body['source_hashes'])==11
        sentinel=object();prior={key:sys.modules.get(key,sentinel) for key in ('adapter','common')}
        try:
            spec=importlib.util.spec_from_file_location('_external_frozen_body_adapter',m.BODY/'adapter.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);sys.modules['adapter']=adapter
            spec=importlib.util.spec_from_file_location('_external_frozen_body_common',m.BODY/'common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common);sys.modules['common']=common
            for name in adapter.ENTRIES:
                ns=adapter.namespace(name);checks[name+'_frozen_newROOT_source3_globals']=ns['ROOT']==m.BODY and ns['SOURCE_CANDIDATE']==3 and all(value.__globals__.get('ROOT')==m.BODY and value.__globals__.get('SOURCE_CANDIDATE')==3 for value in ns.values() if isinstance(value,types.FunctionType) and 'SOURCE_CANDIDATE' in value.__globals__)
            checks['BODY_CAL_remains_original']=common.CAL==m.CAL and adapter.FINAL_STEP==900 and common.FORMAT==m.FORMAT
        finally:
            for key,value in prior.items():
                if value is sentinel:sys.modules.pop(key,None)
                else:sys.modules[key]=value
        def rejects(call):
            try:call()
            except (ValueError,FileExistsError,FileNotFoundError):return True
            return False
        checks['negative_unknown_SHA_rejected']=rejects(lambda:m.original_deadline('UNKNOWN'))
        real_read=m.read
        try:
            m.read=lambda path: [{'status':'RUNNING'}] if Path(path)==m.CAL/'JOBS.json' else real_read(path)
            checks['negative_original_RUNNING_rejected']=rejects(lambda:m.original_deadline('a'*64))
            m.read=lambda path: [{'status':'COMPLETED','exit_code':0} for _ in range(3)] if Path(path)==m.CAL/'JOBS.json' else real_read(path)
            checks['negative_original_all3success_rejected']=rejects(lambda:m.original_deadline('a'*64))
            def missing_batch(path):
                if Path(path)==m.BATCH/'FROZEN.json':raise FileNotFoundError('Negativefixture: no finalizedbatchsource')
                return real_read(path)
            m.read=missing_batch;checks['negative_missing_batch_proof_rejected']=rejects(lambda:m.batch_qualification('a'*64))
        finally:m.read=real_read
        frozen=m.read(m.BATCH/'FROZEN.json');m.verify(frozen['source_hashes']);m.verify(frozen['frozen_science_dependencies']);checks['actual_batch_FIRSTREADY_FROZEN_SHA_pinned']=m.sha(m.BATCH/'FROZEN.json')==m.BATCH_FROZEN_SHA
        path=m.BATCH/'contracts.py';spec=importlib.util.spec_from_file_location('_external_actual_batch_contract_STD',path);contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
        checks['actual_batch_STD_API_source_read']=contract.ROOT==m.BATCH and contract.PRODUCER==m.CAL and contract.MODE=='ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED' and contract.WALL_S==1800 and contract.RSS_MIB==1800 and tuple(__import__('inspect').signature(contract.accept_run).parameters)==('directory',) and tuple(__import__('inspect').signature(contract.final_qualification).parameters)==('checkpoint','digest')
        checks['STD_import_boundary_no_numeric']=not forbidden.intersection(sys.modules)
    finally:builtins.__import__=original_import
    missing={}
    def inspect_code(code,ns):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in ns and not hasattr(builtins,ins.argval):missing.setdefault(code.co_name,[]).append(ins.argval)
        for val in code.co_consts:
            if isinstance(val,types.CodeType):inspect_code(val,ns)
    for val in vars(m).values():
        if isinstance(val,types.FunctionType):inspect_code(val.__code__,val.__globals__)
    checks['all_external_function_globals_resolve']=not missing
    tree=ast.parse((ROOT/'prepare.py').read_text());calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    checks['no_NN_Core_or_originalbodyprepare_calls']=not any((isinstance(node.func,ast.Attribute) and node.func.attr in ('load_checkpoint','step','step_details','prime_old','Core','mj_step','producer_context')) for node in calls)
    checks['only_exclusive_BODY_INPUT_write']='save(BODY / \'INPUT_BINDINGS.json\', binding)' in ast.unparse(tree)
    print(json.dumps(dict(status='ONLYCODE_EXTERNAL_POSTRUN_BINDING_SOURCE_READY_ACTUAL900_UNKNOWN',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(ROOT/n):m.sha(ROOT/n) for n in FILES},frozen_BODY_source_hashes=body['source_hashes'],BODY_CODECHECK_sha256=m.sha(m.BODY/'CODECHECK.json'),child_numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_forwards=0,Core_constructions=0,physical_steps=0,actual_binding_written=False,actual_postrun_or_batch_qualified=False),ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('STDexternalbindingcheckfailed')


if __name__=='__main__':main()
