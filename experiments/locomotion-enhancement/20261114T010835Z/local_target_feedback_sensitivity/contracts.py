"""Stdlib contracts only. No arrays, checkpoint payload, Torch or physical imports."""
from pathlib import Path
import ast,hashlib,importlib.util,json,struct,sys,zipfile
ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
CAL=SCOPE/'target_feedback_order_repair_v4'
BODY=SCOPE/'target_feedback_motor_body_v3'
CHECKPOINT=CAL/'TRAIN/checkpoint_000600.pt'
CHECKPOINT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
SUPPORT=SCOPE/'final_batch_parity_audit/audit.py'
SUPPORT_SHA='79cb7ddf4b9bcb3849af53265d59fa0d9774fcb082a5d630f1798fb43c2cee05'
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
EPSILONS=(1e-4,2e-4)
POINTS=tuple((case,relative,ms,ms//5,ms-1) for case in ('stop','walk') for relative,ms in ((.5,6300),(4.8,10600),(9.,14800)))

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def ledger(paths):return {str(path):sha(path) for path in dict.fromkeys(paths)}
def merge_ledgers(*ledgers):
    merged={}
    for hashes in ledgers:
        for path,digest in hashes.items():
            require(path not in merged or merged[path]==digest,'Conflicting saved source provenance: '+str(path));merged[path]=digest
    return merged
def verify(hashes):
    for path,digest in hashes.items():require(sha(path)==digest,'Evidence/source identity drift: '+str(path))

def header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        require(stream.read(6)==b'\x93NUMPY','NPY magic')
        version=tuple(stream.read(2));require(version in ((1,0),(2,0),(3,0)),'NPY header version')
        width=2 if version==(1,0) else 4;size=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
        require(size<=65536,'NPY header size')
        value=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    require(not value['fortran_order'] and isinstance(value['descr'],str) and 'O' not in value['descr'],'Plain C-order non-object arrays required')
    return dict(shape=list(value['shape']),descr=value['descr'])

def check_headers(case):
    folder=BODY/'trials'/case;result={}
    spec={'neural_decisions.npz':{'time':([3080],'<f8'),'physical_step_ms':([3080],'<i8'),'X238':([3080,238],'<f8'),'command':([3080,2],'<f8'),'mean':([3080,42],'<f8'),'residual':([3080,42],'<f8'),'hidden_before':([3080,352],'<f8'),'hidden_after':([3080,352],'<f8'),'policy_action':([3080,42],'<f8'),'actually_applied_action':([3080,42],'<f8'),'control_source':([3080],'<i8')},'trace_1ms.npz':{'time':([15400],'<f8'),'qpos':([15400,49],'<f8'),'target':([15400,42],'<f8')}}
    for filename,fields in spec.items():
        result[filename]={}
        for key,(shape,dtype) in fields.items():
            value=header(folder/filename,key);require(value==dict(shape=shape,descr=dtype),'Actual source header mismatch: '+case+'/'+filename+'/'+key);result[filename][key]=value
    return result

def completion_success(result,completion):
    return result.get('status')=='COMPLETED' and result.get('diagnostic_integrity_pass') is True and completion.get('pass') is True and completion.get('wall_s',float('inf'))<=300 and completion.get('RSS_MiB',float('inf'))<=1800

def load_support():
    require(sha(SUPPORT)==SUPPORT_SHA,'Immutable final-batch support source drift')
    spec=importlib.util.spec_from_file_location('_local_target_sensitivity_frozen_final_support',SUPPORT);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(module.V4==CAL,'Frozen support target scope differs')
    return module

def body_qualification():
    """Only final body JSON and byte hashes; behavior FAIL is retained as evidence."""
    jobs=read(BODY/'JOBS.json');summary=read(BODY/'SUMMARY.json');binding=read(BODY/'INPUT_BINDINGS.json')
    require(len(jobs)==9 and all(row['status']=='COMPLETED' and row['exit_code']==0 for row in jobs),'Wait for all nine body stages to complete exit0')
    expected=tuple((case,stage) for case in ('walk','stop','left_pivot','right_pivot') for stage in ('PHYSICS','FRESH_NN_SAVED_AUDIT'))
    require(tuple((row.get('condition'),row['stage']) for row in jobs[:8])==expected and jobs[8]['stage']=='SUMMARY','Exact four-body/eight-audit plus summary cohort required')
    require(jobs[8]['summary_sha256']==sha(BODY/'SUMMARY.json'),'Final body summary journal SHA differs')
    require(summary['checkpoint_sha256']==binding['checkpoint_sha256']==CHECKPOINT_SHA and binding['checkpoint']==str(CHECKPOINT),'Actual final600 body identity required')
    require(len(summary['rows'])==4 and {row['condition'] for row in summary['rows']}=={'walk','stop','left_pivot','right_pivot'},'All four body proof rows required')
    saved_sources=summary.get('source_hashes');require(isinstance(saved_sources,dict) and bool(saved_sources),'Nonempty saved body source provenance required')
    require(isinstance(binding.get('source_hashes'),dict) and bool(binding['source_hashes']) and all(saved_sources.get(path)==digest for path,digest in binding['source_hashes'].items()),'Saved body source binding differs from final summary')
    paths=[BODY/'JOBS.json',BODY/'SUMMARY.json',BODY/'INPUT_BINDINGS.json'];selected={}
    for row in summary['rows']:
        case=row['condition'];folder=BODY/'trials'/case;result=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
        require(row['result']==str(folder/'RESULT.json') and row['audit']==str(folder/'VERIFY.json'),'Body summary paths differ')
        require(row['result_sha256']==audit['result_sha256']==sha(folder/'RESULT.json') and row['audit_sha256']==sha(folder/'VERIFY.json'),'Body result/VERIFY proof SHA differs')
        require(audit['integrity_all_pass'] is True and bool(audit['checks']) and all(value is True for value in audit['checks'].values()),'All four fresh saved-body integrity audits must PASS')
        require(result['condition']==case and result['checkpoint']==str(CHECKPOINT) and result['checkpoint_sha256']==audit['checkpoint_sha256']==CHECKPOINT_SHA and result['final_step']==600 and result['state_dims']==[128,64,64,96] and result['state_total']==352,'Body public model/state ABI differs')
        require(result.get('source_hashes')==saved_sources,'Saved body RESULT/summary source provenance differs')
        paths.extend((folder/'RESULT.json',folder/'VERIFY.json'))
        if case in ('stop','walk'):
            require(result['full_duration'] is True and result['actual_ms']==15400 and result['neural_records']==3080,'Both selected cases require full saved duration')
            for key,filename in (('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz')):
                require(result[key]['path']==str(folder/filename) and result[key]['sha256']==sha(folder/filename),'Actual selected body artifact SHA differs');paths.append(folder/filename)
            identity=read(folder/'IDENTITY.json');canonical={key:value for key,value in identity.items() if key!='identity_sha256'}
            require(hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==identity['identity_sha256']==result['identity_sha256'],'Compiled body identity canonical SHA differs')
            require(identity['config']['motor_sha256']==CHECKPOINT_SHA and identity['config']['state_dims']==[128,64,64,96],'Compiled body config identity differs')
            require(identity['config'].get('source_hashes')==saved_sources,'Compiled body identity/saved source provenance differs')
            paths.append(folder/'IDENTITY.json');selected[case]=dict(result=result,identity=identity,headers=check_headers(case))
    verify(saved_sources)
    return ledger(paths),selected,saved_sources
