"""Independent final600-only batch parity gate; stdlib-only until explicit execution."""
from pathlib import Path
from contextlib import contextmanager
import argparse, hashlib, importlib, json, os, resource, signal, sys, threading, time, traceback
ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'target_feedback_order_repair_v4'
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
NAMES=('mean','residual','h96','pretarget','posttarget','action')
ALIASES=('adapter','sources','common','model','learning','scan','data','utils')
STAGES=('RESUME301_TO600_AND_FRESH300_450_600','FINAL600_EVALUATION','INDEPENDENT_SAVED_METRICS')
GATE=1e-12
WALL=300.
RSS=1800.

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read_json(path):return json.loads(Path(path).read_text())
def save_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def verify_hashes(ledger):
    require(isinstance(ledger,dict) and bool(ledger),'Nonempty hash ledger required')
    for path,digest in ledger.items():require(sha(path)==digest,'Source or evidence drift: '+str(path))

def final_qualification(checkpoint,digest):
    """Read final JSON/SHA only; no deserialization or speculative final identity."""
    require(checkpoint==V4/'TRAIN/checkpoint_000600.pt','Only exact v4 final600 checkpoint path admitted')
    require(sha(checkpoint)==digest,'Actual final checkpoint SHA differs')
    jobs=read_json(V4/'JOBS.json');stage=read_json(V4/'STAGE_VERIFICATION.json');tr=read_json(V4/'TRAIN/RESULT.json')
    require(len(jobs)==3 and tuple(row['stage'] for row in jobs)==STAGES,'Exactly three ordered v4 final stages required')
    for row,script in zip(jobs,('train.py','evaluate.py','audit_eval.py')):
        require(row['status']=='COMPLETED' and row['exit_code']==0,'All three main stages must finish exit0 before audit')
        require(str(V4/script) in row['command'],'Main-stage command must target v4 scientific entry')
    require(stage['all_stages_exit0'] is True and stage['train_updates']==600 and stage['new_updates']==300,'Stage final600/new300 qualification required')
    require(tr['updates']==600 and tr['total_updates']==600 and tr['new_updates']==300 and tr['already_completed_updates']==300,'Exactly original300 plus remaining300 updates required')
    for record,sha_key in ((stage,'checkpoint_sha256'),(tr,'sha256')):
        require(record['checkpoint']==str(checkpoint) and record[sha_key]==digest,'Final stage/TRAIN identity mismatch')
    paths=[V4/'JOBS.json',V4/'STAGE_VERIFICATION.json',V4/'TRAIN/RESULT.json']
    for step in (300,450,600):
        proof_path=V4/'TRAIN'/f'VERIFY_{step:06d}.json';proof=read_json(proof_path)
        saved=V4/'TRAIN'/f'checkpoint_{step:06d}.pt'
        require(proof['step']==step and proof['integrity_all_pass'] is True and bool(proof['checks']) and all(value is True for value in proof['checks'].values()),'Every new saved300/450/600 verification must PASS')
        require(proof['checkpoint']==str(saved) and sha(saved)==proof['checkpoint_sha256'],'Saved checkpoint proof SHA/path mismatch')
        if step==600:require(proof['checkpoint_sha256']==digest,'Saved final600 identity differs')
        paths.append(proof_path)
    evaluation=read_json(V4/'EVALUATION.json');audit=read_json(V4/'EVALUATION_VERIFY.json')
    require(evaluation['step']==600 and evaluation['checkpoint']==str(checkpoint) and evaluation['checkpoint_sha256']==digest,'Final evaluation checkpoint mismatch')
    require(audit['integrity_all_pass'] is True and bool(audit['checks']) and all(value is True for value in audit['checks'].values()),'Independent final evaluation audit must PASS')
    require(audit['checkpoint_sha256']==digest and audit['evaluation_sha256']==sha(V4/'EVALUATION.json') and stage['evaluation_sha256']==audit['evaluation_sha256'],'Final evaluation evidence SHA mismatch')
    require(evaluation['arrays']['path']==str(V4/'EVALUATION_ARRAYS.npz') and sha(evaluation['arrays']['path'])==evaluation['arrays']['sha256'],'Final evaluation array identity mismatch')
    require(evaluation['source_hashes']==tr['source_hashes'] and evaluation['data_hashes']==tr['data_hashes'],'Final evaluation/TRAIN provenance differs')
    paths.extend((V4/'EVALUATION.json',V4/'EVALUATION_VERIFY.json',V4/'EVALUATION_ARRAYS.npz'))
    return {str(path):sha(path) for path in paths},tr

@contextmanager
def v4_context():
    """Bind all ambiguous public imports to frozen v4, restore exact caller objects."""
    sentinel=object();previous={name:sys.modules.get(name,sentinel) for name in ALIASES};oldpath=list(sys.path)
    try:
        for name in ALIASES:sys.modules.pop(name,None)
        sys.path.insert(0,str(V4));modules={}
        for name in ALIASES:
            value=importlib.import_module(name)
            require(Path(value.__file__).resolve()==V4/(name+'.py'),'Wrong v4 public module: '+name)
            modules[name]=value
        yield modules
    finally:
        sys.path[:]=oldpath
        for name,value in previous.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        require(sys.path==oldpath and all(sys.modules.get(name,sentinel) is value for name,value in previous.items()),'Caller aliases/path restoration failed')

def execute(checkpoint,digest):
    require(Path(sys.executable).absolute()==PYTHON.absolute(),'Use qualified 125712 Python')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    require(not any(name in sys.modules for name in ('torch','numpy','mujoco','physics','flygym')),'Audit requires a fresh neural-only process')
    out=ROOT/('RUN_'+digest);out.mkdir(exist_ok=False)
    started=time.monotonic();rows=[];ledger={};checks={};result=dict(kind='INDEPENDENT_FINAL600_BATCH_PARITY_AUDIT_V1',status='FAILED',qualified_final600=False,checkpoint=str(checkpoint),checkpoint_sha256=digest,gate=GATE,rows=12,ticks_each=1920,training_updates=0,optimizer_operations=0,gradient_operations=0,Core_constructions=0,physical_steps=0)
    stop=threading.Event()
    def gate(*_):
        if time.monotonic()-started>=WALL:raise RuntimeError('Audit300s wall limit exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>RSS:raise RuntimeError('Audit1800MiB RSS limit exceeded')
    def budget_signal(*_):raise RuntimeError('Audit resource watchdog limit reached')
    old_alarm=signal.signal(signal.SIGALRM,budget_signal);old_usr=signal.signal(signal.SIGUSR1,budget_signal);signal.alarm(300)
    def monitor():
        while not stop.wait(.2):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>RSS:os.kill(os.getpid(),signal.SIGUSR1);return
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        qualification,tr=final_qualification(checkpoint,digest);ledger['final_qualification_before']=qualification
        source_before=dict(tr['source_hashes']);verify_hashes(source_before);ledger['v4_source_before']=source_before
        own_before={str(ROOT/name):sha(ROOT/name) for name in ('audit.py','PREREG.json','README.zh-CN.md','staticcheck.py')};ledger['audit_source_before']=own_before
        print('FINAL600_QUALIFICATION_OK',flush=True);gate()
        with v4_context() as modules:
            learning=modules['learning'];learning.cpu1()
            import torch
            torch.set_num_interop_threads(1)
            result['environment']=dict(python=sys.executable,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),affinity=sorted(os.sched_getaffinity(0)),env={name:os.environ[name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
            require(result['environment']['threads']==1 and len(result['environment']['affinity'])==1,'Actual CPU1 required')
            rng=learning.rng_snapshot();common=modules['common'];model=modules['model']
            m,payload=common.load_checkpoint(checkpoint,digest,final=True)
            checks['strict_final600']=payload['step']==600
            checks['strict_load_RNG_unchanged']=learning.rng_equal(rng,learning.rng_snapshot())
            require(payload['config']['source_hashes']==source_before,'Loaded final payload source differs from final provenance')
            weights=model.state_digest(m.state_dict());result['weights_digest_before']=weights
            print('STRICT_FINAL600_LOAD_OK',round(time.monotonic()-started,3),flush=True);gate()
            with torch.no_grad():
                data=modules['data'].prepare(m);scan=modules['scan'].make_scan(m)
                require(len(data['raw'])==12 and tuple(data['physical'].shape)==(1920,12,112),'All12 full1920 source schema required')
                references=[];old_exact=[]
                def compare(outputs,reference):
                    require(len(outputs)==len(reference)==6,'Six scan outputs required')
                    finite=all(bool(torch.isfinite(value).all()) for value in (*outputs,*reference))
                    errors={name:float((a-b).abs().max()) for name,a,b in zip(NAMES,outputs,reference)}
                    require(finite,'Nonfinite numerical output')
                    return errors
                for i,raw in enumerate(data['raw']):
                    full,old=modules['utils'].production_feedback(m,raw,data,i)
                    require(all(value.shape[0]==1920 for value in full),'Full independent1920 production history required')
                    references.append(full);exact=torch.equal(old,data['oldafter'][:,i]);old_exact.append(exact)
                    one=learning.forward(scan,learning.slice_data(data,i,i+1))
                    errors=compare(tuple(value[:,0] for value in one),full)
                    rows.append(dict(id=raw['entry']['id'],index=i,rows=1920,old256exact=exact,B1=errors))
                    checks[f'{i}_{raw["entry"]["id"]}_B1']=exact and max(errors.values())<=GATE
                    del one,old;gate()
                batch=learning.forward(scan,data)
                for i,row in enumerate(rows):
                    errors=compare(tuple(value[:,i] for value in batch),references[i]);row['B12']=errors
                    checks[f'{i}_{row["id"]}_B12']=old_exact[i] and max(errors.values())<=GATE
                del batch;gate()
                for lo in (0,4,8):
                    micro=learning.forward(scan,learning.slice_data(data,lo,lo+4))
                    for j in range(4):
                        i=lo+j;errors=compare(tuple(value[:,j] for value in micro),references[i]);rows[i]['micro4']=errors;rows[i]['micro_batch_start']=lo
                        checks[f'{i}_{rows[i]["id"]}_micro4']=old_exact[i] and max(errors.values())<=GATE
                    del micro;gate()
            result['weights_digest_after']=model.state_digest(m.state_dict());checks['all_weights_unchanged']=result['weights_digest_after']==weights
            checks['frozen_old_verified']=m.verify_old();checks['no_gradients_created']=all(p.grad is None for p in m.parameters())
            checks['entire_audit_RNG_unchanged']=learning.rng_equal(rng,learning.rng_snapshot())
            checks['no_Core_or_physical_backend']=not any(name in sys.modules for name in ('mujoco','physics','flygym','static_teacher','student_wave'))
            result['all36_comparisons_exact']=all(all(value==0. for value in row[mode].values()) for row in rows for mode in ('B1','B12','micro4'))
        checks['v4_alias_context_restored']=True
        ledger['v4_source_after']={path:sha(path) for path in source_before};checks['v4_source_unchanged']=ledger['v4_source_after']==source_before
        ledger['audit_source_after']={path:sha(path) for path in own_before};checks['audit_source_unchanged']=ledger['audit_source_after']==own_before
        ledger['final_qualification_after']={path:sha(path) for path in qualification};checks['final_qualification_unchanged']=ledger['final_qualification_after']==qualification
        checks['checkpoint_SHA_unchanged']=sha(checkpoint)==digest;gate()
        result['status']='PASS' if all(checks.values()) else 'FAIL'
        result['qualified_final600']=result['status']=='PASS'
        result['body_admission_additional_batch_gate']=result['qualified_final600']
    except BaseException as error:
        result.update(status='FAILED',exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:
        stop.set();signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGUSR1,old_usr)
        result.update(checks=checks,wall_s=time.monotonic()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,physical_behavior_qualified=False)
        save_json(out/'ROWS.json',rows);save_json(out/'SOURCE_LEDGER.json',ledger)
        result['artifacts']={name:dict(path=str(out/name),sha256=sha(out/name)) for name in ('ROWS.json','SOURCE_LEDGER.json')}
        save_json(out/'VERIFY.json',result);print('FINAL_BATCH_AUDIT',result['status'],str(out/'VERIFY.json'),flush=True)
    return 0 if result['qualified_final600'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--execute-parent-final-batch-audit',action='store_true');args=parser.parse_args()
    if not args.execute_parent_final_batch_audit:parser.error('Explicit parent execution flag required; ONLYCODE by default')
    if len(args.sha256)!=64 or any(c not in '0123456789abcdef' for c in args.sha256):parser.error('Supply actual lowercase64 SHA256; unknown or guessed identities are not accepted')
    checkpoint=Path(args.checkpoint).resolve()
    if checkpoint!=V4/'TRAIN/checkpoint_000600.pt':parser.error('Only final v4 checkpoint_000600.pt; diagnostic300/450 unsupported')
    return execute(checkpoint,args.sha256)
if __name__=='__main__':sys.exit(main())
