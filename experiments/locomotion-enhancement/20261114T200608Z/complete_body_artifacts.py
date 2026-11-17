"""Wait for original nine body stages, then report/render saved states sequentially.
No training, NN or physical integration is performed by this orchestration script.
Desktop delivery remains a separate explicit parent image-review step.
"""
from pathlib import Path
import argparse,ctypes,datetime,hashlib,json,os,select,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
BODY=ROOT/'state_aligned_motor_body'
VIDEO=ROOT/'postrun_state_aligned_body_video'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
SHA='e9829e4b94f235889b44d495397d16fb072c9d43f8d2d9cffe91569d46956500'
CASES=('walk','stop','left_pivot','right_pivot')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('Source/proof changed: '+str(p))
def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def body_ready():
    jobs=read(BODY/'JOBS.json')
    if any(j.get('status')=='FAILED' for j in jobs):raise RuntimeError('Original body stage failed; do not retry or render incomplete cohort')
    return len(jobs)==9 and all(j.get('status')=='COMPLETED' and j.get('exit_code')==0 for j in jobs) and (BODY/'SUMMARY.json').is_file()
def wait_body():
    lib=ctypes.CDLL(None,use_errno=True);lib.inotify_init1.argtypes=[ctypes.c_int];lib.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
    fd=lib.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if lib.inotify_add_watch(fd,os.fsencode(BODY),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        end=time.monotonic()+18000
        while not body_ready():
            left=end-time.monotonic()
            if left<=0:raise TimeoutError('Artifact orchestration waiting limit exceeded; body process remains unchanged')
            ready,_,_=select.select([fd],[],[],left)
            if not ready:raise TimeoutError('Artifact orchestration waiting limit exceeded')
            os.read(fd,65536)
    finally:os.close(fd)
def main():
    p=argparse.ArgumentParser();p.add_argument('--finish-after-verified-body',action='store_true');a=p.parse_args()
    if not a.finish_after_verified_body or sys.executable!=PYTHON:raise ValueError('Explicit qualified-parent execution required')
    journal=ROOT/'BODY_ARTIFACT_FINALIZATION_JOBS.json';completion=ROOT/'BODY_ARTIFACT_FINALIZATION.json'
    if any(p.exists() for p in (journal,completion,ROOT/'BODY_ARTIFACT_FINALIZATION_FAILURE.json')):raise FileExistsError('No retries/overwrites')
    ready=read(VIDEO/'READY.json')
    if not ready['FIRSTREADY'] or not all(ready['checks'].values()):raise ValueError('Frozen video code required')
    source={**ready['source_hashes'],**ready['reference_source_hashes'],str(ROOT/'report_functional_motion.py'):sha(ROOT/'report_functional_motion.py'),str(Path(__file__)):sha(__file__)};verify(source)
    print('WAITING_FOR_ORIGINAL_NINE_BODY_STAGES; no polling, no extra simulations',flush=True);wait_body()
    summary=read(BODY/'SUMMARY.json');binding=read(BODY/'INPUT_BINDINGS.json')
    if summary['checkpoint_sha256']!=SHA or binding['checkpoint_sha256']!=SHA or summary['actual_new_native_steps']>4928000:raise ValueError('Same actual900 cohort and finite native budget required')
    if len(summary['rows'])!=4 or tuple(r['condition'] for r in summary['rows'])!=CASES:raise ValueError('Exact four-case cohort required')
    for case in CASES:
        folder=BODY/'trials'/case;result=read(folder/'RESULT.json');proof=read(folder/'VERIFY.json')
        if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or proof['result_sha256']!=sha(folder/'RESULT.json') or result['checkpoint_sha256']!=SHA:raise ValueError('Actual saved metrics and neural replay proof required')
    verify(source)
    proof_before={str(BODY/n):sha(BODY/n) for n in ('JOBS.json','SUMMARY.json','INPUT_BINDINGS.json')}
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',LP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl')
    rows=[]
    for name,script,flag,limit in (('FUNCTIONAL_REPORT',ROOT/'report_functional_motion.py','--report-completed-body',60),('SAVED_STATE_VIDEO',VIDEO/'render.py','--execute-release',1800)):
        verify(source);verify(proof_before)
        logpath=ROOT/(name+'_FINALIZATION.log');command=[PYTHON,'-B',str(script),flag];row=dict(stage=name,command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(logpath),status='RUNNING');rows.append(row)
        temp=journal.with_suffix('.tmp')
        with temp.open('x') as f:json.dump(rows,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,journal)
        with logpath.open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=limit)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with temp.open('x') as f:json.dump(rows,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,journal)
        if code!=0:raise RuntimeError('Artifact stage failed; preserve outputs and do not rerun physics')
    verify(source);verify(proof_before)
    artifacts={str(ROOT/'FUNCTIONAL_MOTION_REPORT.json'):sha(ROOT/'FUNCTIONAL_MOTION_REPORT.json')}
    for case in CASES:
        folder=VIDEO/'OUTPUT'/case;v=read(folder/'VIDEO.json')
        if v['checkpoint_sha256']!=SHA or v['ffmpeg_exit']!=0 or v['full_decode_exit']!=0 or v['decoded_frames']!=v['frames'] or not v['all71opaque'] or not v['allframe_wholebody'] or any(v[k]!=0 for k in ('physics_steps','NN_forwards','Core_constructions')):raise ValueError('Actual all-frame same-body video integrity required')
        for name in ('VIDEO.json','ACTUAL_SAVED_MOTOR_COMPONENT.mp4','DECODED_MONTAGE.png','SAVED_DIAGNOSTICS.png'):artifacts[str(folder/name)]=sha(folder/name)
    save(completion,dict(status='RENDERED_AWAITING_PARENT_IMAGE_REVIEW',checkpoint_sha256=SHA,original_body_all9stages_exit0=True,artifact_stages=rows,source_hashes=source,body_proof_hashes=proof_before,artifacts=artifacts,training_updates=0,physical_integration_steps=0,NN_forwards=0,Core_constructions=0,desktop_delivery_performed=False,parent_review_required='Open all4 DECODED_MONTAGE.png and all4 SAVED_DIAGNOSTICS.png, then run frozen postrun video deliver.py --parent-reviewed-all4-montages-and4-graphs',body_behavior_pass_not_inferred_from_video=True))
    print('ALL_BODY_ARTIFACTS_READY_FOR_PARENT_ACTUAL_IMAGE_REVIEW',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        p=ROOT/'BODY_ARTIFACT_FINALIZATION_FAILURE.json'
        if not p.exists():save(p,dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),original_body_and_weights_unchanged=True,desktop_delivery_performed=False))
        raise
