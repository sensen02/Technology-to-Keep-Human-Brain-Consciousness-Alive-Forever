"""Event-driven preregistered candidate confirmation; no repeated polling or training."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TRAIN=ROOT/'20261110T152714Z'
PYTHON=ROOT/'20261110T125712Z/env/bin/python'
AUDITED=ROOT/'20261110T131413Z/audited_eval.py'
WRAPPER=ROOT/'20261110T152725Z/final_native_audit.py'


def wait_file(target):
    """Subscribe to Linux filesystem notifications; no sleep/existence polling."""
    libc=ctypes.CDLL(None,use_errno=True)
    while not target.exists():
        folder=target.parent
        while not folder.is_dir():folder=folder.parent
        fd=libc.inotify_init1(os.O_CLOEXEC)
        if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        try:
            wd=libc.inotify_add_watch(fd,os.fsencode(folder),0x80|0x100)
            if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
            if target.exists():return
            # Re-arm deeper watch when a required parent appears.
            while not target.exists() and (not target.parent.is_dir() or folder==target.parent):
                os.read(fd,65536)
                if folder!=target.parent and target.parent.is_dir():break
        finally:os.close(fd)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save_status(value):
    temp=HERE/'CONFIRMATION_STATUS.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(HERE/'CONFIRMATION_STATUS.json')


def command(argv,label):
    print(json.dumps(dict(phase=label,command=[str(x) for x in argv])),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (HERE/(label+'.log')).open('x') as log:
        result=subprocess.run([str(PYTHON),'-B']+[str(x) for x in argv],env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(phase=label,exit_code=result.returncode)),flush=True)
    return result.returncode


def main():
    selected=None
    save_status(dict(phase='waiting for preregistered nominal2500/3000',physics_launched=False,script_sha256=digest(__file__)))
    for step in (2500,3000):
        baseline=TRAIN/f'EVAL_ROLE_{step}'
        wait_file(baseline/'RESULT.json')
        result=json.loads((baseline/'RESULT.json').read_text())
        print(json.dumps(dict(nominal=step,gates=result['autonomous']['gates'],cycles=result['autonomous_cycles']['complete_per_leg'])),flush=True)
        if result['autonomous']['all_gates'] and result['autonomous_cycles']['all_legs_completed']:
            selected=(step,baseline);break
    if selected is None:
        save_status(dict(phase='no nominal candidate passed',physics_launched=False,full_goal_passed=False))
        return 0
    step,baseline=selected
    identity=json.loads((baseline/'IDENTITY.json').read_text())
    checkpoint=Path(identity['config']['cli']['checkpoint']).resolve()
    if checkpoint.parent!=TRAIN/'BC_GRU_ROLE_3000':raise ValueError('unexpected candidate checkpoint')
    sys.path.insert(0,str(TRAIN))
    from identity import verify
    verify(identity)
    out=HERE/f'NATIVE_ROLE_{step}'
    status=dict(phase='native independent recording',selected_step=step,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),baseline=str(baseline),baseline_result_sha256=digest(baseline/'RESULT.json'),native=str(out),physics_launched=True,full_goal_passed=False)
    save_status(status)
    rc=command([AUDITED,'--checkpoint',checkpoint,'--out',out,'--native'],f'native_{step}')
    if rc:raise RuntimeError('native recorder failed '+str(rc))
    native=json.loads((out/'RESULT.json').read_text())
    status.update(phase='native recording complete',native_engineering=native['autonomous']['gates']);save_status(status)
    # Original complete native audit runs even if native gate differs from fast candidate.
    audit=HERE/f'NATIVE_ROLE_{step}_AUDIT.json'
    rc=command([WRAPPER,'--run',out,'--baseline',baseline,'--out',audit],f'audit_{step}')
    status.update(phase='native audit complete',audit=str(audit),audit_exit_code=rc);save_status(status)
    if rc: return rc
    perturbations=[]
    for seed in (20261111,20261112):
        perturb=HERE/f'PERTURB_ROLE_{step}_SEED_{seed}'
        rc=command([HERE/'confirm_perturb.py','--checkpoint',checkpoint,'--baseline',baseline,'--out',perturb,'--seed',seed],f'perturb_{step}_{seed}')
        if rc:raise RuntimeError('perturbation evidence failed '+str(rc))
        r=json.loads((perturb/'RESULT.json').read_text())
        perturbations.append(dict(seed=seed,path=str(perturb),all_gates=r['autonomous']['all_gates'],cycles=r['autonomous_cycles']['complete_per_leg'],slip=r['autonomous']['slip_ratio_max']))
        status.update(phase='perturbation confirmation',perturbations=perturbations);save_status(status)
    rc=command([HERE/'render_actual.py','--run',out,'--out',HERE/f'VIDEO_NATIVE_ROLE_{step}'],f'video_{step}')
    if rc:raise RuntimeError('same-run rendering failed '+str(rc))
    status.update(phase='native audit, perturbations and video complete; parent visual/scope review pending',video=str(HERE/f'VIDEO_NATIVE_ROLE_{step}'),full_goal_passed=False)
    save_status(status)
    print(json.dumps(status),flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),flush=True)
        raise
