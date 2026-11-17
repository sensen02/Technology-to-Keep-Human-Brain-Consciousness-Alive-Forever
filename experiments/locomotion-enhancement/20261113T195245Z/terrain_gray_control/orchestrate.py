"""Parent finite gray-world trial + fresh audit + saved matched statistics.

Brown163, plane and flat-box remain archives. No rephysics, retries or training.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def save(path,value,mode='x'):
    with Path(path).open(mode) as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-resource-released',action='store_true')
    a = p.parse_args()
    if not a.parent_resource_released:
        print('SOURCE_ONLY no trial started')
        return
    import runtime
    runtime.verify()
    proof = runtime.read(HERE/'PREFLIGHT.json')
    if not proof['all_passed'] or not all(proof['checks'].values()) or proof['actual_Core_constructions']!=1 or proof['gray_native_steps']!=1600 or proof['brown_physical_steps']!=0:
        raise ValueError('Actual grayCore physical equality and bounded contact preflight required')
    if (HERE/'JOBS.json').exists() or (HERE/'trials'/runtime.CASE).exists():
        raise FileExistsError('No overwrite or repeated gray scene trial')
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1')
    stages = [('trial',[str(HERE/'run.py'),'--mode','trial','--parent-resource-released']),
              ('audit',[str(HERE/'run.py'),'--mode','audit','--parent-resource-released']),
              ('matched_saved_statistics',[str(HERE/'matched.py'),'--parent-read-saved-arrays'])]
    jobs = []
    save(HERE/'JOBS.json',jobs)
    for name,arguments in stages:
        log = HERE/(name+'.log')
        command = [sys.executable,'-B',*arguments]
        row = dict(stage=name,command=command,status='RUNNING',start_epoch_s=time.time(),log=str(log))
        jobs.append(row)
        save(HERE/'JOBS.json',jobs,'w')
        with log.open('x') as stream:
            process = subprocess.run(command,cwd=HERE,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        row.update(status='FINISHED',exit_code=process.returncode,end_epoch_s=time.time())
        save(HERE/'JOBS.json',jobs,'w')
        if process.returncode:
            raise SystemExit(process.returncode)
    folder = HERE/'trials'/runtime.CASE
    result,audit = runtime.read(folder/'RESULT.json'),runtime.read(folder/'VERIFY.json')
    matched = runtime.read(HERE/'MATCHED.json')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=runtime.sha(folder/'RESULT.json'):
        raise ValueError('Actual gray audit identity mismatch')
    if result['physical_steps']!=1232000 or result['native']['post_native_steps']!=768000:
        raise ValueError('Fixed complete trial duration mismatch')
    runtime.verify()
    summary = dict(kind=runtime.SUMMARY_KIND,case=runtime.CASE,new_body_native_steps=1232000,
        new_preflight_native_steps=1600,brown_reference_physics_repeated=0,NN_training_updates=0,
        integrity_all_pass=True,color_control=runtime.color_metadata(),
        result_path=str(folder/'RESULT.json'),result_sha256=runtime.sha(folder/'RESULT.json'),
        verify_path=str(folder/'VERIFY.json'),verify_sha256=runtime.sha(folder/'VERIFY.json'),
        matched_saved_path=str(HERE/'MATCHED.json'),matched_saved_sha256=runtime.sha(HERE/'MATCHED.json'),
        terrain_summary=result['terrain_summary'],cycles=result['autonomous_cycles'],native=result['native'],
        legacy_allterrain_autonomous=result['autonomous'],matched_comparison=matched['comparison'],
        terrain_contact_tested=result['terrain_summary']['terrain_contact_tested'],rough_terrain_pass=result['terrain_summary']['rough_terrain_pass'],
        conclusion_scope='Only this color-only world intervention with unchanged336 policy/geometry; not a new352 deployment, claw model, unique slip explanation or general terrain validation')
    save(HERE/'SUMMARY.json',summary)
    print(json.dumps(dict(integrity_all_pass=True,terrain_contact_tested=summary['terrain_contact_tested'],rough_terrain_pass=summary['rough_terrain_pass'],matched_comparison=summary['matched_comparison'])))


if __name__=='__main__':
    main()
