"""Parent-only one repaired full trial + one fresh fourNN/source audit.

Qualified old plane/flat-box controls are referenced as immutable archives;
this entry never reruns them. Sources/scene are fixed before behavior is seen.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path,value,*,exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no trial or audit subprocess')
        return
    import runtime
    runtime.verify()
    preflight = runtime.read(HERE/'PREFLIGHT.json')
    if not preflight['all_passed'] or not all(preflight['checks'].values()) or preflight['actual_Core_constructions']!=1 or preflight['actual_native_steps']!=1600:
        raise ValueError('Actual new scene oneCore1600step preflight required')
    if preflight['actual_xml_sha256']!=runtime.scene_metadata()['new_scene_xml_sha256']:
        raise ValueError('Preflight scene XML mismatch')
    if (HERE/'JOBS.json').exists() or (HERE/'trials'/runtime.CASE).exists():
        raise FileExistsError('No blind rerun or overwrite of repaired scene')
    jobs = []
    save(HERE/'JOBS.json',jobs)
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1')
    for mode in ('trial','audit'):
        command = [sys.executable,'-B',str(HERE/'run.py'),'--mode',mode,'--parent-resource-released']
        log = HERE/(mode+'.log')
        stage = dict(mode=mode,case=runtime.CASE,command=command,status='RUNNING',start_epoch_s=time.time(),log=str(log))
        jobs.append(stage)
        save(HERE/'JOBS.json',jobs,exclusive=False)
        with log.open('x') as stream:
            process = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=HERE,env=environment)
        stage.update(status='FINISHED',exit_code=process.returncode,end_epoch_s=time.time())
        save(HERE/'JOBS.json',jobs,exclusive=False)
        if process.returncode:
            raise SystemExit(process.returncode)
    folder = HERE/'trials'/runtime.CASE
    result,audit = runtime.read(folder/'RESULT.json'),runtime.read(folder/'VERIFY.json')
    if result['kind']!=runtime.KIND or not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=runtime.sha(folder/'RESULT.json'):
        raise ValueError('Repaired full-trial result/audit identity mismatch')
    if result['physical_steps']!=1232000 or result['native']['post_native_steps']!=768000:
        raise ValueError('Repaired full-trial fixed duration mismatch')
    scene = runtime.scene_metadata()
    controls = {}
    for case,record in scene['archived_controls'].items():
        old_result = runtime.read(record['result_path'])
        old_audit = runtime.read(record['verify_path'])
        if runtime.sha(record['result_path'])!=record['result_sha256'] or runtime.sha(record['verify_path'])!=record['verify_sha256'] or not old_audit['integrity_all_pass'] or not all(old_audit['checks'].values()):
            raise ValueError('Archived control drift '+case)
        controls[case] = dict(record,terrain_summary=old_result['terrain_summary'],physical_steps_repeated=0)
    runtime.verify()
    summary = dict(kind='SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_SUMMARY_V1',scene_repair=scene,
        completed_new_trials=1,completed_new_audits=1,new_body_native_steps=1232000,
        prior_new_scene_preflight_native_steps=1600,NN_training_updates=0,
        integrity_all_pass=True,archived_controls=controls,
        result_path=str(folder/'RESULT.json'),result_sha256=runtime.sha(folder/'RESULT.json'),
        audit_path=str(folder/'VERIFY.json'),audit_sha256=runtime.sha(folder/'VERIFY.json'),
        terrain_summary=result['terrain_summary'],cycles=result['autonomous_cycles'],native=result['native'],
        legacy_allterrain_autonomous=result['autonomous'],functional_stable_body=result['functional_stable_body'],
        terrain_relative_functional_stable_body=result['terrain_relative_functional_stable_body'],
        new_scene_autonomous_result='CONTACT_TESTED' if result['terrain_summary']['terrain_contact_tested'] else 'NOT_TESTED',
        rough_terrain_pass=result['terrain_summary']['rough_terrain_pass'],
        old_stopped_scene_reclassified_as_pass=False,original_D1_D8_or_claw_pulvilli_claim=False)
    save(HERE/'SUMMARY.json',summary)
    print(json.dumps(dict(integrity_all_pass=True,completed_new_trials=1,new_body_native_steps=1232000,
        terrain_contact_tested=result['terrain_summary']['terrain_contact_tested'],rough_terrain_pass=summary['rough_terrain_pass'])))


if __name__ == '__main__':
    main()
