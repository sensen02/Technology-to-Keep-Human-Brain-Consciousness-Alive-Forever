"""Parent managed-job entry: exactly3 complete terrain trials and3 fresh audits.

Finite geometry/frozen policies, no training. Integrity failure stops subsequent
stages; scientific behavior failures remain in successful result files unchanged.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CASES = ('plane_control','flat_box_control','mild_blocks_seed17')


def record_jobs(jobs):
    # This mutable status file is an experiment output, never a frozen source.
    with (HERE/'JOBS.json').open('w') as stream:
        json.dump(jobs,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no process/model/physics started')
        return
    import runtime
    runtime.verify()
    proof = runtime.read(HERE/'PREFLIGHT.json')
    if not proof['all_passed'] or not all(proof['checks'].values()) or proof['Core_constructions'] != 0:
        raise ValueError('Actual zeroCore fourNN/source/camera-queue preflight required')
    if (HERE/'JOBS.json').exists() or any((HERE/'trials'/case).exists() for case in CASES):
        raise FileExistsError('No blind restart or overwrite of full terrain cases')
    jobs = []
    record_jobs(jobs)
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl')
    for case in CASES:
        for entry in ('runner.py','audit.py'):
            log = HERE/(case+'_'+entry.replace('.py','')+'.log')
            command = [sys.executable,str(HERE/entry),'--case',case,'--parent-resource-released']
            stage = dict(case=case,entry=entry,command=command,status='RUNNING',start_epoch_s=time.time(),log=str(log))
            jobs.append(stage)
            record_jobs(jobs)
            with log.open('x') as stream:
                process = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,cwd=HERE)
            stage.update(status='FINISHED',exit_code=process.returncode,end_epoch_s=time.time())
            record_jobs(jobs)
            if process.returncode:
                raise SystemExit(process.returncode)
    results = {}
    for case in CASES:
        folder = HERE/'trials'/case
        result = runtime.read(folder/'RESULT.json')
        audit = runtime.read(folder/'VERIFY.json')
        if not audit['integrity_all_pass'] or audit['result_sha256'] != runtime.sha(folder/'RESULT.json') or not all(audit['checks'].values()):
            raise ValueError('Completed audit identity mismatch '+case)
        results[case] = dict(result_path=str(folder/'RESULT.json'),result_sha256=runtime.sha(folder/'RESULT.json'),
            audit_sha256=runtime.sha(folder/'VERIFY.json'),actual_native_steps=result['physical_steps'],
            legacy_allterrain_five=result['autonomous'],terrain=result['terrain_summary'],
            cycles=result['autonomous_cycles'],native=result['native'],yaw_change_rad=result['yaw_change_rad'])
    runtime.verify()
    summary = dict(kind='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_SUMMARY_V1',cases=results,
        completed_trial_count=3,completed_fresh_audit_count=3,
        actual_native_steps=sum(value['actual_native_steps'] for value in results.values()),
        expected_native_steps=3696000,neural_training_updates=0,
        integrity_all_pass=True,rough_contact_tested=results['mild_blocks_seed17']['terrain']['terrain_contact_tested'],
        rough_terrain_pass=results['mild_blocks_seed17']['terrain']['rough_terrain_pass'],
        plane_behavior_pass=results['plane_control']['terrain']['behavior_diagnostic_pass'],
        sourcefreeze_sha256=runtime.sha(HERE/'FROZEN.json'),
        biological_hook_pulvilli_or_original_D1_D8_claim=False)
    if summary['actual_native_steps'] != summary['expected_native_steps']:
        raise ValueError('Three-trial native step total mismatch')
    with (HERE/'SUMMARY.json').open('x') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(completed=3,integrity_all_pass=True,rough_contact_tested=summary['rough_contact_tested'],rough_terrain_pass=summary['rough_terrain_pass'])))


if __name__ == '__main__':
    main()
