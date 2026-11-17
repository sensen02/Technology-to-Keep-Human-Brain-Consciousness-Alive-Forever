"""Parent only: exactly one matched flat trial, fresh336 audit, diagnostic summary."""
import argparse
import os
import subprocess
import sys
import time
import runtime as r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-resource-released',action='store_true')
    a=p.parse_args()
    if not a.parent_resource_released:
        print('ONLYCODE no execution');return
    r.verify()
    guard=r.read(r.HERE/'PARENT_ZERO_STEP_GUARD.json')
    if not guard['all_passed'] or guard['Core_constructions']!=1 or guard['physics_steps']!=0:
        raise ValueError('Actual compiled oneCore zero-step guard required')
    if guard['freeze_sha256']!=r.sha(r.HERE/'FROZEN.json'):
        raise ValueError('Compiled guard source binding mismatch')
    jobs=[];r.save(r.HERE/'JOBS.json',jobs)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl')
    for mode in ('trial','audit'):
        command=[sys.executable,'-B',str(r.HERE/'run.py'),'--mode',mode,'--parent-resource-released']
        stage=dict(mode=mode,command=command,start_epoch_s=time.time(),status='RUNNING');jobs.append(stage)
        (r.HERE/'JOBS.json').write_text(__import__('json').dumps(jobs,indent=2)+'\n')
        with (r.HERE/(mode+'.log')).open('x') as stream:
            completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,cwd=r.HERE)
        stage.update(exit_code=completed.returncode,status='FINISHED',end_epoch_s=time.time())
        (r.HERE/'JOBS.json').write_text(__import__('json').dumps(jobs,indent=2)+'\n')
        if completed.returncode: raise SystemExit(completed.returncode)
    folder=r.HERE/'trials'/r.CASE
    result=r.read(folder/'RESULT.json');audit=r.read(folder/'VERIFY.json')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=r.sha(folder/'RESULT.json'):
        raise ValueError('Fresh336 audit required')
    if result['physical_steps']!=r.TOTAL_NATIVE or result['native']['post_native_steps']!=r.POST_NATIVE or result['capture_count']!=r.FRAMES or result['neural_records']!=r.DECISIONS:
        raise ValueError('Full unchanged 44.2s schedule required')
    from diagnostics import summarize
    diagnostic=summarize(folder,result)
    r.verify()
    r.save(r.HERE/'SUMMARY.json',dict(kind=r.KIND+'_SUMMARY',completed_new_trials=1,completed_fresh336_audits=1,actual_new_native_steps=r.TOTAL_NATIVE,post_native_steps=r.POST_NATIVE,result_path=str(folder/'RESULT.json'),result_sha256=r.sha(folder/'RESULT.json'),audit_path=str(folder/'VERIFY.json'),audit_sha256=r.sha(folder/'VERIFY.json'),compiled_guard_sha256=r.sha(r.HERE/'PARENT_ZERO_STEP_GUARD.json'),rough_terrain_pass=None,obstacle_coverage='NOT_APPLICABLE_FLAT_CONTROL',reference_original_five_gates=result['terrain_summary']['relative_five'],reference_behavior_diagnostic_pass=result['terrain_summary']['behavior_diagnostic_pass'],functional_stable_body=result['functional_stable_body'],autonomous_cycles=result['autonomous_cycles'],long_term_diagnostic=diagnostic,scene=r.scene_metadata(),neural_biological_validity='NOT_ESTABLISHED',old_physics_rerun=False))
    print(r.HERE/'SUMMARY.json')
if __name__=='__main__': main()
