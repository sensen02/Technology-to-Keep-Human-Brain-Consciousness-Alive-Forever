"""Parent serial one new long case + fresh336 audit; immutable old archives."""
import argparse
import os
import subprocess
import sys
import time
import runtime as r

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('CODE_ONLY no execution')
        return
    r.verify()
    jobs = []
    r.save(r.HERE/'JOBS.json',jobs)
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl')
    for mode in ('trial','audit'):
        command = [sys.executable,'-B',str(r.HERE/'run.py'),'--mode',mode,'--parent-resource-released']
        stage = dict(mode=mode,command=command,start_epoch_s=time.time(),status='RUNNING')
        jobs.append(stage)
        (r.HERE/'JOBS.json').write_text(__import__('json').dumps(jobs,indent=2)+'\n')
        with (r.HERE/(mode+'.log')).open('x') as stream:
            completed = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,cwd=r.HERE)
        stage.update(exit_code=completed.returncode,status='FINISHED',end_epoch_s=time.time())
        (r.HERE/'JOBS.json').write_text(__import__('json').dumps(jobs,indent=2)+'\n')
        if completed.returncode:
            raise SystemExit(completed.returncode)
    folder = r.HERE/'trials'/r.CASE
    result = r.read(folder/'RESULT.json')
    audit = r.read(folder/'VERIFY.json')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=r.sha(folder/'RESULT.json'):
        raise ValueError('Fresh336 audit failed')
    if result['physical_steps']!=r.TOTAL_NATIVE or result['native']['post_native_steps']!=r.POST_NATIVE:
        raise ValueError('Full25000ms/1536000 post-native window required')
    counts = result['native']['post_native_contacts_by_terrain']
    ledger = r.read(r.HERE/'PREPARED/MODEL_LEDGER.json')['cases'][r.CASE]
    blocks = [item for item in ledger['terrain'] if item['name'].startswith('terrain_block_')]
    coverage = {item['name']:dict(native_solver_contacts=counts.get(item['name'],0),status='CONTACTED' if counts.get(item['name'],0)>0 else 'NOTTESTED') for item in blocks}
    old = r.read(r.REPAIR/'trials'/r.OLD_CASE/'RESULT.json')
    oldcounts = old['native']['post_native_contacts_by_terrain']
    newblocks = [name for name in coverage if counts.get(name,0)>0 and oldcounts.get(name,0)==0]
    # Native contact alone cannot establish traversal. A conservative full-body
    # clearance proof uses saved full-mesh projection in the later parent renderer.
    summary = dict(kind=r.KIND+'_SUMMARY',completed_new_trials=1,completed_fresh336_audits=1,
        actual_new_native_steps=r.TOTAL_NATIVE,post_native_steps=r.POST_NATIVE,
        result_path=str(folder/'RESULT.json'),result_sha256=r.sha(folder/'RESULT.json'),
        audit_path=str(folder/'VERIFY.json'),audit_sha256=r.sha(folder/'VERIFY.json'),
        per_block_coverage=coverage,newly_contacted_blocks_vs_old=newblocks,
        more_blocks_tested=bool(newblocks),unreached_blocks_status='NOTTESTED',
        all_blocks_contacted=all(row['status']=='CONTACTED' for row in coverage.values()),
        whole_course_traversal='PENDING_SAVED_FULL_BODY_CLEARANCE_CHECK',
        terrain_summary=result['terrain_summary'],cycles=result['autonomous_cycles'],
        rough_terrain_pass=result['terrain_summary']['rough_terrain_pass'],
        old_rough_rerun=False,gray_or_brown_control_rerun=False,hook_claim=False,
        old_rough_slip=old['terrain_summary']['relative_five']['slip_ratio_max'],
        NN_training_updates=0,scene=r.scene_metadata())
    r.verify()
    r.save(r.HERE/'SUMMARY.json',summary)
    print(str(r.HERE/'SUMMARY.json'))

if __name__=='__main__':
    main()
