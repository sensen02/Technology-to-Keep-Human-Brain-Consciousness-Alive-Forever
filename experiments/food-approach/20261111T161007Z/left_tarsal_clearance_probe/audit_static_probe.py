"""Independent saved-result audit; default NOT_RUN and stdlib-only imports."""
import argparse
import json
from pathlib import Path
from probe_identity import HERE,FREEZE,verify_sources,sha256,exclusive_result,cpu1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result')
    args=parser.parse_args();verify_sources()
    if not args.result:
        print(json.dumps(dict(status='NOT_RUN',reason='Supply explicit new RESULT_STATIC_PROBE path after parent static execution; no result audited')));return
    path=Path(args.result).resolve()
    if path.parent!=HERE or not path.name.startswith('RESULT_STATIC_PROBE_'):raise RuntimeError('Audit only accepts explicit result in new ROOT')
    cpu1()
    import numpy as np
    r=json.loads(path.read_text());checks={};numerical=[]
    def check(name,value):checks[name]=bool(value)
    def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=float)))
    def close(a,b,tol=1e-9):return bool(np.shape(a)==np.shape(b) and np.max(np.abs(np.asarray(a)-np.asarray(b)))<=tol)
    check('result_sourcehash',r['sourcehash_sha256']==sha256(FREEZE))
    check('completed_fixed16rows',r['status']=='STATIC_PROBE_COMPLETE' and len(r['rows'])==16 and r['row_count']==16)
    check('zero_integration',r['physical_integration_steps']==0 and r['calls']['integration']==0)
    check('model_identity',r['compiled_identity_verified_before_after'] and r['model_and_collision_classification_unchanged'])
    check('live_core_unchanged',r['live_core_data_unchanged'] and r['live_core_digest_before']==r['live_core_digest_after'])
    mapping=r['geometry_mapping']
    check('nonfoot_not_reclassified',mapping['geom_id']==72 and mapping['body_id']==61 and mapping['geom_classification']==-1 and mapping['geom_contype']==2 and mapping['geom_conaffinity']==1)
    config=json.loads((HERE/'PREREG.json').read_text())
    pairs={(float(row['requested_time_s']),row['pose_variant']) for row in r['rows']}
    check('fixed_time_variant_grid',pairs=={(float(t),v) for t in config['sample_times_s'] for v in config['pose_variants']})
    for i,row in enumerate(r['rows']):
        check(f'row{i}_nearest_saved_time',abs(row['actual_saved_time_s']-row['requested_time_s'])<=.000501)
        if row['status']=='ROW_FAILURE_RETAINED':numerical.append(False);continue
        J=np.asarray(row['Jpad']);Z=np.asarray(row['null_basis']);g=np.asarray(row['clearance_gradient'])
        singular=np.linalg.svd(J,compute_uv=False);tol=max(1e-12,float(singular[0])*1e-9);rank=int((singular>tol).sum())
        check(f'row{i}_svd_rank_null',rank==row['pad_rank'] and Z.shape==(7,7-rank) and row['null_dimension']==7-rank)
        check(f'row{i}_null_orthogonal',np.max(np.abs(J@Z))<1e-8 and np.max(np.abs(Z.T@Z-np.eye(Z.shape[1])))<1e-8)
        check(f'row{i}_gZ',close(g@Z,row['clearance_gradient_times_null']) and abs(norm(g@Z)-row['gZ_norm'])<1e-9)
        fd=row['finite_difference'];err=float(np.max(np.abs(np.asarray(fd['attachedpoint_gradient_fd'])-g)))
        check(f'row{i}_fd_error_report',abs(err-fd['attachedpoint_jacobian_max_abs_error'])<1e-9 and (err<1e-6)==row['attachedpoint_fd_pass'])
        check(f'row{i}_support_distance_report',abs(abs(row['geometry_crosscheck']['geomDistance']-row['clearance'])-row['geometry_crosscheck']['support_distance_error'])<1e-9)
        numerical.append(bool(row['attachedpoint_fd_pass'] and row['geometry_crosscheck_pass'] and row['pad_fd_status']!='FAIL'))
        if 'kinematic_sequence' not in row:continue
        seq=row['kinematic_sequence'];q0=np.asarray(seq['q_initial']);qf=np.asarray(seq['q_final']);delta=qf-q0
        check(f'row{i}_cumulative_bound',np.max(np.abs(delta))<=.10000000001 and seq['step_count']<=20)
        check(f'row{i}_rate_time_lower_bound',abs(seq['rate2rad_s_time_lower_bound_s']-np.max(np.abs(delta))/2.)<1e-12)
        prev=q0
        for j,step in enumerate(seq['steps']):
            check(f'row{i}_step{j}_chain',close(step['q_initial'],prev,1e-12))
            if 'q_final' not in step:continue
            now=np.asarray(step['q_final'])
            check(f'row{i}_step{j}_trust',np.max(np.abs(now-prev))<=.00500000001)
            if step.get('local_improvement'):
                check(f'row{i}_step{j}_pad',step['pad_error_norm']<=1e-5 and close(step['fixed_primary_target'],row['pad_primary'],1e-12))
                check(f'row{i}_step{j}_bounds',step['original_joint_bounds_respected'])
                for name,hnew in step['other_RH_heights_after'].items():
                    if name==mapping['geom_name']:continue
                    check(f'row{i}_step{j}_other_{name}',hnew>=min(step['other_RH_heights_before'][name],0.)-1.0001e-8)
                prev=now
        if seq['safe_margin_reached']:
            check(f'row{i}_safe_margin',seq['final_height']>=seq['safe_height'] and seq['pad_error_norm']<=1e-5)
    check('summary_count_local',r['summary']['rows_with_local_improvement']==sum(bool(x.get('secondary',{}).get('local_improvement')) for x in r['rows']))
    audit=dict(status='STATIC_RESULT_AUDITED',integrity_all_pass=all(checks.values()),numerical_geometry_all_pass=bool(numerical and all(numerical)),checks=checks,
        result=str(path),result_sha256=sha256(path),sourcehash_sha256=sha256(FREEZE),summary=r['summary'],
        sequence_rows_reaching_safe_margin=sum(bool(x.get('kinematic_sequence',{}).get('safe_margin_reached')) for x in r['rows']),
        scope='Independent saved-result arithmetic/identity checks only; no FK, physics or force prediction; feasibility and integrity separate; no behavior qualification')
    out=exclusive_result('AUDIT_STATIC_PROBE',audit);print(json.dumps(dict(audit=str(out),integrity_all_pass=audit['integrity_all_pass'],numerical_geometry_all_pass=audit['numerical_geometry_all_pass'],checks=len(checks))))
    if not audit['integrity_all_pass']:raise RuntimeError('Static-result integrity checks failed; audit retained')


if __name__=='__main__':main()
