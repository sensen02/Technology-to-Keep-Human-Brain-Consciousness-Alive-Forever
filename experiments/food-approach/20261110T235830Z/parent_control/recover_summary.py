"""Recover aggregation from completed recorded controls. No new physics or optimization."""
from pathlib import Path
import json
import evaluate_parent as ep
import verify_actual

HERE=ep.HERE

def main():
    original=HERE/'SUMMARY.json';saved=HERE/'SUMMARY_INTERRUPTED_ORIGINAL.json'
    if ep.sha256(original)!=ep.sha256(saved):raise ValueError('historical summary copy mismatch')
    history=json.loads(original.read_text())
    if history['status']!='RUNNING' or set(history['conditions'])!={'left'}:raise ValueError('unexpected original interruption scope')
    for p,h in history['source_hashes'].items():
        if ep.sha256(p)!=h:raise ValueError('frozen source drift '+p)
    verification=verify_actual.module();reports={}
    for c in ('left','right'):
        run=HERE/('EVAL_'+c);rp=run/'RESULT.json';ap=HERE/('VERIFY_'+c+'.json')
        r=json.loads(rp.read_text());a=json.loads(ap.read_text())
        fresh=verification.verify(run)
        if not a['integrity_all_pass'] or not all(a['checks'].values()) or not fresh['integrity_all_pass']:raise ValueError('saved actual verification failed '+c)
        out=HERE/('RECOVERED_VERIFY_'+c+'.json')
        with out.open('x') as f:json.dump(fresh,f,indent=2);f.write('\n')
        reports[c]=dict(condition=c,result=str(rp),result_sha256=ep.sha256(rp),audit=str(ap),audit_sha256=ep.sha256(ap),recovery_audit=str(out),recovery_audit_sha256=ep.sha256(out),audit_checks=len(fresh['checks']),integrity=True,eligible=bool(r['engineering_pass'] and fresh['engineering_pass']),criteria=r['criteria'],autonomous=r['autonomous'],cycles=r['autonomous_cycles']['complete_per_leg'],yaw_change_rad=r['yaw_change_rad'],source_hashes=r['source_hashes'],parent_control_neutral=r['parent_control_neutral'],execution_terminal_status='original left evaluator and audit rc0 recorded in original summary' if c=='left' else 'original terminal return code unavailable; full saved result, log and original audit exist and fresh saved-trajectory audit passed')
    summary=dict(status='COMPLETE_RECOVERED_FROM_FULL_SAVED_RESULTS',checkpoint=str(ep.CHECKPOINT),checkpoint_sha256=ep.SHA,conditions=reports,all_conditions_eligible=all(x['eligible'] for x in reports.values()),saved_trajectory_verification=all(x['integrity'] for x in reports.values()),original_job=dict(id='bash-239',runtime_status='unknown job handle after session interruption; terminal code unavailable',original_summary=str(original),original_summary_sha256=ep.sha256(original),historical_copy=str(saved),historical_copy_sha256=ep.sha256(saved),original_recorded_conditions=list(history['conditions']),no_live_duplicate_process='parent confirmed no parent_control evaluation processes before recovery'),recovery_source={str(Path(__file__).resolve()):ep.sha256(__file__)},new_physics_integration_steps=0,new_optimizer_updates=0,source_freeze_unchanged=True,scope='Matched parent500 fixed1s-ramp controls only, no native or food qualification. Existing complete trajectories audited without repeating physics.')
    with (HERE/'RECOVERED_SUMMARY.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=summary['status'],all_conditions_eligible=summary['all_conditions_eligible'],saved_trajectory_verification=True,conditions={c:dict(slip=v['autonomous']['slip_ratio_max'],forward=v['autonomous']['forward_signed_native_per_s'],yaw=v['yaw_change_rad'],cycles=v['cycles'],checks=v['audit_checks']) for c,v in reports.items()},new_physics_steps=0)))

if __name__=='__main__':main()
