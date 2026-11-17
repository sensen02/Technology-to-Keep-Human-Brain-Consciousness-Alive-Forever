"""Parent saved-only v1 audit plus archived-control prefix/parameter comparison."""
from adapter import run
from common import *


def main():
    # Exact v1 independent metrics/legacygates/cycles/schema audit, new ROOT.
    run('audit_saved.py')
    import numpy as np
    folder,old_result,old_audit=archived_reference()
    trial=ROOT/'trials/stop';new_result=read(trial/'RESULT.json');new_audit=read(trial/'VERIFY.json');checks={}
    pairs=(('trace_1ms.npz',5800),('geometry_1ms.npz',4800),('teacher_samples_5ms.npz',960))
    for name,count in pairs:
        with np.load(folder/name,allow_pickle=False) as aa,np.load(trial/name,allow_pickle=False) as bb:
            checks[name+'_same_keys']=set(aa.files)==set(bb.files)
            for key in aa.files:
                a=aa[key];b=bb[key]
                checks[name+'_'+key+'_prefix_exact']=len(a)>=count and len(b)>=count and np.array_equal(a[:count],b[:count])
            if name=='teacher_samples_5ms.npz':checks['candidate_first_action_same_initial_state']=len(aa['Y42'])>960 and len(bb['Y42'])>960 and np.array_equal(aa['Y42'][960],bb['Y42'][960])
    old_identity=read(folder/'IDENTITY.json');new_identity=read(trial/'IDENTITY.json')
    checks['compiled_model_arrays_exact']=old_identity['compiled_arrays']==new_identity['compiled_arrays']
    checks['finite_PD_actuators_exact']=old_identity['actuator']==new_identity['actuator']
    checks['RNG_initial_exact']=read(folder/'RNG_INITIAL.json')==read(trial/'RNG_INITIAL.json')
    checks['HANDOVER_actual_snapshots_exact']=read(folder/'HANDOVER.json')==read(trial/'HANDOVER.json')
    checks['body_length_geometry_exact']=old_result['body_length']==new_result['body_length']
    checks['both_saved_audits_qualified']=old_audit['integrity_all_pass'] and new_audit['integrity_all_pass'] and new_audit['result_sha256']==sha(trial/'RESULT.json')
    checks['one_new_case_budget']=new_result['actual_native_steps']<=1232000
    checks['no_neural_stop_claim']=(new_result['candidate_metrics'] is None or new_result['candidate_metrics']['task_pass_candidate'] is None) and new_result['no_neural_models'] and new_result['no_optimizer']
    m=new_result['candidate_metrics']
    comparison=dict(kind='ARCHIVED_ACTUAL_VS_NEW_FIXED_PRIVATE_SHADOW_STOP_V1',integrity_all_pass=bool(all(checks.values())),checks=checks,archived_control=dict(result=str(folder/'RESULT.json'),sha256=sha(folder/'RESULT.json'),actual_ms=old_result['actual_ms'],candidate_metrics=old_result['candidate_metrics'],native=old_result['native']),new_fixed=dict(result=str(trial/'RESULT.json'),sha256=sha(trial/'RESULT.json'),actual_ms=new_result['actual_ms'],candidate_metrics=m,native=new_result['native']),new_support_full_window_qualified=bool(m is not None and all(m['gates'].values())),no_stop_speed_PASS_defined=True,physical_steps=0,neural_forwards=0,scope='One new fixedprivate-shadow factor vs immutable archived actual-shadow stop; exact5800ms prefix and model/actuator/RNG checks required. Failure does not prove impossibility; supportfullwindow qualification is not stopped-speed/pivot/neuralcontrol success.')
    save(ROOT/'MATCHED_COMPARISON.json',comparison)
    if not all(checks.values()):raise ValueError('Matched archived-control audit failed; retained evidence, no retry')
    print(json.dumps(dict(comparison=str(ROOT/'MATCHED_COMPARISON.json'),integrity_all_pass=True,new_support_full_window_qualified=comparison['new_support_full_window_qualified'])))


if __name__=='__main__':main()
