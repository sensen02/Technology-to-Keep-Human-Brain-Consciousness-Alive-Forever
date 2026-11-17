"""Saved-only v1 independent metrics plus same-direction archivedprefix audit."""
import argparse
from adapter import run
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=('left_pivot','right_pivot'),required=True);p.add_argument('--execute-parent-saved-audit',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_audit:p.error('Parent saved-only audit flagrequired')
    run('audit_saved.py')
    import numpy as np
    folder,old_result,old_audit=archived_reference(a.condition)
    trial=ROOT/'trials'/a.condition;new_result=read(trial/'RESULT.json');new_audit=read(trial/'VERIFY.json');checks={}
    for name,count in (('trace_1ms.npz',5800),('geometry_1ms.npz',4800),('teacher_samples_5ms.npz',960)):
        with np.load(folder/name,allow_pickle=False) as aa,np.load(trial/name,allow_pickle=False) as bb:
            checks[name+'_same_keys']=set(aa.files)==set(bb.files)
            for key in aa.files:
                x=aa[key];y=bb[key];checks[name+'_'+key+'_prefix_exact']=len(x)>=count and len(y)>=count and np.array_equal(x[:count],y[:count])
            if name=='teacher_samples_5ms.npz':checks['candidate_first_action_same_initial_state']=len(aa['Y42'])>960 and len(bb['Y42'])>960 and np.array_equal(aa['Y42'][960],bb['Y42'][960])
    old_identity=read(folder/'IDENTITY.json');new_identity=read(trial/'IDENTITY.json')
    checks['compiled_model_arrays_exact']=old_identity['compiled_arrays']==new_identity['compiled_arrays']
    checks['finite_PD_actuators_exact']=old_identity['actuator']==new_identity['actuator']
    checks['RNG_initial_exact']=read(folder/'RNG_INITIAL.json')==read(trial/'RNG_INITIAL.json')
    checks['HANDOVER_actual_snapshots_exact']=read(folder/'HANDOVER.json')==read(trial/'HANDOVER.json')
    checks['body_length_geometry_exact']=old_result['body_length']==new_result['body_length']
    checks['both_saved_audits_qualified']=old_audit['integrity_all_pass'] and new_audit['integrity_all_pass'] and new_audit['result_sha256']==sha(trial/'RESULT.json')
    checks['per_case_budget']=new_result['actual_native_steps']<=1232000
    checks['teacher_only_no_neural_claim']=new_result['no_neural_models'] and new_result['no_optimizer'] and new_result['kind']=='FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2'
    m=new_result['candidate_metrics']
    comparison=dict(kind='ARCHIVED_ACTUAL_VS_NEW_FIXED_PRIVATE_SHADOW_PIVOT_V2',condition=a.condition,integrity_all_pass=bool(all(checks.values())),checks=checks,archived_control=dict(result=str(folder/'RESULT.json'),sha256=sha(folder/'RESULT.json'),actual_ms=old_result['actual_ms'],candidate_metrics=old_result['candidate_metrics'],native=old_result['native']),new_fixed=dict(result=str(trial/'RESULT.json'),sha256=sha(trial/'RESULT.json'),actual_ms=new_result['actual_ms'],candidate_metrics=m,native=new_result['native']),teacher_pivot_task_pass=bool(m is not None and m['task_pass_candidate']),no_neural_pivot_claim=True,physical_steps=0,neural_forwards=0,scope='One new same-direction fixedprivate-shadow factor vs frozen archived v1; require exactwarm5800ms/model/PD/RNG/handover. Teacher physicaltask qualification is not ANN/neck/nativefly ability; failures retained, no automatic parameter search.')
    target=ROOT/('MATCHED_'+a.condition+'.json');save(target,comparison)
    if not all(checks.values()):raise ValueError('Same-direction archivedprefix comparison failed; evidence retained')
    print(json.dumps(dict(condition=a.condition,comparison=str(target),integrity_all_pass=True,teacher_pivot_task_pass=comparison['teacher_pivot_task_pass'])))


if __name__=='__main__':main()
