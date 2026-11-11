"""Pure arithmetic/NumPy/reference/AST preflight only; no Core/FK/model imports."""
import ast
import json
import os
import sys
import numpy as np
from guard_identity import HERE,FREEZE,verify_sources,sha256,exclusive
from guard_contract import feedback,support_flags


def main():
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    verify_sources();checks={}
    def check(k,v):checks[k]=bool(v)
    a=feedback(961,.003,.020)
    check('actual_low_shadow_safe_still_triggers',a['trigger'] and abs(a['actual_deficit']-.009)<1e-15 and abs(a['desired_shadow_height']-.029)<1e-15)
    a=feedback(961,.020,-.001)
    check('actual_safe_shadow_low_no_feedback',not a['trigger'] and a['applied_deficit']==0 and a['desired_shadow_height']==-.001)
    check('prime_feedback_exactly_inactive',all(not feedback(i,-.001,.02)['trigger'] and feedback(i,-.001,.02)['applied_deficit']==0 and feedback(i,-.001,.02)['desired_shadow_height']==.02 for i in range(961)))
    check('guard_threshold_no_trigger_at_equality',not feedback(961,.012,.02)['trigger'])
    check('lower_actual_means_more_positive_target',feedback(961,-.002,.01)['desired_shadow_height']>feedback(961,.002,.01)['desired_shadow_height']>.01)
    check('deficit_never_lowers_q_des_target',all(feedback(1000,h,.003)['desired_shadow_height']>=.003 for h in np.linspace(-.01,.03,101)))
    check('no_qdes_absolute_height_gate',feedback(1000,.002,.1)['trigger'] and feedback(1000,.002,-.1)['trigger'])
    try:feedback(961,float('nan'),.02);invalid=False
    except ValueError:invalid=True
    check('nonfinite_actual_rejected',invalid)
    check('unsupported_flags_loaded_boundary',support_flags(.1,.004,1.,1.25)['actual_RH_loaded_ground'])
    check('unloaded_support_context',support_flags(.05,.003,1.,1.25)['actual_RH_unloaded'] and support_flags(.05,.003,1.,1.25)['actual_RH_unsupported'])
    check('high_pad_not_loaded_ground',support_flags(.2,.005,1.,1.25)['actual_RH_unsupported'])
    import turning_teacher as t
    check('frozen_constraints_unchanged',t.CONFIG==t._parent.CONFIG and t.CONFIG['per_decision_trust_rad']==.005 and t.CONFIG['local_updates_max']==4)
    prev=np.linspace(-.5,.5,42);primary=prev+np.linspace(-.012,.012,42);final=primary.copy();final[28:35]+=.004
    clipped,raw=t._contract.whole_decision_velocity(prev,final)
    check('whole_decision_velocity_primary_plus_secondary',np.array_equal(raw,(final-prev)/.005) and not np.array_equal(raw,(final-primary)/.005))
    check('original_clip2',np.array_equal(clipped,np.clip(raw,-2,2)))
    p=np.zeros(7);q=np.full(7,.004);pad=np.zeros(3);limits=np.tile([-1.,1.],(7,1))
    check('partial_clearance_improvement_accepted',t._contract.validate_secondary(p,q,pad,pad,limits,0.,.001)[0])
    check('decision_trust_rejection',not t._contract.validate_secondary(p,np.full(7,.00501),pad,pad,limits,0.,.001)[0])
    check('nonlinear_pad_loss_rejection',not t._contract.validate_secondary(p,q,pad,[0.,0.,2e-5],limits,0.,.001)[0])
    sys.path.insert(0,str(t.BASELINE));from wave_reference import WaveReference
    meta=json.loads((HERE.parent/'left_tarsal_clearance_teacher'/'left_yaw_p060'/'REFERENCE_CLEARANCE_TEACHER.json').read_text())
    base=WaveReference(np.asarray(meta['rest_body_feet']),stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    a=t.WiderRadiusReference(base);b=t._parent.WiderRadiusReference(base)
    same=True
    for elapsed in np.arange(2880)*.005:
        x,y=a.sample(elapsed),b.sample(elapsed);same&=all(np.array_equal(x[k],y[k]) for k in x) and a.command(elapsed)==b.command(elapsed)
    check('reference_class_frozen_identity',t.WiderRadiusReference is t._parent.WiderRadiusReference)
    check('full2880_reference_and_commands_exact',same)
    tree=ast.parse((HERE/'turning_teacher.py').read_text());cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ActualGuardTeacher')
    check('only_ik_control_override',{n.name for n in cls.body if isinstance(n,ast.FunctionDef)}=={'__init__','_ik'})
    ik=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_ik')
    primarycalls=[n for n in ast.walk(ik) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='StaticTeacher._ik']
    secondarycalls=[n for n in ast.walk(ik) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bounded_secondary']
    check('original_primary_bypasses_old_guard',len(primarycalls)==1 and len(secondarycalls)==1 and primarycalls[0].lineno<secondarycalls[0].lineno)
    call=secondarycalls[0]
    check('secondary_uses_feedback_target_not_old004',len(call.args)==4 and ast.unparse(call.args[2])=='target_h')
    writes=False
    for node in ast.walk(ik):
        if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            writes|=any('c.d.qpos' in ast.unparse(z) or 'c.d.qvel' in ast.unparse(z) for z in targets)
    check('no_actual_pose_writes',not writes)
    check('inactive_no_memory_clear_source','self.q_des[:] = 0' not in ast.unparse(ik) and 'self.q_des = ' not in ast.unparse(ik))
    prior=json.loads((HERE.parent/'left_tarsal_clearance_teacher'/'left_yaw_p060'/'RESULT.json').read_text())
    check('old86_evidence_retained',prior['clearance_summary']['actual_unsafe_q_des_safe']==86)
    for pth in HERE.glob('*.py'):ast.parse(pth.read_text(),filename=str(pth))
    check('all_sources_parse',True)
    check('no_Core_model_heavy_imports',not any(name in sys.modules for name in ('physics','mujoco','torch','scipy','static_teacher','student_wave')))
    verify_sources()
    report=dict(status='NUMPY_AST_PASS' if all(checks.values()) else 'NUMPY_AST_FAIL',all_pass=all(checks.values()),checks=checks,
        sourcehash_sha256=sha256(FREEZE),cpu_affinity=list(os.sched_getaffinity(0)),actual_Core_FK_physics=False,
        note='Pure arithmetic/reference/AST only. Teacher actual-feedback/geometry and dynamic nativeNF0 remain unexecuted/unqualified.')
    out=exclusive('PREFLIGHT_ACTUAL_GUARD',report);print(json.dumps(dict(report=str(out),status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
    if not report['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
