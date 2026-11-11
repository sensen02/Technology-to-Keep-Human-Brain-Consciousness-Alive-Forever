"""Pure NumPy contracts/reference and AST checks; no Core, model imports or FK."""
import ast
import json
import os
import sys
import numpy as np
from candidate_identity import HERE,FREEZE,verify_sources,sha256,exclusive
from ik_contract import validate_secondary,whole_decision_velocity


def main():
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    verify_sources();checks={}
    def check(k,v):checks[k]=bool(v)
    primary=np.linspace(-.4,.4,7);candidate=primary+.004*np.array([1,-1,1,-1,1,-1,1]);pad=np.array([.1,.2,-.003]);limits=np.tile([-2.,2.],(7,1))
    check('partial_gain_accepted_without_reaching_safeheight',validate_secondary(primary,candidate,pad,pad,limits,-.0007,.0001)==(True,'ACCEPTED'))
    bad=candidate.copy();bad[0]=primary[0]+.005001
    check('over_decision_trust_rejected',validate_secondary(primary,bad,pad,pad,limits,0.,.001)[1]=='DECISION_TRUST')
    check('pad_error_rejected',validate_secondary(primary,candidate,pad,pad+[0.,0.,2e-5],limits,0.,.001)[1]=='NONLINEAR_PAD_ERROR')
    check('pad_within_tolerance_accepted',validate_secondary(primary,candidate,pad,pad+[0.,0.,5e-6],limits,0.,.001)[0])
    bound=limits.copy();bound[0,1]=primary[0]+.002
    check('joint_limit_rejected_not_clipped',validate_secondary(primary,candidate,pad,pad,bound,0.,.001)[1]=='JOINT_BOUNDS')
    bad=candidate.copy();bad[2]=np.nan
    check('nonfinite_rejected',validate_secondary(primary,bad,pad,pad,limits,0.,.001)[1]=='NONFINITE')
    check('no_gain_rejected',validate_secondary(primary,candidate,pad,pad,limits,0.,0.)[1]=='NO_MEANINGFUL_CLEARANCE_GAIN')
    previous=np.linspace(-.5,.5,42);p=previous+np.linspace(-.012,.012,42);final=p.copy();final[28:35]+=.004
    clipped,raw=whole_decision_velocity(previous,final)
    check('raw_qdot_includes_primary_and_secondary',np.array_equal(raw,(final-previous)/.005) and not np.array_equal(raw,(final-p)/.005))
    check('original_clip2_exact',np.array_equal(clipped,np.clip((final-previous)/.005,-2,2)) and np.max(np.abs(clipped))==2.)
    rng=np.random.default_rng(20261111);J=rng.normal(size=(3,7));g=rng.normal(size=7)
    U,s,V=np.linalg.svd(J,full_matrices=True);Z=V[3:].T;direction=Z@(g@Z);direction*=.005/np.max(np.abs(direction))
    check('synthetic_nullspace_direction_preserves_linear_primary',np.linalg.norm(J@direction)<1e-14 and np.max(np.abs(direction))<=.005+1e-16 and g@direction>0.)
    import turning_teacher as t
    sys.path.insert(0,str(t.BASELINE));from wave_reference import WaveReference
    meta=json.loads((HERE.parent/'left_wider_radius'/'left_yaw_p060'/'REFERENCE_WIDER_RADIUS.json').read_text())
    base=WaveReference(np.asarray(meta['rest_body_feet']),stride_native=.36,lift_native=.12,step_s=.8,cycles=100)
    candidate_reference=t.WiderRadiusReference(base);original=t._wider.WiderRadiusReference(base)
    check('reference_class_is_frozen_original',t.WiderRadiusReference is t._wider.WiderRadiusReference)
    same=True
    for elapsed in np.arange(2880)*.005:
        a=candidate_reference.sample(elapsed);b=original.sample(elapsed)
        same&=all(np.array_equal(a[k],b[k]) for k in a) and candidate_reference.command(elapsed)==original.command(elapsed)
    check('full2880_reference_commands_exact',same)
    tree=ast.parse((HERE/'turning_teacher.py').read_text())
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ClearanceStaticTeacher')
    methods={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
    check('only_ik_control_override',methods=={'__init__','_shadow_height_in_actual_root','_actual_height','_ik','close_diagnostics'} and 'action' not in methods and '_allocation' not in methods)
    ik=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_ik')
    secondary_calls=[n for n in ast.walk(ik) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bounded_secondary']
    check('one_secondary_callsite_per_decision',len(secondary_calls)==1)
    primary_calls=[n for n in ast.walk(ik) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_ik']
    check('primary_before_secondary_source_order',len(primary_calls)==1 and primary_calls[0].lineno<secondary_calls[0].lineno)
    writes_actual=False
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            if any('c.d.qpos' in ast.unparse(z) or 'c.d.qvel' in ast.unparse(z) for z in targets):writes_actual=True
    check('no_actual_qpos_qvel_assignments',not writes_actual)
    check('helper_limits_exact',t.CONFIG['per_decision_trust_rad']==.005 and t.CONFIG['local_updates_max']==4 and t.CONFIG['pad_error_norm_max']==1e-5)
    check('prime_activation_source',any(isinstance(n,ast.Compare) and ast.unparse(n)=='decision > 960' for n in ast.walk(ik)))
    files=list(HERE.glob('*.py'))
    for file in files:ast.parse(file.read_text(),filename=str(file))
    check('all_python_sources_parse',True)
    check('no_heavy_model_imports',not any(name in sys.modules for name in ('physics','mujoco','torch','scipy','static_teacher','student_wave')))
    verify_sources()
    report=dict(status='NUMPY_AST_PREFLIGHT_PASS' if all(checks.values()) else 'NUMPY_AST_PREFLIGHT_FAIL',all_pass=all(checks.values()),checks=checks,
        sourcehash_sha256=sha256(FREEZE),python=sys.executable,cpu_affinity=list(os.sched_getaffinity(0)),actual_Core_FK_physics=False,
        limitations='Synthetic algebra/contracts and unchanged reference only. Actual class/helper/model geometry and dynamic tracking are not executed or qualified here.')
    out=exclusive('PREFLIGHT_CLEARANCE_TEACHER',report)
    print(json.dumps(dict(report=str(out),status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
    if not report['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
