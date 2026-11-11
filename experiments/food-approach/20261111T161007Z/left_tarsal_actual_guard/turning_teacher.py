"""Single actual-height guard candidate; imports stay pure NumPy, no Core/FK."""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
from guard_contract import feedback,support_flags

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'left_tarsal_clearance_teacher'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

if 'ik_contract' in sys.modules and Path(sys.modules['ik_contract'].__file__).resolve()!=OLD/'ik_contract.py':
    raise RuntimeError('Conflicting frozen IK contract import')
_contract=load('ik_contract',OLD/'ik_contract.py')
_parent=load('_actual_guard_prior_teacher',OLD/'turning_teacher.py')
BASELINE=_parent.BASELINE
pin_baseline=_parent.pin_baseline
WiderRadiusReference=_parent.WiderRadiusReference
CONFIG=dict(_parent.CONFIG)


def teacher_class():
    Parent=_parent.teacher_class()
    from static_teacher import StaticTeacher
    from physics import W
    class ActualGuardTeacher(Parent):
        def __init__(self,c,config,diagnostic_path):
            super().__init__(c,config,diagnostic_path)
            self.metadata['RH_secondary']=dict(source='Frozen StaticGeometry.bounded_secondary',actual_guard=.012,
                observed_pair_margin=.002,tracking_reserve=.010,config=CONFIG,prime_inactive_through_decision=960,
                trigger='decision>960 and actual_current_height<.012',target='before_h+max(0,.012-actual_h)',
                memory='No reset when inactive; original primary IK uses retained q_des',
                pose='Actual q_des with current actualroot in independent helper, never motor-target',
                deployment='Teacher-only actual-clearance feedback, no extra deployed ANN channel')

        def _ik(self,c,desired):
            previous=self.q_des[self.qa].copy()
            # Explicit original primary: bypass prior q_des-only secondary override.
            achieved,_,_=StaticTeacher._ik(self,c,desired)
            primary=self.q_des.copy();primary_pad=achieved[4].copy()
            decision=int(round((float(c.d.time)-self._start_native_time)/.005));elapsed=decision*.005
            actual_h=self._actual_height(c);before_h=self._shadow_height_in_actual_root(c)
            command=feedback(decision,actual_h,before_h);trigger=command['trigger'];target_h=command['desired_shadow_height']
            measured=c.measure()
            actual_support=support_flags(measured['loads'][4]/W,measured['padz'][4],measured['loads'].sum()/W,measured['com'])
            after_h=before_h;accepted=False;outcome=None;helper_pad_offset=None
            status='PRIME_INACTIVE' if decision<=960 else 'ACTUAL_ABOVE_GUARD'
            if trigger:
                pose=primary.copy();pose[:7]=c.d.qpos[:7]
                b=self._clear_engine.pose(pose);helper_pad_offset=float(np.linalg.norm(b['pad']-primary_pad))
                if helper_pad_offset>1e-8 or abs(b['h']-before_h)>1e-8:
                    status='HELPER_ORIGINAL_SHADOW_FRAME_MISMATCH'
                else:
                    # Exactly one anchored total-trust .005 call, max4 local updates.
                    outcome=self._clear_engine.bounded_secondary(primary[self.qa[self._rh]].copy(),b,target_h,CONFIG)
                    outcome['actual_teacher_rate_feasibility']='Whole q_des raw/clipped chain saved; actual feedback does not establish dynamic tracking'
                    if outcome.get('local_improvement',False):
                        candidate=np.asarray(outcome['q_final'],dtype=float)
                        self.q_des[self.qa[self._rh]]=candidate;self._fk()
                        new_pad=self._leg_kinematics(c,4,False);after_h=self._shadow_height_in_actual_root(c)
                        accepted,status=_contract.validate_secondary(primary[self.qa[self._rh]],candidate,primary_pad,new_pad,self.bounds[self._rh],before_h,after_h)
                        if not accepted:self.q_des[:]=primary;self._fk();after_h=before_h
                    else:status=outcome['status']
            if not np.array_equal(self.q_des[:7],primary[:7]):raise RuntimeError('Original private root changed')
            other=np.ones(42,dtype=bool);other[self._rh]=False
            if not np.array_equal(self.q_des[self.qa][other],primary[self.qa][other]):raise RuntimeError('Secondary changed non-RH q_des')
            achieved=np.asarray([self._leg_kinematics(c,li,False) for li in range(6)])
            clipped,raw=_contract.whole_decision_velocity(previous,self.q_des[self.qa],self.config['command_dt_s'],2.)
            delta=self.q_des[self.qa[self._rh]]-primary[self.qa[self._rh]]
            record=dict(decision=decision,elapsed_s=elapsed,native_time_s=float(c.d.time),geometry_pose_kind='Actual teacher q_des with actualroot, never motor-target',
                previous_q_des=previous.tolist(),primary_q_des=primary[self.qa].tolist(),final_q_des=self.q_des[self.qa].tolist(),qdot_raw=raw.tolist(),qdot_clipped=clipped.tolist(),
                actual_guard=.012,actual_current_height=actual_h,actual_deficit=command['actual_deficit'],applied_deficit=command['applied_deficit'],desired_shadow_height=target_h,
                q_des_height_before=before_h,q_des_height_after=after_h,trigger=trigger,accepted=accepted,status=status,helper_pad_offset=helper_pad_offset,
                primary_RH_pad=primary_pad.tolist(),final_RH_pad=achieved[4].tolist(),pad_error_norm=float(np.linalg.norm(achieved[4]-primary_pad)),
                RH_delta=delta.tolist(),max_abs_secondary_delta=float(np.max(np.abs(delta))),RH_bounds=self.bounds[self._rh].tolist(),
                actual_unsafe_q_des_safe=bool(decision>960 and actual_h<.004 and before_h>=.004),
                actual_below_guard_shadow_above_guard=bool(decision>960 and actual_h<.012 and before_h>=.012),
                root_unchanged=True,other_q_des_unchanged=True,helper_outcome=outcome,**actual_support)
            self._sink.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');self._records+=1
            if self._records%160==0:self._sink.flush()
            return achieved,clipped,raw
    return ActualGuardTeacher


def make_teacher(core,*,speed_scale=1.,yaw_rate=.06,prime_s=4.8,diagnostic_path=None):
    if diagnostic_path is None:raise ValueError('Exclusive new diagnostic path required')
    pin_baseline();from student_wave import teacher_config
    teacher=teacher_class()(core,teacher_config(100),diagnostic_path)
    teacher.reference=WiderRadiusReference(teacher.reference,speed_scale,yaw_rate,prime_s)
    teacher.metadata['conditional_twist']=teacher.reference.metadata
    return teacher
