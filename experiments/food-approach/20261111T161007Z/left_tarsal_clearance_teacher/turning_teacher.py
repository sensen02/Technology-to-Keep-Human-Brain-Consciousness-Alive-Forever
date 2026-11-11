"""One q_des-shadow clearance teacher. Import stays pure NumPy; no Core/FK.

Only _ik changes control behavior; StaticTeacher.action/allocation/finite PD are
inherited. Frozen references and static helper are loaded by absolute path.
"""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
from ik_contract import validate_secondary,whole_decision_velocity

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

_wider=load('_clearance_frozen_wider_reference',HERE.parent/'left_wider_radius'/'turning_teacher.py')
_geometry=load('_clearance_frozen_static_geometry',HERE.parent/'left_tarsal_clearance_probe'/'probe_geometry.py')
BASELINE=_wider.BASELINE
pin_baseline=_wider.pin_baseline
WiderRadiusReference=_wider.WiderRadiusReference
CONFIG=dict(per_decision_trust_rad=.005,decision_s=.005,pad_error_norm_max=1e-5,clearance_improvement_min=1e-8,
            local_updates_max=4,backtracks_max=8,primary_reprojection_iterations_max=5)


def teacher_class():
    pin_baseline()
    from static_teacher import StaticTeacher
    class ClearanceStaticTeacher(StaticTeacher):
        def __init__(self,c,config,diagnostic_path):
            super().__init__(c,config)
            self._clear_engine=_geometry.StaticGeometry(c,self.initial_qpos)
            self._start_native_time=float(c.d.time)
            self._rh=np.array(self.leg_act[4],dtype=int)
            if not np.array_equal(self._rh,self._clear_engine.rh):raise RuntimeError('RH mapping mismatch')
            if self.config['command_dt_s']!=.005:raise RuntimeError('Original decision clock changed')
            self._sink=Path(diagnostic_path).open('x',encoding='utf8')
            self._records=0
            self.metadata['RH_secondary']=dict(source='Frozen StaticGeometry.bounded_secondary',safe_height=.004,config=CONFIG,
                pose='Actual q_des plus only current actualroot in independent private helper; never motor-target',
                prime_inactive_through_decision=960,trigger='q_des_actualroot_height<.004 only',
                actual_height='read-only diagnostics, not a separate controller trigger')

        def _shadow_height_in_actual_root(self,c):
            e=self._clear_engine
            v=e.mesh_cache[e.geom]@self.shadow.geom_xmat[e.geom].reshape(3,3).T+self.shadow.geom_xpos[e.geom]
            R=c.d.xmat[c.th].reshape(3,3)@self.initial_rot.T
            actual=(v-self.initial_origin)@R.T+c.d.xpos[c.th]
            return float(np.min((actual-e.plane_p)@e.plane_n))

        def _actual_height(self,c):
            e=self._clear_engine
            v=e.mesh_cache[e.geom]@c.d.geom_xmat[e.geom].reshape(3,3).T+c.d.geom_xpos[e.geom]
            return float(np.min((v-e.plane_p)@e.plane_n))

        def _ik(self,c,desired):
            previous=self.q_des[self.qa].copy()
            achieved,_,_=super()._ik(c,desired)
            primary=self.q_des.copy();primary_pad=achieved[4].copy()
            decision=int(round((float(c.d.time)-self._start_native_time)/.005))
            elapsed=decision*.005
            actual_h=self._actual_height(c);before_h=self._shadow_height_in_actual_root(c)
            after_h=before_h;trigger=decision>960 and before_h<.004;accepted=False
            status='PRIME_INACTIVE' if decision<=960 else 'Q_DES_ALREADY_SAFE'
            outcome=None;helper_pad_offset=None
            if trigger:
                pose=primary.copy();pose[:7]=c.d.qpos[:7]
                b=self._clear_engine.pose(pose)
                helper_pad_offset=float(np.linalg.norm(b['pad']-primary_pad))
                if helper_pad_offset>1e-8 or abs(b['h']-before_h)>1e-8:
                    status='HELPER_ORIGINAL_SHADOW_FRAME_MISMATCH'
                else:
                    # Exactly one call per full decision; helper total trust .005,
                    # up to4 local updates inside this same anchored trust region.
                    outcome=self._clear_engine.bounded_secondary(primary[self.qa[self._rh]].copy(),b,.004,CONFIG)
                    outcome['actual_teacher_rate_feasibility']='Actual q_des chain raw/clipped qdot recorded separately; dynamic tracking not established'
                    if outcome.get('local_improvement',False):
                        candidate=np.asarray(outcome['q_final'],dtype=float)
                        self.q_des[self.qa[self._rh]]=candidate
                        self._fk()
                        new_pad=self._leg_kinematics(c,4,False)
                        after_h=self._shadow_height_in_actual_root(c)
                        accepted,status=validate_secondary(primary[self.qa[self._rh]],candidate,primary_pad,new_pad,
                            self.bounds[self._rh],before_h,after_h)
                        if not accepted:
                            self.q_des[:]=primary;self._fk();after_h=before_h
                    else:status=outcome['status']
            if not np.array_equal(self.q_des[:7],primary[:7]):raise RuntimeError('Original private q_des root changed')
            other=np.ones(42,dtype=bool);other[self._rh]=False
            if not np.array_equal(self.q_des[self.qa][other],primary[self.qa][other]):raise RuntimeError('Secondary changed non-RH q_des')
            achieved=np.asarray([self._leg_kinematics(c,li,False) for li in range(6)])
            clipped,raw=whole_decision_velocity(previous,self.q_des[self.qa],self.config['command_dt_s'],2.)
            rh_delta=self.q_des[self.qa[self._rh]]-primary[self.qa[self._rh]]
            record=dict(decision=decision,elapsed_s=elapsed,native_time_s=float(c.d.time),
                geometry_pose_kind='Actual teacher q_des with copied current actual root; never motor-target',
                previous_q_des=previous.tolist(),primary_q_des=primary[self.qa].tolist(),final_q_des=self.q_des[self.qa].tolist(),
                qdot_raw=raw.tolist(),qdot_clipped=clipped.tolist(),actual_current_height=actual_h,q_des_height_before=before_h,q_des_height_after=after_h,
                trigger=trigger,accepted=accepted,status=status,helper_pad_offset=helper_pad_offset,
                primary_RH_pad=primary_pad.tolist(),final_RH_pad=achieved[4].tolist(),pad_error_norm=float(np.linalg.norm(achieved[4]-primary_pad)),
                RH_delta=rh_delta.tolist(),max_abs_secondary_delta=float(np.max(np.abs(rh_delta))),RH_bounds=self.bounds[self._rh].tolist(),
                actual_unsafe_q_des_safe=bool(decision>960 and actual_h<.004 and before_h>=.004),
                root_unchanged=True,other_q_des_unchanged=True,helper_outcome=outcome)
            self._sink.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');self._records+=1
            if self._records%160==0:self._sink.flush()
            return achieved,clipped,raw

        def close_diagnostics(self):
            if not self._sink.closed:self._sink.flush();self._sink.close()
    return ClearanceStaticTeacher


def make_teacher(core,*,speed_scale=1.,yaw_rate=.06,prime_s=4.8,diagnostic_path=None):
    if diagnostic_path is None:raise ValueError('Exclusive new diagnostic_path required')
    pin_baseline()
    from student_wave import teacher_config
    teacher=teacher_class()(core,teacher_config(100),diagnostic_path)
    teacher.reference=WiderRadiusReference(teacher.reference,speed_scale,yaw_rate,prime_s)
    teacher.metadata['conditional_twist']=teacher.reference.metadata
    return teacher
