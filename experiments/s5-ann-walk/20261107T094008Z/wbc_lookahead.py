"""One bounded QP-to-incremental-PD clock correction, no new body/force authority.

The frozen-state inverse torque target is only exact at the current state; actual
joint state moves during a5ms hold. Use the solved acceleration to predict the
end-of-hold joint position/velocity before converting optimal torque to a target.
This is a declared teacher-interface experiment, not a neural deployment claim.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T092817Z/wbc_teacher.py')
spec=importlib.util.spec_from_file_location('frozen_qp_teacher',SOURCE)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T094008Z')
w.OUT=OUT;w.b.OUT=OUT
Original=w.WholeBodyTeacher


class LookaheadTeacher(Original):
    def desired(self,d,t):
        _,meta=super().desired(d,t)
        row=self.rows[-1]
        actuator=w.ACTUATORS[-1]
        q=d.qpos[self.p.qa];v=d.qvel[self.p.va];acc=row['qacc'][self.p.va]
        horizon=w.b.PDT
        predicted_q=q+horizon*v+.5*horizon*horizon*acc
        predicted_v=v+horizon*acc
        target=predicted_q+(row['tau_optimal']+actuator.kd*predicted_v)/actuator.kp
        nominal=self.ref.warm(t)
        alpha=row['blend']
        desired=nominal if row['fallback'] else (1-alpha)*nominal+alpha*target
        desired=np.clip(desired,self.ref.lo,self.ref.hi)
        assert np.isfinite(desired).all()
        rate=np.clip((desired-actuator.target_rad)/horizon,-w.b.CAP,w.b.CAP)
        reachable=np.clip(actuator.target_rad+horizon*rate,self.ref.lo,self.ref.hi)
        old_target=row['torque_target'].copy()
        frozen_prediction=np.clip(actuator.kp*(reachable-q)-actuator.kd*v,-actuator.torque_limit,actuator.torque_limit)
        end_prediction=np.clip(actuator.kp*(reachable-predicted_q)-actuator.kd*predicted_v,-actuator.torque_limit,actuator.torque_limit)
        row.update(frozen_state_inverse_target=old_target,predicted_endpoint_q=predicted_q.copy(),
            predicted_endpoint_v=predicted_v.copy(),endpoint_prediction_tau=end_prediction,
            endpoint_prediction_minus_optimal=end_prediction-row['tau_optimal'],
            torque_target=target.copy(),commanded_target=desired.copy(),target_rate=rate,
            tau_predicted_reachable=frozen_prediction,
            tau_reachable_minus_optimal=frozen_prediction-row['tau_optimal'],
            torque_conversion='solved_qacc_5ms_endpoint_prediction')
        return desired,meta


w.b.AnchorTeacher=LookaheadTeacher
_original_prereg=w.b.prereg


def prereg(folder,args):
    _original_prereg(folder,args)
    path=folder/'PREREGISTRATION.json'
    p=json.loads(path.read_text())
    p.update(stage='S5 matched QP-to-incremental-servo endpoint prediction correction',
        actual_before_run_evidence='Original QP at1.1s solved and frozenreachabletauerror0 but actual5msendpointtau errorRMS1.436 vsoptimalRMS1.69, whilebodydropped; numericalsolvealone notphysicalrealization.',
        change='Only torque-to-target conversion: predictq_end=q+.005v+.5*.005²qacc; v_end=v+.005qacc; target=q_end+(tau_opt+kd*v_end)/kp, with SAME.1snominalblend/jointbounds/30rad/s/nativecaps.',
        original_tasks_unchanged=True,original_model_unchanged=True,
        no_success_claim='Continuous algorithmteacher only, fixed4s comparison; originalD/materialslip/stepping/heading gates unchanged. Allfailingphases retained.',
        diagnostic_caution='Prediction assumes solvedqacc remainsvalidthroughhold, which contacts mayinvalidate. Frozenstate reachabletau and predictedendpointtau savedseparately, actualBODYendpoint measuredseparately.',
        new_source_hashes={str(q):w.n.sha(q) for q in [Path(__file__),SOURCE]})
    w.n.dump(path,p)


w.b.prereg=prereg

if __name__=='__main__':w.main()
