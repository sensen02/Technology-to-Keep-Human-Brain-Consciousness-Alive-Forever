"""Training-only nominal-preserving contact-velocity residual teacher.

Retains the accepted nominal wave target and servo support. A bounded learned-
teacher residual integrates tangential contact velocity correction during loaded
stance and decays in nominal swing. No physical state, root trajectory or contact
parameter is ever written by the teacher. This is algorithm assistance, not ANN.
"""
import importlib.util
from pathlib import Path
import numpy as np
import mujoco as mj
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T091243Z/anchor_teacher.py')
spec=importlib.util.spec_from_file_location('nominal_anchor_screen_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T092322Z')
b.OUT=OUT
PARAMETERS=dict(tangential_velocity_gain=1.,residual_joint_bound_rad=.05,
                residual_rate_cap_rad_s=5.,swing_decay_s=.03,
                contact_threshold_weight=.005,jacobian_damping=.02,
                physical_policy_dt=.005)
BaseTeacher=b.AnchorTeacher


class ResidualTeacher(BaseTeacher):
    def __init__(self,p,ref):
        super().__init__(p,ref)
        self.delta=np.zeros(42)
        self.previous_stance=np.zeros(6,bool)
        self.ready=False

    def capture(self,d,t):
        super().capture(d,t)
        self.delta[:]=0.

    def desired(self,d,t):
        stance=(self.phase(t)+self.offset)%1<5/6
        entries=stance & ~self.previous_stance
        self.anchor_pos[entries]=d.xpos[self.p.bids][entries]
        self.anchor_rot[entries]=d.xmat[self.p.bids].reshape(6,3,3)[entries]
        self.previous_stance=stance.copy()
        load=np.zeros(6); weighted_J=np.zeros((6,2,42));weighted_v=np.zeros((6,2))
        forces=np.zeros(6);jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        for i in range(d.ncon):
            co=d.contact[i];ga,gb=int(co.geom1),int(co.geom2)
            if not (self.p.planes[ga] or self.p.planes[gb]):continue
            geom=gb if self.p.planes[ga] else ga
            leg=int(self.p.feet[geom])
            if leg<0:continue
            mj.mj_contactForce(self.m,d,i,forces)
            normal=float((co.frame.reshape(3,3).T@forces[:3])[2])*(1 if self.p.planes[ga] else -1)
            if normal<=0:continue
            bid=int(self.m.geom_bodyid[geom])
            mj.mj_jac(self.m,d,jp,jr,co.pos,bid)
            load[leg]+=normal
            weighted_J[leg]+=normal*jp[:2,self.p.va]
            weighted_v[leg]+=normal*(jp@d.qvel)[:2]
        velocity=np.zeros((6,2));rate=np.zeros(42);residual=np.zeros((6,2))
        for li,ix in enumerate(self.indices):
            if stance[li] and load[li]>.005*n.dsm.WEIGHT:
                J=weighted_J[li,:,ix].T/load[li]
                # Numpy advanced indexing places the indexed joint axis first.
                assert J.shape==(2,7),J.shape
                velocity[li]=weighted_v[li]/load[li]
                # Protect nominal support: only a small joint-space correction,
                # no absolute joint replacement or body/root pose instruction.
                step=J.T@np.linalg.solve(J@J.T+.02**2*np.eye(2),-velocity[li])
                rate[ix]=np.clip(step,-5.,5.)
            elif not stance[li]:
                rate[ix]=-self.delta[ix]/.03
            self.delta[ix]=np.clip(self.delta[ix]+.005*rate[ix],-.05,.05)
            residual[li]=[np.linalg.norm(velocity[li]),np.linalg.norm(self.delta[ix])]
        desired=np.clip(self.ref.warm(t)+self.delta,self.ref.lo,self.ref.hi)
        return desired,dict(stance=stance,stance_entry=entries,virtual_root=d.qpos[:7].copy(),
            reference_body_position=self.reference_position.copy(),anchor_position=self.anchor_pos.copy(),
            anchor_rotation=self.anchor_rot.copy(),foot_goal_position=d.xpos[self.p.bids].copy(),
            foot_goal_rotation=d.xmat[self.p.bids].reshape(6,3,3).copy(),ik_residual=residual,
            body_feedback_velocity=np.zeros(2),contact_material_velocity=velocity,
            contact_load=load,residual_joint_correction=self.delta.copy())


# Same metadata keys/shapes before warmup and afterwards are required by runtime.
# Extra residual diagnostics are emitted through a separate sampler below.
_original_desired=ResidualTeacher.desired

def compatible_desired(self,d,t):
    desired,meta=_original_desired(self,d,t)
    self.latest_residual={key:meta.pop(key) for key in ['contact_material_velocity','contact_load','residual_joint_correction']}
    return desired,meta

ResidualTeacher.desired=compatible_desired
b.AnchorTeacher=ResidualTeacher
original_prereg=b.prereg


def prereg(folder,args):
    original_prereg(folder,args)
    path=folder/'PREREGISTRATION.json'
    protocol=__import__('json').loads(path.read_text())
    protocol.update(stage='S5 bounded nominal-preserving contact-velocity residual teacher',
        algorithm='Accepted nominal target plus residual jointcorrection. Loadednominalstance integrates negative force-weighted actualtangential material velocity through localJpinv; nominal swing decays correction.',
        parameters=PARAMETERS,
        previous_anchor_algorithm='not used: no fullposeIK and no virtualrootforwardtarget',
        diagnostic_field='TEACHER.ik_residual stores [tangential material speed, jointresidualnorm] perleg for thisarm, not IK errors',
        causal_question='Can bounded actualcontact velocity feedback reduce materialslip while retaining nominal support?',
        additional_sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE]},
        limitations='Teacher-only feedback is non-neural; deploy requires trainedreplacementANN afterteacherwithdrawal. Boundedsmallresidual can saturate and failure is retained.')
    n.dump(path,protocol)

b.prereg=prereg

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=4.)
    ap.add_argument('--tag',default='residual_screen')
    ap.add_argument('--feedback-off',action='store_true')
    args=ap.parse_args();b.simulate(args)
