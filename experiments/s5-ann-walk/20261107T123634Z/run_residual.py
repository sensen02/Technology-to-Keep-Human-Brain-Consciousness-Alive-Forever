"""Strict full-native execution of two learned networks, no teacher after1s.
The frozen history-based motor ANN and learned proprioceptive residual share the
original bounded incremental actuator. No contact feedback formula is deployed.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import mujoco as mj
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T111149Z/run_velocity.py'
spec=importlib.util.spec_from_file_location('frozen_neural_motor_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
PROPRIO=ROOT/'experiments/s5-ann-walk/20261107T091243Z/proprioception.py'
spec=importlib.util.spec_from_file_location('declared_physical_proprioception',PROPRIO)
pmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(pmod)
OriginalPolicy=n.Policy;OriginalActuator=n.IncrementalActuator
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z')
CONTEXT={};POLICIES=[]


class RuntimeActuator(OriginalActuator):
    def __init__(self,m,d,**kwargs):
        super().__init__(m,d,**kwargs)
        names=[m.geom(i).name or '' for i in range(m.ngeom)]
        legs=('LF','LM','LH','RF','RM','RH')
        p=types.SimpleNamespace(m=m,jids=m.actuator_trnid[:,0].copy(),
            qa=m.jnt_qposadr[m.actuator_trnid[:,0]].copy(),va=m.jnt_dofadr[m.actuator_trnid[:,0]].copy(),
            q0=m.key_qpos[0].copy(),thorax=m.body('Thorax').id)
        p.planes=m.geom_type==mj.mjtGeom.mjGEOM_PLANE
        p.feet=np.array([next((j for j,leg in enumerate(legs) if name.startswith(leg) and 'Tars' in name),-1) for name in names])
        CONTEXT.update(plant=p,data=d,actuator=self)


class ResidualPolicy(OriginalPolicy):
    def __init__(self,path):
        super().__init__(path)
        self.rows=[];self.last_rate=None;POLICIES.append(self)
        w=self.w
        assert w['actor_w0'].shape==(128,477) and w['actor_w1'].shape==(128,128) and w['actor_w2'].shape==(6,128)
        assert np.asarray(w['indices']).shape==(6,) and float(w['delta_scale'])==.003
        assert len(pmod.FEATURE_NAMES)==183

    def observe(self,target):
        super().observe(target)
        p=CONTEXT['plant'];d=CONTEXT['data'];m=p.m
        force=np.zeros(6);loads=np.zeros(6)
        for i in range(d.ncon):
            co=d.contact[i];ga,gb=int(co.geom1),int(co.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga;leg=int(p.feet[geom])
            if leg<0:continue
            mj.mj_contactForce(m,d,i,force)
            normal=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if p.planes[ga] else -1)
            loads[leg]+=max(0.,normal)
        features=pmod.features(p,d,target,loads)
        history=self.buf[self.lags].reshape(-1).copy()
        inputs=np.r_[features,history].astype(np.float32)
        w=self.w
        h=np.tanh(inputs@w['actor_w0'].T+w['actor_b0'])
        h=np.tanh(h@w['actor_w1'].T+w['actor_b1'])
        mean=h@w['actor_w2'].T+w['actor_b2']
        residual=np.tanh(mean)*float(w['delta_scale'])
        base_pred=self.predict_normalized(self.buf[None])[0]
        base_next=w['center']+w['scale']*base_pred
        next_target=np.asarray(base_next,dtype=np.float64).copy()
        next_target[np.asarray(w['indices'],int)]+=residual
        self.last_rate=np.clip((next_target-target)/.005,-30.,30.)
        self.rows.append(dict(time=float(d.time),features=features.copy(),input=inputs.copy(),history=history,
            mean=np.asarray(mean).copy(),residual_target=np.asarray(residual).copy(),
            base_prediction=np.asarray(base_next).copy(),predicted_rate=self.last_rate.copy(),
            actual_q=d.qpos[p.qa].copy(),actual_qvel=d.qvel[p.va].copy(),foot_load=loads))

    def action(self,target):
        # The model has already observed the current state at this policy tick.
        assert self.last_rate is not None
        return self.last_rate.copy()


n.IncrementalActuator=RuntimeActuator;n.Policy=ResidualPolicy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=4.);ap.add_argument('--tag',default='ann_residual_screen')
    ap.add_argument('--control',choices=['ann','hold','motor_cut','teacher'],default='ann')
    ap.add_argument('--disturb',action='store_true');args=ap.parse_args()
    n.OUT=OUT/'evaluation';n.OUT.mkdir(parents=True,exist_ok=True)
    protocol=n.OUT/(args.tag+'_PROTOCOL.json')
    if protocol.exists():raise FileExistsError('preserve prior residual evaluation')
    n.dump(protocol,dict(importance='核心必需',stage='S5 two-ANN learned physicalfeedback strict native evaluation',
        model=str(args.model),model_sha256=n.sha(args.model),
        base='Frozen111149velocity-losslearnedmotorANN, real40framehistory7lags. LearnedresidualANN477->128tanh->128tanh->6 givesdeltaq<=.003rad sixoriginaljointaxes.',
        inputs='183declaredcurrentphysicalproprioception THEN294causalown-targethistory. No time/phase/worldposition/rootheight/yawreference/anchors/gaitlookup/teacherstate supplied.',
        deployment='First1scontinuousalgorithmphysicalwarmupwithgenuineANNhistory, after1s learnedbase+learnedresidual only. No PDcontactformula/staticforceplan/NoSlipchange.',
        motor='Original42rate bounded30rad/s integrated1ms, finitePD128 andoriginalnativeforcecaps,12.5usstep/5mspolicy.',
        body='Full71mesh49/48/42neq0/free root/originalgravity/contact/friction/masses/inertia andnoslip0 unchanged; qpos/qvelwritesonlyinitialization.',
        neural_cut='hold ormotorcut usesoriginalkernel; networkforwardmaybeloggedbutoutputnotapplied after1s. Doesnotclaimoriginalconnectomeablation.',
        controller=args.control,seconds=args.seconds,disturb=args.disturb,
        features=pmod.contract(),assessment='All originalD/actualsixlegsteps/heading/perlegmaterialslip<=.5/independentphysicsreplay andindependentlyreconstructedBOTHANNs; trainingrewardneveracceptance.',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,PROPRIO,Path(n.__file__),n.GAIT,args.model]},
        units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED'))
    result=None
    try:result=n.simulate(args.model,args.tag,args.seconds,args.control,args.disturb)
    finally:
        rows=[row for policy in POLICIES for row in policy.rows]
        if rows:
            folder=n.OUT/args.tag
            trace=folder/'RESIDUAL_POLICY.npz'
            np.savez_compressed(trace,**{key:np.asarray([row[key] for row in rows]) for key in rows[0]})
            if result is not None:
                result.update(neural_architecture='frozenlearnedhistorymotorANN +learnedproprioceptiveresidualANN',
                    residual_trace_sha256=n.sha(trace),proprioception_contract=pmod.contract())
                n.dump(folder/'RESULT.json',result)


if __name__=='__main__':main()
