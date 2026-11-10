"""One-variable contact-damping diagnostic: exact20ms low-pass velocity filter.

Preserves frozen122002 K5, contacts, nominalshape, PD128/nativebounds and ramp.
This powered joint controller is not added physical friction or guaranteed passive.
"""
import hashlib
import json
from pathlib import Path
import types

SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T122002Z/contact_damping.py')
SOURCE_SHA256='c9e74385520b5023f578301798b2b39e75d01fce6bddecab54b954b521e37083'
HERE=Path(__file__).resolve()
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T122432Z')
TAU=.02
source=SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256,'Frozen damping source changed'
patches=[
    ("20261107T122002Z')\nb.OUT=OUT","20261107T122432Z')\nb.OUT=OUT"),
    ('class DampingTeacher(f.SupportTeacher):\n    def desired(self,d,t):',
     """class DampingTeacher(f.SupportTeacher):
    def __init__(self,p,ref):
        super().__init__(p,ref)
        self.filtered_velocity=np.zeros((6,3))
        self.filter_active=np.zeros(6,dtype=bool)
        self.filter_last_time=None

    def desired(self,d,t):"""),
    ("        raw=-DAMPING*np.einsum('lkj,lk->j',J[:,:2,self.p.va],vel[:,:2])",
     """        dt=.005 if self.filter_last_time is None else float(t-self.filter_last_time)
        if not np.isfinite(dt) or dt<0:raise ValueError('Nonmonotonic filter time')
        beta=float(-np.expm1(-dt/.02))
        newly_active=active & ~self.filter_active
        self.filtered_velocity[~active | newly_active]=0.
        self.filtered_velocity[active]+=beta*(vel[active]-self.filtered_velocity[active])
        self.filter_active=active.copy();self.filter_last_time=float(t)
        filtered=self.filtered_velocity.copy()
        raw=-DAMPING*np.einsum('lkj,lk->j',J[:,:2,self.p.va],filtered[:,:2])"""),
    ('            contact_velocity=vel,contact_J=J,velocity_crosscheck=check,',
     """            contact_velocity=vel,filtered_contact_velocity=filtered,filter_state=filtered.copy(),
            filter_dt=dt,filter_beta=beta,newly_active_contact=newly_active,filter_active=active.copy(),
            joint_damping_power_raw=float(raw@d.qvel[self.p.va]),
            full_virtual_damping_power=float(-DAMPING*np.sum(filtered[:,:2]*vel[:,:2])),
            contact_J=J,velocity_crosscheck=check,"""),
    ("default='contact_damping_screen'","default='filtered_damping_screen'"),
    ("trace=folder/'CONTACT_DAMPING.npz'","trace=folder/'FILTERED_DAMPING.npz'"),
]
effective=source
for old,new in patches:
    assert effective.count(old)==1,('Expected one frozen source match',old)
    effective=effective.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(effective.encode()).hexdigest()
runtime=types.ModuleType('filtered_contact_damping_runtime');runtime.__file__=str(SOURCE)
exec(compile(effective,str(HERE)+'::effective','exec'),runtime.__dict__)
original_prereg=runtime.b.prereg


def prereg(folder,args):
    artifact=folder/'EFFECTIVE_DAMPING.py'
    if artifact.exists():raise FileExistsError(artifact)
    artifact.write_text(effective)
    original_prereg(folder,args)
    path=folder/'PREREGISTRATION.json';value=json.loads(path.read_text())
    value.update(stage='S5 contact damping measuredvelocity20msfilter singlemechanismtest',
        velocity_filter=dict(time_constant_s=TAU,update='beta=1-exp(-(t-last_t)/.02),exactexpm1;firstdt.005s',
            activation='Everynewloadednominalstancecontactstartsfilterstate0 thenfirstexponentialupdate. Inactive/unloaded/swingstateexactlyreset0, no carriedswingvelocity.',
            data='ActualFnweightedworldcontactcentroidvelocity3D, sameactualJ/load/contactmath. OnlyfilteredXYentersunchangedK5torquebias.',
            diagnostics='FILTERED_DAMPING.npz keepsmeasuredcontact_velocity,filtered_contact_velocity,filterstate,beta/dt/newactive/activeplusalloriginalJ/load/torque/target/actualstatefields.'),
        unchanged='K5,originalnativecorrectionbound.0015625rad,fullnominal101114+.02RHyaw,PD128,alpha1s+.1sramp,actualcontacts/J/friction/solref/solimp/cone/Newton/noslip0/free-root; no physicsstatewrites.',
        power_caveat='Unfilteredfull-generalizedJ.T(-K*v)hasnonpositivevirtualpower, butfree-rootDOFsareunactuatedsoappliedjointpowerneednotbenegative. Filterlagcanadditionallygivepositiveinstantaneousvirtualpowerduringvelocityreversal. Poweredinternaljointfeedback, notphysicalfrictionincrease orpassivityclaim.',
        adaptation=dict(baseline=str(SOURCE),baseline_sha256=SOURCE_SHA256,wrapper=str(HERE),wrapper_sha256=runtime.n.sha(HERE),
            effective_source=str(artifact),effective_sha256=EFFECTIVE_SHA256,
            patches=[dict(old=old,new=new,required_matches=1) for old,new in patches]),
        assessment='OriginalfullnativeD/realsteps/heading/materialslipcriteria unchanged; priorunfiltereddampingD5failure retained. No improvementbeforeparentphysicaltest.')
    value['parameters']['velocity_filter_tau_s']=TAU
    value['additional_sources'][str(HERE)]=runtime.n.sha(HERE)
    runtime.n.dump(path,value)


runtime.b.prereg=prereg


if __name__=='__main__':runtime.main()
