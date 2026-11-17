"""One bounded local-teacher candidate: swing-only terminal-foot touch feedback."""
import types
from context import *
import reach
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation

config=dict(change='Append tarsus5 obstacle sensing during swing only; preserve original three-link stumbling and all original gains.',
     scenes=['flat','block015'],seeds=[119,227],physical_steps=68000,
     input_trace_sha256=sha(reach.TRACE),teacher_only=True,training_updates=0,
     anatomical_joint_limits='UNKNOWN_NOT_CONFIGURED',
     gates='Same reach.py diagnostic gates; do not relax after run.',
     source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'reach.py'):sha(HERE/'reach.py')})
write(HERE/'FOOT_TEACHER_CONFIG.json',config)
original_make=reach.make_body
def make(scene,seed):
    b=original_make(scene,seed)
    # reach.run deep-copies this motor, preserving bound methods on the new object.
    def mask(self,obs):
        projection=np.dot(obs.stumbling_contact_forces,obs.fly_heading)
        original=(projection[:,:3]<self.stumbling_force_threshold).any(axis=1)
        phase=self.cpg_network.curr_phases%(2*np.pi)
        swing=np.array([self.preprogrammed_steps.swing_period[leg][0]<phase[i]<
                        self.preprogrammed_steps.swing_period[leg][1]+self.swing_extension for i,leg in enumerate(self.legs)])
        return original | ((projection[:,3]<self.stumbling_force_threshold)&swing)
    b.motor._get_stumbling_mask=types.MethodType(mask,b.motor)
    def teacher(command):
        obs=HybridControllerObservation.from_sim(b.sim,b.name,legs=b.motor.legs,
                 stumbling_links=('tibia','tarsus1','tarsus2','tarsus5'))
        return b.motor.step(command,obs).joint_angles
    b.teacher=teacher
    return b
reach.make_body=make
results=[]
for scene in config['scenes']:
    for seed in config['seeds']:
        results.append(reach.run(scene,'teacher',seed,HERE/'FOOT_TEACHER'/f'{scene}_{seed}'))
write(HERE/'FOOT_TEACHER_RESULT.json',dict(results=results,physical_steps=sum(r['steps'] for r in results),
     status='LOCAL_TEACHER_CANDIDATE_ONLY',training_updates=0))
