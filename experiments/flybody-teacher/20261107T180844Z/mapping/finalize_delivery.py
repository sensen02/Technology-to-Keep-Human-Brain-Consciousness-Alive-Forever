"""Package this completed audit in its two authorized directories and verify bytes."""
import json,hashlib,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
D=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z/mapping')
B=json.loads((P/'current_model.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):(P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
interface=dict(status='PROPOSED_NEXT_INTERFACE_NOT_IMPLEMENTED_OR_TRAINED',body_source=B['runtime']['model_source'],body_source_sha256=B['runtime']['model_source_sha256'],
    output=dict(shape=[42],ordered_joint_names=[a['target_name'] for a in B['actuators']],units='rad/s',rate_clip=[-30.,30.],command_period_s=.001,policy_period_s=.005,target_bounds='current compiled hinge ranges',torque_semantics='unchanged bounded incremental target then finite PD128; no native activation filter'),
    retained_body='complete71mesh, free root, no equality/root actuator, originalgravity/contact/friction/force caps',
    student_inputs=dict(local_motor_module='causal local proprioception/contact observations plus anatomically constrained descending commands; exact sensory channels and latencies require a separately frozen contract',brain='restricted ascending neural interface and internal-state signals; no direct full joint vector or 42-joint command bypass around neck',training_only='registered official motion landmarks/segment orientations; teacher labels and actual failed episodes distinguished; privileged reference not deployed'),
    supervision=dict(source='official measured102joint motion / teacher trajectories separately tagged',method='bounded continuous landmark/orientation retargeting to current42; not59-action cropping',geometric_validation='candidate regional mapping only; holdout retarget residual and current angle/rate limits required',dynamic_validation='current-body original physics, no adhesive actuator; actual support, slip, nonfoot contact, saturation, release and causal controls'),
    unmapped_official_controls=['six adhesion','three head','two abdominal tendons','six tarsus hinges','six distal-tarsal tendons'],
    feeding_choice='not solved by walking interface: current mouth/head/abdomen fixed; explicit ingestion/environment/internal-state interface still required',
    surgery='deferred until full feeding/choice/locomotion platform and preoperative parameters frozen')
dump('STUDENT_INTERFACE.json',interface)
execution=dict(scope='this delegated mapping audit only',jobs=[
    dict(id='bash-158',operation='official extraction first version',exit_code=0,forward=410,jac=204,keyframe_reset=1),
    dict(id='bash-159',operation='current extraction first version',exit_code=0,forward=170,jac=84,keyframe_reset=1),
    dict(id='bash-165',operation='official extraction corrected tendon decoder',exit_code=0,forward=410,jac=204,keyframe_reset=1),
    dict(id='bash-166',operation='current extraction corrected tendon decoder',exit_code=0,forward=170,jac=84,keyframe_reset=1),
    dict(id='bash-168',operation='saved-array tables and plot',exit_code=0,forward=0,jac=0,keyframe_reset=0)],
    initial_directory_probe='ls authorized output directories returned2 because neither existed yet; write tool then created output parent normally',
    initial_document_probe='ROADMAP.md and TRACKER.md names absent; AGENTS directed actual GOAL_ROADMAP.zh-CN.md and GOAL_TRACKER.zh-CN.md, both read',
    environment_steps=0,physics_steps=0,training_updates=0,cumulative_mj_forward=1160,cumulative_mj_jac=576,cumulative_scratch_keyframe_reset=4,
    visualization_review='KINEMATIC_REFERENCE.png actually opened; six leg line chains and differing reference poses visible; not a physical video or full-mesh rendering',
    cleanup='Superseded first model JSONs replaced after tendon-decoder correction; no extra experiment directory or tracked-file edit; required desktop copy retained')
dump('EXECUTION_LEDGER.json',execution)
files=sorted(p for p in P.iterdir() if p.is_file() and p.name not in ('MANIFEST.json','DELIVERY_VERIFICATION.json'))
manifest=dict(workspace=str(P),desktop=str(D),files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
dump('MANIFEST.json',manifest)
D.mkdir(parents=True,exist_ok=True)
for p in files+[P/'MANIFEST.json']:shutil.copy2(p,D/p.name)
assert all(sha(p)==sha(D/p.name) for p in files+[P/'MANIFEST.json'])
delivery=dict(passed=True,files_verified=len(files)+1,workspace=str(P),desktop=str(D),manifest_sha256=sha(P/'MANIFEST.json'),no_step=True)
dump('DELIVERY_VERIFICATION.json',delivery);shutil.copy2(P/'DELIVERY_VERIFICATION.json',D/'DELIVERY_VERIFICATION.json')
print(json.dumps(delivery,indent=2))
