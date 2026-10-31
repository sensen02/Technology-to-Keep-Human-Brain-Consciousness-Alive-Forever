"""P3: real-derived NMF step FK only; zero integration or neural simulation."""
from pathlib import Path
import json,hashlib,pickle
import numpy as np
import mujoco
from flygym.compose import NeuroMechFly,KinematicPosePreset
from flygym.anatomy import Skeleton,AxisOrder,JointPreset,ActuatedDOFPreset,BodySegment
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
O=Path('/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z')
PKL=Path('/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl')
REAL=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z/REFERENCE.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False);counts=dict(model_compile=0,forward=0,physics_step=0,neural_step=0)
 npkl=pickle.loads(PKL.read_bytes());steps=PreprogrammedSteps();fly=NeuroMechFly(name='reference')
 # add_joints builds a neutral keyframe via one internal compile; count explicitly.
 skeleton=Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,joint_preset=JointPreset.LEGS_ONLY)
 fly.add_joints(skeleton,neutral_pose=KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL));counts['model_compile']+=1
 m,d=fly.compile();counts['model_compile']+=1
 dofs=skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY);adr=np.array([m.jnt_qposadr[m.joint(fly.jointdof_to_mjcfjoint[x].name).id] for x in dofs]);neutral=fly._get_neutral_qpos(m)
 legs=list(steps.legs);ids=[m.body(fly.bodyseg_to_mjcfbody[BodySegment(f'{leg}_tarsus5')].name).id for leg in legs];hipids=[m.body(fly.bodyseg_to_mjcfbody[BodySegment(f'{leg}_coxa')].name).id for leg in legs];thorax=m.body(fly.bodyseg_to_mjcfbody[BodySegment('c_thorax')].name).id
 phases=np.linspace(0,2*np.pi,33);feet=[];hips=[];allq=[];allangles=[]
 for phase in phases:
  angles=steps.get_joint_angles_by_dof_order(np.full(6,phase),output_dof_order=dofs);d.qpos[:]=neutral;d.qpos[adr]=angles;mujoco.mj_forward(m,d);counts['forward']+=1
  rot=d.xmat[thorax].reshape(3,3);feet.append((d.xpos[ids]-d.xpos[thorax])@rot);hips.append((d.xpos[hipids]-d.xpos[thorax])@rot);allq.append(d.qpos.copy());allangles.append(angles)
 feet=np.array(feet);hips=np.array(hips);hip_width=float(abs(hips[0,legs.index('lf'),1]-hips[0,legs.index('rf'),1]));assert hip_width>0
 # Exact segment origin, not claw surface/contact location; save definition prominently.
 np.testing.assert_allclose(feet[0],feet[-1],rtol=0,atol=1e-12)
 z=np.load(REAL);f=z['foot_body'];realcheck=dict(left_fore_y_range=[float(f[:,0,1].min()),float(f[:,0,1].max())],right_fore_y_range=[float(f[:,3,1].min()),float(f[:,3,1].max())],all685_left_positive=bool(np.all(f[:,0,1]>0)),all685_right_negative=bool(np.all(f[:,3,1]<0)),left_greater_right_fraction=float(np.mean(f[:,0,1]>f[:,3,1])),units='SOURCE_UNITS_UNKNOWN')
 np.savez_compressed(O/'STANDARD_GAIT_FK.npz',phase=phases,foot_body=feet,hip_body=hips,foot_body_over_forehip_width=feet/hip_width,angles_in_anatomical_dof_order=np.array(allangles),source_qpos=np.array(allq))
 report=dict(source='Real untethered straight walking, NMF2 processed single steps; symmetrized/closed/smoothed, not raw trajectory',pickle=str(PKL),pickle_sha256=sha(PKL),pickle_meta=npkl['meta'],samples_per_step=steps._length,sample_dt_s=steps._timestep,processed_cycle_duration_s=steps.duration,natural_cycle_frequency_hz=steps.frequency if hasattr(steps,'frequency') else 1/steps.duration,phase_samples=33,legs=legs,coordinate='thorax body frame: +x anterior,+y left,+z dorsal; source NMF native numerical length, do not assume SI',foot_definition='origin of tarsus5 body (proximal fifth-tarsus point), NOT exact distal claw or contact location',normalization='divide all foot coordinates by LF/RF coxa-origin lateral separation; no claim equal body morphology',forehip_width_native=hip_width,source_dof_order=[x.name for x in dofs],source_units_SI='NOT_ESTABLISHED here',right_side_sign='PreprogrammedSteps flips legacy right roll/yaw; native model right roll/yaw axes also anatomical-symmetric; use API, do not raw-copy legacy angles',current_model_warning='Current90muscle model has trochanter yaw/roll and no tarsus1 hinge; source joint DOFs are NOT one-to-one. Cartesian retarget and native FK/IK required.',real_freewalking_sidecheck=realcheck,counts=counts,paper='https://doi.org/10.1038/s41592-024-02497-y',paper_postprint='https://www.epfl.ch/labs/ramdya-lab/wp-content/uploads/2024/08/NMF2_postprint.pdf',software_license='FlyGym2.1.0 distribution Apache-2.0; do not infer unrelated external freewalking data relicensing',source_hashes={str(Path(__file__)):sha(Path(__file__)),str(PKL):sha(PKL),str(REAL):sha(REAL)})
 (O/'REFERENCE_METADATA.json').write_text(json.dumps(report,indent=2)+'\n');(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
