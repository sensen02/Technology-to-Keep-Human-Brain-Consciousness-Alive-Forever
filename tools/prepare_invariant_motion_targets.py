"""Construct invariant partial bending targets from recorded markers; no IK or dynamics."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from engine.embodied.kinematic_loss import geometric_interior_angles,interior_angle_loss
S=Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z/NMF2_CONTINUOUS.npz')
T=Path('/home/sensen/Desktop/engine修复验收/s4-real-motion-retarget/20261102T011240Z/RETARGETED_TARGETS.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert (O/'PREREGISTRATION.json').exists() and not (O/'RESULT.json').exists()
 inputs=[S,T,Path(__file__),R/'engine/embodied/kinematic_loss.py',R/'test_kinematic_loss.py'];hashes={str(p):sha(p) for p in inputs}
 dump(O/'SOURCE_IDENTITIES.json',hashes)
 s=np.load(S,allow_pickle=False);t=np.load(T,allow_pickle=False);points=s['xyz_source'][:,:,:4];mask=s['marker_valid'][:,:,:4]
 angles,valid=geometric_interior_angles(points,mask)
 assert angles.shape==(37,6,2) and valid.all()
 reference,_=geometric_interior_angles(t['reference_points'][:,:,:4],t['model_target_mask'][:,:,:4])
 source_error=float(abs(angles-s['geometric_interior_angle_rad'][:,:,:2]).max());retarget_error=float(abs(angles-reference).max())
 assert source_error<1e-12 and retarget_error<1e-12
 rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
 transformed=points@rotation*2.75+np.array([3.,-11.,20.]);values,vm=geometric_interior_angles(transformed,mask)
 invariant_error=float(abs(values-angles).max());assert invariant_error<1e-12 and np.array_equal(vm,valid)
 independent=[]
 for frame in range(37):
  legs=[]
  for leg in range(6):
   output=[]
   for k in range(2):
    u=points[frame,leg,k]-points[frame,leg,k+1];v=points[frame,leg,k+2]-points[frame,leg,k+1]
    dot=math.fsum(float(a*b) for a,b in zip(u,v));cross=np.cross(u,v);length=math.sqrt(math.fsum(float(x)**2 for x in cross));output.append(math.atan2(length,dot))
   legs.append(output)
  independent.append(legs)
 scalar_error=float(abs(np.array(independent)-angles).max());assert scalar_error<1e-12
 comparator=interior_angle_loss(t['achieved_points'][:,:,:4],points,mask)
 model,_=geometric_interior_angles(t['achieved_points'][:,:,:4],mask)
 assert abs(comparator['loss']-float(np.mean((model-angles)**2)))<1e-12
 np.savez_compressed(O/'INVARIANT_BENDING_TARGETS.npz',angles_rad=angles,angle_valid=valid,source_frame=s['source_frame'],time_relative_s=s['time_relative_s'],trial_id=s['trial_id'],split=s['split'],leg_names=s['leg_names'],triplet_names=np.array(['ThC-CTr-FTi','CTr-FTi-TiTa']),observed_keypoints=points,observed_marker_valid=mask,model_IK_bending_diagnostic=model,root_valid=np.zeros(37,bool),native_joint_angle_truth=np.array(False),runtime_enabled=np.array(False))
 before=json.loads((O/'PREREGISTRATION.json').read_text())['before_sha256'];changes=[]
 for path,h in before.items():
  backup=O/('BEFORE_'+Path(path).name);assert sha(backup)==h;changes.append(dict(path=path,before_sha256=h,after_sha256=sha(Path(path)),reason='Add invariant three-marker bending error; existing losses unchanged, no runtime caller added.'))
 dump(O/'SOURCE_CHANGE_LEDGER.json',dict(changes=changes))
 dump(O/'RESULT.json',dict(passed=True,source_sha256=hashes,frames=37,legs=6,valid_angles=444,source_acos_comparison_max_abs_rad=source_error,retarget_rotation_length_change_max_abs_rad=retarget_error,positive_scale_rigid_frame_invariance_max_abs_rad=invariant_error,independent_scalar_atan2_max_abs_rad=scalar_error,existing_IK_bending_error=comparator,not_full_pose=True,native_joint_angle_truth=False,root_ground_force_truth=False,raw_time_axis='Nominal120Hz,not independent acquisition timestamps',source_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),calls=dict(model_compile=0,network=0,body_step=0,neural_step=0,optimizer=0,training=0)))
 print((O/'RESULT.json').read_text())
if __name__=='__main__':main()
