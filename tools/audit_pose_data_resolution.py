"""Audit what anatomical resolution each real/derived pose dataset actually has.
No models, no simulation: only field/level inventory. Absence of muscle fields is
asserted, so a later claim of muscle-level pose data must break this test first.
"""
from pathlib import Path
import json,hashlib
import numpy as np
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s1-pose-data-resolution/20261102T190000Z'
SOURCES={
 'real_freewalk_3d':'real-freewalking-3d/20261027T150000Z/REFERENCE.npz',
 'nmf2_continuous':'s1-continuous-fly-kinematics/20261102T004326Z/NMF2_CONTINUOUS.npz',
 'invariant_interior_angles':'s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz',
 'caltech_20_derived_angles':'s1-continuous-fly-kinematics/20261102T004326Z/caltech_angles/train_neutral.npz',
 'standard_gait_model_fk':'s5-standard-gait-reference/20261028T040000Z/STANDARD_GAIT_FK.npz',
}
MUSCLE_TOKENS=('muscle','act','activation','fmax','lengthrange','attachment','site','innervation','emg')
MARKER_TOKENS=('keypoint','marker','foot','landmark','xyz','raw_keypoints')
JOINT_TOKENS=('angle','joint','qpos','dof','phase')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False)
 freeze={name:dict(path=str(D/rel),sha256=sha(D/rel)) for name,rel in SOURCES.items()}
 records=[]
 for name,rel in SOURCES.items():
  p=D/rel
  with np.load(p,allow_pickle=False) as z:
   fields={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
   keys=[k.lower() for k in z.files]
  muscle_hits=[k for k in keys if any(t in k for t in MUSCLE_TOKENS)]
  marker_hits=[k for k in keys if any(t in k for t in MARKER_TOKENS)]
  joint_hits=[k for k in keys if any(t in k for t in JOINT_TOKENS)]
  records.append(dict(dataset=name,source_sha256=freeze[name]['sha256'],n_fields=len(fields),fields=fields,landmark_fields=marker_hits,joint_angle_fields=joint_hits,muscle_like_fields=muscle_hits,anatomical_level=('landmark + derived joint angles' if joint_hits else 'landmark only') if marker_hits else ('model DOF' if joint_hits else 'other')))
  if muscle_hits:raise AssertionError(('muscle-resolved field unexpectedly present; claim must be revisited',name,muscle_hits))
 result=dict(passed=True,datasets=records,conclusion='No project dataset carries measured muscle activation, muscle length, attachment coordinates or innervation; the finest measured level is external 3D landmarks (up to 50 keypoints and 6 foot points, 685 frames) plus derived joint angles. Muscle-level quantities exist only as model/mapping-derived values.',implications=['Fitting cannot be supervised at muscle level: any muscle-level target would be our own model output, not animal measurement.','Available supervised pose targets are leg landmark positions, 2 or 3 interior angles per leg, 20 derived Caltech angles, and model-FK 42 DOF from a processed reference.','Therefore a muscle-parameter fit is under-determined and must be declared engineering calibration, not anatomical identification.'],calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,linprog=0))
 (O/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({r['dataset']:r['anatomical_level'] for r in records},indent=2),flush=True)
if __name__=='__main__':main()
