"""Pure-array semantic audit on actual recorded/retargeted motion, no new rollout."""
from pathlib import Path
import sys,json,hashlib,math,argparse
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from engine.embodied.kinematic_loss import masked_cartesian_huber,segment_direction_loss,masked_velocity_huber
S=Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z/NMF2_CONTINUOUS.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s4-kinematic-loss-readiness/20261102T011240Z')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 inputs=[S,Path(__file__),R/'engine/embodied/kinematic_loss.py',R/'test_kinematic_loss.py'];hashes={str(f):sha(f) for f in inputs}
 dump(O/'PREREGISTRATION.json',dict(task='S4 real-record loss and masks readiness, not fitted movement or runtime action',source_sha256=hashes,checks='Actual37frame record self-identity zero, positive scale/translation direction invariance, explicit bad-mask rejection, adjacent-frame calculation with original nominal dt. Independent scalar reconstruction of baseline temporal variation, not a neural prediction.',physics_calls=0,neural_calls=0,training=False))
 z=np.load(S,allow_pickle=False);p=z['xyz_source'];m=z['marker_valid'];t=z['time_relative_s'];ids=z['trial_id']
 normalizer=np.median(z['bone_lengths_source'],axis=(0,2))[:,None]
 identity=masked_cartesian_huber(p,p,m,normalizer=normalizer)
 direction=segment_direction_loss(p,p,m)
 velocity=masked_velocity_huber(p,p,m,t,normalizer_per_second=normalizer,max_gap_s=1/120*1.01,trial_ids=ids)
 invariant=segment_direction_loss(p*1.5+np.array([3.,-7.,19.]),p,m)
 assert identity['loss']==direction['loss']==velocity['loss']==0 and invariant['loss']<1e-25
 # An explicit frozen-first-record comparator shows geometry differs over time.
 frozen=np.broadcast_to(p[:1],p.shape)
 stationary=masked_cartesian_huber(frozen,p,m,normalizer=normalizer)
 stationary_direction=segment_direction_loss(frozen,p,m)
 scalar_points=[];scalar_angles=[]
 for frame in range(len(p)):
  for leg in range(6):
   for marker in range(5):
    if not m[frame,leg,marker]:continue
    res=math.sqrt(math.fsum(float(p[0,leg,marker,a]-p[frame,leg,marker,a])**2 for a in range(3)))/float(normalizer[leg,0])
    scalar_points.append(.5*res*res if res<=1 else res-.5)
   for seg in range(4):
    a=p[0,leg,seg+1]-p[0,leg,seg];b=p[frame,leg,seg+1]-p[frame,leg,seg]
    la=math.sqrt(math.fsum(float(v)**2 for v in a));lb=math.sqrt(math.fsum(float(v)**2 for v in b));dot=math.fsum(float(x*y) for x,y in zip(a,b))/(la*lb)
    cross=np.cross(a,b);sn=math.sqrt(math.fsum(float(v)**2 for v in cross))/(la*lb);scalar_angles.append(math.atan2(sn,dot)**2)
 assert abs(math.fsum(scalar_points)/len(scalar_points)-stationary['loss'])<1e-12
 assert abs(math.fsum(scalar_angles)/len(scalar_angles)-stationary_direction['loss'])<1e-12
 no_valid=masked_cartesian_huber(p,p,np.zeros_like(m),normalizer=normalizer)
 assert not no_valid['evaluable'] and no_valid['loss'] is None
 bad=p.copy();bad[0,0,0,0]=np.nan;rejected=False
 try:masked_cartesian_huber(bad,p,m,normalizer=normalizer)
 except ValueError:rejected=True
 assert rejected
 report=dict(passed=True,source_sha256=hashes,record_frames=len(p),record_coordinate_values=p.size,identity=identity,direction_identity=direction,velocity_identity=velocity,direction_positive_scale_translation_invariance=invariant,frozen_first_frame_cartesian=stationary,frozen_first_frame_direction=stationary_direction,missing_all_evaluable=False,nonfinite_prediction_rejected=True,normalizers_from_entire_record_for_audit_only=normalizer.tolist(),scope='All-source median segment lengths used ONLY to normalize this diagnostic, not calibrated body transform or trained normalizer. Runtime/model fitting must use training-only calibration. Stationary comparator is arithmetic benchmark, never deployed controller.',calls=dict(model_load=0,network=0,body_step=0,neural_step=0,training=0))
 assert all(sha(Path(f))==h for f,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
