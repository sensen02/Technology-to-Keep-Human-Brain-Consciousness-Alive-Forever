"""Offline warmup trajectory checks. No simulation imports or new integration."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();o=Path(args.directory)
 with np.load(o/'BODY.npz') as f:b={k:f[k] for k in f.files}
 def arr(*keys):return next(b[k] for k in keys if k in b)
 t=arr('time');q=arr('qpos');v=arr('qvel');act=arr('activation');assist=arr('assist_torque','assist');external=arr('external_force','external');fc=arr('foot_contact').astype(bool);nf=arr('nonfoot_floor_count','nonfoot_count');ff=arr('ground_force_foot_world','foot_force');warnings=arr('warnings','warning');com=arr('com_position','com');feet=arr('foot_position_world','foot_origins');n=len(t)
 assert n>=2 and np.all(np.diff(t)>0)
 assert q.shape==(n,49) and v.shape==(n,48) and assist.shape==(n,48)
 assert all(np.isfinite(x).all() for x in b.values() if np.issubdtype(x.dtype,np.number))
 assert not assist[:,:6].any(),'Root force assistance is forbidden'
 assert not external.any(),'External body force must be absent'
 assert not warnings.any(),'Numerical warnings invalidate this check'
 norm=np.linalg.norm(q[:,3:7],axis=1);np.testing.assert_allclose(norm,1,rtol=0,atol=1e-8)
 # Rotate world offsets into body coordinates with the inverse free-root quaternion.
 w=q[:,3];xyz=q[:,4:7];delta=feet-q[:,:3,None].transpose(0,2,1)
 twice=2*np.cross(-xyz[:,None,:],delta)
 local=delta+w[:,None,None]*twice+np.cross(-xyz[:,None,:],twice)
 side=(local[:,0,1]>0)&(local[:,3,1]<0)
 liftoff=np.sum(np.diff(fc.astype(int),axis=0)==-1,axis=0)
 touchdown=np.sum(np.diff(fc.astype(int),axis=0)==1,axis=0)
 gait=t>=.25;late=t>=max(t[0],t[-1]-.2)
 weight=json.loads((o/'GAINS.json').read_text())['weight'];ratio=ff[:,:,2].sum(axis=1)/weight
 dx=float(q[-1,0]-q[0,0]);height_ratio=float(com[-1,2]/com[0,2])
 gait_start=int(np.searchsorted(t,.25));gait_start=min(gait_start,n-1)
 gait_dx=float(q[-1,0]-q[gait_start,0])
 # Count sustained air bouts independently, without modifying the simulation/contact data.
 dt_sample=float(np.median(np.diff(t)));durable=[];max_air=[]
 for j in range(6):
  air=(~fc[:,j])&gait;pad=np.r_[False,air,False].astype(int)
  starts=np.flatnonzero(np.diff(pad)==1);ends=np.flatnonzero(np.diff(pad)==-1)
  durations=(ends-starts)*dt_sample
  durable.append(int((durations>=.010-1e-9).sum()))
  max_air.append(float(durations.max()) if len(durations) else 0.)
 # This is a conservative necessary screen, not a sufficient walking certification.
 screen=bool(t[-1]-t[0]>=.85 and dx>.3 and not nf[gait].any() and np.all(np.array(durable)>=3) and side.all() and ratio[late].mean()>.7 and height_ratio>.7)
 outcome=dict(saved_states=n,physical_duration_s=float(t[-1]-t[0]),new_physics_steps=0,new_neural_steps=0,finite=True,root_assistance_zero=True,external_force_zero=True,numerical_warning_zero=True,forefoot_body_side_valid_fraction=float(side.mean()),forefoot_body_side_min=[float(local[:,0,1].min()),float((-local[:,3,1]).min())],forward_displacement_native=dx,post_ramp_forward_displacement_native=gait_dx,com_final_initial_height_ratio=height_ratio,contact_liftoffs=liftoff.tolist(),contact_touchdowns=touchdown.tolist(),durable_air_bouts_ge10ms=durable,max_air_duration_s=max_air,walking_nonfoot_fraction=float((nf[gait]>0).mean()) if gait.any() else None,last200ms_mean_foot_load_ratio=float(ratio[late].mean()),necessary_walking_screen=screen,walking_certified=False,note='Necessary offline screen only. Contact transitions may include chatter; require actual video and stance/swing support review. Cartesian forefoot sides do not certify all mesh clearance.',body_sha256=hashlib.sha256((o/'BODY.npz').read_bytes()).hexdigest())
 target=o/'PARENT_PREWARM_VERIFICATION.json';assert not target.exists();target.write_text(json.dumps(outcome,indent=2)+'\n');print(json.dumps(outcome,indent=2))
if __name__=='__main__':main()
