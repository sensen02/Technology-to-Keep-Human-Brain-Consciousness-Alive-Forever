"""Frozen teacher-relative command calibration metrics, full physical-state records."""
from scope import *
from scipy.signal import find_peaks

def velocity(a,first,last):
 p=a['position'][first:last,:2];yaw=a['yaw'][first:last]
 direction=np.c_[np.cos(yaw[:-1]),np.sin(yaw[:-1])]
 return float(np.sum(np.diff(p,axis=0)*direction)/(DT*(len(p)-1)))

def excursion(a,first,last):
 values=[]
 for foot in a['foot_forward'][first:last].T:
  smooth=np.convolve(foot,np.ones(51)/51,mode='valid');peaks,_=find_peaks(smooth,distance=400,prominence=.02)
  cycles=[float(np.ptp(smooth[x:y+1])) for x,y in zip(peaks[:-1],peaks[1:])]
  values.append(float(np.median(cycles)) if cycles else None)
 if any(v is None for v in values):return values,None
 return values,float(np.median(values[3:])/np.median(values[:3]))

def measure(a,m,meta):
 yaw=np.unwrap(a['yaw']);start=meta['start'];action=meta['action'];off=meta['off'];family=meta['family']
 q=a['qpos'];j=np.flatnonzero(m.jnt_limited);angles=q[:,m.jnt_qposadr[j]]
 excess=np.maximum(0,np.maximum(m.jnt_range[j,0]-angles,angles-m.jnt_range[j,1]))
 mweight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));pos=a['position'];rms=float(np.sqrt(np.mean(np.sum((np.diff(pos[-2000:,:2],axis=0)/DT)**2,axis=1))))
 active=(action+1800,off-300) if family.startswith(('arc','stride')) else (start+1800,off-300)
 forward=velocity(a,*active);angle=float(np.rad2deg(yaw[off-1]-yaw[action if family.startswith(('arc','stride')) else start]))
 amps,ratio=excursion(a,*active)
 actuator_ids=np.where(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)[0]
 limits=m.actuator_ctrlrange[actuator_ids]
 gates=dict(completed=len(q)==20000,finite=all(np.isfinite(v).all() for v in a.values()),upright=float(a['upright'].min())>=.8,
   height=float(pos[:,2].min())>=.7*a['initial_position'][2],nonfoot=float(a['nonfoot'].max()/mweight)<=.2,
   joint_limit=float(excess.max())<=.02,target_limits=bool((a['control']>=limits[:,0]-1e-12).all() and (a['control']<=limits[:,1]+1e-12).all()),stand=rms<=.30)
 if family.startswith(('arc','stride')):gates['direction']=angle*(1 if family.endswith('left') else -1)>0 and forward>0
 elif family=='back':gates['direction']=forward<0
 speed=[]
 if family=='speed':
  on=meta['speed_on'];down=meta['speed_off']
  speed=[velocity(a,start+1800,on-300),velocity(a,on+1800,down-300),velocity(a,down+1800,off-300)]
  gates['speed_changes']=speed[1]>speed[0] and speed[1]>speed[2] and min(speed)>0
 return dict(absolute_pass=bool(all(gates.values())),gates=gates,forward_speed=forward,yaw_degrees=angle,stride_by_leg=amps,
   stride_right_left_ratio=ratio,speed_windows=speed,stand_rms=rms,min_height=float(pos[:,2].min()),min_upright=float(a['upright'].min()),
   nonfoot_peak=float(a['nonfoot'].max()/mweight),joint_excess=float(excess.max()),dx=float(pos[-1,0]-a['initial_position'][0]))

def compare(t,s):
 gates=dict(speed=abs(s['forward_speed']-t['forward_speed'])<=max(1.,.25*abs(t['forward_speed'])),
  yaw=abs(s['yaw_degrees']-t['yaw_degrees'])<=max(10.,.25*abs(t['yaw_degrees'])),
  stand=s['stand_rms']<=max(.30,t['stand_rms']+.10),height=s['min_height']>=t['min_height']-.10,
  upright=s['min_upright']>=min(.8,t['min_upright']-.05),nonfoot=s['nonfoot_peak']<=max(.2,t['nonfoot_peak']+.05),
  joint_limit=s['joint_excess']<=max(.02,t['joint_excess']+.01),target=s['gates']['target_limits'],finite=s['gates']['finite'] and s['gates']['completed'])
 if t['family'].startswith(('arc','stride')):gates['direction']=s['yaw_degrees']*t['yaw_degrees']>0 and s['forward_speed']*t['forward_speed']>0
 elif t['family']=='back':gates['direction']=s['forward_speed']*t['forward_speed']>0
 if t['family'].startswith('stride'):
  tr=t['stride_right_left_ratio'];sr=s['stride_right_left_ratio']
  gates['stride_ratio']=tr is not None and sr is not None and abs(tr-sr)<=.20
  gates['stride_direction']=tr is not None and sr is not None and (tr-1)*(sr-1)>0
 if t['family']=='speed':
  gates['speed_windows']=all(abs(sv-tv)<=max(1.,.25*abs(tv)) for tv,sv in zip(t['speed_windows'],s['speed_windows']))
  tv=t['speed_windows'];sv=s['speed_windows'];gates['speed_direction']=(sv[1]-sv[0])*(tv[1]-tv[0])>0 and (sv[2]-sv[1])*(tv[2]-tv[1])>0
 return dict(relative_pass=bool(all(gates.values())),relative_gates=gates,teacher_absolute=t['absolute_pass'],student_absolute=s['absolute_pass'],
   status='CONSISTENT' if all(gates.values()) else 'STUDENT_DEVIATION',family=t['family'],variant=t['variant'])
