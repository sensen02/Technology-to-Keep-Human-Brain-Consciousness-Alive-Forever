from scope import *

def window(a,lo,hi,weight):
 pos=a['position'][lo:hi,:2];yaw=a['yaw'][lo:hi];diff=np.diff(pos,axis=0)/DT
 return dict(rms=float(np.sqrt(np.mean(np.sum(diff**2,axis=1)))),drift=float(np.linalg.norm(pos-pos[0],axis=1).max()),
  support=float(np.mean((a['normal_load'][lo:hi]>=.02*weight).sum(1)>=3)),
  forward=float(np.mean(np.sum(diff*np.c_[np.cos(yaw[:-1]),np.sin(yaw[:-1])],axis=1))))

def measure(a,m,meta):
 pos=a['position'];yaw=np.unwrap(a['yaw']);weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));j=np.flatnonzero(m.jnt_limited);q=a['qpos'][:,m.jnt_qposadr[j]]
 excess=np.maximum(0,np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]));end=window(a,len(pos)-2000,len(pos),weight)
 ids=np.where(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)[0];limits=m.actuator_ctrlrange[ids]
 angle=float(np.rad2deg(yaw[meta['off']-1]-yaw[meta['start']]))
 peak=float(np.linalg.norm(pos[:,:2]-a['initial_position'][:2],axis=1).max())
 g=dict(completed=len(pos)==20000,finite=all(np.isfinite(v).all() for v in a.values()),upright=a['upright'].min()>=.8,
  height=pos[:,2].min()>=.7*a['initial_position'][2],nonfoot=a['nonfoot'].max()/weight<=.2,joint_limit=excess.max()<=.02,
  target_limits=(a['control']>=limits[:,0]-1e-12).all() and (a['control']<=limits[:,1]+1e-12).all(),stand=end['rms']<=.30)
 extra={}
 if meta['family'].startswith('pivot'):
  g.update(direction=angle*(1 if meta['family'].endswith('left') else -1)>=18,pivot=peak<=2.25)
 else:
  hold=window(a,meta['action']+2500,meta['off']-500,weight);before=window(a,meta['start']+1800,meta['action']-300,weight);first=None
  for tick in range(meta['action'],meta['off']-2000,25):
   test=window(a,tick,tick+2000,weight)
   if test['rms']<=.30 and test['drift']<=.10 and test['support']>=.95:first=(tick-meta['action'])*DT;break
  cmd=HERE/'COMMANDS'/f"{meta['family']}_{meta['variant']}"/'TRACE.npz'
  with np.load(cmd,allow_pickle=False) as z:
   indices=[i for i,p in enumerate(meta['ports']) if p['group']=='forward'];forward_arrived=int(z['arrived'][meta['action']+2500:meta['off']-500,indices].sum())
   maxcmd=float(abs(z['command'][meta['action']+2500:meta['off']-500]).max())
  g.update(walking_before=before['forward']>=2.5,active_hold=hold['rms']<=.30 and hold['drift']<=.10 and hold['support']>=.95,
   command_stop=maxcmd<=.035,forward_still_arrives=forward_arrived>0)
  extra=dict(hold=hold,before=before,first_still_delay=first,forward_arrived_during_hold=forward_arrived,max_hold_command=maxcmd)
 g={k:bool(v) for k,v in g.items()}
 return dict(absolute_pass=all(g.values()),gates=g,yaw_degrees=angle,peak_excursion=peak,stand_rms=end['rms'],min_height=float(pos[:,2].min()),
  min_upright=float(a['upright'].min()),nonfoot_peak=float(a['nonfoot'].max()/weight),joint_excess=float(excess.max()),**extra)

def compare(t,s):
 g=dict(yaw=abs(t['yaw_degrees']-s['yaw_degrees'])<=max(10.,.25*abs(t['yaw_degrees'])),stand=s['stand_rms']<=max(.30,t['stand_rms']+.10),
  height=s['min_height']>=t['min_height']-.10,upright=s['min_upright']>=min(.8,t['min_upright']-.05),nonfoot=s['nonfoot_peak']<=max(.2,t['nonfoot_peak']+.05),
  joint=s['joint_excess']<=max(.02,t['joint_excess']+.01),target=s['gates']['target_limits'],finite=s['gates']['finite'] and s['gates']['completed'])
 if t['family'].startswith('pivot'):
  g.update(direction=t['yaw_degrees']*s['yaw_degrees']>0,translation=s['peak_excursion']<=max(2.25,t['peak_excursion']+.25))
 else:
  g.update(before_speed=abs(s['before']['forward']-t['before']['forward'])<=max(1.,.25*abs(t['before']['forward'])),
   hold_rms=s['hold']['rms']<=max(.30,t['hold']['rms']+.10),hold_drift=s['hold']['drift']<=max(.10,t['hold']['drift']+.05),
   active=s['forward_arrived_during_hold']>0 and s['max_hold_command']<=.035,
   delay=s['first_still_delay'] is not None and (t['first_still_delay'] is None or s['first_still_delay']<=t['first_still_delay']+.10))
 g={k:bool(v) for k,v in g.items()}
 return dict(family=t['family'],variant=t['variant'],relative_pass=all(g.values()),relative_gates=g,teacher_absolute=t['absolute_pass'],student_absolute=s['absolute_pass'])
