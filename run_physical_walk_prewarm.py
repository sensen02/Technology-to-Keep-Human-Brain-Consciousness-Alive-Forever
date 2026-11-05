"""Same 90-muscle full body, declared algorithmic joint-torque warmup. No brain yet."""
from pathlib import Path
import os,json,time,sys,traceback,hashlib,xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path(__file__).parent
BASE=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z')
STANCE=Path('/home/sensen/Desktop/engine修复验收/s5-noncrossing-static-prototype/20261028T041000Z/STATIC_STANCE.npz')
REF=Path('/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z/STANDARD_GAIT_FK.npz')
LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 def conv(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)
 Path(p).write_text(json.dumps(x,indent=2,default=conv)+'\n')
def build(o,dt,counts,muscle_activation_timeconstants=None,base_xml=None):
 src=Path(base_xml) if base_xml is not None else BASE/'INTEGRATED_MODEL.xml'
 tree=ET.parse(src);root=tree.getroot();m=_load_mjcf(str(src)).compile();counts['model_load']+=1
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);counts['forward']+=1;oldL=d.actuator_length.copy();q=np.load(STANCE)['qpos'].copy();d.qpos[:]=q;mujoco.mj_forward(m,d);counts['forward']+=1
 # Extend rear support under measured COM (-.80), retaining native bounds and toe height.
 qa0=np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]);va0=np.array([m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]);js=[m.dof_jntid[v] for v in va0];bids=[m.body(l+'Tarsus5').id for l in LEGS];target=d.xpos[bids].copy();target[[1,4],0]-=.35;target[[2,5],0]-=.85;cache={}
 def up(v):
  if 'v' not in cache or not np.array_equal(cache['v'],v):d.qpos[:]=q;d.qpos[qa0]=v;mujoco.mj_forward(m,d);counts['forward']+=1;cache['v']=v.copy()
 def ff(v):up(v);return np.r_[((d.xpos[bids]-target)*10).ravel(),.005*(v-q[qa0])]
 def jj(v):
  up(v);rr=[]
  for bid in bids:
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,d.xpos[bid],bid);rr.append(jp[:,va0]*10)
  return np.vstack(rr+[.005*np.eye(42)])
 res=least_squares(ff,np.clip(q[qa0],m.jnt_range[js,0]+1e-7,m.jnt_range[js,1]-1e-7),jac=jj,bounds=(m.jnt_range[js,0]+1e-7,m.jnt_range[js,1]-1e-7),max_nfev=100);up(res.x);q[qa0]=res.x
 dump(o/'STANCE_SUPPORT_RETARGET.json',dict(targets=target,actual=d.xpos[bids],errors=np.linalg.norm(d.xpos[bids]-target,axis=1),reason='041 COM x=-.80 at hind support boundary caused backwards tipping in physical051; extend hind support by .85 and middle by .35 native units',optimizer_success=res.success))
 lr=m.actuator_lengthrange.copy() if base_xml is not None else m.actuator_lengthrange*(d.actuator_length/oldL)[:,None]
 for j in root.iter('joint'):
  if j.get('name') and j.get('type')!='free':j.set('springref',format(q[m.jnt_qposadr[m.joint(j.get('name')).id]],'.17g'))
 for i,a in enumerate(root.findall('./actuator/*')):a.set('lengthrange',' '.join(format(v,'.17g') for v in lr[i]))
 if muscle_activation_timeconstants is not None:
  tau=np.asarray(muscle_activation_timeconstants,dtype=float)
  if tau.shape!=(2,) or not np.isfinite(tau).all() or np.any(tau<=0):raise ValueError('two finite positive muscle activation timeconstants required')
  changes=[]
  for a in root.findall('./actuator/*'):
   old=a.get('dynprm');values=old.split();values[:2]=[format(v,'.17g') for v in tau];a.set('dynprm',' '.join(values));changes.append(dict(actuator=a.get('name'),old_dynprm=old,new_dynprm=a.get('dynprm')))
  dump(o/'ACTIVATION_PARAMETER_LEDGER.json',dict(changes=changes,reason='Declared engineering activation-memory bridge, not physiological calibration',physiology='NOT_ESTABLISHED'))
 # Explicit engineering contact calibration: reduce soft-contact settling that prevented hind-foot lift.
 for g in root.iter('geom'):g.set('solref','.002 1')
 root.find('option').set('timestep',format(dt,'.17g'));keys=root.find('keyframe')
 for k in list(keys):keys.remove(k)
 ET.SubElement(keys,'key',name='noncrossing_warmup',qpos=' '.join(format(v,'.17g') for v in q));tree.write(o/'MODEL.xml',encoding='utf8',xml_declaration=True)
 m=_load_mjcf(str(o/'MODEL.xml')).compile();counts['model_load']+=1
 assert (m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,90,71,0)
 dump(o/'MODEL_CHANGES.json',dict(base=str(BASE/'INTEGRATED_MODEL.xml'),base_sha256=sha(src),new_sha256=sha(o/'MODEL.xml'),lengthrange_handling=('provided base lengthrange preserved' if base_xml is not None else 'rescaled to keyframe operating fraction'),base_model=str(src),changes=['41 noncrossing initial pose','42 springrefs aligned once','90 lengthranges rescaled newL/oldL once preserving032 operating fractions','body timestep'],lr=lr,geometry_gain_mass_unchanged=True,contact_solref='.002 1 all geoms, explicit engineering calibration; old implicit default .02 1',physiological_units='NOT_ESTABLISHED'))
 return m,q

def prepare_gait(m,q,o,counts,swing_height=None):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);counts['forward']+=1
 joints=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE];qa=np.array([m.jnt_qposadr[j] for j in joints]);va=np.array([m.jnt_dofadr[j] for j in joints]);lo=m.jnt_range[joints,0]+1e-7;hi=m.jnt_range[joints,1]-1e-7
 bids=[m.body(l+'Tarsus5').id for l in LEGS];baseline=d.xpos[bids].copy();ref=np.load(REF);phase=ref['phase'];foot=ref['foot_body'];shape=np.empty_like(foot)
 for j in range(6):
  # Relative shape, not coordinate transplantation: actual source point matches native Tarsus5 origin.
  for a,amp in [(0,.30),(1,.06),(2,.45)]:
   vals=foot[:,j,a];span=np.ptp(vals);v=(vals-vals[16])/max(span,1e-9)*amp
   if a==2:v=(vals-vals.min())/max(span,1e-9)*amp
   shape[:,j,a]=v
 # Align each source leg's highest-swing phase to LF peak before imposing tripod offsets.
 aligned=shape.copy();source_peak=phase[np.argmax(foot[:-1,:,2],axis=0)];lfpeak=source_peak[0]
 raw=CubicSpline(phase,shape,axis=0,bc_type='periodic')
 for j in range(6):aligned[:,j]=raw((phase+source_peak[j]-lfpeak)%(2*np.pi))[:,j]
 # Use extracted LF swing-height profile for all legs after phase alignment; explicitly engineering tripod retarget.
 aligned[:,:,2]=shape[:,0,2,None]
 if swing_height is not None:
  # Declared engineering correction: give EVERY leg the same explicit swing-lift
  # window shape, because the transplanted reference made the hind feet almost
  # never rise (measured: hind lift 0.06 vs front 0.51). x/y stay from the reference.
  h=float(swing_height)
  if not np.isfinite(h) or h<0:raise ValueError('nonnegative finite swing height required')
  zref=shape[:,0,2];peak=float(zref.max());lift=zref>=.6*peak
  idx=np.flatnonzero(lift)
  for j in range(6):
   prof=np.zeros(len(phase));prof[idx]=h*np.sin(np.linspace(0,np.pi,len(idx)))**.8
   aligned[:-1,j,2]=prof[:-1]
 aligned[-1]=aligned[0];spl=CubicSpline(phase,aligned,axis=0,bc_type='periodic');offset=np.array([0,np.pi,0,np.pi,0,np.pi]);qs=[];errs=[];x=np.clip(q[qa],lo,hi)
 for phi in phase:
  target=baseline+np.array([spl((phi+offset[j])%(2*np.pi))[j] for j in range(6)])
  cache={}
  def update(v):
   if 'x' not in cache or not np.array_equal(cache['x'],v):d.qpos[:]=q;d.qpos[qa]=v;mujoco.mj_forward(m,d);counts['forward']+=1;cache['x']=v.copy()
  def fun(v):update(v);return np.r_[((d.xpos[bids]-target)*10).ravel(),.008*(v-q[qa])]
  def jac(v):
   update(v);rows=[]
   for bid in bids:
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,d.xpos[bid],bid);rows.append(jp[:,va]*10)
   return np.vstack(rows+[.008*np.eye(42)])
  r=least_squares(fun,x,jac=jac,bounds=(lo,hi),max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7);x=r.x;update(x);qs.append(x.copy());errs.append(np.linalg.norm(d.xpos[bids]-target,axis=1))
 qs=np.array(qs);qs[-1]=qs[0];gait=CubicSpline(phase,qs,axis=0,bc_type='periodic');np.savez_compressed(o/'GAIT.npz',phase=phase,qpos_hinges=qs,target_baseline_body_origins=baseline,errors=errs,qpos_addresses=qa,dof_addresses=va)
 dump(o/'GAIT_RETARGET.json',dict(max_cartesian_error=float(np.max(errs)),phase_offsets=offset,source_swing_peak_phase=source_peak,engineering_phase_alignment='each source x/y shifted by z-peak to LF peak; all legs use extracted LF swing-height profile before tripod offsets',amplitude_native_xyz=[.30,.06,.45],swing_height_override=swing_height,source_point='tarsus5 body origin both source/target',engineering_approximation='Source trajectory variations normalized per leg/axis around native041 footprint; not measured absolute anatomy or direct joint-angle copy',source_sha=sha(REF)))
 return qa,va,bids,gait

def contacts(m,d):
 foot=np.zeros((6,3));non=np.zeros(3);contact=np.zeros(6,bool);noncount=0
 for k in range(d.ncon):
  co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
  if not(pa or pb):continue
  g=b if pa else a;name=m.geom(g).name;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);f=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1);leg=next((i for i,l in enumerate(LEGS) if name.startswith(l) and 'Tars' in name),None)
  if leg is None:non+=f;noncount+=1
  else:foot[leg]+=f;contact[leg]=True
 return foot,non,contact,noncount

def run(o,duration,kp,clip,dt,neural_duration=0.,neural_command_dt_s=.005,neural_motor_config=None,neural_motor_output_connected=True,neural_sensory_channel_weights=None,neural_descending_config=None,warmup_settle_s=0.,warmup_hold_s=0.,muscle_activation_timeconstants=None,warmup_actuator_command=None,warmup_activation_start_s=None,warmup_settle_target=None,neural_prewarm_s=0.,assistance_fade_s=0.,base_xml=None,keep_assistance_after_handover=False,swing_height=None,allow_signed_sensory_weights=False,sample_dt_s=.001):
 if not np.isfinite([warmup_settle_s,warmup_hold_s]).all() or min(warmup_settle_s,warmup_hold_s)<0:raise ValueError('nonnegative finite appended warmup durations required')
 if not np.isfinite(neural_prewarm_s) or neural_prewarm_s<0 or neural_prewarm_s>duration+warmup_settle_s+warmup_hold_s:raise ValueError('invalid neural prewarm duration')
 if neural_prewarm_s and (not neural_duration or warmup_actuator_command is not None or neural_descending_config is not None):raise ValueError('prewarm requires neural continuation without fixed commands or descending pulses')
 if neural_prewarm_s and abs(neural_prewarm_s/neural_command_dt_s-round(neural_prewarm_s/neural_command_dt_s))>1e-8:raise ValueError('prewarm must align with command clock')
 if not np.isfinite(assistance_fade_s) or assistance_fade_s<0 or assistance_fade_s>neural_prewarm_s:raise ValueError('assistance fade requires continuous neural prewarm of at least equal duration')
 if warmup_actuator_command is not None:
  warmup_actuator_command=np.asarray(warmup_actuator_command,dtype=float)
  if warmup_actuator_command.shape!=(90,) or not np.isfinite(warmup_actuator_command).all() or np.any(warmup_actuator_command<0) or np.any(warmup_actuator_command>1):raise ValueError('finite bounded90 warmup command required')
  if warmup_activation_start_s is None or not np.isfinite(warmup_activation_start_s) or warmup_activation_start_s<0 or warmup_activation_start_s>=duration+warmup_settle_s+warmup_hold_s:raise ValueError('warmup activation start must precede handover')
 o.mkdir(parents=True,exist_ok=False);counts=dict(model_load=0,forward=0,body_step=0,neural_step=0);started=time.time();error=None;rows=[]
 dump(o/'PREREGISTRATION.json',dict(assistance_fade_s=assistance_fade_s,neural_prewarm_s=neural_prewarm_s,duration_s=duration,neural_duration_s=neural_duration,neural_command_dt_s=neural_command_dt_s,neural_motor_output_connected=bool(neural_motor_output_connected),neural_sensory_channel_weights=neural_sensory_channel_weights,neural_descending_config=__import__('dataclasses').asdict(neural_descending_config) if neural_descending_config is not None else None,neural_dt_s=.0005,motor_output_config=__import__('dataclasses').asdict(neural_motor_config) if neural_motor_config is not None else 'unchanged default tau5ms/full-scale100Hz/gain1',warmup_settle_s=warmup_settle_s,warmup_hold_s=warmup_hold_s,muscle_activation_timeconstants=muscle_activation_timeconstants,warmup_actuator_command=warmup_actuator_command,warmup_activation_start_s=warmup_activation_start_s,swing_height=swing_height,warmup_settle_target=warmup_settle_target,handover_time_s=duration+warmup_settle_s+warmup_hold_s if neural_duration else None,dt=dt,hold_s=.15,ramp_s=.10,frequency_hz=5.,kp=kp,torque_clip=clip,damping='2sqrt(kp*initial diagonal mass)',controller='leg-joint PD + bias - passive compensation; root6 always zero',contact_calibration='solref .002 1 on all geoms',stage=('ALGORITHM ASSISTED with live neural prewarm then SAME warmed neural runtime' if neural_prewarm_s and not keep_assistance_after_handover else 'ALGORITHM ASSIST KEPT THROUGHOUT: hybrid assisted neural closed loop, explicitly NOT autonomous weight bearing' if keep_assistance_after_handover else 'ALGORITHM ASSISTED then fresh resting neural-only on same MjData') if neural_duration else 'ALGORITHM ASSISTED ONLY, no neural model instantiated',sample_dt_s=sample_dt_s,no_runtime_qpos_qvel_writes=True,keep_assistance_after_handover=bool(keep_assistance_after_handover),no_body_external_force=True,success_screen='nonfoot absent after hold; positive forward displacement; repeated real contact/liftoff; not neural autonomy',base_xml=str(base_xml) if base_xml is not None else None,source_hashes={str(x):sha(x) for x in [Path(__file__),STANCE,REF,Path(base_xml) if base_xml is not None else BASE/'INTEGRATED_MODEL.xml',P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/embodied/sensory_channels.py',P/'engine/neural_cond.py']}))
 try:
  m,q=build(o,dt,counts,muscle_activation_timeconstants=muscle_activation_timeconstants,base_xml=base_xml);qa,va,bids,gait=prepare_gait(m,q,o,counts,swing_height=swing_height);d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);counts['forward']+=1
  mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM);diag=np.diag(mass)[va];kd=2*np.sqrt(kp*diag);dump(o/'GAINS.json',dict(kp=kp,kd=kd,mass_diagonal=diag,clip=clip,weight=m.body_mass.sum()*abs(m.opt.gravity[2])))
  n=round((duration+warmup_settle_s+warmup_hold_s)/dt);stride=round(sample_dt_s/dt);thorax=m.body('Thorax').id
  settle_target=q[qa] if warmup_settle_target is None else np.asarray(warmup_settle_target,dtype=float)
  if settle_target.shape!=(42,) or not np.isfinite(settle_target).all():raise ValueError('finite42 hinge settle target required')
  settle_joint_ids=m.dof_jntid[va]
  if np.any(settle_target<m.jnt_range[settle_joint_ids,0]) or np.any(settle_target>m.jnt_range[settle_joint_ids,1]):raise ValueError('settle target violates native hinge limits')
  dump(o/'SETTLE_TARGET.json',dict(joint_names=[m.joint(int(j)).name for j in settle_joint_ids],qpos_addresses=qa,target=settle_target,source='original base stance' if warmup_settle_target is None else 'declared offline candidate hinge target; physical root never overwritten',sensory_reference='unchanged original pre-warmup qpos'))
  def save(refresh=True):
   if refresh:mujoco.mj_forward(m,d);counts['forward']+=1
   f,nf,fc,nc=contacts(m,d);rows.append(dict(time=d.time,qpos=d.qpos.copy(),qvel=d.qvel.copy(),activation=d.act.copy(),command=d.ctrl.copy(),thorax=d.xpos[thorax].copy(),com=d.subtree_com[thorax].copy(),foot_origins=d.xpos[bids].copy(),foot_force=f,nonfoot_force=nf,foot_contact=fc,nonfoot_count=nc,assist=d.qfrc_applied.copy(),external=d.xfrc_applied.copy(),warning=d.warning.number.copy(),muscle_force=d.actuator_force.copy()))
  save();print('DYNAMICS_START',o,'gait maxerror',json.loads((o/'GAIT_RETARGET.json').read_text())['max_cartesian_error'],flush=True)
  for step in range(n-round(neural_prewarm_s/dt)):
   t=float(d.time);ramp=np.clip((t-.15)/.10,0,1);phi=max(0,t-.15)*2*np.pi*5;des=(1-ramp)*q[qa]+ramp*gait(phi%(2*np.pi));vel=ramp*gait(phi%(2*np.pi),1)*2*np.pi*5+(gait(phi%(2*np.pi))-q[qa])*(10. if .15<t<.25 else 0.)
   if warmup_settle_s+warmup_hold_s>0 and step>=round(duration/dt):
     u=np.clip((step*dt-duration)/warmup_settle_s,0.,1.) if warmup_settle_s>0 else 1.
     blend=10*u**3-15*u**4+6*u**5;blend_velocity=(30*u**2-60*u**3+30*u**4)/warmup_settle_s if warmup_settle_s>0 and u<1 else 0.
     vel=(1-blend)*vel-blend_velocity*(des-settle_target);des=(1-blend)*des+blend*settle_target
   torque=kp*(des-d.qpos[qa])+kd*(vel-d.qvel[va])+d.qfrc_bias[va]-d.qfrc_passive[va]
   d.qfrc_applied[:]=0;d.qfrc_applied[va]=np.clip(torque,-clip,clip);d.xfrc_applied[:]=0;d.ctrl[:]=0
   if warmup_actuator_command is not None and step*dt>=warmup_activation_start_s-1e-12:d.ctrl[:]=warmup_actuator_command
   assert not d.qfrc_applied[:6].any();mujoco.mj_step(m,d);counts['body_step']+=1
   if (step+1)%stride==0:save()
   if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite body state')
  # With a live neural prewarm this state is the prewarm start, NOT the handover boundary.
  np.savez_compressed(o/('NEURAL_PREWARM_START.npz' if neural_prewarm_s else 'HANDOVER_STATE.npz'),qpos=d.qpos,qvel=d.qvel,activation=d.act,command=d.ctrl,time=d.time,qacc_warmstart=d.qacc_warmstart,assist=d.qfrc_applied)
  if neural_duration>0:
   import run_neural_muscle_closure as c
   from run_integrated_repaired_closure import identities
   from engine.embodied.runtime_pool_guard import load_runtime_pool
   from engine.embodied.continuous_handover import run_neural_continuous
   np.random.seed(0);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0))
   if neural_descending_config is not None:
    from engine.embodied.descending_command import augment_dnp09
    built=augment_dnp09(built)
    from scipy.sparse import save_npz
    save_npz(o/'AUGMENTED_We.npz',built['edges']['We']);save_npz(o/'AUGMENTED_Wi.npz',built['edges']['Wi']);np.savez_compressed(o/'AUGMENTED_IDENTITIES.npz',root_ids=built['tier']['root_ids'],global_index=built['global_index']);dump(o/'AUGMENTATION.json',built['coverage']['descending_augmentation'])
   manifest=c.build_motor_manifest(load_runtime_pool(c.POOLS),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids']);joints,geom=identities(m);dump(o/'MAPPING.json',manifest)
   # BODY boundary row is withdrawal-after; before-state persists separately above and in API snapshots.
   rows.pop();neural_callback_count=[-1]
   def observe(mm,dd):
    neural_callback_count[0]+=1
    if neural_callback_count[0]%stride==0:save(False)
   runtime_state=None
   if neural_prewarm_s:
    def assisted(mm,dd):
     step=round(dd.time/dt);t=float(dd.time);ramp=np.clip((t-.15)/.10,0,1);phi=max(0,t-.15)*2*np.pi*5;des=(1-ramp)*q[qa]+ramp*gait(phi%(2*np.pi));vel=ramp*gait(phi%(2*np.pi),1)*2*np.pi*5+(gait(phi%(2*np.pi))-q[qa])*(10. if .15<t<.25 else 0.)
     if warmup_settle_s+warmup_hold_s>0 and step>=round(duration/dt):
      u=np.clip((step*dt-duration)/warmup_settle_s,0.,1.) if warmup_settle_s>0 else 1.;blend=10*u**3-15*u**4+6*u**5;bv=(30*u**2-60*u**3+30*u**4)/warmup_settle_s if warmup_settle_s>0 and u<1 else 0.
      vel=(1-blend)*vel-bv*(des-settle_target);des=(1-blend)*des+blend*settle_target
     dd.qfrc_applied[va]=np.clip(kp*(des-dd.qpos[qa])+kd*(vel-dd.qvel[va])+dd.qfrc_bias[va]-dd.qfrc_passive[va],-clip,clip)
     if assistance_fade_s>0:
      remain=(duration+warmup_settle_s+warmup_hold_s-dd.time)/assistance_fade_s
      dd.qfrc_applied[va]*=float(np.clip(remain,0.,1.))
    warm=run_neural_continuous(m,d,built,manifest,groups,neural_prewarm_s,sensory_reference_qpos=q,joints=joints,geom_leg=geom,body_dt_s=dt,command_dt_s=neural_command_dt_s,motor_config=neural_motor_config,body_observer=observe,retain_currents=True,sensory_channel_weights=neural_sensory_channel_weights,assisted_prewarm_callback=assisted,allow_signed_sensory_weights=allow_signed_sensory_weights)
    runtime_state=warm['runtime_state'];dump(o/'PREWARM_NEURAL_RECORDS.json',warm['records'])
    warmed_arrays={name:np.array(value,copy=True) for name,value in vars(runtime_state['runtime'][0]).items() if isinstance(value,np.ndarray)}
    warmed_arrays['decoder_rates']=runtime_state['runtime'][2].rates.copy();warmed_arrays['pending']=runtime_state['pending'].copy();np.savez_compressed(o/'PREWITHDRAW_NEURAL_STATE.npz',**warmed_arrays)
    np.savez_compressed(o/'WARMED_RUNTIME.npz',pending=runtime_state['pending'],decoder_rates=runtime_state['runtime'][2].rates,neural_time_s=runtime_state['runtime'][0].t_ms/1000,body_time=d.time)
    counts['body_step']+=warm['counts']['body_steps'];counts['neural_step']+=warm['counts']['neural_steps'];counts['forward']+=warm['counts']['forward']
    rows.pop();neural_callback_count[0]=-1
   ret=run_neural_continuous(m,d,built,manifest,groups,neural_duration,runtime_state=runtime_state,sensory_reference_qpos=q,joints=joints,geom_leg=geom,body_dt_s=dt,command_dt_s=neural_command_dt_s,motor_config=neural_motor_config,body_observer=observe,retain_currents=True,motor_output_connected=neural_motor_output_connected,sensory_channel_weights=neural_sensory_channel_weights,descending_command=neural_descending_config,assisted_prewarm_callback=(assisted if keep_assistance_after_handover and neural_prewarm_s else None),assisted_neural_closedloop=bool(keep_assistance_after_handover and neural_prewarm_s))
   np.savez_compressed(o/'NEURAL_BOUNDARY.npz',**ret['runtime_boundary'])
   np.savez_compressed(o/'HANDOVER_BEFORE.npz',**ret['handover_before']);np.savez_compressed(o/'HANDOVER_AFTER.npz',**ret['handover_after_withdrawal']);dump(o/'NEURAL_RECORDS.json',ret['records']);np.savez_compressed(o/'NEURAL.npz',commands=ret['commands'],spike_counts=ret['spike_counts'],sensory_current_nA=np.array([r['sensory_current_nA'] for r in ret['records']]))
   counts['body_step']+=ret['counts']['body_steps'];counts['neural_step']+=ret['counts']['neural_steps'];counts['forward']+=ret['counts']['forward'];counts['network_instances']=int(neural_prewarm_s>0)+ret['counts']['network_instances'];dump(o/'HANDOVER.json',{k:v for k,v in ret.items() if k not in ('runtime_boundary','runtime_state','handover_before','handover_after_withdrawal','records','commands','spike_counts')})
 except BaseException as exc:
  error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
 finally:
  if rows:
   arrays={k:np.array([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(o/'BODY.npz',**arrays)
   b=arrays;mask=b['time']>.25;first=np.flatnonzero(b['nonfoot_count']>0);weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));dump(o/'OUTCOME.json',dict(initial_root=b['qpos'][0,:3],final_root=b['qpos'][-1,:3],forward_displacement=float(b['qpos'][-1,0]-b['qpos'][0,0]),initial_com=b['com'][0],final_com=b['com'][-1],first_nonfoot_s=None if not len(first) else float(b['time'][first[0]]),last_nonfoot_count=int(b['nonfoot_count'][-1]),walking_nonfoot_fraction=float((b['nonfoot_count'][mask]>0).mean()) if mask.any() else None,last_foot_load_ratio=float(b['foot_force'][-1,:,2].sum()/weight),assist_max=float(abs(b['assist']).max()),root_assist_max=float(abs(b['assist'][:,:6]).max()),warnings=int(b['warning'][-1].sum()),contact_liftoff_counts=np.sum(np.diff(b['foot_contact'].astype(int),axis=0)==-1,axis=0),algorithm_only=not bool(neural_duration),walking_certified=False))
  dump(o/'EXECUTION.json',dict(counts=counts,runtime_s=time.time()-started,exit_code=0 if error is None else 1,error=error));(o/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 print('COMPLETE',o,flush=True)
if __name__=='__main__':
 o=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T050000Z');duration=float(sys.argv[2]) if len(sys.argv)>2 else .15;kp=float(sys.argv[3]) if len(sys.argv)>3 else 100.;clip=float(sys.argv[4]) if len(sys.argv)>4 else 50.;run(o,duration,kp,clip,.0000125,float(sys.argv[5]) if len(sys.argv)>5 else 0.)
