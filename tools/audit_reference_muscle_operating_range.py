"""S1/S2/S4 frozen original reference cycle. No integration/optimization/network.
run: one original compile, reuse exact cached states, forward missing states only.
verify: independent scalar arithmetic in a new process; zero compile/forward.
Never seals: parent review must precede sealing.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib,math,csv,traceback,shutil
import numpy as np
R=Path(__file__).resolve().parents[1]
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s2-reference-muscle-operating-range/20261102T155412Z'
B=D/'s5-banc888-physical-handover/20261101T220018Z'
E=D/'s2-early-muscle-support/20261102T053639Z'
H=D/'s74-muscle-operating-range/20261023T160536Z'
REF=D/'s5-standard-gait-reference/20261028T040000Z'
SHA='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def dump(name,x):
 (O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def error(a,b):return float(np.max(abs(np.asarray(a)-np.asarray(b))))
def table(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def terms(length,velocity,lr,acc,p):
 r0,r1,force,scale,lo,hi,vmax,fp,fv=p[:9]
 force=scale/max(1e-15,acc) if force<0 else force
 l0=(lr[1]-lr[0])/max(1e-15,r1-r0);lt=lr[0]-r0*l0
 L=r0+(length-lr[0])/max(1e-15,l0);V=velocity/max(1e-15,l0*vmax)
 a=(lo+1)/2;b=(hi+1)/2
 if L<lo or L>hi:fl=0.
 elif L<=a:fl=.5*((L-lo)/max(1e-15,a-lo))**2
 elif L<=1:fl=1-.5*((1-L)/max(1e-15,1-a))**2
 elif L<=b:fl=1-.5*((L-1)/max(1e-15,b-1))**2
 else:fl=.5*((hi-L)/max(1e-15,hi-b))**2
 y=fv-1;vv=0. if V<=-1 else (V+1)**2 if V<=0 else fv-(y-V)**2/max(1e-15,y) if V<=y else fv
 bias=0. if L<=1 else -force*fp*.5*((L-1)/max(1e-15,b-1))**2 if L<=b else -force*fp*(.5+(L-b)/max(1e-15,b-1))
 return l0,lt,L,V,fl,vv,force,-force*fl*vv,bias

def run():
 import mujoco as mj
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 from scipy.interpolate import CubicSpline
 import xml.etree.ElementTree as ET
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'AUDIT.npz').exists(),'refuse science overwrite'
 calls=dict(model_compile=0,forward=0,muscleGain_actual=0,muscleGain_zero_velocity=0,muscleBias=0,body_step=0,neural_step=0,network=0,training=0,optimization=0,LP=0,contactForce=0)
 paths=[Path(__file__),R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',R/'tools_mirror_muscles.py',R/'audit_muscle_geometry_provenance.py',R/'prepare_standard_gait_reference.py',E/'COMPILED_MODEL.npz',E/'TRAIN_000_BASELINE_AUDIT.npz',E/'IDENTITY_AND_MAPPING.json',H/'formulas.py',H/'engine_util_misc.c',REF/'REFERENCE_METADATA.json',REF/'STANDARD_GAIT_FK.npz',REF/'prepare_standard_gait_reference.py']+[B/'RUN'/n for n in ['MODEL.xml','BODY.npz','GAIT.npz','GAIT_RETARGET.json','STANCE_SUPPORT_RETARGET.json','GAINS.json','MODEL_CHANGES.json','MAPPING.json','run_physical_walk_prewarm.py','HANDOVER_BEFORE.npz','HANDOVER_AFTER.npz']]
 hashes={str(p):sha(p) for p in paths};assert hashes[str(B/'RUN/MODEL.xml')]==SHA
 dump('SOURCE_HASHES_BEFORE.json',hashes)
 dump('PREREGISTRATION.json',dict(head='0f6da6442f0c0b8cf756c20de1828a006cee315d',scope='S1/S2/S4 original3470/root888 saved reference1.1..1.3,401frames at0.5ms; no new trajectories',thresholds=dict(gain_absolute=1e-12,relative_actual_to_stationary_gain=1e-3,activation_floor_atol=1e-12,joint_range_fraction_near_edge=.01,torque_envelope_abs=1e-12),threshold_meaning='engineering numerical counts, not physiology or behavior gates; no retuning',activation_envelope='independent90 a in[1e-4,1]; mapped66 tied hypothetical equal activation, unresolved fixed1e-4; coordinate extrema are separate relaxations, not jointly feasible vectors. Actual same-command activation need not be tied.',budget='1compile; at most401 saved forward, exact cache reuse required; actual+vel0 native gain per frame/muscle, native bias; zero integration/network/LP',boundary='BODY1.3 is AFTER withdrawal: q/v/act unchanged, command and assist may differ from last assisted prestep; preserved not relabeled',force_truth='Original saved muscle_force/foot_force; cloned qstateforward not exact saved warmstart constraint solve',formula_source=str(H/'engine_util_misc.c')+':844..945',source_sha256=hashes))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 try:
  b=ar(B/'RUN/BODY.npz');idx=np.arange(2200,2601);assert error(b['time'][idx],1.1+np.arange(401)*.0005)<1e-9
  a={k:v[idx] for k,v in b.items()};a['source_indices']=idx
  old=ar(E/'TRAIN_000_BASELINE_AUDIT.npz');meta=js(E/'IDENTITY_AND_MAPPING.json');assert meta['model_sha256']==SHA
  m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();calls['model_compile']+=1
  compiled={k:getattr(m,k).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)}
  previous=ar(E/'COMPILED_MODEL.npz');match={k:np.array_equal(v,previous[k]) for k,v in compiled.items()};assert all(match.values())
  np.savez_compressed(O/'COMPILED_ACTUAL_ARRAYS.npz',**compiled);dump('COMPILED_REUSE_IDENTITY.json',match)
  nj={k:int(getattr(m,k)) for k in dir(m) if k.startswith('n') and isinstance(getattr(m,k),(int,np.integer))}
  anames=[m.actuator(i).name for i in range(m.nu)];jids=np.flatnonzero(m.jnt_type==mj.mjtJoint.mjJNT_HINGE);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];jnames=[m.joint(int(j)).name for j in jids]
  assert (m.nq,m.nv,m.nu,m.na,m.nmesh,m.nmocap,m.neq)==(49,48,90,90,71,0,0)
  mapping=js(B/'RUN/MAPPING.json')['rows'];assert anames==[r['actuator'] for r in mapping]
  groups={};pg=np.full(90,-1,int)
  for r in mapping:
   key=tuple(r['local_indices'])
   if key:
    if key not in groups:groups[key]=len(groups)
    pg[r['actuator_index']]=groups[key]
  assert len(groups)==66 and np.sum(pg<0)==16
  partial=np.array([r['status']=='partial' for r in mapping]);assert partial.sum()==7
  keys=['length','velocity','force','gain','bias','gain_zero_velocity','moment','qfrc_actuator','offline_com','offline_foot_origins']
  collected={k:[] for k in keys};reused=[]
  for t,i in enumerate(idx):
   hit=[k for k,tt in enumerate(old['time']) if abs(tt-b['time'][i])<1e-10 and all(np.array_equal(old[s][k],b[s][i]) for s in ['qpos','qvel','activation','command'])]
   if hit:
    k=hit[0];length=old['length'][k];velocity=old['velocity'][k];force=old['force'][k];moment=old['moment_90x48'][k];qfrc=old['qfrc_actuator'][k];com=old['offline_com'][k];feet=old['foot_marker_position'][k];reused.append(dict(frame=t,source=str(E/'TRAIN_000_BASELINE_AUDIT.npz'),source_row=k,all_state_exact=True))
   else:
    d=mj.MjData(m)
    for dest,src in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('qfrc_applied','assist'),('xfrc_applied','external')]:getattr(d,dest)[:]=b[src][i]
    d.time=b['time'][i];mj.mj_forward(m,d);calls['forward']+=1
    length=d.actuator_length.copy();velocity=d.actuator_velocity.copy();force=d.actuator_force.copy();qfrc=d.qfrc_actuator.copy();com=d.subtree_com[m.body('Thorax').id].copy();feet=d.xpos[[m.body(l+'Tarsus5').id for l in LEGS]].copy()
    moment=np.zeros((90,48))
    if d.actuator_moment.shape==moment.shape:moment[:]=d.actuator_moment
    else:
     for u in range(90):
      start=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);moment[u,d.moment_colind[start:start+n]]=d.actuator_moment[start:start+n]
   gain=np.array([mj.mju_muscleGain(length[u],velocity[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain_actual']+=90
   zero=np.array([mj.mju_muscleGain(length[u],0.,m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain_zero_velocity']+=90
   bias=np.array([mj.mju_muscleBias(length[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_biasprm[u,:9]) for u in range(90)]);calls['muscleBias']+=90
   for k,v in dict(length=length,velocity=velocity,force=force,gain=gain,bias=bias,gain_zero_velocity=zero,moment=moment,qfrc_actuator=qfrc,offline_com=com,offline_foot_origins=feet).items():collected[k].append(v)
  a.update({k:np.asarray(v) for k,v in collected.items()});a['pool_group']=pg;a['partial_mask']=partial
  param=m.actuator_gainprm;lr=m.actuator_lengthrange
  tt=np.array([[terms(a['length'][t,u],a['velocity'][t,u],lr[u],m.actuator_acc0[u],param[u]) for u in range(90)] for t in range(401)])
  for j,k in enumerate(['L0','LT','normalized_fiber_length','normalized_fiber_velocity','FL','FV','effective_Fmax','formula_gain','formula_bias']):a[k]=tt[:,:,j]
  a['range_fraction']=(a['length']-lr[:,0])/(lr[:,1]-lr[:,0]);a['active_force']=a['gain']*a['activation']
  a['force_length_zeroed']=abs(a['gain_zero_velocity'])<=1e-12
  a['force_velocity_zeroed']=(abs(a['gain'])<=1e-12)&~a['force_length_zeroed']
  a['force_velocity_suppressed']=(abs(a['gain'])<=1e-3*abs(a['gain_zero_velocity']))&~a['force_length_zeroed']
  a['native_gain_nearzero']=abs(a['gain'])<=1e-12
  a['activation_at_floor']=abs(a['activation']-1e-4)<=1e-12
  a['command_zero']=abs(a['command'])<=1e-12
  a['moment_42x90']=a['moment'][:,:,va].transpose(0,2,1);A=a['moment_42x90']*a['gain'][:,None,:];a['coefficient_42x90']=A
  a['active_joint_torque']=np.einsum('tju,tu->tj',a['moment_42x90'],a['active_force']);passive=np.einsum('tju,tu->tj',a['moment_42x90'],a['bias']);a['passive_joint_torque']=passive
  a['total_joint_torque']=a['qfrc_actuator'][:,va];a['historical_assist']=a['assist'][:,va];a['historical_total_input']=a['total_joint_torque']+a['historical_assist']
  floor=passive+A.sum(2)*1e-4;a['independent_lower']=floor+np.minimum(A,0).sum(2)*.9999;a['independent_upper']=floor+np.maximum(A,0).sum(2)*.9999
  poolA=np.stack([A[:,:,pg==g].sum(2) for g in range(66)],axis=2);a['pool_coefficient_42x66']=poolA
  a['tied_lower']=floor+np.minimum(poolA,0).sum(2)*.9999;a['tied_upper']=floor+np.maximum(poolA,0).sum(2)*.9999
  for family in ['independent','tied']:
   for target in ['historical_assist','historical_total_input']:
    a[family+'_'+target+'_outside']=(a[target]<a[family+'_lower']-1e-12)|(a[target]>a[family+'_upper']+1e-12)
  gait=ar(B/'RUN/GAIT.npz');phase=np.maximum(0,a['time']-.15)*10*np.pi%(2*np.pi);a['phase_rad']=phase;a['reference_joint_target']=CubicSpline(gait['phase'],gait['qpos_hinges'],axis=0,bc_type='periodic')(phase);a['reference_joint_error']=a['qpos'][:,qa]-a['reference_joint_target']
  a['joint_range_fraction']=(a['qpos'][:,qa]-m.jnt_range[jids,0])/(m.jnt_range[jids,1]-m.jnt_range[jids,0]);a['joint_near_range']=(a['joint_range_fraction']<=.01)|(a['joint_range_fraction']>=.99)
  a['joint_names']=np.array(jnames);a['actuator_names']=np.array(anames);a['joint_qpos_addresses']=qa;a['joint_dof_addresses']=va
  errors=dict(saved_force=error(a['force'],a['muscle_force']),native_gain_bias_force=error(a['gain']*a['activation']+a['bias'],a['force']),scalar_formula_gain=error(a['formula_gain'],a['gain']),scalar_formula_bias=error(a['formula_bias'],a['bias']),zero_velocity_FL=error(-a['effective_Fmax']*a['FL'],a['gain_zero_velocity']),moment_savedforce_qfrc=error(np.einsum('tuv,tu->tv',a['moment'],a['muscle_force']),a['qfrc_actuator']),active_passive_joint=error(a['active_joint_torque']+passive,a['total_joint_torque']),com_saved=error(a['offline_com'],a['com']),foot_origin_saved=error(a['offline_foot_origins'],a['foot_origins']),fiber_LT_identity=error((a['length']-a['LT'])/a['L0'],a['normalized_fiber_length']))
  assert max(errors.values())<1e-9,errors
  np.savez_compressed(O/'AUDIT.npz',**a)
  dump('REUSED_STATES.json',dict(count=len(reused),rows=reused,compiled_arrays_exact=len(match),old_scope='s74 different body/history;083 sparse poses different model key; current early cache exact matching root888/model3470 q/v/act/ctrl only. No existing identical401-frame census found in inspected tools and archives. Sparse capacity tables lack length/velocity and cannot supply this decomposition.'))
  root=ET.parse(B/'RUN/MODEL.xml').getroot();sites={s.get('name'):dict(body=body.get('name'),attributes=s.attrib) for body in root.iter('body') for s in body.findall('site')};tendons={x.get('name'):dict(attributes=x.attrib,path=[dict(tag=c.tag,attributes=c.attrib,site_definition=sites.get(c.get('site'))) for c in x]) for x in root.find('tendon')}
  muscles=[]
  for u,r in enumerate(mapping):
   xml=list(root.find('actuator'))[u]
   muscles.append(dict(index=u,name=anames[u],leg=r['leg'],mapping_status=r['status'],pool_group=int(pg[u]),pool_local_indices=r['local_indices'],pool_root_ids=r['tier_root_ids'],missing_roots=r['missing_tier_roots'],xml=xml.attrib,tendon=tendons.get(xml.get('tendon')),lengthrange=lr[u],gainprm=param[u],biasprm=m.actuator_biasprm[u],dynprm=m.actuator_dynprm[u],ctrlrange=m.actuator_ctrlrange[u],actrange=m.actuator_actrange[u],forcerange=m.actuator_forcerange[u],ctrllimited=int(m.actuator_ctrllimited[u]),actlimited=int(m.actuator_actlimited[u]),forcelimited=int(m.actuator_forcelimited[u]),acc0=m.actuator_acc0[u],length0=m.actuator_length0[u],L0=a['L0'][0,u],LT=a['LT'][0,u],effective_Fmax=a['effective_Fmax'][0,u],source_deficit='Inherited foreleg template; LF/RF source anatomy does not establish measured six-leg force-length/velocity calibration. Other legs copied/scaled; mid/hind Trochanter site->Femur hypothesis. Keypose lengthrange rescale is engineering operating-fraction preservation.'))
  dump('MUSCLE_PATHS_PARAMETERS_SOURCE_DEFICITS.json',muscles)
  dump('MODEL_METADATA.json',dict(model_sha256=SHA,mujoco_version=mj.__version__,allcounts=nj,actuator_names=anames,joint_names=jnames,mapping=mapping,pool_group=pg,shared_pool_groups=[dict(group=g,muscles=np.flatnonzero(pg==g)) for g in range(66) if np.sum(pg==g)>1],joint_range=m.jnt_range[jids],joint_springref=m.qpos_spring[qa],joint_stiffness=m.jnt_stiffness[jids],weight=float(m.body_mass.sum()*abs(m.opt.gravity[2])),formula=dict(L0='(lengthrange[1]-lengthrange[0])/(gainprm[1]-gainprm[0])',LT='lengthrange[0]-gainprm[0]*L0',L='gainprm[0]+(length-lengthrange[0])/L0=(length-LT)/L0',V='velocity/(L0*gainprm[6])',gain='-Fmax*FL(L)*FV(V)',bias='native separate passive length curve',sign='negative velocity=path shortening; gain and tensile force negative; moment.T maps to generalized torque',interpretation='Effective native fiber coordinate, not measured fascicle length; LT is equivalent fixed offset, not explicit compliant tendon slack identification'),boundary='401 includes1.3 after withdrawal; last400 rows1.1..1.2995 auxiliary; endpoint command/assist preserved.',limitations='No integration;0.5ms grid cannot resolve12.5us events; no per-frame warmstart => constraint force not reconstructed; original saved footforce retained. No claim biological calibration or stance feasibility.'))
  summarize(a,errors,calls)
  unchanged=hashes=={str(p):sha(p) for p in paths};assert unchanged
  dump('SOURCE_HASHES_AFTER.json',hashes);dump('EXECUTION.json',dict(calls=calls,exit_code=0,source_unchanged=unchanged,reused_frames=len(reused)))
  print(json.dumps(dict(errors=errors,calls=calls,reused_frames=len(reused))))
 except Exception:
  dump('EXECUTION_FAILURE.json',dict(calls=calls,error=traceback.format_exc(),exit_code=1));raise

def summarize(a,errors,calls):
 meta=js(O/'MODEL_METADATA.json');mapping=meta['mapping'];W=meta['weight'];pg=a['pool_group']
 masks={'all90':np.ones(90,bool),'unresolved16':pg<0,'partial7':a['partial_mask']}
 masks.update({l:np.array([r['leg']==l for r in mapping]) for l in LEGS})
 masks.update({'pool_'+str(g):pg==g for g in range(66)})
 maskkeys=['native_gain_nearzero','force_length_zeroed','force_velocity_zeroed','force_velocity_suppressed','activation_at_floor','command_zero']
 phases={'cycle_auxiliary_400':np.arange(400),'withdrawal_boundary':np.array([400])}
 for k in range(4):phases['time_quarter_'+str(k)]=np.arange(k*100,(k+1)*100)
 details={};rows=[]
 for title,ix in phases.items():
  groups={}
  for group,mask in masks.items():
   n=int(mask.sum());view=np.ix_(ix,mask)
   groups[group]=dict(muscles=n,samples=len(ix),counts={k:int(a[k][view].sum()) for k in maskkeys},activation_mean=float(a['activation'][view].mean()),command_mean=float(a['command'][view].mean()),active_force_L1_mean=float(abs(a['active_force'][view]).sum(1).mean()),L_min=float(a['normalized_fiber_length'][view].min()),L_max=float(a['normalized_fiber_length'][view].max()),V_min=float(a['normalized_fiber_velocity'][view].min()),V_max=float(a['normalized_fiber_velocity'][view].max()))
  details[title]=dict(groups=groups,assist_L1_mean=float(abs(a['historical_assist'][ix]).sum(1).mean()),active_torque_L1_mean=float(abs(a['active_joint_torque'][ix]).sum(1).mean()),joint_target_RMS=float(np.sqrt(np.mean(a['reference_joint_error'][ix]**2))),foot_Fz_over_W_mean=float(a['foot_force'][ix,:,2].sum(1).mean()/W),nonfoot_samples=int((a['nonfoot_count'][ix]>0).sum()),near_joint_range_coordinate_samples=int(a['joint_near_range'][ix].sum()),envelope_outside_coordinate_samples={family+'_'+target:int(a[family+'_'+target+'_outside'][ix].sum()) for family in ['independent','tied'] for target in ['historical_assist','historical_total_input']})
  for j,name in enumerate(a['joint_names']):
   rows.append(dict(phase=title,joint=str(name),assist_mean=float(a['historical_assist'][ix,j].mean()),assist_L1_mean=float(abs(a['historical_assist'][ix,j]).mean()),muscle_mean=float(a['total_joint_torque'][ix,j].mean()),active_mean=float(a['active_joint_torque'][ix,j].mean()),independent_lo_min=float(a['independent_lower'][ix,j].min()),independent_hi_max=float(a['independent_upper'][ix,j].max()),total_input_outside=int(a['independent_historical_total_input_outside'][ix,j].sum()),tied_total_input_outside=int(a['tied_historical_total_input_outside'][ix,j].sum()),range_fraction_min=float(a['joint_range_fraction'][ix,j].min()),range_fraction_max=float(a['joint_range_fraction'][ix,j].max()),joint_error_RMS=float(np.sqrt(np.mean(a['reference_joint_error'][ix,j]**2)))))
 table('JOINT_PHASE_QUANTITIES.csv',rows)
 rows=[]
 for u,name in enumerate(a['actuator_names']):
  row=dict(index=u,name=str(name),leg=mapping[u]['leg'],status=mapping[u]['status'],pool=int(pg[u]),Lmin=float(a['normalized_fiber_length'][:,u].min()),Lmax=float(a['normalized_fiber_length'][:,u].max()),Vmin=float(a['normalized_fiber_velocity'][:,u].min()),Vmax=float(a['normalized_fiber_velocity'][:,u].max()),activation_mean=float(a['activation'][:,u].mean()),active_abs_max=float(abs(a['active_force'][:,u]).max()))
  row.update({k+'_counts':int(a[k][:,u].sum()) for k in maskkeys});rows.append(row)
 table('MUSCLE_COUNTS.csv',rows)
 dump('RESULT.json',dict(errors=errors,calls=calls,phases=details,whole401_counts={k:int(a[k].sum()) for k in maskkeys},interpretation='Zero stationary gain=>force-length numerical zero. Nonzero stationary and zero actual=>force-velocity numerical zero. Suppressed ratio<=1e-3 excludes stationaryzero; counts overlap by definition explicitly. Envelope comparison reproduces historical controller inputs, not necessarytrue torque/support impossibility.'))

def verify():
 # Scalar verification deliberately does not import MuJoCo or the run helper.
 a=ar(O/'AUDIT.npz');m=ar(O/'COMPILED_ACTUAL_ARRAYS.npz');n=401;maxerr={k:0. for k in ['gain','bias','force','stationary','moment','fiber_coordinates','envelope','pool_envelope']};counts=dict(force_length_zeroed=0,force_velocity_zeroed=0,force_velocity_suppressed=0,native_gain_nearzero=0)
 for t in range(n):
  for u in range(90):
   l0,lt,L,V,fl,fv,F,g,b=terms(float(a['length'][t,u]),float(a['velocity'][t,u]),m['actuator_lengthrange'][u],float(m['actuator_acc0'][u]),m['actuator_gainprm'][u])
   for k,v in [('gain',abs(g-a['gain'][t,u])),('bias',abs(b-a['bias'][t,u])),('force',abs(g*a['activation'][t,u]+b-a['muscle_force'][t,u])),('stationary',abs(-F*fl-a['gain_zero_velocity'][t,u])),('fiber_coordinates',abs((float(a['length'][t,u])-lt)/l0-a['normalized_fiber_length'][t,u]))]:maxerr[k]=max(maxerr[k],float(v))
   z=abs(-F*fl)<=1e-12;v=abs(g)<=1e-12 and not z;s=abs(g)<=1e-3*abs(-F*fl) and not z
   for k,bit in [('force_length_zeroed',z),('force_velocity_zeroed',v),('force_velocity_suppressed',s),('native_gain_nearzero',abs(g)<=1e-12)]:assert bit==bool(a[k][t,u]);counts[k]+=int(bit)
  for j in range(42):
   torque=math.fsum(float(a['moment_42x90'][t,j,u])*float(a['muscle_force'][t,u]) for u in range(90));maxerr['moment']=max(maxerr['moment'],abs(torque-float(a['total_joint_torque'][t,j])))
   coeff=[float(a['moment_42x90'][t,j,u])*float(a['gain'][t,u]) for u in range(90)];passive=math.fsum(float(a['moment_42x90'][t,j,u])*float(a['bias'][t,u]) for u in range(90));base=passive+1e-4*math.fsum(coeff)
   pool=[math.fsum(coeff[u] for u in range(90) if a['pool_group'][u]==g) for g in range(66)]
   for family,C,key in [('independent',coeff,'envelope'),('tied',pool,'pool_envelope')]:
    lo=base+.9999*math.fsum(min(0,x) for x in C);hi=base+.9999*math.fsum(max(0,x) for x in C);maxerr[key]=max(maxerr[key],abs(lo-a[family+'_lower'][t,j]),abs(hi-a[family+'_upper'][t,j]))
 assert max(maxerr.values())<1e-9,maxerr
 assert js(O/'RESULT.json')['whole401_counts']=={**js(O/'RESULT.json')['whole401_counts'],**counts}
 hashes=js(O/'SOURCE_HASHES_BEFORE.json');assert all(sha(p)==v for p,v in hashes.items())
 dump('INDEPENDENT_SCALAR_VERIFICATION.json',dict(process='fresh process, scalar loops math.fsum, no MuJoCo import/compile/forward',max_errors=maxerr,whole_counts=counts,frames=n,muscles=90,joints=42,source_hashes_unchanged=True,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0),exit_code=0));print(json.dumps(maxerr))

if __name__=='__main__':
 {'run':run,'verify':verify}[sys.argv[1]]()
