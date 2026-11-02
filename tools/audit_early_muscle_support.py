"""S2/S4: saved-state muscle/support audit; no integration/network/training.
run: one original MODEL3470 compile, exactly 101 saved frames per variant.
verify: independent process, arrays/stdlib only, no MuJoCo import.
Parent reviews before an external one-time seal; this tool never seals.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,math,traceback,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
O=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z')
S=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
LEGS=['LF','LM','LH','RF','RM','RH'];LABELS=['TRAIN_000_BASELINE','TRAIN_017_MINUS']
MODEL_SHA='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
def js(p):return json.loads(Path(p).read_text())
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def dump(p,x):
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def err(a,b):return float(np.max(abs(np.asarray(a)-np.asarray(b))))
def rotation(q):
 q=np.asarray(q)/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def so3log(M):
 v=np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])*.5
 a=math.atan2(float(np.linalg.norm(v)),float((np.trace(M)-1)/2))
 return v if a<1e-12 else v*a/np.linalg.norm(v)
def statistics(a,weight,names):
 t=a['relative_time_s'];phases={}
 for title,lo,hi in [('prewarm_last20ms',-.02,0),('release_0_5ms',0,.005),('release_5_10ms',.005,.01),('release_10_18ms',.01,.018),('release_18_30ms',.018,.03000001)]:
  k=(t>=lo-1e-10)&(t<hi-1e-10 if hi<=.018 else t<=hi);ix=np.flatnonzero(k)
  # Use nominal grid bounds rather than floating-point body clocks.
  if title=='prewarm_last20ms':ix=np.arange(40)
  if title=='release_0_5ms':ix=np.arange(40,50)
  if title=='release_5_10ms':ix=np.arange(50,60)
  if title=='release_10_18ms':ix=np.arange(60,76)
  if title=='release_18_30ms':ix=np.arange(76,101)
  q=a['qfrc_actuator'][ix,6:];active=a['qfrc_active'][ix,6:];passive=a['qfrc_muscle_passive'][ix,6:]
  target=a['removed_assist'][6:];order=np.argsort(abs(target))[::-1][:10]
  phases[title]=dict(frames=len(ix),footFz_over_W_mean=a['native_foot_force'][ix,:,2].mean(0)/weight,footFz_over_W_first=a['native_foot_force'][ix[0],:,2]/weight,footFz_over_W_last=a['native_foot_force'][ix[-1],:,2]/weight,COMz_first_last=a['native_com'][[ix[0],ix[-1]],2],COM_velocity_mean=a['com_velocity'][ix].mean(0),root_SO3_last=a['root_SO3_relative'][ix[-1]],act_mean=float(a['activation'][ix].mean()),act_max=float(a['activation'][ix].max()),act_lt001_count_mean=float((a['activation'][ix]<.01).sum(1).mean()),command_mean=float(a['command'][ix].mean()),active_torque_L1_mean=float(abs(active).sum(1).mean()),passive_muscle_torque_L1_mean=float(abs(passive).sum(1).mean()),per_leg_active_torque_L1_mean=[float(abs(active[:,[j for j,n in enumerate(names) if n.startswith(l)]]).sum(1).mean()) for l in LEGS],assist_joint_comparison=[dict(joint=names[j],removed_assist=float(target[j]),muscle_total_mean=float(q[:,j].mean()),muscle_active_mean=float(active[:,j].mean()),muscle_passive_mean=float(passive[:,j].mean()),joint_velocity_mean=float(a['qvel'][ix,6+j].mean())) for j in order])
 return phases

def run():
 import mujoco
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 counts=dict(model_compile=0,forward=0,contactForce=0,Jacobian=0,kinematics_explicit=0,body_step=0,neural_step=0,network=0,training=0,LP=0,muscleGain=0,muscleBias=0)
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'AUDIT.npz').exists() and not (O/'RESULT.json').exists(),'refuse overwriting science'
 paths=[Path(__file__),R/'GOAL_TRACKER.zh-CN.md',R/'GOAL_ROADMAP.zh-CN.md',S/'EXACT_GATE.json',S/'PARENT_REVIEW.json',B/'IDENTITIES.npz',B/'RUN/MAPPING.json',S/'PARAMETER_GROUPS.npz']
 for label in LABELS+['GATE_NEUTRAL_50MS']:
  paths.extend(S/label/n for n in ['RUN/MODEL.xml','RUN/BODY.npz','RUN/MAPPING.json','PARAMETERS.npz','RASTER.npz','RUN/HANDOVER_BEFORE.npz','RUN/HANDOVER_AFTER.npz','RUN/NEURAL_RECORDS.json','RUN/PREWARM_NEURAL_RECORDS.json','BODYLOSS_NATIVE.npz'])
 frozen={str(p):sha(p) for p in paths};dump(O/'SOURCE_HASHES_BEFORE.json',frozen)
 dump(O/'PREREGISTRATION.json',dict(stage='S2/S4 saved-state early muscle support, research only',head='4598bae93fe58e4a21d7507558c77a8ba3fd795e',variants=LABELS,window_absolute_s=[1.28,1.33],grid_s=.0005,frames_per_variant=101,zero_body_neural_training=True,ground_truth='Original saved foot_force, COM and muscle_force; BODY no per-frame warmstart, never claim exactsolvercontactstate',reconstruction='Fresh clone MjData per saved frame with qpos/qvel/act/ctrl/applied inputs; native gain*activation+bias; no state integration; spot compile1 forward6 previously disclosed',phases_ms=[[0,5],[5,10],[10,18],[18,30]],assist='Saved lastprewarm BODY.assist plus full HANDOVER_BEFORE.qfrc_applied: historical algorithm torque, not neural truth or an independently inferred required load',model=MODEL_SHA,legacy='Old root_id stage5 muscle histories excluded; LP evidence citation only, no solves',checks=dict(force_atol=1e-10,COM_atol=1e-12,matrix_atol=1e-10),source_hashes=frozen))
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 try:
  p=S/LABELS[0]/'RUN/MODEL.xml';assert sha(p)==MODEL_SHA
  m=_load_mjcf(str(p)).compile();counts['model_compile']+=1
  assert (m.nq,m.nv,m.nu,m.na,m.nmesh,m.nmocap,m.neq)==(49,48,90,90,71,0,0)
  assert np.all(m.actuator_gaintype==mujoco.mjtGain.mjGAIN_MUSCLE) and np.all(m.actuator_biastype==mujoco.mjtBias.mjBIAS_MUSCLE)
  compiled={k:np.asarray(getattr(m,k)).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)};compiled['gravity']=m.opt.gravity.copy();np.savez_compressed(O/'COMPILED_MODEL.npz',**compiled)
  W=float(m.body_mass.sum()*abs(m.opt.gravity[2]));thorax=m.body('Thorax').id
  joints=[m.joint(i).name for i in range(1,m.njnt)];anames=[m.actuator(i).name for i in range(90)]
  geomnames=[m.geom(i).name or '' for i in range(m.ngeom)];planes=m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
  geomlegs=np.array([next((j for j,l in enumerate(LEGS) if n.startswith(l) and 'Tars' in n),-1) for n in geomnames]);bids=[m.body(l+'Tarsus5').id for l in LEGS]
  mapping=js(B/'RUN/MAPPING.json');roots=arr(B/'IDENTITIES.npz')['root_ids'];current_roots=arr(S/'PARAMETER_GROUPS.npz')['root_ids'];assert np.array_equal(roots,current_roots)
  poolkeys=[];poolgroup=np.full(90,-1,int)
  for row in mapping['rows']:
   i=row['actuator_index'];assert anames[i]==row['actuator'];assert row['tier_root_ids']==[int(roots[k]) for k in row['local_indices']]
   key=tuple(row['local_indices'])
   if key:
    if key not in poolkeys:poolkeys.append(key)
    poolgroup[i]=poolkeys.index(key)
  assert len(poolkeys)==66
  dump(O/'IDENTITY_AND_MAPPING.json',dict(model_sha256=MODEL_SHA,legs=LEGS,joint_names=joints,actuator_names=anames,mapping=mapping,shared_pool_ids=poolgroup,distinct_nonempty_pools=len(poolkeys),root_ids_source=str(B/'IDENTITIES.npz'),old_root_id_history_used=False,root_identity_exact=True,foot_marker='Tarsus5 body origin; geometry only, not a force application point',units='model native units; biological SI scale NOT_ESTABLISHED'))
  result={};allarrays={}
  gate=arr(S/'GATE_NEUTRAL_50MS/RUN/BODY.npz');base=arr(S/LABELS[0]/'RUN/BODY.npz')
  gate_exact={k:np.array_equal(v[:len(gate[k])],gate[k]) for k,v in base.items()};assert all(gate_exact.values());dump(O/'GATE_IDENTITY.json',gate_exact)
  for label in LABELS:
   assert sha(S/label/'RUN/MODEL.xml')==MODEL_SHA and js(S/label/'RUN/MAPPING.json')==mapping
   b=base if label==LABELS[0] else arr(S/label/'RUN/BODY.npz');times=1.28+np.arange(101)*.0005;idx=np.array([np.argmin(abs(b['time']-t)) for t in times]);assert err(b['time'][idx],times)<1e-10 and len(set(idx))==101
   a={k:v[idx] for k,v in b.items()};a['native_foot_force']=a.pop('foot_force');a['native_com']=a.pop('com');a['saved_muscle_force']=a.pop('muscle_force');a['source_indices']=idx;a['relative_time_s']=np.arange(-40,61)*.0005
   before=arr(S/label/'RUN/HANDOVER_BEFORE.npz');after=arr(S/label/'RUN/HANDOVER_AFTER.npz');a['removed_assist']=before['qfrc_applied'];a['lastprewarm_assist']=a['assist'][39];assert np.all(a['assist'][40:]==0) and np.all(after['qfrc_applied']==0)
   collected={k:[] for k in ['force','actuator_gain','actuator_bias','active_force','passive_force','length','velocity','moment_90x48','qfrc_actuator','qfrc_active','qfrc_muscle_passive','qfrc_passive','qfrc_bias','offline_foot_force','offline_com','foot_marker_position','com_velocity','root_SO3_relative','root_roll_pitch','offline_contact_count']}
   contactrows=[]
   for ii,i in enumerate(idx):
    d=mujoco.MjData(m)
    for dst,src in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('qfrc_applied','assist'),('xfrc_applied','external')]:getattr(d,dst)[:]=b[src][i]
    d.time=float(b['time'][i]);mujoco.mj_forward(m,d);counts['forward']+=1
    mom=np.zeros((90,48))
    if d.actuator_moment.shape==mom.shape:mom[:]=d.actuator_moment
    else:
     for u in range(90):
      adr=int(d.moment_rowadr[u]);n=int(d.moment_rownnz[u]);mom[u,d.moment_colind[adr:adr+n]]=d.actuator_moment[adr:adr+n]
    gain=np.array([mujoco.mju_muscleGain(d.actuator_length[u],d.actuator_velocity[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);counts['muscleGain']+=90
    bias=np.array([mujoco.mju_muscleBias(d.actuator_length[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_biasprm[u,:9]) for u in range(90)]);counts['muscleBias']+=90
    af=gain*d.act;pf=bias;ff=np.zeros((6,3));f=np.zeros(6)
    for cidx in range(d.ncon):
     c=d.contact[cidx];g1=int(c.geom1);g2=int(c.geom2)
     if not(planes[g1] or planes[g2]):continue
     mujoco.mj_contactForce(m,d,cidx,f);counts['contactForce']+=1
     gf=g2 if planes[g1] else g1;leg=int(geomlegs[gf]);force=(c.frame.reshape(3,3).T@f[:3])*(1 if planes[g1] else -1)
     if leg>=0:ff[leg]+=force
     contactrows.append([ii,cidx,g1,g2,leg,float(c.dist),*c.pos,*force])
    # Exact instantaneous COM velocity from saved qvel and mass-weighted COM Jacobian.
    jac=np.zeros((3,48));mujoco.mj_jacSubtreeCom(m,d,jac,thorax);counts['Jacobian']+=1
    rot=rotation(d.qpos[3:7]);relative=so3log(rotation(a['qpos'][40,3:7]).T@rot)
    vals=dict(force=d.actuator_force,actuator_gain=gain,actuator_bias=bias,active_force=af,passive_force=pf,length=d.actuator_length,velocity=d.actuator_velocity,moment_90x48=mom,qfrc_actuator=d.qfrc_actuator,qfrc_active=mom.T@af,qfrc_muscle_passive=mom.T@pf,qfrc_passive=d.qfrc_passive,qfrc_bias=d.qfrc_bias,offline_foot_force=ff,offline_com=d.subtree_com[thorax],foot_marker_position=d.xpos[bids],com_velocity=jac@d.qvel,root_SO3_relative=relative,root_roll_pitch=np.array([math.atan2(rot[2,1],rot[2,2]),math.atan2(-rot[2,0],math.hypot(rot[2,1],rot[2,2]))]),offline_contact_count=d.ncon)
    for k,v in vals.items():collected[k].append(np.asarray(v).copy())
   for k,v in collected.items():a[k]=np.asarray(v)
   a['moment_42x90']=a['moment_90x48'][:,:,6:].transpose(0,2,1);a['offline_ground_contacts']=np.asarray(contactrows).reshape(-1,12)
   a['foot_marker_velocity_fd']=np.gradient(a['foot_marker_position'],.0005,axis=0);a['native_COM_velocity_fd']=np.gradient(a['native_com'],.0005,axis=0)
   errors=dict(muscle_force_saved=err(a['force'],a['saved_muscle_force']),gain_bias_force=err(a['force'],a['active_force']+a['passive_force']),matrix_qfrc=err(np.einsum('tuv,tu->tv',a['moment_90x48'],a['force']),a['qfrc_actuator']),activepassive_qfrc=err(a['qfrc_active']+a['qfrc_muscle_passive'],a['qfrc_actuator']),foot_force_saved=err(a['offline_foot_force'],a['native_foot_force']),COM_saved=err(a['offline_com'],a['native_com']))
   assert errors['muscle_force_saved']<1e-10 and errors['gain_bias_force']<1e-10 and errors['matrix_qfrc']<1e-10 and errors['COM_saved']<1e-12
   native=arr(S/label/'BODYLOSS_NATIVE.npz')['rows'];a['native_release_rows']=native[(native[:,0]>=1.3-1e-10)&(native[:,0]<=1.33+1e-10)]
   assert a['native_release_rows'].shape[0]==2401
   nr=a['native_release_rows'];first=float(nr[np.flatnonzero(nr[:,7]>0)[0],0]-1.3);nativephases={}
   for lo,hi in [(0,5),(5,10),(10,18),(18,30)]:
    mask=(nr[:,0]>1.3+lo/1000+1e-10)&(nr[:,0]<=1.3+hi/1000+1e-10);v=nr[mask];nativephases[f'{lo}_{hi}ms']=dict(endpoints=len(v),footFz_over_W_mean=v[:,1:7].mean(0)/W,positive_nonfootFz_over_W_mean=float(v[:,7].mean()/W),positive_nonfoot_endpoints=int((v[:,7]>0).sum()),COMz_first_last=v[[0,-1],10])
   records=js(S/label/'RUN/PREWARM_NEURAL_RECORDS.json')+js(S/label/'RUN/NEURAL_RECORDS.json');records=[r for r in records if r['absolute_end_s']>=1.28-1e-10 and r['absolute_start_s']<=1.33+1e-10]
   dump(O/(label+'_COMMAND_CLOCK.json'),records)
   raster=arr(S/label/'RASTER.npz');tick=np.arange(len(raster['packed']));ticktime=.2+(tick+1)*.0005;keep=(ticktime>=1.28-1e-10)&(ticktime<=1.33+1e-10)
   a['MN_pulse_bits']=np.unpackbits(raster['packed'][keep],axis=1)[:,:len(roots)][:,raster['motor_indices']];a['MN_local_indices']=raster['motor_indices'];a['MN_root_ids']=roots[raster['motor_indices']];a['neural_endpoint_body_time']=ticktime[keep];a['neural_endpoint_time']=ticktime[keep]-.2;a['neural_release_time']=ticktime[keep]-1.3;a['pool_group']=poolgroup
   rel=a['neural_release_time'];mn=a['MN_pulse_bits'];pos=np.flatnonzero((rel>1e-10)&(mn.sum(1)>0));firstMN=float(rel[pos[0]]) if len(pos) else None
   result[label]=dict(errors=errors,first_nonfoot_ms=first*1000,first_postrelease_MN_endpoint_ms=None if firstMN is None else firstMN*1000,phases=statistics(a,W,joints),native_phases=nativephases,release=dict(act_mean=float(a['activation'][40].mean()),command_mean=float(a['command'][40].mean()),act_lt001=int((a['activation'][40]<.01).sum()),active_torque_L1=float(abs(a['qfrc_active'][40,6:]).sum()),passive_muscle_torque_L1=float(abs(a['qfrc_muscle_passive'][40,6:]).sum()),removedassist_L1=float(abs(a['removed_assist'][6:]).sum()),removedassist_lastpre_maxdiff=err(a['removed_assist'],a['lastprewarm_assist']),footFz_over_W=a['native_foot_force'][40,:,2]/W,COM_velocity=a['com_velocity'][40]),exact_solver_contact_state=False)
   np.savez_compressed(O/(label+'_AUDIT.npz'),**a)
   print(label,json.dumps(result[label]['errors']),flush=True)
  assert all(sha(p)==h for p,h in frozen.items());dump(O/'SOURCE_HASHES_AFTER.json',{str(p):sha(p) for p in frozen})
  final={k:np.asarray(getattr(m,k)) for k in compiled if k!='gravity'};assert all(np.array_equal(v,final[k] if k!='gravity' else m.opt.gravity) for k,v in compiled.items())
  dump(O/'RESULT.json',dict(weight=W,model_identity=MODEL_SHA,compiled_arrays_unchanged=True,variants=result,counts=counts,spot_counts=js(O/'SPOT_RESULT.json'),status='RECONSTRUCTION_VALIDATED_NO_AUTONOMOUS_SUPPORT_CLAIM'))
 except BaseException:
  dump(O/'FAILURE.json',dict(traceback=traceback.format_exc(),counts=counts));raise
 finally:dump(O/'RESOURCE_ACCOUNTING.json',counts)

def verify():
 # New process, no mujoco/flygym import. Scalar loops independently check products.
 meta=js(O/'IDENTITY_AND_MAPPING.json');res=js(O/'RESULT.json');checks={};maxmatrix=0.;maxsplit=0.
 roots=arr(B/'IDENTITIES.npz')['root_ids'];mapping=js(B/'RUN/MAPPING.json');keys=[]
 for row in mapping['rows']:
  assert row['tier_root_ids']==[int(roots[i]) for i in row['local_indices']]
  if row['local_indices'] and row['local_indices'] not in keys:keys.append(row['local_indices'])
 assert len(keys)==66 and sum(r['status']=='complete' for r in mapping['rows'])==67 and sum(r['status']=='partial' for r in mapping['rows'])==7
 for label in LABELS:
  a=arr(O/(label+'_AUDIT.npz'));b=arr(S/label/'RUN/BODY.npz');assert len(a['time'])==101
  for k,src in [('qpos','qpos'),('qvel','qvel'),('activation','activation'),('command','command'),('native_foot_force','foot_force'),('native_com','com'),('saved_muscle_force','muscle_force'),('assist','assist')]:assert np.array_equal(a[k],b[src][a['source_indices']])
  for t in range(101):
   for v in range(48):
    value=math.fsum(float(a['moment_90x48'][t,u,v])*float(a['force'][t,u]) for u in range(90));maxmatrix=max(maxmatrix,abs(value-float(a['qfrc_actuator'][t,v])))
   for u in range(90):maxsplit=max(maxsplit,abs(float(a['actuator_gain'][t,u])*float(a['activation'][t,u])+float(a['actuator_bias'][t,u])-float(a['force'][t,u])))
  assert np.array_equal(a['moment_42x90'],a['moment_90x48'][:,:,6:].transpose(0,2,1))
  for row in mapping['rows']:
   u=row['actuator_index'];group=meta['shared_pool_ids'][u];assert group==(-1 if not row['local_indices'] else keys.index(row['local_indices']))
  # Rebuild selected phase statistics and original high-rate event without model.
  phase=statistics(a,res['weight'],meta['joint_names']);phase=json.loads(json.dumps(phase,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()));assert phase==res['variants'][label]['phases']
  nr=a['native_release_rows'];first=(float(nr[np.flatnonzero(nr[:,7]>0)[0],0])-1.3)*1000;assert abs(first-res['variants'][label]['first_nonfoot_ms'])<1e-10
  assert np.array_equal(roots[a['MN_local_indices']],a['MN_root_ids']);assert err(a['neural_endpoint_body_time']-.2,a['neural_endpoint_time'])==0
  counts=np.unpackbits(arr(S/label/'RASTER.npz')['packed'],axis=1)[:,:len(roots)];idx=np.rint((a['neural_endpoint_body_time']-.2)/.0005).astype(int)-1;assert np.array_equal(counts[idx][:,a['MN_local_indices']],a['MN_pulse_bits'])
  checks[label]=dict(source_state_exact=True,phase_statistics_exact=True,native_event_exact=True,root_and_pulses_exact=True)
 assert maxmatrix<1e-10 and maxsplit<1e-10
 frozen=js(O/'SOURCE_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in frozen.items());assert frozen==js(O/'SOURCE_HASHES_AFTER.json')
 dump(O/'INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,checks=checks,matrix_scalar_maxerr=maxmatrix,gain_bias_scalar_maxerr=maxsplit,shared_pools=66,complete=67,partial=7,unresolved=16,source_hashes_unchanged=True,model_compile=0,forward=0,body_step=0,neural_step=0,network=0,method='fresh process scalar fsum matrix products; original state/root/pulses/nativeevent; phase recomputation shared helper explicitly disclosed'))
 print('INDEPENDENT_ARRAY_VERIFICATION_PASS',maxmatrix,maxsplit)
if __name__=='__main__':
 {'run':run,'verify':verify}[sys.argv[1]]()
