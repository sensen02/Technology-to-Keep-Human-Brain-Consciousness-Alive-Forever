"""Saved native force/momentum audit; no simulation stepping. Parent reviews before seal."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,math,csv,shutil,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
O=Path('/home/sensen/Desktop/engine修复验收/s2-handover-support-dynamics/20261102T071409Z')
C=Path('/home/sensen/Desktop/engine修复验收/s2-early-muscle-support/20261102T053639Z/COMPILED_MODEL.npz')
LEGS=['LF','LM','LH','RF','RM','RH']
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def dump(p,x):
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def tab(p,cols,values):
 with Path(p).open('w') as f:
  w=csv.writer(f);w.writerow(cols);w.writerows(values)
def prepare():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'PREREGISTRATION.json').exists()
 z=arr(C);mass=float(z['body_mass'].sum());g=z['gravity'];assert abs(mass-z['body_subtreemass'][2])<1e-18
 paths=[Path(__file__),C,B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'CONTACT_SCHEMA.json',B/'RUN/HANDOVER_BEFORE.npz',B/'RUN/HANDOVER_AFTER.npz',R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',R/'tools/run_banc888_physical_handover.py',R/'run_physical_walk_prewarm.py']+sorted((B/'NATIVE').glob('*.npz'))
 frozen={str(p):sha(p) for p in paths}
 dump(O/'SOURCE_HASHES_BEFORE.json',frozen);shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 dump(O/'STATIC_DATA_CHECK.json',dict(total_mass=mass,thorax_subtree_mass=z['body_subtreemass'][2],thorax_own_mass=z['body_mass'][2],rootfree_bodyid=z['jnt_bodyid'][0],gravity=g,weight=mass*abs(g[2]),COM_definition='saved d.subtree_com[Thorax]; subtree contains every positive model mass',native_qvel=False,body_qvel_grid_s=.0005,source=str(C),no_model_calls=True))
 dump(O/'PREREGISTRATION.json',dict(stage='S1/S2/S5 mechanism diagnosis only',head='2ccb699964a084c848a3652bd44d05b10d1655f0',baseline=str(B),native_window_s=[1.2,1.32],body_window_s=[1.2,1.32],saved_frames=241,model_compile=1,forward=241,COMJac=241,foot_body_Jac=1446,body_step=0,neural_step=0,LP=0,optimization=0,training=0,source_hashes=frozen,quadratures=['left','right','trapezoid'],quadrature_definition='unshifted physical endpoint forces; dv=sum a_endpoint dt per chosen quadrature, z integrates predicted v by trapezoid; analytic gravity term exact. No fitted evaluation offset.',acceleration='(signed sum foot+nonfoot Fz / W - 1)*abs(gz), full fly COM only; internal joint PD cannot change total linear momentum; verify root/external zero',events='first native candidate, per-contact positive ground nonfoot, aggregate signed positive NF separately; deadband0 and 1e-8W; floor membership required excludes self contacts',references='analytic z0+v0*t-g*t²/2 and z0+v0*t (groundFz=W), not new simulations or unique cause evidence',geometry='original Tarsus5 origin world/body position/velocity; no material slip inference; mesh vertex minimum world height geometric only, no LP/support recomputation',tolerances=dict(native_ground_reconstruction=1e-10,COM=1e-12,scalar_integral=1e-10),solver='original mj_step followed observer copied-state mj_forward; post-state force is not guaranteed effective force used by preceding Euler step. No force-time offset selection. No per-native-qvel; compare at .5ms shared times.'))
 print('PREPARED',mass,g.tolist(),flush=True)
def run():
 import mujoco
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 counts=dict(model_compile=0,forward=0,COMJac=0,foot_body_Jac=0,body_step=0,neural_step=0,LP=0,optimization=0,training=0,contactForce=0)
 assert (O/'PREREGISTRATION.json').exists() and not (O/'ARRAYS.npz').exists()
 try:
  frozen=json.loads((O/'SOURCE_HASHES_BEFORE.json').read_text());assert all(sha(p)==h for p,h in frozen.items())
  m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();counts['model_compile']+=1
  c=arr(C);assert all(np.array_equal(np.asarray(getattr(m,k)),c[k]) for k in ['body_mass','body_subtreemass','body_parentid','jnt_bodyid','jnt_type']);assert np.array_equal(m.opt.gravity,c['gravity'])
  mass=float(m.body_mass.sum());g=float(abs(m.opt.gravity[2]));W=mass*g;thorax=m.body('Thorax').id
  assert thorax==2 and m.jnt_type[0]==0 and m.nmocap==0 and m.neq==0
  metadata=dict(total_mass=mass,rootfree_body='Thorax',rootfree_body_mass=float(m.body_mass[thorax]),rootfree_subtree_mass=float(m.body_subtreemass[thorax]),gravity=m.opt.gravity,weight=W,timestep=m.opt.timestep,integrator=int(m.opt.integrator),viscosity=m.opt.viscosity,density=m.opt.density,wind=m.opt.wind,geom_names=[m.geom(i).name for i in range(m.ngeom)],body_names=[m.body(i).name for i in range(m.nbody)],units='native model length,time,mass,force; SI NOT_ESTABLISHED')
  assert m.opt.viscosity==0 and m.opt.density==0
  dump(O/'MODEL_METADATA.json',metadata)
  # Select unique endpoint rows; preserve BOTH assisted and withdrawn boundary rows explicitly.
  selected={};contacts=[];offsets=[0]
  for p in sorted((B/'NATIVE').glob('*.npz')):
   z=arr(p);keep=np.flatnonzero((z['step']>=96000)&(z['step']<=105600))
   for i in keep:
    for k in z:
     if k not in ['contact_rows','contact_offsets']:selected.setdefault(k,[]).append(z[k][i])
    contacts.extend(z['contact_rows'][z['contact_offsets'][i]:z['contact_offsets'][i+1]])
    offsets.append(len(contacts))
  a={'native_'+k:np.asarray(v) for k,v in selected.items()};a['contact_rows']=np.asarray(contacts).reshape(-1,30);a['contact_offsets']=np.asarray(offsets)
  schema=json.loads((B/'CONTACT_SCHEMA.json').read_text());floors=set(schema['floor_geom_indices']);signed=np.zeros((len(offsets)-1,3));positive=np.zeros(len(signed));candidate=np.zeros(len(signed),bool);first={}
  eventrows=[]
  for i in range(len(signed)):
   for row in a['contact_rows'][offsets[i]:offsets[i+1]]:
    g1,g2=int(row[0]),int(row[1])
    if g1 not in floors and g2 not in floors:continue
    force=row[15:18]*(1 if g1 in floors else -1);signed[i]+=force
    if row[4]>=0:continue
    candidate[i]=True;positive[i]+=max(float(force[2]),0)
    for title,match in [('candidate',True),('positive_contact',force[2]>0),('positive_contact_deadband',force[2]>1e-8*W)]:
     if match and title not in first and a['native_step'][i]>=104000 and (a['native_step'][i]>104000 or a['native_kind'][i]==1):
      other=g2 if g1 in floors else g1;first[title]=dict(native_index=i,step=int(a['native_step'][i]),relative_ms=float((a['native_time'][i]-1.3)*1000),geom=m.geom(other).name,body=m.body(int(m.geom_bodyid[other])).name,geom_id=other,body_id=int(m.geom_bodyid[other]),signed_force=force,row=row)
   if candidate[i]:eventrows.append([i,int(a['native_step'][i]),int(a['native_kind'][i]),a['native_time'][i]-1.3,positive[i],a['native_nonfoot_force'][i,2]])
  a['ground_force']=a['native_foot_force'].sum(1)+a['native_nonfoot_force'];a['ground_force_from_contacts']=signed;a['positive_nonfoot_contact_sum']=positive;a['nonfoot_candidate']=candidate
  assert np.max(abs(signed-a['ground_force']))<1e-10
  a['acceleration_z']=(a['ground_force'][:,2]/W-1)*g
  b=arr(B/'RUN/BODY.npz');ix=np.arange(2400,2641);assert np.max(abs(b['time'][ix]-(1.2+np.arange(241)*.0005)))<1e-10
  for k,v in b.items():a['body_'+k]=v[ix]
  a['body_source_index']=ix
  velocities=[];coms=[];footp=[];footv=[];bodyp=[];bodyv=[];clearances=[];nonfootmin=[];support_hull=[]
  bids=[m.body(l+'Tarsus5').id for l in LEGS];geomlegs=[next((j for j,l in enumerate(LEGS) if (m.geom(i).name or '').startswith(l) and 'Tars' in m.geom(i).name),-1) for i in range(m.ngeom)]
  for ii in ix:
   d=mujoco.MjData(m)
   for dst,src in [('qpos','qpos'),('qvel','qvel'),('act','activation'),('ctrl','command'),('qfrc_applied','assist'),('xfrc_applied','external')]:getattr(d,dst)[:]=b[src][ii]
   d.time=float(b['time'][ii]);mujoco.mj_forward(m,d);counts['forward']+=1
   J=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,J,thorax);counts['COMJac']+=1;velocities.append(J@d.qvel);coms.append(d.subtree_com[thorax].copy())
   rot=d.xmat[thorax].reshape(3,3);pos=d.xpos[bids].copy();vs=[]
   for bid in bids:
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,bid);counts['foot_body_Jac']+=1;vs.append(jp@d.qvel)
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,thorax);counts['foot_body_Jac']+=1
   vs=np.array(vs);omega=jr@d.qvel;rel=pos-d.xpos[thorax];footp.append(pos);footv.append(vs);bodyp.append(rel@rot);bodyv.append((vs-jp@d.qvel-np.cross(omega,rel))@rot)
   heights=np.full(m.ngeom,np.nan)
   for gi in range(m.ngeom):
    if m.geom_type[gi]==mujoco.mjtGeom.mjGEOM_MESH:
     mi=int(m.geom_dataid[gi]);start=int(m.mesh_vertadr[mi]);n=int(m.mesh_vertnum[mi]);verts=m.mesh_vert[start:start+n];heights[gi]=np.min((verts@d.geom_xmat[gi].reshape(3,3).T+d.geom_xpos[gi])[:,2])
   clearances.append([min(heights[j] for j,l in enumerate(geomlegs) if l==k) for k in range(6)])
   ids=[j for j,l in enumerate(geomlegs) if l<0 and j not in floors and np.isfinite(heights[j])];j=min(ids,key=lambda j:heights[j]);nonfootmin.append([j,heights[j]])
  a.update(com_velocity=np.asarray(velocities),offline_com=np.asarray(coms),foot_world_position=np.asarray(footp),foot_world_velocity=np.asarray(footv),foot_body_position=np.asarray(bodyp),foot_body_velocity=np.asarray(bodyv),foot_mesh_min_world_z=np.asarray(clearances),nonfoot_mesh_min_world_z=np.asarray(nonfootmin))
  assert np.max(abs(a['offline_com']-a['body_com']))<1e-12
  assert np.all(a['native_assist_root_max']==0) and np.all(a['native_external_max']==0) and np.all(a['body_assist'][:,:6]==0) and np.all(a['body_external']==0)
  a['body_native_index']=np.array([np.flatnonzero((a['native_step']==int(round(t/1.25e-5)))&((a['native_kind']==1) if abs(t-1.3)<1e-9 else (a['native_kind']==0)))[0] for t in a['body_time']])
  # Separate continuous prewarm and released integrations; boundary force is discontinuous.
  results={};integrals=[]
  for title,mask,bi0 in [('prewarm',(a['native_step']<=104000)&(a['native_kind']==0),0),('release',(a['native_step']>=104000)&((a['native_step']>104000)|(a['native_kind']==1)),200)]:
   ni=np.flatnonzero(mask);t=a['native_time'][ni];dt=np.diff(t);assert np.max(abs(dt-1.25e-5))<1e-10
   acc=a['acceleration_z'][ni];v0=a['com_velocity'][bi0,2];z0=a['body_com'][bi0,2];sub={}
   for mode in ['left','right','trapezoid']:
    increments=dt*(acc[:-1] if mode=='left' else acc[1:] if mode=='right' else (acc[:-1]+acc[1:])*.5)
    v=np.r_[v0,v0+np.cumsum(increments)];z=np.r_[z0,z0+np.cumsum(dt*(v[:-1]+v[1:])*.5)]
    a[title+'_'+mode+'_native_index']=ni;a[title+'_'+mode+'_velocity']=v;a[title+'_'+mode+'_height']=z
    shared=np.flatnonzero((a['body_time']>=t[0]-1e-10)&(a['body_time']<=t[-1]+1e-10));local=np.array([np.argmin(abs(t-a['body_time'][k])) for k in shared]);assert np.max(abs(t[local]-a['body_time'][shared]))<1e-10
    ev=v[local]-a['com_velocity'][shared,2];ez=z-a['native_com'][ni,2]
    sub[mode]=dict(max_velocity_error=float(abs(ev).max()),end_velocity_error=float(ev[-1]),max_height_error=float(abs(ez).max()),end_height_error=float(ez[-1]),end_predicted_velocity=float(v[-1]),end_predicted_height=float(z[-1]))
   tt=t-t[0];a[title+'_analytic_freefall_height']=z0+v0*tt-.5*g*tt**2;a[title+'_analytic_supported_height']=z0+v0*tt;a[title+'_analytic_freefall_velocity']=v0-g*tt
   a[title+'_analytic_time']=tt
   sub.update(duration_s=float(tt[-1]),initial_vz=float(v0),final_vz=float(a['com_velocity'][shared[-1],2]),COM_first_last=a['native_com'][ni[[0,-1]],2],mean_foot_W=float(a['native_foot_force'][ni[1:],:,2].sum(1).mean()/W),mean_nonfoot_W=float(a['native_nonfoot_force'][ni[1:],2].mean()/W),positive_nonfoot_endpoints=int((positive[ni[1:]]>0).sum()),COM_peak_to_peak=float(np.ptp(a['native_com'][ni,2])),vz_minmax=[float(a['com_velocity'][shared,2].min()),float(a['com_velocity'][shared,2].max())],gravity_impulse=float(-mass*g*tt[-1]),ground_impulse_right=float(np.sum(a['ground_force'][ni[1:],2]*dt)),ground_impulse_trapezoid=float(np.sum((a['ground_force'][ni[:-1],2]+a['ground_force'][ni[1:],2])*.5*dt)),measured_momentum_change=float(mass*(a['com_velocity'][shared[-1],2]-v0)),analytic_freefall_end_height=float(a[title+'_analytic_freefall_height'][-1]),analytic_supported_end_height=float(a[title+'_analytic_supported_height'][-1]))
   results[title]=sub
  phases=[]
  for lo,hi in [(-100,-75),(-75,-50),(-50,-25),(-25,0),(0,5),(5,10),(10,18),(18,18.9125),(18.9125,20)]:
   keep=(a['native_time']>1.3+lo/1000+1e-10)&(a['native_time']<=1.3+hi/1000+1e-10)&(a['native_kind']==0);ii=np.flatnonzero(keep)
   phases.append(dict(lo_ms=lo,hi_ms=hi,endpoints=len(ii),mean_foot_W=float(a['native_foot_force'][ii,:,2].sum(1).mean()/W),mean_NF_W=float(a['native_nonfoot_force'][ii,2].mean()/W),mean_ground_W=float(a['ground_force'][ii,2].mean()/W),ground_impulse=float(np.sum(a['ground_force'][ii,2])*1.25e-5),gravity_impulse=float(-W*len(ii)*1.25e-5),positive_NF=int((positive[ii]>0).sum())))
  for name,field in [('aggregate_positive',a['native_nonfoot_force'][:,2]>0),('aggregate_deadband',a['native_nonfoot_force'][:,2]>W*1e-8)]:
   keep=np.flatnonzero(field & (a['native_step']>104000));i=int(keep[0]);first[name]=dict(native_index=i,step=int(a['native_step'][i]),relative_ms=float((a['native_time'][i]-1.3)*1000))
  np.savez_compressed(O/'ARRAYS.npz',**a)
  dump(O/'RESULT.json',dict(model=metadata,counts=counts,quadrature=results,phases=phases,events=first,force_rows_maxerror=float(np.max(abs(signed-a['ground_force']))),COM_maxerror=float(np.max(abs(a['offline_com']-a['body_com']))),release_foot_clearance=a['foot_mesh_min_world_z'][200],release_nonfoot_min=a['nonfoot_mesh_min_world_z'][200],source_boundary=dict(step=104000,body_s=1.3,neural_s=1.1,duplicate_kind0_assisted_kind1_released=True),root_external_zero=True,exact_solver_evaltime='UNKNOWN; no offset fitted',autonomous_walking=False))
  tab(O/'NATIVE.csv',['index','step','time_s','relative_ms','kind','phase','footFz','nonfootFz','groundFz','groundW','acceleration_z','COMz','rootz','positiveNF'],[[i,a['native_step'][i],a['native_time'][i],(a['native_time'][i]-1.3)*1000,a['native_kind'][i],a['native_phase'][i],a['native_foot_force'][i,:,2].sum(),a['native_nonfoot_force'][i,2],a['ground_force'][i,2],a['ground_force'][i,2]/W,a['acceleration_z'][i],a['native_com'][i,2],a['native_root'][i,2],positive[i]] for i in range(len(signed))])
  tab(O/'SHARED_BODY.csv',['source_index','time_s','relative_ms','COMz','COMvx','COMvy','COMvz','assist_joint_L2','rootvx','rootvy','rootvz'],[[ix[i],a['body_time'][i],(a['body_time'][i]-1.3)*1000,a['body_com'][i,2],*a['com_velocity'][i],np.linalg.norm(a['body_assist'][i,6:]),*a['body_qvel'][i,:3]] for i in range(len(ix))])
  tab(O/'PHASES.csv',list(phases[0]),[list(p.values()) for p in phases]);tab(O/'NONFOOT_EVENTS.csv',['native_index','step','kind','relative_s','sum_percontact_positiveNFz','signed_aggregateNFz'],eventrows)
  assert all(sha(p)==h for p,h in frozen.items());dump(O/'SOURCE_HASHES_AFTER.json',{p:sha(p) for p in frozen})
  print(json.dumps(results,default=lambda v:v.tolist()),flush=True)
 except BaseException:
  dump(O/'FAILURE.json',dict(traceback=traceback.format_exc(),counts=counts));raise
 finally:dump(O/'RESOURCE_ACCOUNTING.json',counts)
def verify():
 # Fresh process pure arrays. Scalar integration independently, no imported model or run helpers.
 a=arr(O/'ARRAYS.npz');res=json.loads((O/'RESULT.json').read_text());W=res['model']['weight'];mass=res['model']['total_mass'];g=abs(res['model']['gravity'][2]);errors={};ground=[];first=None
 floors={0}
 for i in range(len(a['native_time'])):
  rows=a['contact_rows'][a['contact_offsets'][i]:a['contact_offsets'][i+1]];components=[[],[],[]];pos=[]
  for row in rows:
   if int(row[0]) not in floors and int(row[1]) not in floors:continue
   sign=1 if int(row[0]) in floors else -1
   for j in range(3):components[j].append(sign*float(row[15+j]))
   if row[4]<0:
    z=sign*float(row[17]);pos.append(max(0,z))
    if z>0 and first is None and a['native_step'][i]>104000:first=(i,int(row[1]) if sign==1 else int(row[0]))
  ground.append([math.fsum(v) for v in components]);assert abs(math.fsum(pos)-a['positive_nonfoot_contact_sum'][i])<1e-10
 errors['ground_scalar']=float(np.max(abs(np.asarray(ground)-a['ground_force'])));assert errors['ground_scalar']<1e-10
 e=res['events']['positive_contact'];assert first==(e['native_index'],e['geom_id']);assert res['model']['geom_names'][first[1]]==e['geom'];assert abs(e['relative_ms']-(a['native_time'][first[0]]-1.3)*1000)<1e-10
 for title,start in [('prewarm',0),('release',200)]:
  for mode in ['left','right','trapezoid']:
   ni=a[title+'_'+mode+'_native_index'];v=float(a['com_velocity'][start,2]);z=float(a['body_com'][start,2]);vs=[v];zs=[z]
   for p,q in zip(ni[:-1],ni[1:]):
    dt=float(a['native_time'][q]-a['native_time'][p]);acc0=(float(a['ground_force'][p,2])/mass-g);acc1=float(a['ground_force'][q,2])/mass-g
    an=acc0 if mode=='left' else acc1 if mode=='right' else (acc0+acc1)/2
    vn=v+dt*an;z+=dt*(v+vn)/2;v=vn;vs.append(v);zs.append(z)
   ev=float(np.max(abs(np.array(vs)-a[title+'_'+mode+'_velocity'])));ez=float(np.max(abs(np.array(zs)-a[title+'_'+mode+'_height'])));assert ev<1e-10 and ez<1e-10;errors[title+'_'+mode]=dict(velocity=ev,height=ez)
  ni=a[title+'_right_native_index'];J=math.fsum(float(a['ground_force'][i,2])*(float(a['native_time'][i])-float(a['native_time'][p])) for p,i in zip(ni[:-1],ni[1:]));assert abs(J-res['quadrature'][title]['ground_impulse_right'])<1e-10
 frozen=json.loads((O/'SOURCE_HASHES_BEFORE.json').read_text());assert all(sha(p)==h for p,h in frozen.items());assert frozen==json.loads((O/'SOURCE_HASHES_AFTER.json').read_text())
 dump(O/'INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,errors=errors,first_contact_identity=True,sourcehash_unchanged=True,method='fresh process scalar force fsum and recursive velocity/height integration, original contactrow identity',model_compile=0,forward=0,body_step=0,neural_step=0));print('PURE_ARRAY_PASS',errors)
if __name__=='__main__':{'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
