"""S2/P3 saved-only early support geometry, no integration or neural simulation."""
from pathlib import Path
import json,sys,os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_double_muscle_force import restore,c,BASE,sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s2-early-support-region/20261027T200000Z')
FORCE_TOL=1e-9
GEOM_TOL=1e-10
LEGS=['lf','lm','lh','rf','rm','rh']
def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def region(points,com):
 p=np.unique(np.asarray(points,dtype=float).reshape(-1,2),axis=0);n=len(p)
 if n==0:return dict(kind='empty',dimension=-1,vertices=[],area=0.,signed_boundary_distance=None,unsigned_distance=None,inside_2d=None)
 if n==1:return dict(kind='point',dimension=0,vertices=p,area=0.,signed_boundary_distance=None,unsigned_distance=float(np.linalg.norm(com-p[0])),inside_2d=None)
 def half(seq):
  h=[]
  for pt in seq:
   while len(h)>=2 and cross(h[-1]-h[-2],pt-h[-1])<=GEOM_TOL*np.linalg.norm(h[-1]-h[-2]):h.pop()
   h.append(pt)
  return h
 h=np.array(half(p)[:-1]+half(p[::-1])[:-1]);dim=1 if len(h)<3 else 2
 pairs=[(h[0],h[-1])] if dim==1 else list(zip(h,np.roll(h,-1,axis=0)))
 distances=[];side=[]
 for a,b in pairs:
  v=b-a;length=np.linalg.norm(v);q=a+np.clip(np.dot(com-a,v)/np.dot(v,v),0,1)*v;distances.append(np.linalg.norm(com-q));side.append(cross(v,com-a)/length)
 distance=float(min(distances));inside=bool(min(side)>=-GEOM_TOL) if dim==2 else None
 return dict(kind='polygon' if dim==2 else 'segment_or_collinear',dimension=dim,vertices=h,area=abs(sum(cross(a,b) for a,b in pairs))/2 if dim==2 else 0.,signed_boundary_distance=(distance if inside else -distance) if dim==2 else None,unsigned_distance=distance,inside_2d=inside)

def tests():
 assert region([],np.zeros(2))['dimension']==-1
 assert region([[0,0]],np.ones(2))['signed_boundary_distance'] is None
 assert region([[0,0],[1,0],[2,0]],np.ones(2))['dimension']==1
 assert region([[0,0],[1,0]],np.ones(2))['area']==0
 square=[[0,0],[1,0],[1,1],[0,1]]
 assert region(square,np.array([.5,.5]))['signed_boundary_distance']==.5
 assert region(square,np.array([2.,.5]))['signed_boundary_distance']==-1
 assert region(square,np.array([0.,.5]))['signed_boundary_distance']==0
 return dict(exit_code=0,empty_point_segment_collinear_polygon_inside_outside_boundary=True)

def main():
 O.mkdir(parents=True,exist_ok=False);p=BASE/'normal';sources=[Path(__file__),Path(c.cp.CANONICAL_XML),p/'BODY.npz',p/'CONTACTS.json',BASE/'MANIFEST_SHA256.json',Path('/run/media/sensen/Data2/cell_wound_prototype/run_double_muscle_force.py'),Path('/run/media/sensen/Data2/cell_wound_prototype/analyze_s294_support_ratio.py')];hashes={str(f):sha(f) for f in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='historical normal0-40ms all saved samples ONLY',body_neural_steps=0,force_threshold=FORCE_TOL,threshold_source='analyze_s294_support_ratio.py positive fz>1e-9 contact-bearing convention, model force not SI',geometry_distance_tolerance=GEOM_TOL,geometric_definition='all MuJoCo plane-tarsal contact records including zero-force contacts; separated NONFOOT',bearing_definition='individual foot plane contact local compressive normal force>1e-9, not absolute/sum force',resolution_s=.0000125,signed_distance='positive inside nondegenerate 2D polygon, negative outside, zero boundary; degenerates null with unsigned distance',interpretation='geometric containment neither necessary nor sufficient dynamic stability; no controller or parameter changes'))
 dump(O/'TESTS.json',tests());manifest=json.loads((BASE/'MANIFEST_SHA256.json').read_text());assert sha(p/'BODY.npz')==manifest['normal/BODY.npz'];assert sha(p/'CONTACTS.json')==manifest['normal/CONTACTS.json']
 z=np.load(p/'BODY.npz');contacts=json.loads((p/'CONTACTS.json').read_text());m=c.cp.load_model();m.opt.timestep=.0000125;up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);seed=np.eye(3)[np.argmin(abs(up))];ex=np.cross(seed,up);ex/=np.linalg.norm(ex);ey=np.cross(up,ex);basis=np.array([ex,ey]);origin=np.zeros(3);foot={}
 for g in range(m.ngeom):
  gn=(m.geom(g).name or '').lower();bn=(m.body(int(m.geom_bodyid[g])).name or '').lower();matches=[l for l in LEGS if (gn.startswith(l) or bn.startswith(l)) and 'tars' in gn+bn]
  if len(matches)==1:foot[g]=matches[0]
 dump(O/'PROJECTION_AND_CLASSIFICATION.json',dict(gravity=m.opt.gravity,up=up,basis=basis,origin=origin,foot_geoms={str(g):dict(name=m.geom(g).name,leg=l) for g,l in foot.items()},mass=m.body_mass,total_mass=float(m.body_mass.sum()),COM='sum(body_mass*xipos)/sum(body_mass), all model bodies; saved COM reused after equality check each state',normal='MuJoCo contact local normal force, world reaction frame.T @ local with plane side sign',no_foot_centres=True))
 indices=np.flatnonzero(z['time']<=.040+1e-10);first_nonfoot=np.flatnonzero(z['nonfoot_floor_count']);pre=int(first_nonfoot[0]-1);focus=[0,800,1200,2000,pre];rows=[];errors=[];calls=0
 for idx in indices:
  d=restore(m,z,int(idx));calls+=1;com=(m.body_mass[:,None]*d.xipos).sum(0)/m.body_mass.sum();comerr=float(abs(com-z['com_position'][idx]).max());projected=basis@(com-origin);actual=[];perleg={l:dict(points_world=[],forces_world=[],normal_loads=[]) for l in LEGS+['NONFOOT']};fs=np.zeros(3);geom=[];bearing=[]
  assert len(contacts[idx])==d.ncon
  for k,saved in enumerate(contacts[idx]):
   co=d.contact[k];np.testing.assert_allclose(co.pos,saved[4:7],atol=1e-10,rtol=0);a,b=int(co.geom1),int(co.geom2);assert a==int(saved[1]) and b==int(saved[2]);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=b if pa else a;leg=foot.get(g,'NONFOOT');local=np.asarray(saved[7:13]);force=(co.frame.reshape(3,3).T@local[:3])*(1 if pa else -1);pos=np.asarray(saved[4:7]);xy=basis@(pos-origin);normal=float(local[0]);fs+=force
   perleg[leg]['points_world'].append(pos);perleg[leg]['forces_world'].append(force);perleg[leg]['normal_loads'].append(normal)
   actual.append(dict(contact_index=k,fly_geom=g,leg=leg,position_world=pos,position_projected=xy,reaction_world=force,compressive_normal_load=normal,bearing=bool(normal>FORCE_TOL)))
   if leg!='NONFOOT':
    geom.append(xy)
    if normal>FORCE_TOL:bearing.append(xy)
  for leg,v in perleg.items():v.update(geometric_contact_count=len(v['points_world']),bearing_contact_count=sum(f>FORCE_TOL for f in v['normal_loads']),world_force_sum=np.array(v['forces_world']).reshape(-1,3).sum(0),normal_load_sum=sum(v['normal_loads']))
  forceerr=float(abs(fs-z['ground_force_world'][idx]).max());errors.append(dict(index=int(idx),com_maxabs=comerr,world_force_maxabs=forceerr));assert max(comerr,forceerr)<1e-9
  rows.append(dict(index=int(idx),time_s=float(z['time'][idx]),com_world=com,com_projected=projected,com_height=float(np.dot(com-origin,up)),thorax_height=float(np.dot(z['thorax_position'][idx]-origin,up)),com_vertical_velocity=float(np.dot(z['com_velocity'][idx],up)),contacts=actual,perleg=perleg,geometric_region=region(geom,projected),bearing_region=region(bearing,projected)))
 dump(O/'SUPPORT_REGION_TIME_SERIES.json',rows);dump(O/'RESTORE_ERRORS.json',errors)
 def first(condition):
  r=next((r for r in rows if condition(r)),None);return None if r is None else dict(index=r['index'],time_s=r['time_s'])
 events=dict(first_fewer_than_six_geometric_legs=first(lambda r:sum(r['perleg'][l]['geometric_contact_count']>0 for l in LEGS)<6),first_fewer_than_six_bearing_legs=first(lambda r:sum(r['perleg'][l]['bearing_contact_count']>0 for l in LEGS)<6),first_geometric_outside=first(lambda r:r['geometric_region']['inside_2d'] is False),first_bearing_outside=first(lambda r:r['bearing_region']['inside_2d'] is False),first_bearing_degenerate=first(lambda r:r['bearing_region']['dimension']<2),first_nonfoot=dict(index=int(first_nonfoot[0]),time_s=float(z['time'][first_nonfoot[0]])),focus_indices=focus,resolution_s=.0000125)
 dump(O/'EVENTS.json',events);dump(O/'FOCUS_STATES.json',[rows[i] for i in focus]);fig=plt.figure(figsize=(18,12));gs=fig.add_gridspec(3,5)
 for col,idx in enumerate(focus):
  r=rows[idx];ax=fig.add_subplot(gs[0,col]);colors=plt.get_cmap('tab10')
  for j,l in enumerate(LEGS+['NONFOOT']):
   ps=np.array(r['perleg'][l]['points_world']).reshape(-1,3)@basis.T
   if len(ps):ax.scatter(ps[:,0],ps[:,1],s=10,c=[colors(j)],label=l)
  for key,style in [('geometric_region','--'),('bearing_region','-')]:
   h=np.array(r[key]['vertices']).reshape(-1,2)
   if len(h):v=np.vstack([h,h[:1]]) if len(h)>=3 else h;ax.plot(v[:,0],v[:,1],style,label=key)
  ax.plot(*r['com_projected'],'kx',ms=9);ax.set_aspect('equal');ax.set_title(f"{r['time_s']*1000:.4f}ms\nG:{r['geometric_region']['kind']} B:{r['bearing_region']['kind']}",fontsize=9);ax.grid(alpha=.2)
  if col==0:ax.legend(fontsize=6)
 t=np.array([r['time_s'] for r in rows])*1000;ax=fig.add_subplot(gs[1,:]);ax.plot(t,[r['com_height'] for r in rows],label='full mass-weighted COM height');ax.plot(t,[r['thorax_height'] for r in rows],label='thorax height');ax.legend();ax.grid(alpha=.2);ax.set_xlabel('saved physical ms')
 ax=fig.add_subplot(gs[2,:3])
 for key in ['geometric_region','bearing_region']:ax.plot(t,[r[key]['signed_boundary_distance'] if r[key]['signed_boundary_distance'] is not None else np.nan for r in rows],label=key+' signed margin')
 ax.axhline(0,color='k',lw=.5);ax.set_xlabel('saved physical ms; gaps=degenerate');ax.legend();ax.grid(alpha=.2)
 ax=fig.add_subplot(gs[2,3:]);ax.plot(t,[sum(r['perleg'][l]['geometric_contact_count']>0 for l in LEGS) for r in rows],label='geometric foot legs');ax.plot(t,[sum(r['perleg'][l]['bearing_contact_count']>0 for l in LEGS) for r in rows],label='bearing foot legs');ax.plot(t,[r['perleg']['NONFOOT']['geometric_contact_count'] for r in rows],label='NONFOOT contacts');ax.legend(fontsize=7);ax.grid(alpha=.2);ax.set_xlabel('saved physical ms')
 fig.suptitle('Historical normal0-40ms / contact-point hulls exclude NONFOOT / containment is NOT dynamic stability');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'SUPPORT_REGION_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_EARLY_SUPPORT_REGION.png').write_bytes((O/'SUPPORT_REGION_COMPARISON.png').read_bytes());assert all(sha(Path(f))==v for f,v in hashes.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,mj_forward=calls,mj_step=0,neural_step=0,states=len(rows),check_resolution_s=.0000125,max_com_restore_error=max(r['com_maxabs'] for r in errors),max_world_force_restore_error=max(r['world_force_maxabs'] for r in errors),failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(events))
if __name__=='__main__':main()
