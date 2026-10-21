"""One forward, unchanged saved pose: qualify native contact mapping, not standing."""
from pathlib import Path
import json,hashlib,time,itertools,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import mujoco
 from engine import canonical_physics as cp
 from flygym.compose.fly import musculoskeletal as mus
 from run_s2_fixed_stimulus import dense_moment
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-support-input-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',forward_calls=0,steps=0,manual_research_not_product=True,checks={},contacts=[])
 files=[Path(__file__),ROOT/'S2_SUPPORT_INPUT_QUALIFICATION_PROTOCOL.zh-CN.md',Path(cp.__file__),Path(mus.__file__),ROOT/'run_s2_fixed_stimulus.py',cp.CANONICAL_XML,OLD/'raw.npz',OLD/'metadata.json']
 hashes={str(p.resolve()):sha(p) for p in files}
 try:
  spec=mus._load_mjcf(str(cp.CANONICAL_XML));meshes=sorted({str(Path(mesh.file).resolve()) for mesh in spec.meshes});hashes.update({p:sha(p) for p in meshes});m=cp.load_model();d=mujoco.MjData(m)
  with np.load(OLD/'raw.npz',allow_pickle=False) as z: raw={k:z[k].copy() for k in z.files}
  meta=json.loads((OLD/'metadata.json').read_text());d.qpos[:]=raw['qpos'];d.qvel[:]=0;d.ctrl[:]=0;d.act[:]=0
  r['forward_calls']=1;mujoco.mj_forward(m,d)
  E=np.asarray(d.efc_J).reshape(d.nefc,m.nv).copy();moment=dense_moment(m,d)
  np.savez_compressed(out/'endpoint.npz',qpos=d.qpos,qvel=d.qvel,act=d.act,ctrl=d.ctrl,efc_J=E,efc_force=d.efc_force,efc_type=d.efc_type,efc_id=d.efc_id,qfrc_constraint=d.qfrc_constraint,qfrc_bias=d.qfrc_bias,qfrc_passive=d.qfrc_passive,qfrc_actuator=d.qfrc_actuator,moment=moment,qfrc_applied=d.qfrc_applied,xfrc_applied=d.xfrc_applied)
  def check(name,a,b):
   a=np.asarray(a);b=np.asarray(b);ok=a.shape==b.shape and np.allclose(a,b,rtol=1e-10,atol=1e-10);err=float(np.max(np.abs(a-b),initial=0)) if a.shape==b.shape else None;r['checks'][name]=dict(ok=bool(ok),max_abs_error=err);return ok
  r['compiled']=dict(cone=int(m.opt.cone),cone_name=mujoco.mjtCone(int(m.opt.cone)).name,jacobian=int(m.opt.jacobian),solver=int(m.opt.solver),disableflags=int(m.opt.disableflags),enableflags=int(m.opt.enableflags),integrator=int(m.opt.integrator),timestep=float(m.opt.timestep),gravity=m.opt.gravity.tolist(),nefc=d.nefc,ncon=d.ncon,neq=m.neq,nmocap=m.nmocap,nplugin=m.nplugin,mesh_count=m.nmesh,unique_mesh_files=len(meshes),constraint_types=np.unique(d.efc_type,return_counts=True)[0].tolist(),constraint_type_counts=np.unique(d.efc_type,return_counts=True)[1].tolist())
  for field in ('actuator_forcelimited','jnt_actfrclimited','body_gravcomp','dof_frictionloss','tendon_frictionloss','actuator_actearly','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_trntype','actuator_actadr','actuator_actnum','actuator_ctrlrange','actuator_actrange','actuator_actlimited','jnt_type','jnt_limited','jnt_range','jnt_margin','tendon_limited','tendon_range','tendon_margin'):
   r['compiled'][field]=np.asarray(getattr(m,field)).tolist()
  maps=[];covered=[]
  for idx,c in enumerate(d.contact):
   entry=dict(index=idx,geom=[int(c.geom1),int(c.geom2)],geom_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)],efc_address=int(c.efc_address),dim=int(c.dim),exclude=int(c.exclude),dist=float(c.dist),includemargin=float(c.includemargin),friction=np.asarray(c.friction).tolist(),pos=np.asarray(c.pos).tolist(),frame=np.asarray(c.frame).tolist(),geom_margin=m.geom_margin[[c.geom1,c.geom2]].tolist(),geom_gap=m.geom_gap[[c.geom1,c.geom2]].tolist())
   r['contacts'].append(entry)
   if c.efc_address<0: continue
   if int(m.opt.cone)!=int(mujoco.mjtCone.mjCONE_PYRAMIDAL) or c.dim!=3: raise ValueError('not pyramidal3')
   J=np.zeros((3,m.nv))
   for g,sign in ((c.geom2,1),(c.geom1,-1)):
    jp=np.zeros_like(J);jr=np.zeros_like(J);mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[g]));J+=sign*jp
   frame=np.asarray(c.frame).reshape(3,3);mu=np.asarray(c.friction)[:2];B=np.array([frame[0]+x*mu[0]*frame[1]+y*mu[1]*frame[2] for x,y in [(1,0),(-1,0),(0,1),(0,-1)]]).T;C=J.T@B;maps.append(C)
   adr=int(c.efc_address);rows=E[adr:adr+4].T;covered.extend(range(adr,adr+4));perm=min(itertools.permutations(range(4)),key=lambda p:np.max(np.abs(C-rows[:,p])))
   entry['native_row_permutation']=list(perm);check('native_contact_rows_'+str(idx),C,rows[:,perm])
  C=np.hstack(maps);check('old_contact_map',C,raw['contact_diamond']);check('old_moment',moment,raw['moment']);check('old_base',d.qfrc_actuator,raw['base']);check('old_bias',d.qfrc_bias,raw['bias']);check('old_passive',d.qfrc_passive,raw['passive']);check('constraint_force_factorization',E.T@d.efc_force,d.qfrc_constraint);check('zero_external',np.r_[d.qfrc_applied,d.xfrc_applied.ravel()],np.zeros(m.nv+6*m.nbody))
  active=[c for c in r['contacts'] if c['efc_address']>=0]
  r['checks']['old_contact_count']=dict(ok=len(active)==len(meta['active_contacts']))
  for c,o in zip(active,meta['active_contacts']):
   k=c['index'];check('old_pos_'+str(k),c['pos'],o['pos']);check('old_frame_'+str(k),c['frame'],np.array(o['frame']).ravel());check('old_mu_'+str(k),c['friction'][:2],o['mu']);check('old_distance_'+str(k),c['dist'],o['dist']);check('old_pair_'+str(k),c['geom'],[o['geom1'],o['geom2']])
  r['checks']['no_missing_reactions']=dict(ok=sorted(covered)==list(range(d.nefc)) and len(covered)==d.nefc)
  r['checks']['no_warnings']=dict(ok=not np.any(d.warning.number))
  r['checks']['no_force_coupling']=dict(ok=not any(np.any(getattr(m,k)) for k in ('actuator_forcelimited','jnt_actfrclimited','body_gravcomp','dof_frictionloss','tendon_frictionloss','actuator_actearly')) and m.nplugin==0 and m.neq==0 and m.nmocap==0)
  r['status']='PASS_INPUT_MAPPING' if all(x['ok'] for x in r['checks'].values()) else 'FAIL_INPUT_MAPPING'
 except Exception as e: r.update(status='FAIL',error=type(e).__name__+': '+str(e))
 finally:
  r['source_input_hashes_before']=hashes;r['source_input_hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=hashes==r['source_input_hashes_after'];r['environment']=dict(python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__)
  if not r['hashes_unchanged']: r['status']='FAIL_HASH_DRIFT'
  dump(out/'report.json',r)
 print(out);print(r['status']);print([(k,v) for k,v in r['checks'].items() if not v['ok']]);return 0 if r['status']=='PASS_INPUT_MAPPING' else 1
if __name__=='__main__':raise SystemExit(main())
