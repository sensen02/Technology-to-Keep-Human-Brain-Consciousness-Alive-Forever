"""Parent-only14new actualRGBpairs appended to112immutable realframes; no training."""
from pathlib import Path
import argparse,hashlib,json,os,sys,copy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT/'frame_library_v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent-resource-released',action='store_true');args=parser.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no arrays/models/render');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 if (HERE/'LIBRARY.json').exists() or (HERE/'RGB.npz').exists():raise FileExistsError('inspectexistingoutputs,no blind regeneration')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw
 sys.path.insert(0,str(ROOT));from camera import eye_frames,render_eye_pair,add_decor
 pr=read(HERE/'PREREG.json');cp=read(ROOT/'PREREG_CAMERA_PROBE.json');old=read(OLD/'LIBRARY.json');oa=read(OLD/'VERIFY.json')
 if old['kind']!='VISION_STATIC_HEAD_LIBRARY_V1' or not oa['integrity_all_pass'] or not all(oa['checks'].values()) or oa['library_sha256']!=sha(OLD/'LIBRARY.json') or oa['rgb_sha256']!=sha(OLD/'RGB.npz'):raise ValueError('originalindependentRGBproof required')
 hashes={}
 for proof in (old,oa):
  for path,h in proof['source_hashes'].items():
   if sha(path)!=h:raise ValueError('oldsourceidentitydrift '+path)
   hashes[path]=h
 base=Path(cp['body_base']);run=Path(cp['trajectory']);sys.path.insert(0,str(base));from physics import Core
 from identity import verify,verify_runtime,capture
 ident=read(run/'IDENTITY.json');verify(ident)
 for path in [HERE/'PREREG.json',HERE/'generate.py',HERE/'audit.py',ROOT/'camera.py',ROOT/'PREREG_CAMERA_PROBE.json',OLD/'LIBRARY.json',OLD/'RGB.npz',OLD/'VERIFY.json',OLD/'PHYSICAL_IDENTITY.json',run/'IDENTITY.json',run/'trace_1ms.npz',Path(cp['head_geometry_audit'])]:hashes[str(path)]=sha(path)
 fresh(HERE/'INPUTS_BEFORE.json',hashes)
 with np.load(OLD/'RGB.npz',allow_pickle=False) as z:prefix=z['rgb'].copy()
 if prefix.shape!=(112,2,32,32,3) or prefix.dtype!=np.uint8:raise ValueError('original112frameprefix')
 rows=copy.deepcopy(old['rows'])
 for i,row in enumerate(rows):
  if row['frame_id']!=i or hashlib.sha256(prefix[i].tobytes()).hexdigest()!=row['rgb_sha256']:raise ValueError('originalRGBrowidentity')
  row['source_split']=row['split'];row['split']='train' if row['config_id'].startswith('T') else 'development';row['parent_frame_id']=i
  if row['split']=='development':row['selected_for_policy_sequences']=False;row['unused_by_policytraining']=True
 calls={'integration_attempts':0}
 def forbidden(*args,**kw):calls['integration_attempts']+=1;raise RuntimeError('RGB library prohibits physical integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 c=Core(adhesion=False);model=c.plant;verify_runtime(ident,c);mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
 if (model.nq,model.nv,model.nu)!=(49,48,42) or mesh.sum()!=71 or not np.all(model.geom_rgba[mesh,3]==1):raise ValueError('full71opaque required')
 newident=capture(c,dict(kind=pr['kind'],source_hashes=hashes,physical_steps=0,newRGBpairs=14,reusedRGBpairs=112),extra_paths=list(hashes));fresh(HERE/'PHYSICAL_IDENTITY.json',newident)
 indices=[cfg['saved_pose_index'] for cfg in pr['new_configs']]
 with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:qp=z['qpos'][indices].copy();qv=z['qvel'][indices].copy();times=z['time'][indices].copy()
 spec=cp['eyes'];head=model.body('Head').id;geometry=read(cp['head_geometry_audit']);targetspec=cp['visual_targets'];renderer=None;newframes=[];diagnostics=[];panels=[]
 try:
  renderer=mj.Renderer(model,height=32,width=32)
  for j,cfg in enumerate(pr['new_configs']):
   c.d.qpos[:]=qp[j];c.d.qvel[:]=qv[j];c.d.time=float(times[j]);mj.mj_forward(model,c.d)
   if not np.array_equal(c.d.qpos,qp[j]) or not np.array_equal(c.d.qvel,qv[j]):raise ValueError('savedposechanged')
   hp=c.d.xpos[head].copy();hr=c.d.xmat[head].reshape(3,3).copy();eyes=eye_frames(hp,hr,geometry,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=hr[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
   wall=mid+cfg['forward']*.5*ex;wall[2]=1.4;caseframes={};panel=Image.new('RGB',(448,64),color='white')
   for col,stimulus in enumerate(pr['stimuli']):
    sign=1 if stimulus in ('left','occluded_left') else -1 if stimulus in ('right','occluded_right') else 0;present=stimulus in ('center','left','right','occluded_left','occluded_right');occluded=stimulus.startswith('occluded_') or stimulus=='occluder_only';center=mid+cfg['forward']*ex+sign*cfg['side']*ey;center[2]=1.2
    def decor(scene):
     if present:add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],center,np.eye(3),targetspec['color_rgba'])
     if occluded:add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,[.04,1.4,1.4],wall,basis,[.12,.12,.12,1.])
    image,meta=render_eye_pair(renderer,model,c.d,eyes,spec,decor);fid=112+len(newframes);newframes.append(image.copy());caseframes[stimulus]=image
    rows.append(dict(frame_id=fid,config_id=cfg['config_id'],split='validation',source_split='newvalidation',stimulus=stimulus,pose_id=cfg['pose_id'],actual_saved_pose_index=cfg['saved_pose_index'],capture_time_s=float(times[j]),actual_head_position=hp.tolist(),actual_head_rotation=hr.tolist(),actualheadpose=dict(xpos=hp.tolist(),xmat=hr.tolist()),bearing_rad=float(np.arctan2(sign*cfg['side'],cfg['forward'])) if present else None,target_visible=present,environment_target_world_xyz=center.tolist() if present else None,decor=dict(target_present=present,occluder_present=occluded,target_radius=.12,target_height=2.4,target_rgba=targetspec['color_rgba'],occluder_center=wall.tolist() if occluded else None,occluder_halfsize=[.04,1.4,1.4] if occluded else None),camera_metadata=meta,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),selected_for_policy_sequences=True,unused_by_policytraining=False))
    panel.paste(Image.fromarray(np.concatenate((image[0],image[1]),axis=1)),(col*64,0))
   for row in rows[-7:]:
    if row['stimulus']=='occluded_left':row['target_visible']=not np.array_equal(caseframes['occluded_left'],caseframes['occluder_only'])
    if row['stimulus']=='occluded_right':row['target_visible']=not np.array_equal(caseframes['occluded_right'],caseframes['occluder_only'])
   diagnostics.append(dict(config_id=cfg['config_id'],pose_id=cfg['pose_id'],left_visible_meanabs=float(np.abs(caseframes['left'].astype(float)-caseframes['off']).mean()),right_visible_meanabs=float(np.abs(caseframes['right'].astype(float)-caseframes['off']).mean()),center_visible_meanabs=float(np.abs(caseframes['center'].astype(float)-caseframes['off']).mean()),occluded_left_equals_wall=bool(np.array_equal(caseframes['occluded_left'],caseframes['occluder_only'])),occluded_right_equals_wall=bool(np.array_equal(caseframes['occluded_right'],caseframes['occluder_only']))))
   panels.append((cfg,panel))
 finally:
  if renderer is not None:renderer.close()
 verify_runtime(ident,c);verify_runtime(newident,c)
 for path,h in hashes.items():
  if sha(path)!=h:raise ValueError('sourceidentitydrift duringnew14 '+path)
 rgb=np.concatenate((prefix,np.stack(newframes)),axis=0)
 if rgb.shape!=(126,2,32,32,3) or not np.array_equal(rgb[:112],prefix):raise ValueError('actual126/old112exactprefix required')
 np.savez_compressed(HERE/'RGB.npz',rgb=rgb);blocked=old['target_fully_occluded_all'] and all(d['occluded_left_equals_wall'] and d['occluded_right_equals_wall'] for d in diagnostics)
 result=dict(kind=pr['kind'],rgb_path=str(HERE/'RGB.npz'),rgb_sha256=sha(HERE/'RGB.npz'),rows=rows,eye_order=pr['eye_order'],camera_spec=spec,source_hashes=hashes,physical_identity=str(HERE/'PHYSICAL_IDENTITY.json'),physical_identity_sha256=sha(HERE/'PHYSICAL_IDENTITY.json'),compiled_unchanged=True,compiled_physics_before_after_unchanged=True,physics_steps=0,integration_attempts=calls['integration_attempts'],Core_constructions=1,model_forward_calls=0,optimizer_updates=0,parent_library=str(OLD/'LIBRARY.json'),parent_library_sha256=sha(OLD/'LIBRARY.json'),parent_RGB_sha256=sha(OLD/'RGB.npz'),old112pixelprefixexact=True,reused_binocular_frames=112,new_binocular_frames=14,new_raw_RGB_renders=28,unique_binocular_frames=126,occlusion_diagnostics=old['occlusion_diagnostics']+diagnostics,target_fully_occluded_all=blocked,invisible_label_training_admitted=blocked,policy_train_configs=old['policy_train_configs'],policy_validation_configs=[dict(config_id=c['config_id'],pose_id=c['pose_id']) for c in pr['new_configs']],old_V0_V1_development_not_fresh_validation=True,scope='STATIC_HEAD_IMAGE_CLAMP_PRETRAIN_ONLY_NOT_MOVING_BODY_VISION',biological_compound_eye_validity=False)
 fresh(HERE/'LIBRARY.json',result)
 preview=Image.new('RGB',(896,240),color='white');draw=ImageDraw.Draw(preview)
 for i,(cfg,panel) in enumerate(panels):
  draw.text((4,i*120+4),cfg['config_id']+' '+cfg['pose_id']+' off / center / left / right / occludedL / occludedR / wall; cells LEFT|RIGHT',fill='black');preview.paste(panel.resize((896,96),Image.Resampling.NEAREST),(0,i*120+24))
 preview.save(HERE/'NEW14_PANEL.png');print(json.dumps(dict(total_frames=126,old112pixelprefixexact=True,newframes=14,all_occlusion_exact=blocked,physical_steps=0,panel=str(HERE/'NEW14_PANEL.png'))))
if __name__=='__main__':main()
