"""Parent-only real static-head RGB library. No heavy imports until explicit execution."""
from pathlib import Path
import argparse,hashlib,json,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def fresh(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')

def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no rendering or data loaded');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 if (HERE/'RGB.npz').exists() or (HERE/'LIBRARY.json').exists():raise FileExistsError('inspect existing library; no blind regenerate')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw
 sys.path.insert(0,str(ROOT));from camera import eye_frames,render_eye_pair,add_decor
 pr=read(HERE/'PREREG.json');camera_pr=read(ROOT/'PREREG_CAMERA_PROBE.json');old_probe=read(ROOT/'CAMERA_PROBE/PREFLIGHT.json')
 for path,h in old_probe['source_hashes'].items():
  if sha(path)!=h:raise ValueError('successful camera input drift '+path)
 base=Path(camera_pr['body_base']);run=Path(camera_pr['trajectory']);sys.path.insert(0,str(base));from physics import Core
 from identity import verify,verify_runtime,capture
 ident=read(run/'IDENTITY.json');verify(ident)
 inputs=[HERE/n for n in ('PREREG.json','generate.py','audit.py')]+[ROOT/'camera.py',ROOT/'PREREG_CAMERA_PROBE.json',ROOT/'CAMERA_PROBE/PREFLIGHT.json',run/'IDENTITY.json',run/'trace_1ms.npz',Path(camera_pr['head_geometry_audit'])]
 hashes={str(p):sha(p) for p in inputs};fresh(HERE/'INPUTS_BEFORE.json',hashes)
 calls={'integration_attempts':0}
 def forbidden(*a,**kw):calls['integration_attempts']+=1;raise RuntimeError('statichead library forbids physical integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 c=Core(adhesion=False);model=c.plant;verify_runtime(ident,c)
 mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
 if (model.nq,model.nv,model.nu)!=(49,48,42) or mesh.sum()!=71 or not np.all(model.geom_rgba[mesh,3]==1.):raise ValueError('complete71opaque body required')
 identity=capture(c,dict(kind=pr['kind'],source_hashes=hashes,physics_steps=0),extra_paths=inputs);fresh(HERE/'PHYSICAL_IDENTITY.json',identity)
 with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
  qp=z['qpos'][pr['pose_indices']].copy();qv=z['qvel'][pr['pose_indices']].copy();times=z['time'][pr['pose_indices']].copy()
 head=model.body('Head').id;geometry=read(camera_pr['head_geometry_audit']);spec=camera_pr['eyes'];targetspec=camera_pr['visual_targets'];rows=[];frames=[];diagnostics=[];renderer=None;panels=[]
 try:
  renderer=mj.Renderer(model,height=32,width=32)
  for cfg in pr['configs']:
   for pid in cfg['pose_ids']:
    c.d.qpos[:]=qp[pid];c.d.qvel[:]=qv[pid];c.d.time=float(times[pid]);mj.mj_forward(model,c.d)
    if not np.array_equal(c.d.qpos,qp[pid]) or not np.array_equal(c.d.qvel,qv[pid]):raise ValueError('saved pose changed')
    hp=c.d.xpos[head].copy();hr=c.d.xmat[head].reshape(3,3).copy();eyes=eye_frames(hp,hr,geometry,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2
    ex=hr[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
    wall=mid+cfg['forward']*.5*ex;wall[2]=1.4;caseframes={};panel=Image.new('RGB',(448,80),color='white');draw=ImageDraw.Draw(panel)
    for col,stimulus in enumerate(pr['stimuli']):
     sign=1 if stimulus in ('left','occluded_left') else -1 if stimulus in ('right','occluded_right') else 0
     present=stimulus in ('left','right','center','occluded_left','occluded_right');occluded=stimulus.startswith('occluded_') or stimulus=='occluder_only'
     center=mid+cfg['forward']*ex+sign*cfg['side']*ey;center[2]=targetspec['center_z']
     def decor(scene):
      if present:add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[targetspec['radius'],targetspec['height']/2,0],center,np.eye(3),targetspec['color_rgba'])
      if occluded:add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,[.04,1.4,1.4],wall,basis,[.12,.12,.12,1.])
     rgb,camera_metadata=render_eye_pair(renderer,model,c.d,eyes,spec,decor)
     fid=len(frames);frames.append(rgb);caseframes[stimulus]=rgb;row=dict(frame_id=fid,config_id=cfg['config_id'],split=cfg['split'],stimulus=stimulus,pose_id='pose'+str(pid),actual_saved_pose_index=pr['pose_indices'][pid],capture_time_s=float(times[pid]),actual_head_position=hp.tolist(),actual_head_rotation=hr.tolist(),actualheadpose=dict(xpos=hp.tolist(),xmat=hr.tolist()),target_visible=present,camera_metadata=camera_metadata,bearing_rad=float(np.arctan2(sign*cfg['side'],cfg['forward'])) if present else None,environment_target_world_xyz=center.tolist() if present else None,decor=dict(target_present=present,occluder_present=occluded,target_radius=targetspec['radius'],target_height=targetspec['height'],target_rgba=targetspec['color_rgba'],occluder_center=wall.tolist() if occluded else None,occluder_halfsize=[.04,1.4,1.4] if occluded else None),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),selected_for_policy_sequences=pid==cfg['policy_selected_pose_id'],unused_by_policytraining=pid!=cfg['policy_selected_pose_id'])
     rows.append(row);draw.text((col*64,0),stimulus.replace('occluded_','occ_')[:10],fill='black');panel.paste(Image.fromarray(np.concatenate((rgb[0],rgb[1]),axis=1)),(col*64,16))
    diagnostics.append(dict(config_id=cfg['config_id'],pose_id=pid,left_visible_meanabs=float(np.abs(caseframes['left'].astype(float)-caseframes['off']).mean()),right_visible_meanabs=float(np.abs(caseframes['right'].astype(float)-caseframes['off']).mean()),center_visible_meanabs=float(np.abs(caseframes['center'].astype(float)-caseframes['off']).mean()),occluded_left_equals_wall=bool(np.array_equal(caseframes['occluded_left'],caseframes['occluder_only'])),occluded_right_equals_wall=bool(np.array_equal(caseframes['occluded_right'],caseframes['occluder_only']))))
    for row in rows[-7:]:
     if row['stimulus']=='occluded_left':row['target_visible']=not np.array_equal(caseframes['occluded_left'],caseframes['occluder_only'])
     if row['stimulus']=='occluded_right':row['target_visible']=not np.array_equal(caseframes['occluded_right'],caseframes['occluder_only'])
    panels.append((cfg['config_id'],pid,panel))
 finally:
  if renderer is not None:renderer.close()
 verify_runtime(ident,c);verify_runtime(identity,c)
 for path,h in hashes.items():
  if sha(path)!=h:raise ValueError('source/input drift during optical library '+path)
 if len(frames)!=112:raise ValueError('fixed112 binocular exposures')
 path=HERE/'RGB.npz';np.savez_compressed(path,rgb=np.stack(frames));allblocked=all(d['occluded_left_equals_wall'] and d['occluded_right_equals_wall'] for d in diagnostics)
 manifest=dict(kind=pr['kind'],rgb_path=str(path),rgb_sha256=sha(path),rows=rows,camera_spec=spec,eye_order=pr['eye_order'],source_hashes=hashes,physical_identity=str(HERE/'PHYSICAL_IDENTITY.json'),physical_identity_sha256=sha(HERE/'PHYSICAL_IDENTITY.json'),compiled_unchanged=True,compiled_physics_before_after_unchanged=True,physics_steps=0,integration_attempts=calls['integration_attempts'],model_forward_calls=0,optimizer_updates=0,Core_constructions=1,unique_binocular_frames=112,raw_RGB_renders=224,occlusion_diagnostics=diagnostics,target_fully_occluded_all=allblocked,invisible_label_training_admitted=allblocked,policy_train_configs=[dict(config_id=c['config_id'],pose_id='pose'+str(c['policy_selected_pose_id'])) for c in pr['configs'] if c['split']=='train'],policy_validation_configs=[dict(config_id=c['config_id'],pose_id='pose'+str(c['policy_selected_pose_id'])) for c in pr['configs'] if c['split']=='validation'],scope='STATIC_HEAD_IMAGE_CLAMP_PRETRAIN_ONLY_NOT_REAL_BODY_VISION_CLOSED_LOOP',biological_compound_eye_validity=False)
 fresh(HERE/'LIBRARY.json',manifest)
 image=Image.new('RGB',(896,100*len(panels)),color='white');draw=ImageDraw.Draw(image)
 for i,(cid,pid,panel) in enumerate(panels):
  draw.text((4,100*i),f'{cid} pose{pid}: off / center / left / right / occludedL / occludedR / wallOnly; each cell L|R',fill='black');image.paste(panel.resize((896,80),Image.Resampling.NEAREST),(0,100*i+20))
 image.save(HERE/'LIBRARY_PANEL.png')
 print(json.dumps(dict(frames=112,physics_steps=0,compiled_unchanged=True,all_occlusion_exact=allblocked,manifest=str(HERE/'LIBRARY.json'),panel=str(HERE/'LIBRARY_PANEL.png'))))
if __name__=='__main__':main()
