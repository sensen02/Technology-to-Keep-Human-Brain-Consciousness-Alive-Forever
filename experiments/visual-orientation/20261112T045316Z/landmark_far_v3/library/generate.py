"""Parent CPU1EGL only:42 actual binocular far-landmark RGB pairs, zero integration."""
from pathlib import Path
import argparse,json,hashlib,os,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');a=ap.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no rendering');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 if any((HERE/n).exists() for n in ('LIBRARY.json','RGB.npz','INPUTS_BEFORE.json')):raise FileExistsError('inspect partial/completed library, no blind rerun')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw
 sys.path.insert(0,str(ROOT));from camera import eye_frames,render_eye_pair,add_decor
 pr=read(SCOPE/'PREREG.json');cfgs=pr['library']['configs'];cp=read(ROOT/'PREREG_CAMERA_PROBE.json');base=Path(cp['body_base']);run=Path(cp['trajectory']);sys.path.insert(0,str(base));from physics import Core
 from identity import capture,verify,verify_runtime
 old=read(run/'IDENTITY.json');verify(old)
 paths=[SCOPE/'PREREG.json',HERE/'generate.py',HERE/'audit.py',ROOT/'camera.py',ROOT/'PREREG_CAMERA_PROBE.json',ROOT/'CAMERA_PROBE/PREFLIGHT.json',run/'IDENTITY.json',run/'trace_1ms.npz',Path(cp['head_geometry_audit'])]
 hashes={str(p):sha(p) for p in paths};fresh(HERE/'INPUTS_BEFORE.json',hashes)
 calls={'attempts':0}
 def forbidden(*aa,**kw):calls['attempts']+=1;raise RuntimeError('far library forbids integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 core=Core(adhesion=False);model=core.plant;verify_runtime(old,core);mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
 if (model.nq,model.nv,model.nu)!=(49,48,42) or mesh.sum()!=71 or not np.all(model.geom_rgba[mesh,3]==1):raise ValueError('complete71opaque frozen body required')
 ident=capture(core,dict(kind=pr['library']['kind'],source_hashes=hashes,physics_steps=0),extra_paths=paths);fresh(HERE/'PHYSICAL_IDENTITY.json',ident)
 with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:qp=z['qpos'][[c['saved_pose_index'] for c in cfgs]].copy();qv=z['qvel'][[c['saved_pose_index'] for c in cfgs]].copy();times=z['time'][[c['saved_pose_index'] for c in cfgs]].copy()
 geometry=read(cp['head_geometry_audit']);spec=cp['eyes'];head=model.body('Head').id;rows=[];frames=[];diagnostics=[];panels=[];renderer=None
 try:
  renderer=mj.Renderer(model,height=32,width=32)
  for j,cfg in enumerate(cfgs):
   core.d.qpos[:]=qp[j];core.d.qvel[:]=qv[j];core.d.time=float(times[j]);mj.mj_forward(model,core.d)
   if not np.array_equal(core.d.qpos,qp[j]) or not np.array_equal(core.d.qvel,qv[j]):raise ValueError('saved pose changed')
   hp=core.d.xpos[head].copy();hr=core.d.xmat[head].reshape(3,3).copy();eyes=eye_frames(hp,hr,geometry,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=hr[:,0].copy();ex[2]=0;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
   centers={}
   for name,sign in (('left',1),('right',-1),('center',0)):
    p=mid+cfg['forward']*ex+sign*cfg['forward']*np.tan(cfg['angle'])*ey;p[2]=1.2;centers[name]=p
   boxes=np.array([centers['left'].copy(),centers['right'].copy()]);boxes[:,2]=1.4;half=np.array([.18,.18,1.4]);clearance=[]
   for eye in eyes:
    for box in boxes:clearance.append(float(np.linalg.norm(np.maximum(np.abs((eye['pos']-box)@basis)-half,0.))))
   eyeoutside=min(clearance)>.005;case={};panel=Image.new('RGB',(448,64),'white')
   for col,stim in enumerate(pr['library']['stimuli']):
    sign=1 if stim in ('left','occluded_left') else -1 if stim in ('right','occluded_right') else 0;present=stim not in ('off','occluder_only');hidden=stim.startswith('occluded_') or stim=='occluder_only';target=centers['left' if sign==1 else 'right' if sign==-1 else 'center']
    def decor(scene):
     if present:add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],target,np.eye(3),[1.,.08,.03,1.])
     if hidden:
      for box in boxes:add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,half,box,basis,[.12,.12,.12,1.])
    rgb,meta=render_eye_pair(renderer,model,core.d,eyes,spec,decor);fid=len(frames);frames.append(rgb.copy());case[stim]=rgb
    rows.append(dict(frame_id=fid,config_id=cfg['config_id'],split=cfg['split'],stimulus=stim,pose_id=cfg['pose_id'],actual_saved_pose_index=cfg['saved_pose_index'],capture_time_s=float(times[j]),actual_head_position=hp.tolist(),actual_head_rotation=hr.tolist(),camera_metadata=meta,bearing_rad=sign*cfg['angle'] if present else None,target_visible=present,environment_target_world_xyz=target.tolist() if present else None,closed_box_centers_xyz=boxes.tolist() if hidden else [],closed_box_rotation=basis.tolist(),closed_box_halfsize=half.tolist(),eye_outside_boxes=eyeoutside,eye_box_min_clearance=min(clearance),box_eye_clearances=clearance,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    panel.paste(Image.fromarray(np.concatenate((rgb[0],rgb[1]),axis=1)),(col*64,0))
   for row in rows[-7:]:
    if row['stimulus'] in ('occluded_left','occluded_right'):row['target_visible']=not np.array_equal(case[row['stimulus']],case['occluder_only'])
   diagnostics.append(dict(config_id=cfg['config_id'],left_visible_meanabs=float(np.abs(case['left'].astype(float)-case['off']).mean()),right_visible_meanabs=float(np.abs(case['right'].astype(float)-case['off']).mean()),center_visible_meanabs=float(np.abs(case['center'].astype(float)-case['off']).mean()),occluded_left_equals_boxonly=bool(np.array_equal(case['occluded_left'],case['occluder_only'])),occluded_right_equals_boxonly=bool(np.array_equal(case['occluded_right'],case['occluder_only'])),eye_outside_boxes=eyeoutside,eye_box_min_clearance=min(clearance)))
   panels.append((cfg,panel))
 finally:
  if renderer is not None:renderer.close()
 verify_runtime(old,core);verify_runtime(ident,core)
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('input changed '+p)
 rgb=np.stack(frames)
 if rgb.shape!=(42,2,32,32,3):raise ValueError('42 exact real RGB pairs')
 np.savez_compressed(HERE/'RGB.npz',rgb=rgb)
 blocked=all(d['occluded_left_equals_boxonly'] and d['occluded_right_equals_boxonly'] and d['eye_outside_boxes'] for d in diagnostics)
 result=dict(kind=pr['library']['kind'],rgb_path=str(HERE/'RGB.npz'),rgb_sha256=sha(HERE/'RGB.npz'),rows=rows,eye_order=['LEFT','RIGHT'],camera_spec=spec,source_hashes=hashes,physical_identity=str(HERE/'PHYSICAL_IDENTITY.json'),physical_identity_sha256=sha(HERE/'PHYSICAL_IDENTITY.json'),physics_steps=0,integration_attempts=calls['attempts'],compiled_unchanged=True,Core_constructions=1,model_forward_calls=0,target_fully_occluded_all=blocked,invisible_label_training_admitted=blocked,occlusion_diagnostics=diagnostics,unique_binocular_frames=42,raw_RGB_renders=84,scope='FAR_LANDMARK_STATIC_HEAD_PRETRAIN_ONLY_NOT_NEAR_NAVIGATION',old_near_failure_unchanged=True)
 fresh(HERE/'LIBRARY.json',result);preview=Image.new('RGB',(896,720),'white');draw=ImageDraw.Draw(preview)
 for j,(cfg,panel) in enumerate(panels):
  draw.text((4,j*120+4),cfg['config_id']+' '+cfg['pose_id']+' FAR: off/center/left/right/occludedL/occludedR/bothBoxes LEFT|RIGHT',fill='black');preview.paste(panel.resize((896,96),Image.Resampling.NEAREST),(0,j*120+24))
 preview.save(HERE/'PANEL.png');print(json.dumps(dict(pair_count=42,physics_steps=0,all_occlusion_exact=blocked,panel=str(HERE/'PANEL.png'))))
if __name__=='__main__':main()
