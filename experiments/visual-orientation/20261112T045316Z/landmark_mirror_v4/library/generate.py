"""Parentonly: reuse42actualRGB exactly and append14new farvalidation frames,zero steps."""
from pathlib import Path
import argparse,json,hashlib,os,sys,copy
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'landmark_far_v3/library'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');args=ap.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no rendering');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 if any((HERE/n).exists() for n in ('RGB.npz','LIBRARY.json','INPUTS_BEFORE.json')):raise FileExistsError('inspect existing outputs no blind regenerate')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw
 sys.path.insert(0,str(ROOT));from camera import eye_frames,render_eye_pair,add_decor
 pr=read(SCOPE/'PREREG.json');cp=read(ROOT/'PREREG_CAMERA_PROBE.json');old=read(OLD/'LIBRARY.json');oa=read(OLD/'VERIFY.json')
 if old['kind']!='VISION_FAR_HEAD_LIBRARY_V3' or not oa['integrity_all_pass'] or not all(oa['checks'].values()) or oa['library_sha256']!=sha(OLD/'LIBRARY.json') or oa['rgb_sha256']!=sha(OLD/'RGB.npz'):raise ValueError('old actual42 independent proof')
 hashes={}
 for p in (old,oa):
  for path,h in p['source_hashes'].items():
   if sha(path)!=h:raise ValueError('old source drift '+path)
   hashes[path]=h
 base=Path(cp['body_base']);run=Path(cp['trajectory']);sys.path.insert(0,str(base));from physics import Core
 from identity import capture,verify,verify_runtime
 ident=read(run/'IDENTITY.json');verify(ident)
 for p in [SCOPE/'PREREG.json',HERE/'generate.py',HERE/'audit.py',ROOT/'camera.py',ROOT/'PREREG_CAMERA_PROBE.json',OLD/'LIBRARY.json',OLD/'RGB.npz',OLD/'VERIFY.json',OLD/'PHYSICAL_IDENTITY.json',run/'IDENTITY.json',run/'trace_1ms.npz',Path(cp['head_geometry_audit'])]:hashes[str(p)]=sha(p)
 fresh(HERE/'INPUTS_BEFORE.json',hashes)
 with np.load(OLD/'RGB.npz',allow_pickle=False) as z:prefix=z['rgb'].copy()
 if prefix.shape!=(42,2,32,32,3):raise ValueError('old42prefix')
 rows=copy.deepcopy(old['rows'])
 for i,r in enumerate(rows):
  if r['frame_id']!=i or hashlib.sha256(prefix[i].tobytes()).hexdigest()!=r['rgb_sha256']:raise ValueError('prefix row identity')
  r['source_split']=r['split'];r['split']='train' if r['config_id'].startswith('T') else 'development';r['parent_frame_id']=i
 calls={'attempts':0}
 def forbidden(*aa,**kw):calls['attempts']+=1;raise RuntimeError('library prohibits steps')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 c=Core(adhesion=False);m=c.plant;verify_runtime(ident,c);mesh=m.geom_type==mj.mjtGeom.mjGEOM_MESH
 if (m.nq,m.nv,m.nu)!=(49,48,42) or mesh.sum()!=71 or not np.all(m.geom_rgba[mesh,3]==1):raise ValueError('complete frozen body')
 newident=capture(c,dict(kind=pr['library']['kind'],source_hashes=hashes,physics_steps=0,newRGBpairs14=True),extra_paths=list(hashes));fresh(HERE/'PHYSICAL_IDENTITY.json',newident)
 configs=pr['library']['new_configs'];indices=[cfg['saved_pose_index'] for cfg in configs]
 with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:qp=z['qpos'][indices].copy();qv=z['qvel'][indices].copy();times=z['time'][indices].copy()
 spec=cp['eyes'];ha=read(cp['head_geometry_audit']);head=m.body('Head').id;new=[];diagnostics=[];panels=[];renderer=None
 try:
  renderer=mj.Renderer(m,height=32,width=32)
  for j,cfg in enumerate(configs):
   c.d.qpos[:]=qp[j];c.d.qvel[:]=qv[j];c.d.time=float(times[j]);mj.mj_forward(m,c.d)
   if not np.array_equal(c.d.qpos,qp[j]) or not np.array_equal(c.d.qvel,qv[j]):raise ValueError('saved pose changed')
   hp=c.d.xpos[head].copy();hr=c.d.xmat[head].reshape(3,3).copy();eyes=eye_frames(hp,hr,ha,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=hr[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);basis=np.column_stack((ex,ey,np.array([0.,0.,1.])));centers={}
   for name,sign in (('left',1),('right',-1),('center',0)):
    center=mid+cfg['forward']*ex+sign*cfg['forward']*np.tan(cfg['angle'])*ey;center[2]=1.2;centers[name]=center
   boxes=np.asarray([centers['left'].copy(),centers['right'].copy()]);boxes[:,2]=1.4;half=np.array([.18,.18,1.4]);clearance=[float(np.linalg.norm(np.maximum(abs((e['pos']-b)@basis)-half,0.))) for e in eyes for b in boxes];outside=min(clearance)>.005;case={};panel=Image.new('RGB',(448,64),'white')
   for col,stim in enumerate(pr['library']['stimuli']):
    sign=1 if stim in ('left','occluded_left') else -1 if stim in ('right','occluded_right') else 0;present=stim not in ('off','occluder_only');hidden=stim.startswith('occluded_') or stim=='occluder_only';target=centers['left' if sign>0 else 'right' if sign<0 else 'center']
    def decorate(scene):
     if present:add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],target,np.eye(3),[1.,.08,.03,1.])
     if hidden:
      for b in boxes:add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,half,b,basis,[.12,.12,.12,1.])
    rgb,meta=render_eye_pair(renderer,m,c.d,eyes,spec,decorate);fid=42+len(new);new.append(rgb.copy());case[stim]=rgb
    rows.append(dict(frame_id=fid,config_id=cfg['config_id'],split='validation',source_split='newvalidation',stimulus=stim,pose_id=cfg['pose_id'],actual_saved_pose_index=cfg['saved_pose_index'],capture_time_s=float(times[j]),actual_head_position=hp.tolist(),actual_head_rotation=hr.tolist(),camera_metadata=meta,bearing_rad=sign*cfg['angle'] if present else None,target_visible=present,environment_target_world_xyz=target.tolist() if present else None,closed_box_centers_xyz=boxes.tolist() if hidden else [],closed_box_rotation=basis.tolist(),closed_box_halfsize=half.tolist(),eye_outside_boxes=outside,eye_box_min_clearance=min(clearance),box_eye_clearances=clearance,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    panel.paste(Image.fromarray(np.concatenate((rgb[0],rgb[1]),1)),(col*64,0))
   for r in rows[-7:]:
    if r['stimulus'].startswith('occluded_'):r['target_visible']=not np.array_equal(case[r['stimulus']],case['occluder_only'])
   diagnostics.append(dict(config_id=cfg['config_id'],visible_left_meanabs=float(np.abs(case['left'].astype(float)-case['off']).mean()),visible_right_meanabs=float(np.abs(case['right'].astype(float)-case['off']).mean()),visible_center_meanabs=float(np.abs(case['center'].astype(float)-case['off']).mean()),occluded_left_equals_boxonly=bool(np.array_equal(case['occluded_left'],case['occluder_only'])),occluded_right_equals_boxonly=bool(np.array_equal(case['occluded_right'],case['occluder_only'])),eye_outside_boxes=outside,eye_box_min_clearance=min(clearance)));panels.append((cfg,panel))
 finally:
  if renderer is not None:renderer.close()
 verify_runtime(ident,c);verify_runtime(newident,c)
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('source changed '+p)
 rgb=np.concatenate((prefix,np.stack(new)),axis=0)
 if rgb.shape!=(56,2,32,32,3) or not np.array_equal(rgb[:42],prefix):raise ValueError('old42prefix exact')
 np.savez_compressed(HERE/'RGB.npz',rgb=rgb);blocked=old['target_fully_occluded_all'] and all(x['occluded_left_equals_boxonly'] and x['occluded_right_equals_boxonly'] and x['eye_outside_boxes'] for x in diagnostics)
 manifest=dict(kind=pr['library']['kind'],rgb_path=str(HERE/'RGB.npz'),rgb_sha256=sha(HERE/'RGB.npz'),rows=rows,eye_order=['LEFT','RIGHT'],camera_spec=spec,source_hashes=hashes,physical_identity=str(HERE/'PHYSICAL_IDENTITY.json'),physical_identity_sha256=sha(HERE/'PHYSICAL_IDENTITY.json'),physics_steps=0,integration_attempts=calls['attempts'],compiled_unchanged=True,Core_constructions=1,model_forward_calls=0,unique_binocular_frames=56,reused_actual_pairs=42,new_actual_pairs=14,new_RGB_renders=28,parent_library=str(OLD/'LIBRARY.json'),parent_library_sha256=sha(OLD/'LIBRARY.json'),old42pixelprefixexact=True,target_fully_occluded_all=blocked,invisible_label_training_admitted=blocked,occlusion_diagnostics=old['occlusion_diagnostics']+diagnostics,digital_augmented_exposures=0,scope='ACTUAL_FAR_RGB_LIBRARY_NOT_DIGITALMIRRORS_OR_BODYQUALIFICATION')
 fresh(HERE/'LIBRARY.json',manifest);im=Image.new('RGB',(896,240),'white');draw=ImageDraw.Draw(im)
 for j,(cfg,panel) in enumerate(panels):draw.text((4,j*120+4),cfg['config_id']+' '+cfg['pose_id']+' NEW ACTUAL RGB off/center/left/right/occL/occR/boxonly LEFT|RIGHT',fill='black');im.paste(panel.resize((896,96),Image.Resampling.NEAREST),(0,j*120+24))
 im.save(HERE/'NEW14_PANEL.png');print(json.dumps(dict(total_actual_pairs=56,old42exact=True,new14=True,all_occlusion_exact=blocked,physical_steps=0,panel=str(HERE/'NEW14_PANEL.png'))))
if __name__=='__main__':main()
