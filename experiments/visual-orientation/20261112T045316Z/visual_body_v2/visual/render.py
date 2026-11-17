"""Savedactualbody video with true released/actor RGBinsets. No policyforward or integration."""
from pathlib import Path
import argparse,os,sys,subprocess,hashlib,json,importlib.util
from bisect import bisect_right
from contracts import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS);p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY no arrays/Core/render');return
 require(a.condition in IDS,'explicit condition required')
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw,ImageFont
 case=load(a.condition);empty=load('V_empty');t=case['trace'];g=case['geometry'];d=case['decisions'];rgb=case['rgb'];cap=case['captures'];r=case['result'];hashes=case['source_hashes'];hashes.update(empty['source_hashes']);hashes[str(HERE/'render.py')]=sha(HERE/'render.py');require(np.array_equal(t['time'],empty['trace']['time']),'sameemptyclock')
 camera_path=VIS/'camera.py';spec=importlib.util.spec_from_file_location('_visualbody_delivery_camera',camera_path);camera=importlib.util.module_from_spec(spec);sys.modules[spec.name]=camera;spec.loader.exec_module(camera);hashes[str(camera_path)]=sha(camera_path)
 sys.path.insert(0,str(BASE));from physics import Core
 from identity import verify,verify_runtime
 verify(case['identity']);calls={'attempts':0}
 def forbidden(*aa,**kw):calls['attempts']+=1;raise RuntimeError('video prohibits integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden;c=Core(adhesion=False);model=c.plant;verify_runtime(case['identity'],c);require((model.nq,model.nv,model.nu)==(49,48,42),'fullbodydimensions');mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH;require(mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1),'all71opaque')
 head=model.body('Head').id;look=(t['com'].min(0)+t['com'].max(0))/2;extent=0.
 for i in range(0,15400,20):
  c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(model,c.d);extent=max(extent,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-look,axis=1)+model.geom_rbound[mesh])))
 for capture in cap:
  if capture['stimulus']!='off':extent=max(extent,float(np.linalg.norm(np.asarray(capture['target_center_xyz'])-look)+1.3))
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,extent/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=look;option=mj.MjvOption();option.geomgroup[:]=1;option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
 # Optional occluder geometry is spectator decoration derived from the exact recorded anchor.
 wall_center=None;wall_basis=None
 if a.condition=='V_left_occluded':
  geometry_path=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json');ha=read(geometry_path);hashes[str(geometry_path)]=sha(geometry_path);eyes=camera.eye_frames(np.asarray(cap[290]['actual_head_position']),np.asarray(cap[290]['actual_head_rotation']),ha,read(VIS/'PREREG_CAMERA_PROBE.json')['eyes']);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=np.asarray(cap[290]['actual_head_rotation'])[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);wall_center=mid+.4*ex;wall_center[2]=1.4;wall_basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
 out=HERE/('VIDEO_'+a.condition);out.mkdir(exist_ok=False);path=out/'ACTUAL_VISUAL_BODY_WITH_EYES.mp4';oldbuffer=[int(model.vis.global_.offwidth),int(model.vis.global_.offheight)];model.vis.global_.offwidth=960;model.vis.global_.offheight=640;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);renderer=None;writer=None;mapping=[];boxes=[];capturetimes=[x['capture_time_s'] for x in cap];releasetimes=[x['release_time_s'] for x in cap];queries=d['time'].tolist()
 try:
  renderer=mj.Renderer(model,height=640,width=960);writer=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
  for frame,i in enumerate(range(0,15400,20)):
   now=float(t['time'][i]);elapsed=now-5.8;c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(model,c.d);require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'savedposechanged');require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'COM reconstruction')
   rot=c.d.xmat[c.th].reshape(3,3);angle=float(np.arctan2(rot[1,0],rot[0,0]));require(abs(np.arctan2(np.sin(angle-g['thorax_yaw'][i]),np.cos(angle-g['thorax_yaw'][i])))<1e-10,'yaw reconstruction');require(np.allclose(c.d.xpos[head],g['head_xyz'][i],atol=1e-10,rtol=0) and np.allclose(c.d.xmat[head].reshape(3,3),g['head_rotation'][i],atol=1e-10,rtol=0),'actualhead reconstruction')
   renderer.update_scene(c.d,camera=cam,scene_option=option);renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering();mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));yy,xx=np.where(mask);require(len(xx)>0 and xx.min()>5 and xx.max()<954 and yy.min()>92 and yy.max()<540,'fullbodycropped/headeroverlap');boxes.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
   current=bisect_right(capturetimes,now)-1;require(current>=0,'initialcapture metadatarequired');capture=cap[current]
   if capture['stimulus']!='off':camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],capture['target_center_xyz'],np.eye(3),[1.,.08,.03,1.])
   if wall_center is not None and capture.get('counterfactual_available',False):camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_BOX,[.04,1.4,1.4],wall_center,wall_basis,[.12,.12,.12,1.])
   body=Image.fromarray(renderer.render());bd=ImageDraw.Draw(body);bd.rectangle((0,0,960,90),fill='#f6f6f6');bd.rectangle((0,544,960,640),fill='#f6f6f6');source=int(t['control_source'][i]);phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ANN ONLY'}[source];qi=bisect_right(queries,now)-1;actual=bisect_right(releasetimes,now)-1;actor=None if qi<0 else int(d['image_capture_id'][qi]);raw=None if qi<0 else int(d['raw_released_image_capture_id'][qi]);turn=None if qi<0 else float(d['raw_turn'][qi]);bodyyaw=float(np.degrees(case['yaw'][i]));extra=float(np.degrees(case['yaw'][i]-empty['yaw'][i]))
   lines=[a.condition+' | ACTUAL MOVING-HEAD VISION | '+phase,f'Actual {now:.3f}/15.400s | '+(f'ANN {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN')+' | 1.00x / 50fps','World landmark '+capture['stimulus']+' | RGB camera50Hz / release+20ms / GRU5ms','Neural turn '+('N/A before first query' if turn is None else f'{turn:+.4f}')+' | no olfaction / no scripted steering']
   for j,text in enumerate(lines):bd.text((10,2+22*j),text,font=font,fill='#151515')
   timing='Source on .60 -> released .62; off3.60 -> released3.62; recovery4.12-4.62'
   if a.condition=='V_empty':timing='Empty scene throughout; real camera and delayed release continue'
   elif a.condition in ('V_left_to_right','V_right_to_left'):timing='Source on .60 -> released .62; side switch4.80 -> released4.82'
   elif a.condition=='V_left_occluded':timing='Left on .60; wall3.60 -> release3.62; actual occlusion precondition by3.82'
   elif a.condition=='V_flip_fixedframe':timing='Actual side switch4.80; actor holds capture4.78/release4.80; real camera continues'
   flags='Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+' | '+('Control' if case['iscontrol'] else 'Function')+' '+('PASS' if case['functional_pass'] else 'FAIL');bottom=[f'Body delta yaw {bodyyaw:+.3f} deg | minus matched V_empty {extra:+.3f} deg',f'Slip {r["autonomous"]["slip_ratio_max"]:.6f} (limit .5) | '+flags,timing,'Saved actual physics / substitute ANN / bounded PD / NOT biological vision validation']
   for j,text in enumerate(bottom):bd.text((10,546+22*j),text,font=font,fill='#151515')
   image=Image.new('RGB',(1280,720),'white');image.paste(body,(0,0));draw=ImageDraw.Draw(image);draw.text((974,10),'REAL SAVED EYE RGB',font=font,fill='black');draw.text((974,35),'LEFT / RIGHT (32x32, nearest enlargement)',font=small,fill='black')
   def inset(fid,y,title):
    draw.text((974,y),title,font=font,fill='black')
    if fid is None or fid<0:draw.text((974,y+34),'N/A: not yet released / no actor query',font=small,fill='black');return
    require(cap[fid]['release_time_s']<=now+1e-10,'future eyeimage leak');draw.text((974,y+25),f'ID {fid} cap {cap[fid]["capture_time_s"]:.3f}s',font=small,fill='black');draw.text((974,y+43),f'release {cap[fid]["release_time_s"]:.3f}s',font=small,fill='black')
    for eye in range(2):image.paste(Image.fromarray(rgb[fid,eye]).resize((128,128),Image.Resampling.NEAREST),(980+eye*144,y+67))
   inset(actual,80,'ACTUAL RELEASED IMAGE');inset(actor,335,'ACTOR USED IMAGE');frozen=False if qi<0 else bool(d['actor_frame_frozen'][qi]);draw.text((974,555),'Actor image frozen: '+str(frozen),font=small,fill='black');draw.text((974,578),'Warm genuine images may be released',font=small,fill='black');draw.text((974,596),'before first neural query (actor N/A).',font=small,fill='black');draw.text((12,660),'Eyes are actual captured RGB; target coordinates are spectator annotations only, never policy inputs.',font=font,fill='black')
   pixels=np.asarray(image).tobytes();writer.stdin.write(pixels);mapping.append(dict(frame=frame,trace_index=i,time_s=now,decision_index=None if qi<0 else qi,decision_time_s=None if qi<0 else float(d['time'][qi]),actual_released_image_id=None if actual<0 else actual,actor_used_image_id=actor,decision_raw_released_id=raw,world_scene_capture_id=current,actor_frame_frozen=frozen,actual_RGB_pair_sha256=None if actual<0 else cap[actual]['rgb_sha256'],actor_RGB_pair_sha256=None if actor is None else cap[actor]['rgb_sha256'],qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(t['qvel'][i].tobytes()).hexdigest(),rendered_RGB_sha256=hashlib.sha256(pixels).hexdigest()))
  writer.stdin.close();encode_exit=writer.wait();require(encode_exit==0,'ffmpegencode failed')
 finally:
  if writer is not None and writer.poll() is None:writer.stdin.close();writer.wait()
  if renderer is not None:renderer.close()
  model.vis.global_.offwidth,model.vis.global_.offheight=oldbuffer
 verify_runtime(case['identity'],c);require(calls['attempts']==0,'integrationattempted');unchanged(hashes)
 meta=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True).stdout);stream=meta['streams'][0];require(int(stream['nb_read_frames'])==770 and stream['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001,'video770/50/15.4')
 chosen=[0,289,290,320,430,470,480,530,540,769];montage=Image.new('RGB',(2560,3600));decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);count=0
 while True:
  raw=decoder.stdout.read(1280*720*3)
  if not raw:break
  require(len(raw)==1280*720*3,'shortdecodedframe')
  if count in chosen:j=chosen.index(count);montage.paste(Image.frombytes('RGB',(1280,720),raw),((j%2)*1280,(j//2)*720))
  count+=1
 decoder.stdout.close();decode_exit=decoder.wait();require(decode_exit==0 and count==770,'fulldecode770 failed');montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path);unchanged(hashes);fresh(out/'FFPROBE.json',meta);fresh(out/'FRAME_MAPPING.json',mapping);fresh(out/'VIDEO.json',dict(condition=a.condition,source_run=str(case['run']),source_hashes=hashes,video=str(path),video_sha256=sha(path),montage=str(montage_path),montage_sha256=sha(montage_path),frames=770,decoded_frames=count,fps=50,physical_seconds=15.4,playback_seconds=15.4,resolution=[1280,720],physics_steps=0,integration_attempts=0,model_forward_calls=0,compiled_runtime_unchanged=True,source_unchanged=True,all_qpos_qvel_exact_COM_yaw_head_checked=True,all71meshopaque=True,all_frame_fullbody_containment=True,body_silhouette_bounds=boxes,eyeinsets_actualsavedRGB=True,rawreleased_vs_actorused_separate=True,nonfuturelookup=True,normal_vs_control_separate=True,functional_pass=case['functional_pass'],functional_flags=case['functional_flags'],functional_metrics=case['functional_metrics'],engineering_pass=r['engineering_pass'],ffmpeg_exit=encode_exit,full_decode_exit=decode_exit,parent_actual_review='PENDING'))
 print(str(out/'VIDEO.json'))
if __name__=='__main__':main()
