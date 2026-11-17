"""Same-run body pose and real delayed actor-eye RGB, zero policyforward/integration."""
import argparse,os,sys,subprocess,importlib.util
from contracts import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=VIDEO_IDS,required=True);p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY noarrays/Core/render');return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw,ImageFont
 case=load(a.condition);empty=load('V_empty');t=case['trace'];d=case['decisions'];g=case['geometry'];rgb=case['rgb'];caps=case['captures'];r=case['result'];hashes=case['source_hashes'];hashes.update(empty['source_hashes']);require(np.array_equal(t['time'],empty['trace']['time']),'Matchedemptyclock')
 spec=importlib.util.spec_from_file_location('_neck_rgb_visual_camera',VIS/'camera.py');camera=importlib.util.module_from_spec(spec);sys.modules[spec.name]=camera;spec.loader.exec_module(camera)
 sys.path.insert(0,str(BASE));from physics import Core
 from identity import verify,verify_runtime
 verify(case['identity']);attempts={'count':0}
 def forbidden(*x,**kw):attempts['count']+=1;raise RuntimeError('Savedvideo prohibits integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden;c=Core(adhesion=False);model=c.plant;verify_runtime(case['identity'],c);require((model.nq,model.nv,model.nu)==(49,48,42),'Fullbodydimensions');mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH;require(mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1),'All71opaque');head=model.body('Head').id;look=(t['com'].min(0)+t['com'].max(0))/2;extent=0.
 for i in range(0,15400,20):
  c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(model,c.d);extent=max(extent,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-look,axis=1)+model.geom_rbound[mesh])))
 for cap in caps:
  if cap['stimulus']!='off':extent=max(extent,float(np.linalg.norm(np.asarray(cap['target_center_xyz'])-look)+1.5))
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,extent/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=look;option=mj.MjvOption();option.geomgroup[:]=1;option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
 out=HERE/('VIDEO_'+a.condition);out.mkdir(exist_ok=False);path=out/'ACTUAL_RGB_BRAIN_NECK_BODY.mp4';oldbuffer=[int(model.vis.global_.offwidth),int(model.vis.global_.offheight)];model.vis.global_.offwidth=960;model.vis.global_.offheight=640;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);renderer=None;writer=None;mapping=[];boxes=[]
 try:
  renderer=mj.Renderer(model,height=640,width=960);writer=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
  for frame,i in enumerate(range(0,15400,20)):
   physical_ms=i+1;now=float(t['time'][i]);elapsed=now-5.8;qi=min(3079,physical_ms//5);require(float(d['time'][qi])<=now+1e-8,'Nonfuture5mslookup');current=min(769,physical_ms//20);capture=caps[current];require(capture['capture_time_s']<=now+1e-8,'Currentworldcaptureknown')
   c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(model,c.d);require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'Savedposechanged');require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'COMreconstruction');rot=c.d.xmat[c.th].reshape(3,3);yaw=float(np.arctan2(rot[1,0],rot[0,0]));require(abs(np.arctan2(np.sin(yaw-g['thorax_yaw'][i]),np.cos(yaw-g['thorax_yaw'][i])))<1e-10,'Yawreconstruction');require(np.allclose(c.d.xpos[head],g['head_xyz'][i],atol=1e-10,rtol=0) and np.allclose(c.d.xmat[head].reshape(3,3),g['head_rotation'][i],atol=1e-10,rtol=0),'Actualheadreconstruction')
   renderer.update_scene(c.d,camera=cam,scene_option=option);renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering();mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));yy,xx=np.where(mask);require(len(xx)>0 and xx.min()>5 and xx.max()<954 and yy.min()>92 and yy.max()<540,'Fullbodycroppedorheaderoverlap');boxes.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
   if capture['stimulus']!='off':camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],capture['target_center_xyz'],np.eye(3),[1.,.08,.03,1.])
   covers=bool(capture.get('counterfactual_available',False))
   if covers:
    require(a.condition=='V_left_occluded' and len(capture['cover_rotation'])==3,'Recordedfarclosedcoverevidence');basis=np.asarray(capture['cover_rotation']);require(bool(g['source_anchor_valid'][i]),'Savedworldanchorforcovers')
    for center in g['target_centers_xyz'][i][:2]:
     center=center.copy();center[2]=1.4;camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_BOX,[.18,.18,1.4],center,basis,[.12,.12,.12,1.])
   body=Image.fromarray(renderer.render());bd=ImageDraw.Draw(body);bd.rectangle((0,0,960,90),fill='#f6f6f6');bd.rectangle((0,544,960,640),fill='#f6f6f6');phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ACTUAL RGB BRAIN -> NECK -> OTHER'}[int(t['control_source'][i])];valid=bool(d['image_valid'][qi]);actor=int(d['image_capture_id'][qi]) if valid else None;raw=int(d['raw_released_image_capture_id'][qi]) if valid else None;turn=float(d['raw_turn'][qi]);vu=float(d['visual_raw_turn'][qi]);brainnorm=float(np.linalg.norm(d['brain_hidden_after'][qi]));extra=float(np.degrees(case['yaw'][i]-empty['yaw'][i]));actualyaw=float(np.degrees(case['yaw'][i]));frozen=bool(d['actor_frame_frozen'][qi])
   lines=[a.condition+' | '+phase,f'Actual {now:.3f}/15.400s | '+(f'ANN {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN')+' | 1.00x / 50fps',f'ActualRGB vision {vu:+.4f} -> brain16 -> DN2 -> learned turn {turn:+.4f}',f'304 memories + cache/ports; world scene {capture["stimulus"]}, closed far covers {covers}']
   for j,text in enumerate(lines):bd.text((10,2+22*j),text,font=font,fill='#151515')
   functional='DESCRIPTIVE CUT' if case['functional_pass'] is None else ('PASS' if case['functional_pass'] else 'FAIL');bottom=[f'Body delta yaw {actualyaw:+.3f} deg | minus matched V_empty {extra:+.3f} deg',f'Original slip {r["autonomous"]["slip_ratio_max"]:.6f} (limit .5) | Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+f' | Visual {functional}',f'Body-stable {r["functional_stable_body"]}; camera50Hz/+20ms; neck+2ms/5msneural consumption','OTHER retains local basewalking; NOT full-neck/native retina/general navigation validation']
   for j,text in enumerate(bottom):bd.text((10,546+22*j),text,font=font,fill='#151515')
   image=Image.new('RGB',(1280,720),'white');image.paste(body,(0,0));draw=ImageDraw.Draw(image);draw.text((972,8),'ACTUAL ACTOR EYE RGB',font=font,fill='black');draw.text((972,31),f'valid={valid} actorID={actor} rawID={raw}',font=small,fill='black')
   if valid:
    require(actor is not None and raw is not None and 0<=actor<770 and 0<=raw<770,'NonnegativegenuineimageID');require(caps[actor]['release_time_s']<=float(d['time'][qi])+1e-8 and caps[raw]['release_time_s']<=float(d['time'][qi])+1e-8,'No futureactororrawimage');draw.text((972,50),f'cap {caps[actor]["capture_time_s"]:.3f}s release {caps[actor]["release_time_s"]:.3f}s',font=small,fill='black');draw.text((972,68),f'neural consume {float(d["time"][qi]):.3f}s frozen={frozen}',font=small,fill='black')
    for eye in range(2):image.paste(Image.fromarray(rgb[actor,eye]).resize((128,128),Image.Resampling.NEAREST),(980+eye*144,100))
    draw.text((972,244),f'Raw released RGB ID{raw} (camera continues)',font=small,fill='black')
    for eye in range(2):image.paste(Image.fromarray(rgb[raw,eye]).resize((64,64),Image.Resampling.NEAREST),(1000+eye*128,272))
   else:
    require(int(d['image_capture_id'][qi])==-1 and physical_ms<20,'Onlyinitialinvalid exposure');draw.text((980,114),'NO RELEASED RGB',font=font,fill='black');draw.text((980,143),'Placeholder, not a retinal exposure',font=small,fill='black')
   details=[f'Vision raw {vu:+.6f} (brain input)',f'Brain16 norm {brainnorm:.6f}',f'DN held {d["DN_held"][qi].tolist()}',f'AN held {d["AN_held"][qi].tolist()}',f'Decoded turn {turn:+.6f}',f'Capture/release/consume step (ms):',f'DN {int(d["DN_capture_step"][qi])}/{int(d["DN_release_step"][qi])}/{int(d["physical_step_ms"][qi])}',f'AN {int(d["AN_capture_step"][qi])}/{int(d["AN_release_step"][qi])}/{int(d["physical_step_ms"][qi])}',f'Cut DN={bool(d["DN_cut"][qi])} AN={bool(d["AN_cut"][qi])}', '16CNNcache / AN1 / DN2 separate', 'Real source IDs, engineered transport']
   for j,text in enumerate(details):draw.text((972,362+22*j),text,font=small,fill='black')
   draw.text((12,666),'Same-run saved physics; eye insets are actual delayed captures. Target geometry is spectator-only, native units.',font=font,fill='black');draw.text((12,692),'No policy forward or integration in video. Original slip and functional failures remain visible.',font=small,fill='black');pixels=np.asarray(image).tobytes();writer.stdin.write(pixels)
   mapping.append(dict(frame=frame,trace_index=i,time_s=now,physical_ms=physical_ms,decision_index=qi,decision_time_s=float(d['time'][qi]),world_scene_capture_id=current,world_closedcovers=covers,actor_valid=valid,actor_used_image_id=actor,raw_released_image_id=raw,actor_frame_frozen=frozen,actor_RGB_pair_sha256=None if actor is None else caps[actor]['rgb_sha256'],raw_RGB_pair_sha256=None if raw is None else caps[raw]['rgb_sha256'],qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(t['qvel'][i].tobytes()).hexdigest(),rendered_RGB_sha256=hashlib.sha256(pixels).hexdigest()))
  writer.stdin.close();encode_exit=writer.wait();require(encode_exit==0,'FFmpegencode')
 finally:
  if writer is not None and writer.poll() is None:writer.stdin.close();writer.wait()
  if renderer is not None:renderer.close()
  model.vis.global_.offwidth,model.vis.global_.offheight=oldbuffer
 verify_runtime(case['identity'],c);require(attempts['count']==0,'Integrationattempted');unchanged(hashes)
 meta=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True).stdout);st=meta['streams'][0];require(int(st['nb_read_frames'])==770 and st['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001,'Actual770frame50fps15.4s')
 chosen=[0,1,49,289,290,320,370,470,530,769];montage=Image.new('RGB',(2560,3600));decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);count=0
 while True:
  rawbytes=decoder.stdout.read(1280*720*3)
  if not rawbytes:break
  require(len(rawbytes)==1280*720*3,'Shortdecodedframe')
  if count in chosen:j=chosen.index(count);montage.paste(Image.frombytes('RGB',(1280,720),rawbytes),((j%2)*1280,(j//2)*720))
  count+=1
 decoder.stdout.close();decode_exit=decoder.wait();require(decode_exit==0 and count==770,'Full770decode');mp=out/'DECODED_MONTAGE.png';montage.save(mp);unchanged(hashes);fresh(out/'FFPROBE.json',meta);fresh(out/'FRAME_MAPPING.json',mapping);fresh(out/'VIDEO.json',dict(condition=a.condition,source_run=str(case['run']),source_hashes=hashes,video=str(path),video_sha256=sha(path),montage=str(mp),montage_sha256=sha(mp),frames=770,decoded_frames=count,fps=50,physical_seconds=15.4,playback_seconds=15.4,resolution=[1280,720],physics_steps=0,integration_attempts=0,NN_forward_calls=0,compiled_runtime_unchanged=True,source_unchanged=True,all_qpos_qvel_exact_COM_yaw_head_checked=True,all71meshopaque=True,all_frame_fullbody_containment=True,body_silhouette_bounds=boxes,eyeinsets_actualsavedRGB=True,invalid_initial_no_negative_index=True,rawreleased_vs_actorused_separate=True,nonfuturelookup=True,world_target_and_twofarcover_scene_from_recorded_metadata=True,geometry_units='NATIVE_NOT_SI',neural_rows=3080,neural_state_total=304,functional_pass=case['functional_pass'],functional_flags=case['functional_flags'],functional_metrics=case['functional_metrics'],engineering_pass=r['engineering_pass'],stable_body=r['functional_stable_body'],whole_neck_or_biological_retina_claim=False,ffmpeg_exit=encode_exit,full_decode_exit=decode_exit,parent_actual_review='PENDING'));print(str(out/'VIDEO.json'))
if __name__=='__main__':main()
