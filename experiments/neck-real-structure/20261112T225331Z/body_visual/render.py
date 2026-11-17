"""Saved same-run physics reconstruction; sidebar displays recorded port signals, never fake eyes.
Reliable Core/segmentation/full-decode procedure follows the cited immutable prior renderer.
"""
from pathlib import Path
import argparse,os,sys,subprocess,json,hashlib,math
from contracts import *

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=('P_left_pulse','P_left_DN_cut'),required=True);parser.add_argument('--execute-after-parent-release',action='store_true');args=parser.parse_args()
 if not args.execute_after_parent_release:print('CODE_ONLY no arrays/Core/render');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(k)=='1',k+'=1required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw,ImageFont
 case=load(args.condition);empty=load('P_empty');t=case['trace'];d=case['decisions'];g=case['geometry'];r=case['result'];hashes=case['source_hashes'];hashes.update(empty['source_hashes']);require(np.array_equal(t['time'],empty['trace']['time']),'actualmatchedclock')
 sys.path.insert(0,str(BASE));from physics import Core
 from identity import verify,verify_runtime
 verify(case['identity']);calls={'attempts':0}
 def forbidden(*aa,**kw):calls['attempts']+=1;raise RuntimeError('Savedvideo prohibits integration')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden;c=Core(adhesion=False);model=c.plant;verify_runtime(case['identity'],c);require((model.nq,model.nv,model.nu)==(49,48,42),'fullbodydimensions');mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH;require(mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1),'71opaque')
 head=model.body('Head').id;look=(t['com'].min(0)+t['com'].max(0))/2;extent=0.
 for i in range(0,15400,20):
  c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(model,c.d);extent=max(extent,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-look,axis=1)+model.geom_rbound[mesh])))
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,extent/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=look;option=mj.MjvOption();option.geomgroup[:]=1;option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
 out=HERE/('VIDEO_'+args.condition);out.mkdir(exist_ok=False);path=out/'ACTUAL_ARTIFICIAL_BRAIN_SOURCE_NECK_BODY.mp4';oldbuffer=[int(model.vis.global_.offwidth),int(model.vis.global_.offheight)];model.vis.global_.offwidth=960;model.vis.global_.offheight=640;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);renderer=None;writer=None;mapping=[];boxes=[]
 try:
  renderer=mj.Renderer(model,height=640,width=960);writer=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
  for frame,i in enumerate(range(0,15400,20)):
   now=float(t['time'][i]);elapsed=now-5.8;physical_ms=i+1;qi=min(3079,physical_ms//5);require(float(d['time'][qi])<=now+1e-8 and abs(float(d['time'][qi])-qi*.005)<1e-8,'savedneuralnonfuturemapping');c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(model,c.d)
   require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'savedqposqvelchanged');require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'savedCOMreconstruction');rot=c.d.xmat[c.th].reshape(3,3);angle=float(np.arctan2(rot[1,0],rot[0,0]));require(abs(np.arctan2(np.sin(angle-g['thorax_yaw'][i]),np.cos(angle-g['thorax_yaw'][i])))<1e-10,'savedyawreconstruction');require(np.allclose(c.d.xpos[head],g['head_xyz'][i],atol=1e-10,rtol=0) and np.allclose(c.d.xmat[head].reshape(3,3),g['head_rotation'][i],atol=1e-10,rtol=0),'savedheadreconstruction')
   renderer.update_scene(c.d,camera=cam,scene_option=option);renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering();mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));yy,xx=np.where(mask);require(len(xx)>0 and xx.min()>5 and xx.max()<954 and yy.min()>92 and yy.max()<540,'wholebodycropped/headeroverlap');boxes.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
   body=Image.fromarray(renderer.render());bd=ImageDraw.Draw(body);bd.rectangle((0,0,960,90),fill='#f6f6f6');bd.rectangle((0,544,960,640),fill='#f6f6f6');phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ARTIFICIAL BRAIN + OTHER MOTOR'}[int(t['control_source'][i])];turn=float(d['raw_turn'][qi]);cue=float(d['cue'][qi]);cut=bool(d['DN_cut'][qi]);bodyyaw=float(np.degrees(case['yaw'][i]));extra=float(np.degrees(case['yaw'][i]-empty['yaw'][i]));brainnorm=float(np.linalg.norm(d['brain_hidden_after'][qi]))
   lines=[args.condition+' | '+phase,f'Actual {now:.3f}/15.400s | '+(f'ANN {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN')+' | 1.00x / 50fps','Source-qualified 2 DNp09 / 1 AN08B018 ports | experimental instruction, NOT visual input',f'Learned turn {turn:+.4f} | 272 neural states; no direct brain-to-joint bypass']
   for j,text in enumerate(lines):bd.text((10,2+22*j),text,font=font,fill='#151515')
   timing='Cue ANN .60..3.60; actual 2ms source queue; 5ms neural consumption'
   if args.condition=='P_left_DN_cut':timing='DN cut at ANN1.60 / global7.40s: source silence + receiver0; no brain/motor reset'
   bottom=[f'Body delta yaw {bodyyaw:+.3f} deg | minus matched P_empty {extra:+.3f} deg',f'Slip {r["autonomous"]["slip_ratio_max"]:.6f} (limit .5) | Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+' | body-stable '+str(r['functional_stable_body']),timing,'OTHER retains low-level autonomous walking; this is NOT full-neck or native-function validation']
   for j,text in enumerate(bottom):bd.text((10,546+22*j),text,font=font,fill='#151515')
   image=Image.new('RGB',(1280,720),'white');image.paste(body,(0,0));draw=ImageDraw.Draw(image);draw.text((974,12),'ACTUAL SAVED SOURCE PORTS',font=font,fill='black');draw.text((974,38),'No camera / no synthetic eye images',font=small,fill='black');draw.text((974,58),f'Neural row {qi}/3079 @ {float(d["time"][qi]):.3f}s',font=small,fill='black')
   portmap={}
   for direction,y in [('AN',90),('DN',275)]:
    value=d[direction+'_held'][qi];cap=int(d[direction+'_capture_step'][qi]);release=int(d[direction+'_release_step'][qi]);consume=int(d['physical_step_ms'][qi]);is_cut=bool(d[direction+'_cut'][qi]);draw.text((974,y),direction+' RECEIVER-HELD ACTIVITY',font=font,fill='black');draw.text((974,y+27),'['+', '.join(f'{float(v):+.5f}' for v in value)+']',font=font,fill='#2166ac' if direction=='AN' else '#b2182b')
    if cap<0:message='N/A: initial hold or cut baseline'
    else:require(release==cap+2 and release<=consume and consume*.001<=now+1e-8,'actualportfuturetimestamp');message=f'capture {cap}ms / release {release}ms'
    draw.text((974,y+56),message,font=small,fill='black');draw.text((974,y+78),f'neural consume {consume}ms',font=small,fill='black');draw.text((974,y+100),'cut receiver baseline: '+str(is_cut),font=small,fill='black');portmap[direction]=dict(held=[float(v) for v in value],capture_ms=cap,queue_release_ms=release,neural_consume_ms=consume,cut=is_cut)
   for text,y in [(f'Brain-only instruction cue {cue:+.1f}',465),(f'Brain16 state norm {brainnorm:.6f}',491),(f'DN receiver decoded turn {turn:+.5f}',517),('Brain16 + OTHER motor256 = 272',553),('AN1/DN2 held buffers are separate.',577),('Actual source identity / artificial coupling',607),('Not native physiology or electrode data.',629)]:draw.text((974,y),text,font=small,fill='black')
   draw.text((12,660),'Same-run saved physics. Source ports use real fibre identities; queue timing and learned adapters are engineering assumptions.',font=font,fill='black');pixels=np.asarray(image).tobytes();writer.stdin.write(pixels);mapping.append(dict(frame=frame,trace_index=i,time_s=now,neural_index=qi,neural_time_s=float(d['time'][qi]),source_ports=portmap,brain16_norm=brainnorm,cue=cue,learnedturn=turn,qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(t['qvel'][i].tobytes()).hexdigest(),rendered_RGB_sha256=hashlib.sha256(pixels).hexdigest()))
  writer.stdin.close();encode_exit=writer.wait();require(encode_exit==0,'ffmpegencodefailed')
 finally:
  if writer is not None and writer.poll() is None:writer.stdin.close();writer.wait()
  if renderer is not None:renderer.close()
  model.vis.global_.offwidth,model.vis.global_.offheight=oldbuffer
 verify_runtime(case['identity'],c);require(calls['attempts']==0,'integrationattempted');unchanged(hashes)
 meta=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True).stdout);stream=meta['streams'][0];require(int(stream['nb_read_frames'])==770 and stream['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001,'video770/50/15.4')
 chosen=[0,49,50,289,290,320,370,430,470,769];montage=Image.new('RGB',(2560,3600));decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);count=0
 while True:
  raw=decoder.stdout.read(1280*720*3)
  if not raw:break
  require(len(raw)==1280*720*3,'shortdecodedframe')
  if count in chosen:j=chosen.index(count);montage.paste(Image.frombytes('RGB',(1280,720),raw),((j%2)*1280,(j//2)*720))
  count+=1
 decoder.stdout.close();decode_exit=decoder.wait();require(decode_exit==0 and count==770,'full770decodefailed');montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path);unchanged(hashes)
 fresh(out/'FFPROBE.json',meta);fresh(out/'FRAME_MAPPING.json',mapping);fresh(out/'VIDEO.json',dict(condition=args.condition,source_run=str(case['run']),source_hashes=hashes,video=str(path),video_sha256=sha(path),montage=str(montage_path),montage_sha256=sha(montage_path),frames=770,decoded_frames=count,fps=50,physical_seconds=15.4,playback_seconds=15.4,resolution=[1280,720],physics_steps=0,integration_attempts=0,model_forward_calls=0,compiled_runtime_unchanged=True,source_unchanged=True,all_qpos_qvel_exact_COM_yaw_head_checked=True,all71meshopaque=True,all_frame_fullbody_containment=True,body_silhouette_bounds=boxes,sidepanel_actual_saved_source_ports=True,sidepanel_no_camera_or_target_invention=True,neural_rows=3080,neural_state_total=272,port_buffers_separate=True,nonfuturelookup=True,engineering_pass=r['engineering_pass'],stable_body=r['functional_stable_body'],all6functionalPASS_claim=False,whole_neck_or_native_function_claim=False,ffmpeg_exit=encode_exit,full_decode_exit=decode_exit,parent_actual_review='PENDING'))
 print(str(out/'VIDEO.json'))
if __name__=='__main__':main()
