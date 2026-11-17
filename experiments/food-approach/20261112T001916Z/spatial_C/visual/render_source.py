"""Representative spatial actual-state video; user-scene patch has no model collision."""
import argparse,os,sys,subprocess,hashlib,json
from contracts_visual import HERE,BASE,HEAD,arrays,summary,require,sha,save,unchanged,read

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=('C_left_near','C_right_near'));p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
 if not a.execute_after_parent_release:print('CODE_ONLY spatial renderer not executed');return
 require(a.condition is not None,'explicit representative condition required')
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw,ImageFont
 from bisect import bisect_right
 data=summary(arrays(a.condition));baseline=summary(arrays('C_noodor'));hashes=data['hashes'];hashes.update(baseline['hashes']);hashes[str(HERE/'render_source.py')]=sha(HERE/'render_source.py')
 sys.path.insert(0,str(BASE));from physics import Core
 from identity import verify,verify_runtime
 verify(data['identity']);attempts=[0]
 def forbidden(*a,**k):attempts[0]+=1;raise RuntimeError('render integration forbidden')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 c=Core(adhesion=False);model=c.plant;verify_runtime(data['identity'],c);require((model.nq,model.nv,model.nu)==(49,48,42),'body dimensions')
 mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH;require(mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1),'full71 opaque')
 head=model.body('Head').id;ha=read(HEAD);require(head==ha['head']['body_id'],'head ID');local=np.asarray(ha['sensor_offsets']['local_xyz'])
 t=data['trace'];g=data['geometry'];d=data['decisions'];r=data['result'];center=data['source'];radius=data['radius'];m=r['spatial_metrics'];center3=np.r_[center,0.]
 look=(t['com'].min(0)+t['com'].max(0))/2;extent=0.
 for i in range(0,15400,20):
  c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(model,c.d)
  extent=max(extent,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-look,axis=1)+model.geom_rbound[mesh])))
 extent=max(extent,float(np.linalg.norm(center3-look)+radius))
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,extent/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=look
 opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;opt.geomgroup[:]=1
 out=HERE/('VIDEO_'+a.condition);out.mkdir(exist_ok=False);video=out/'ACTUAL_SPATIAL_SOURCE.mp4';w,h=960,640;oldbuffer=[int(model.vis.global_.offwidth),int(model.vis.global_.offheight)];model.vis.global_.offwidth=w;model.vis.global_.offheight=h
 renderer=None;writer=None;frame_records=[];boxes=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 try:
  renderer=mj.Renderer(model,height=h,width=w)
  writer=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
  for frame,i in enumerate(range(0,15400,20)):
   now=float(t['time'][i]);elapsed=now-5.8;c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(model,c.d)
   require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'saved pose changed')
   require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'COM mismatch')
   rot=c.d.xmat[c.th].reshape(3,3);angle=np.arctan2(rot[1,0],rot[0,0]);require(abs(np.arctan2(np.sin(angle-g['thorax_yaw'][i]),np.cos(angle-g['thorax_yaw'][i])))<1e-10,'yaw mismatch')
   pos=c.d.xpos[head];hr=c.d.xmat[head].reshape(3,3);sites=pos+local@hr.T
   for actual,saved in ((pos,g['head_xyz'][i]),(hr,g['head_rotation'][i]),(sites,g['sensor_xyz'][i])):require(np.allclose(actual,saved,atol=1e-10,rtol=0),'head/sensor mismatch')
   renderer.update_scene(c.d,camera=cam,scene_option=opt)
   # Model-only segmentation precedes decorative geometry; only update_scene assigns model segids.
   renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
   mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));yy,xx=np.where(mask)
   # Render-only user geoms. World patch exists only after actual anchor creation; it never follows head.
   marker_count=0
   if bool(g['source_anchor_valid'][i]):
    scene=renderer.scene
    for j in range(32):
     require(scene.ngeom<scene.maxgeom,'user marker scene capacity')
     p0=np.r_[center+radius*np.array([np.cos(j*2*np.pi/32),np.sin(j*2*np.pi/32)]),.003]
     p1=np.r_[center+radius*np.array([np.cos((j+1)*2*np.pi/32),np.sin((j+1)*2*np.pi/32)]),.003]
     geom=scene.geoms[scene.ngeom];mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.array([.003,.003,.003]),np.zeros(3),np.eye(3).ravel(),np.array([.1,.9,.25,.8],dtype=np.float32));mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,.003,p0,p1);geom.category=int(mj.mjtCatBit.mjCAT_DECOR);geom.objtype=int(mj.mjtObj.mjOBJ_UNKNOWN);geom.objid=-1;geom.segid=-1;scene.ngeom+=1;marker_count+=1
   rgb=renderer.render()  # RGB only: user geoms segid=-1 never enter a segmentation pass.
   require(len(xx)>0 and xx.min()>5 and xx.max()<w-6 and yy.min()>92 and yy.max()<h-100,'body cropped/overlay occlusion');boxes.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
   k=bisect_right(d['time'].tolist(),now)-1;source=int(t['control_source'][i]);phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ANN ONLY'}[source]
   valid=bool(g['source_anchor_valid'][i]);distance=float(g['center_distance'][i]) if valid else None
   confirmed=m['confirmed_elapsed_s'] is not None and elapsed>=m['confirmed_elapsed_s'];entered=m['first_entry_elapsed_s'] is not None and elapsed>=m['first_entry_elapsed_s'];inside=bool(g['inside_patch'][i]) if valid else False
   label='PRE-ANCHOR' if not valid else 'INSIDE / endpoint confirmed' if inside and confirmed else 'INSIDE / not yet confirmed' if inside else 'OUTSIDE / after entry' if entered else 'OUTSIDE / no entry yet'
   values='C / receptors / turn N/A'
   if k>=0:
    require(d['time'][k]<=now,'future decision');z=d['receptors'][k];C=d['concentration'][k];values=f'z causal L/R {z[0]:.4f}/{z[1]:.4f} | held C {C[0]:.4f}/{C[1]:.4f} | turn {float(d["raw_turn"][k]):+.3f}'
   im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle((0,0,w,90),fill='#f6f6f6');draw.rectangle((0,h-96,w,h),fill='#f6f6f6')
   top=[a.condition+' | SPATIAL_GAUSSIAN / DEVELOPMENT | '+phase,f'Actual {now:.3f}/15.400s | '+(f'ANN elapsed {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN')+' | 1.00x',values,'Source FIXED WORLD after handover | ON .60-.80 ANN sec | marker = render-only patch']
   bottom=[('Distance center '+f'{distance:.4f}'+' / edge '+f'{float(g["edge_distance"][i]):.4f}') if valid else 'Source geometry not yet anchored',label+f' | body delta yaw {float(np.degrees(data["yaw"][i])):+.3f} deg',f'Original5 '+('PASS' if r['engineering_pass'] else 'FAIL')+f' | slip {r["autonomous"]["slip_ratio_max"]:.4f} | full-run pass-through {m["overshoot_or_pass_through"]}','Endpoint proxy only / NOT parking or ingestion / no body receptor-cut or D confirmation']
   for row,line in enumerate(top):draw.text((10,2+22*row),line,font=font,fill='#151515')
   for row,line in enumerate(bottom):draw.text((10,h-94+23*row),line,font=font,fill='#151515')
   raw=np.asarray(im).tobytes();writer.stdin.write(raw);frame_records.append(dict(frame=frame,trace_index=i,time_s=now,ANN_elapsed_s=elapsed,decision_index=k,marker_user_geoms=marker_count,source_center_xy=center.tolist() if valid else None,distance_center=distance,inside_patch=inside,endpoint_confirmed=confirmed,body_rgb_sha256=hashlib.sha256(raw).hexdigest(),qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest()))
  writer.stdin.close();encode_exit=writer.wait();require(encode_exit==0,'encode failed')
 finally:
  if writer is not None and writer.poll() is None:writer.stdin.close();writer.wait()
  if renderer is not None:renderer.close()
  model.vis.global_.offwidth,model.vis.global_.offheight=oldbuffer
 verify_runtime(data['identity'],c);require(attempts[0]==0,'integration attempted');unchanged(hashes)
 probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],check=True,capture_output=True,text=True);meta=json.loads(probe.stdout);require(int(meta['streams'][0]['nb_read_frames'])==770 and meta['streams'][0]['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001,'video timing')
 eventframes=[0,289,290,330,430,769]
 for event in (m['first_entry_elapsed_s'],m['confirmed_elapsed_s']):
  if event is not None:eventframes.append(min(769,max(0,int(round((5.8+event)*50)))))
 eventframes=sorted(set(eventframes));montage=Image.new('RGB',(w*2,h*((len(eventframes)+1)//2)));reader=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);count=0
 while True:
  raw=reader.stdout.read(w*h*3)
  if not raw:break
  require(len(raw)==w*h*3,'short decode')
  if count in eventframes:
   j=eventframes.index(count);montage.paste(Image.frombytes('RGB',(w,h),raw),(j%2*w,j//2*h))
  count+=1
 reader.stdout.close();decode_exit=reader.wait();require(decode_exit==0 and count==770,'full decode failed');image=out/'DECODED_MONTAGE.png';montage.save(image);unchanged(hashes)
 save(out/'FFPROBE.json',meta);save(out/'FRAME_MAPPING.json',frame_records);save(out/'VIDEO.json',dict(condition=a.condition,video=str(video),video_sha256=sha(video),montage=str(image),montage_sha256=sha(image),source_hashes=hashes,state_total=288,frames=770,decoded_frames=count,fps=50,physical_seconds=15.4,playback_seconds=15.4,ffmpeg_exit=encode_exit,full_decode_exit=decode_exit,ffprobe_exit=probe.returncode,mesh_count=71,all_opaque_fullbody=True,silhouette_bounds=boxes,physics_steps=0,integration_attempts=0,model_runtime_unchanged=True,head_sensor_geometry_checked=True,render_only_user_patch=True,source_fixed_world=center.tolist(),framebuffer=dict(before=oldbuffer,during=[w,h],after=oldbuffer),spatial_metrics=m,engineering_pass=r['engineering_pass'],parking=False,ingestion=False,visual_review='PENDING_PARENT_OPENING'))
 print(str(out/'VIDEO.json'))
if __name__=='__main__':main()
