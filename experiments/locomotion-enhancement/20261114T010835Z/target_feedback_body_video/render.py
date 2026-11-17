"""One direct MjSpec model, saved-state FK only; standalone, no model/Core/NN imports.
Default CLI only qualifies metadata. Actual backend requires --execute-release.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,sys
from contract import ROOT,BODY,CAL,CASES,XML,sha,read,save,require,verify_map,qualify
from mp4_timing import shorten

def indices(n):return [min(k+20,n)-1 for k in range(0,n,20)]
def phase(i):return 'ZERO RATE SETTLE' if i<1000 else ('ALGORITHM TEACHER WARMUP' if i<5800 else 'PURE MOTOR ANN COMPONENT')
def array_hash(a):return hashlib.sha256(a.tobytes()).hexdigest()
def runtime_arrays(model,ident,np):
 for name,expected in ident['compiled_arrays'].items():
  value=np.asarray(getattr(model,name))
  require(list(value.shape)==expected['shape'] and str(value.dtype)==expected['dtype'] and array_hash(value)==expected['sha256'],'Canonical compiled array mismatch: '+name)
 require((model.nq,model.nv,model.nu)==(49,48,42),'Body dimensions')
 require(model.opt.timestep==ident['physics']['timestep'] and np.array_equal(model.opt.gravity,ident['physics']['gravity']),'Original physical options')

def build_model(ident,mj,np):
 # Same asset selection as original _load_mjcf, resolved ONLY from the actual run file ledger.
 spec=mj.MjSpec.from_file(str(XML));ledger=ident['files']
 # Original _load_mjcf sets the base offscreen buffer640x640 before compilation.
 spec.visual.global_.offheight=640;spec.visual.global_.offwidth=640
 for assets,basedir in ((spec.meshes,XML.parent/spec.meshdir),(spec.textures,XML.parent/spec.texturedir)):
  for asset in assets:
   if not asset.file:continue
   local=Path(asset.file) if Path(asset.file).is_absolute() else (basedir/asset.file).resolve()
   if str(local) not in ledger:
    matches=[Path(p) for p in ledger if Path(p).name==Path(asset.file).name]
    require(len(matches)==1,'Unique actual asset ledger resolution: '+asset.file);local=matches[0]
   require(sha(local)==ledger[str(local)]['sha256'],'Actual asset digest');asset.file=str(local)
 spec.meshdir='';spec.texturedir='';model=spec.compile()
 # Original load_full_model sets geom/material alpha1; original Core sets mu1.5.
 # This is body reconstruction, never visual tuning of physics. ALL arrays must match next.
 mesh=model.geom_type==int(mj.mjtGeom.mjGEOM_MESH);require(int(mesh.sum())==71,'All71mesh')
 model.geom_rgba[mesh,3]=1.;materials=np.unique(model.geom_matid[mesh]);model.mat_rgba[materials[materials>=0],3]=1.
 model.geom_friction[:,0]=1.5
 runtime_arrays(model,ident,np)
 require(np.all(model.geom_rgba[mesh,3]==1.),'Actual all71 opaque')
 return model,mesh

def arrays(path,np):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def restore(model,data,t,i,mj,np):
 data.qpos[:]=t['qpos'][i];data.qvel[:]=t['qvel'][i];data.ctrl[:]=t['actualctrl'][i];data.time=float(t['time'][i]);data.xfrc_applied[:]=0.;data.qfrc_applied[:]=0.
 mj.mj_forward(model,data)
 require(np.array_equal(data.qpos,t['qpos'][i]) and np.array_equal(data.qvel,t['qvel'][i]) and np.array_equal(data.ctrl,t['actualctrl'][i]),'Saved qpos/qvel/ctrl identity')
 # Reconstructed contacts/forces are never read. All metric overlays use saved trace/result.

def graph(out,record,t,g,np,Image,ImageDraw,ImageFont):
 """Four synchronized saved-data panels; no recomputed acceptance or contact inference."""
 image=Image.new('RGB',(1200,1060),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 r=record['result'];n=len(t['time']);times=t['time'];limit=n/1000;colors=['#2266bb','#bb4422','#448844','#994499','#cc9911','#338888']
 draw.text((20,8),f'{record["condition"]} | ACTUAL SAME-RUN SAVED DIAGNOSTICS | {limit:.3f}s | component only',fill='black',font=font)
 def plot(slot,title,ys,colorset):
  x0,y0,x1,y1=80,65+slot*225,1160,235+slot*225
  draw.text((20,y0-25),title,fill='black',font=font);draw.rectangle((x0,y0,x1,y1),outline='#777777')
  values=[np.asarray(v,float) for v in ys];finite=np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.array([])
  if not len(finite):draw.text((x0+10,y0+50),'N/A: no saved measurement in this phase',fill='black',font=font);return
  lo=float(finite.min());hi=float(finite.max());span=max(hi-lo,1e-9);lo-=.05*span;hi+=.05*span
  for values,color in zip(values,colorset):
   pts=[]
   for k in range(0,len(values),max(1,n//1600)):
    if np.isfinite(values[k]):pts.append((x0+float(times[k])/limit*(x1-x0),y1-(float(values[k])-lo)/(hi-lo)*(y1-y0)))
   if len(pts)>1:draw.line(pts,fill=color,width=2)
  for tm in (1.,5.8):
   if tm<=limit:
    x=x0+tm/limit*(x1-x0);draw.line((x,y0,x,y1),fill='#999999');draw.text((x+3,y0+3),f'{tm}s',fill='#555555',font=font)
  draw.text((x0-65,y0),f'{hi:.3g}',fill='black',font=font);draw.text((x0-65,y1-18),f'{lo:.3g}',fill='black',font=font)
 yaw=np.full(n,np.nan);drift=np.full(n,np.nan)
 if len(g.get('time',[])):yaw[1000:]=np.degrees(np.unwrap(g['thorax_yaw']))
 if r['candidate_start'] is not None:drift[5800:]=np.linalg.norm(t['com'][5800:,:2]-np.array(r['candidate_start']['com'])[:2],axis=1)/r['body_length']['length_native']
 plot(0,'Saved thorax yaw (deg); initial settle geometry absent = N/A',[yaw],['#2266bb'])
 plot(1,'Saved COM XY drift / body length from actual handover; before handover N/A',[drift],['#bb4422'])
 plot(2,'Saved actual six foot loads / weight (LF LH LM RF RH RM)',[t['loads'][:,k]/r['weight_native'] for k in range(6)],colors)
 plot(3,'Saved RMS: ctrl(native torque) / target(rad) / action(unitless); common axis',[np.sqrt(np.mean(t[k]**2,axis=1)) for k in ('actualctrl','target','action')],colors[:3])
 metric=r['candidate_metrics'];cycles='N/A before handover' if metric is None else f'pivot {metric["pivot_cycle_counts"]} | walk {metric["walk_cycle_counts"]}'
 draw.text((20,980),'Saved whole-window cycle counts: '+cycles,fill='black',font=font)
 draw.text((20,1008),'All curves use this actual saved trace; contact forces are not reconstructed for measurement.',fill='black',font=font)
 path=out/'SAVED_DIAGNOSTICS.png';image.save(path);return path

def render_case(model,mesh,record,out,mj,np,Image,ImageDraw,ImageFont):
 r=record['result'];folder=Path(record['folder']);ident=record['identity'];runtime_arrays(model,ident,np)
 t=arrays(folder/'trace_1ms.npz',np);g=arrays(folder/'geometry_1ms.npz',np) if 'geometry' in r else {}
 n=record['timing']['samples'];require(len(t['time'])==n,'Actual sample count');require(all(np.isfinite(v).all() for v in t.values()),'Saved finite trace')
 require(np.max(np.abs(t['time']-np.arange(1,n+1)*.001))<1e-8,'Contiguous actual1ms clock')
 expected=np.where(np.arange(n)<1000,0,np.where(np.arange(n)<5800,2,3));require(np.array_equal(t['control_source'],expected),'Source0/2/3 actual labels')
 require(all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')),'No saved extra forces')
 require(np.all(np.abs(t['actualctrl'])<=np.array(ident['actuator']['torque_limit'])[None]+1e-9),'Saved finite torque limits')
 sample=indices(n);require(sample[-1]==n-1 and len(sample)==record['timing']['frames'],'Last actual endpoint included')
 data=mj.MjData(model);th=model.body('Thorax').id;center=(t['com'].min(0)+t['com'].max(0))/2;radius=0.
 for i in sample:
  restore(model,data,t,i,mj,np);radius=max(radius,float(np.max(np.linalg.norm(data.geom_xpos[mesh]-center,axis=1)+model.geom_rbound[mesh])))
 width,height=960,640;before=[int(model.vis.global_.offwidth),int(model.vis.global_.offheight)];model.vis.global_.offwidth=width;model.vis.global_.offheight=height
 renderer=mj.Renderer(model,height=height,width=width);cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,radius/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.65);cam.lookat[:]=center;cam.azimuth=140.;cam.elevation=-25.
 opt=mj.MjvOption();opt.geomgroup[:]=1;opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);out.mkdir();video=out/'ACTUAL_SAVED_MOTOR_COMPONENT.mp4'
 argv=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','960x640','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-bf','0','-preset','fast','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','1000',str(video)]
 writer=subprocess.Popen(argv,stdin=subprocess.PIPE);mapping=[];bounds=[];rc=None
 try:
  for frame,i in enumerate(sample):
   restore(model,data,t,i,mj,np);require(np.allclose(data.subtree_com[th],t['com'][i],atol=1e-10,rtol=0),'Actual saved COM identity')
   rotation=data.xmat[th].reshape(3,3);actual_yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
   if i>=1000:
    require(g and abs(float(g['time'][i-1000])-float(t['time'][i]))<1e-8,'Saved geometry same time')
    require(abs(float(np.angle(np.exp(1j*(actual_yaw-g['thorax_yaw'][i-1000])))))<1e-10,'Actual saved yaw identity')
   renderer.update_scene(data,camera=cam,scene_option=opt);pixels=renderer.render();renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
   ys,xs=np.where(np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))
   require(len(xs)>0 and xs.min()>5 and xs.max()<width-6 and ys.min()>84 and ys.max()<height-90,'Wholebody containment failed; retain output; new render scope required')
   bounds.append([int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]);im=Image.fromarray(pixels);draw=ImageDraw.Draw(im);draw.rectangle((0,0,width,80),fill='white');draw.rectangle((0,height-85,width,height),fill='white')
   metric=r['candidate_metrics'];task='N/A' if metric is None else ('PASS' if metric['task_pass_candidate'] else 'FAIL');legacy='PASS' if r['original_five']['all_gates'] else 'FAIL'
   yawtext='N/A (settle geometry not saved)' if i<1000 else f'{math.degrees(float(g["thorax_yaw"][i-1000])):+.3f} deg'
   lines=[f'{record["condition"].upper()} | {phase(i)}',f'Actual {t["time"][i]:.3f}/{n/1000:.3f}s | 50fps endpoints | phase cmd {([0,0] if i<1000 else ([1,0] if i<5800 else r["command"]))}',f'Actual window task {task} | legacy five {legacy} | '+('FULL15.4s' if r['full_duration'] else 'EARLY STOP / SHORT WINDOW')]
   for j,line in enumerate(lines):draw.text((10,4+j*24),line,font=font,fill='black')
   bottom=[f'Saved yaw {yawtext} | COM z {t["com"][i,2]:.4f} | saved NF {t["nonfoot"][i]:.4f}', 'SOURCE3 / TEST NOT FINAL DEPLOYMENT / CONSTANT [speed,yaw] DIAGNOSTIC', 'Pure motor ANN component / finite PD; NOT artificial brain-real neck deployment']
   for j,line in enumerate(bottom):draw.text((10,height-81+j*25),line,font=font,fill='black')
   raw=np.asarray(im).tobytes();writer.stdin.write(raw);mapping.append(dict(frame=frame,pts_s=frame*.02,duration_s=(record['timing']['last_frame_ms'] if frame==len(sample)-1 else 20)/1000,trace_index=i,actual_endpoint_s=float(t['time'][i]),source=int(t['control_source'][i]),qpos_sha256=array_hash(t['qpos'][i]),qvel_sha256=array_hash(t['qvel'][i]),ctrl_sha256=array_hash(t['actualctrl'][i]),RGB_sha256=hashlib.sha256(raw).hexdigest()))
 finally:
  writer.stdin.close();rc=writer.wait();renderer.close();model.vis.global_.offwidth,model.vis.global_.offheight=before
 require(rc==0,'FFmpeg encoding failed');runtime_arrays(model,ident,np);require([int(model.vis.global_.offwidth),int(model.vis.global_.offheight)]==before,'Visual framebuffer restored')
 timing=shorten(video,n,record['timing']['last_frame_ms'])
 probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-show_frames','-of','json',str(video)],capture_output=True,text=True);require(probe.returncode==0,'ffprobe exit')
 meta=json.loads(probe.stdout);frames=meta['frames'];stream=meta['streams'][0];require(int(stream['nb_read_frames'])==len(sample)==len(frames),'All encoded frames enumerated')
 pts=[float(f['pts_time']) for f in frames];durations=[float(f.get('duration_time',f.get('pkt_duration_time','nan'))) for f in frames]
 require(all(abs(p-k*.02)<1e-6 for k,p in enumerate(pts)),'Actual movie PTS50fps main cadence')
 require(abs(durations[-1]-record['timing']['last_frame_ms']/1000)<1e-6 and abs(pts[-1]+durations[-1]-n/1000)<1e-6 and abs(float(meta['format']['duration'])-n/1000)<1e-6,'Exact actual physical/movie endpoint')
 save(out/'FFPROBE.json',meta);chosen=sorted({min(range(len(sample)),key=lambda k:abs(float(t['time'][sample[k]])-target)) for target in (0.,1.,5.78,5.8,(5.8+n/1000)/2,n/1000)})
 reader=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(video),'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);count=0;montage=Image.new('RGB',(1920,640*math.ceil(len(chosen)/2)),'white')
 while True:
  raw=reader.stdout.read(width*height*3)
  if not raw:break
  require(len(raw)==width*height*3,'Full decoded frame')
  if count in chosen:montage.paste(Image.frombytes('RGB',(width,height),raw),((chosen.index(count)%2)*width,(chosen.index(count)//2)*height))
  count+=1
 reader.stdout.close();decode_rc=reader.wait();require(decode_rc==0 and count==len(sample),'All frames full decoded')
 montage.save(out/'DECODED_MONTAGE.png');graphpath=graph(out,record,t,g,np,Image,ImageDraw,ImageFont);save(out/'FRAME_MAPPING.json',mapping)
 manifest=dict(condition=record['condition'],source_folder=str(folder),checkpoint_sha256=r['checkpoint_sha256'],trace_sha256=r['trace']['sha256'],result_sha256=sha(folder/'RESULT.json'),audit_sha256=sha(folder/'VERIFY.json'),identity_sha256=r['identity_sha256'],video=str(video),video_sha256=sha(video),montage=str(out/'DECODED_MONTAGE.png'),montage_sha256=sha(out/'DECODED_MONTAGE.png'),graph=str(graphpath),graph_sha256=sha(graphpath),frames=len(sample),decoded_frames=count,ffmpeg_exit=rc,ffprobe_exit=probe.returncode,full_decode_exit=decode_rc,physical_seconds=n/1000,movie_seconds=float(meta['format']['duration']),ANN_seconds=record['timing']['ANN_seconds'],last_saved_state_included=True,last_saved_endpoint_s=float(t['time'][-1]),last_frame_endpoint_difference_s=0.,tail_timing=timing,framebuffer={'before':before,'during':[width,height],'after':before},camera={'fixed':True,'center':center.tolist(),'distance':cam.distance},all71opaque=True,allframe_wholebody=True,silhouette_bounds=bounds,all_qpos_qvel_ctrl_exact=True,COM_yaw_exact=True,reconstructed_contact_metrics_used=False,metrics_source='sameactualRESULT/trace/geometry/native peaks only',candidate_metrics=r['candidate_metrics'],original_five=r['original_five'],native=r['native'],early_stop=r['early_stop'],physics_steps=0,NN_forwards=0,Core_constructions=0,visual_review='PENDING_PARENT_ACTUAL_OPEN_ALL_MONTAGES_AND_GRAPHS')
 save(out/'VIDEO.json',manifest);return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-release',action='store_true');a=p.parse_args()
 q=qualify()
 if not a.execute_release:print(json.dumps(dict(status='ACTUAL_METADATA_QUALIFIED_ONLY',checkpoint_sha256=q['checkpoint_sha256'],cases=list(CASES))));return
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});require(not any(k in sys.modules for k in ('torch','physics','model','flygym')),'No NN/Core imports')
 out=ROOT/'OUTPUT';require(not out.exists(),'Fresh output scope required');out.mkdir();save(out/'QUALIFICATION.json',dict(checkpoint_sha256=q['checkpoint_sha256'],source_hashes=q['source_hashes'],cases=list(CASES)))
 import numpy as np
 import mujoco as mj
 from PIL import Image,ImageDraw,ImageFont
 calls={'integration_attempts':0}
 def forbidden(*args,**kwargs):calls['integration_attempts']+=1;raise RuntimeError('Integration forbidden in renderer')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 model,mesh=build_model(q['records'][0]['identity'],mj,np);results=[]
 for record in q['records']:results.append(render_case(model,mesh,record,out/record['condition'],mj,np,Image,ImageDraw,ImageFont))
 require(calls['integration_attempts']==0 and 'torch' not in sys.modules,'ZeroNN/zero integration')
 verify_map(q['source_hashes']);verify_map(read(ROOT/'READY.json')['source_hashes'])
 save(out/'DELIVERY_PENDING_REVIEW.json',dict(status='RENDERED_NOT_USER_ACCEPTED_PENDING_PARENT_OPEN_AND_DESKTOP_HASH',checkpoint_sha256=q['checkpoint_sha256'],cases=results,model_compiles=1,Core_constructions=0,NN_forwards=0,physics_steps=0,integration_attempts=0,source_hashes=q['source_hashes'],source_unchanged=True,desktop_copied=False))
 print(json.dumps(dict(status='RENDERED_PENDING_PARENT_REVIEW',out=str(out),cases=4,checkpoint_sha256=q['checkpoint_sha256'])),flush=True)
if __name__=='__main__':main()
