"""Read-only actual saved states, streaming single-thread video; no integration."""
from pathlib import Path
import sys,json,hashlib,subprocess,argparse
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent/'banc-walking-recovery/20261110T152714Z'
sys.path.insert(0,str(BASE))
from physics import Core
from identity import verify,verify_runtime

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--audit',required=True);p.add_argument('--condition',required=True);a=p.parse_args()
 run=Path(a.run);out=HERE/('VIDEO_'+a.condition);assert not out.exists()
 r=json.loads((run/'RESULT.json').read_text());audit=json.loads(Path(a.audit).read_text());ident=json.loads((run/'IDENTITY.json').read_text())
 assert audit['integrity_all_pass'] and all(audit['checks'].values()) and r['condition']==a.condition
 assert r['checkpoint_sha256']=='6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f'
 assert sha(r['checkpoint'])==r['checkpoint_sha256']
 files=[run/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz')]+[Path(a.audit),Path(__file__).resolve(),HERE/'PREREG.json']
 hashes={str(x):sha(x) for x in files}
 for ledger in (r['source_hashes'],audit['source_hashes']):
  for file,h in ledger.items():assert sha(file)==h
 for n,k in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry')]:assert sha(run/n)==r[k]['sha256']
 verify(ident);assert r['transition']['all_exact'] and r['transition']['no_reset'] and r['teacher_calls_after_handover']==0 and not r['extra_forces']
 calls={'integration_attempts':0}
 def forbidden(*args,**kwargs):
  calls['integration_attempts']+=1;raise RuntimeError('integration forbidden')
 mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
 c=Core(adhesion=False);m=c.plant;verify_runtime(ident,c)
 assert (m.nq,m.nv,m.nu)==(49,48,42)
 mesh=m.geom_type==mj.mjtGeom.mjGEOM_MESH
 assert mesh.sum()==71 and np.all(m.geom_rgba[mesh,3]==1.)
 with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:t={k:z[k] for k in z.files}
 with np.load(run/'geometry_1ms.npz',allow_pickle=False) as z:yaw=np.unwrap(z['thorax_yaw'])
 with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert len(t['time'])==15400 and t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48)
 assert all(np.isfinite(v).all() for v in t.values()) and all(np.isfinite(v).all() for v in d.values())
 assert np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)])
 assert np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8
 assert np.array_equal(t['actual_requested_action'],t['action']) and np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0))
 assert all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max'))
 assert abs(yaw[-1]-yaw[5799]-r['yaw_change_rad'])<1e-12
 out.mkdir();video=out/'ACTUAL_MOTOR_HANDOVER.mp4';width=960;height=640
 original_framebuffer=[int(m.vis.global_.offwidth),int(m.vis.global_.offheight)]
 m.vis.global_.offwidth=width;m.vis.global_.offheight=height  # Visualization only; no physical arrays changed.
 renderer=mj.Renderer(m,height=height,width=width)
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=11.;cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=(t['com'].min(axis=0)+t['com'].max(axis=0))/2
 opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;opt.geomgroup[:]=1
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 ff=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
 writer=subprocess.Popen(ff,stdin=subprocess.PIPE);mapping=[];boxes=[]
 try:
  for frame,i in enumerate(range(0,15400,20)):
   now=float(t['time'][i]);code=int(t['control_source'][i]);phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ANN ONLY'}[code]
   c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(m,c.d)
   assert np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i])
   assert np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0)
   rot=c.d.xmat[c.th].reshape(3,3);actual=float(np.arctan2(rot[1,0],rot[0,0]))
   assert abs(np.arctan2(np.sin(actual-yaw[i]),np.cos(actual-yaw[i])))<1e-10
   renderer.update_scene(c.d,camera=cam,scene_option=opt);pixels=renderer.render()
   renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
   mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));ys,xs=np.where(mask)
   assert len(xs)>0 and xs.min()>5 and xs.max()<width-6 and ys.min()>80 and ys.max()<height-90,(frame,xs.min(),xs.max(),ys.min(),ys.max())
   boxes.append([int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
   image=Image.fromarray(pixels);draw=ImageDraw.Draw(image);draw.rectangle((0,0,width,74),fill=(246,246,246));draw.rectangle((0,height-85,width,height),fill=(246,246,246))
   lines=[f'{a.condition.upper()} TURN | VISUAL REVIEW | {phase}',f'Actual time {now:.3f} / 15.400 s | 1.00x real time | 50 fps',f'ANN window: slip {r["autonomous"]["slip_ratio_max"]:.6f} (original limit 0.5) | yaw {r["yaw_change_rad"]:+.6f} rad']
   for j,line in enumerate(lines):draw.text((12,3+j*23),line,font=font,fill=(25,25,25))
   bottom=[f'Heading {actual:+.6f} rad | COM z {t["com"][i,2]:.4f} | nonfoot {t["nonfoot"][i]:.4f}', '1.0 s settle + 4.8 s teacher + 9.6 s ANN | original gate '+('PASS' if r['engineering_pass'] else 'FAIL'), 'Actual saved physics | all 71 meshes opaque | substitute ANN / finite PD']
   for j,line in enumerate(bottom):draw.text((12,height-81+j*25),line,font=font,fill=(25,25,25))
   raw=np.asarray(image).tobytes();writer.stdin.write(raw)
   mapping.append(dict(frame=frame,trace_index=i,time_s=now,source=code,qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(t['qvel'][i].tobytes()).hexdigest(),rendered_rgb_sha256=hashlib.sha256(raw).hexdigest(),heading_rad=actual))
   if frame%100==0:print(f'{a.condition} rendered {frame+1}/770',flush=True)
 finally:
  writer.stdin.close();rc=writer.wait();renderer.close()
 assert rc==0 and calls['integration_attempts']==0
 probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True);assert probe.returncode==0
 meta=json.loads(probe.stdout);stream=meta['streams'][0];assert int(stream['nb_read_frames'])==770 and stream['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001
 save(out/'FFPROBE.json',meta)
 chosen=[0,50,289,290,365,769];montage=Image.new('RGB',(width*2,height*3));keyframes=[]
 reader=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
 count=0
 while True:
  raw=reader.stdout.read(width*height*3)
  if not raw:break
  assert len(raw)==width*height*3
  if count in chosen:
   im=Image.frombytes('RGB',(width,height),raw);montage.paste(im,((chosen.index(count)%2)*width,(chosen.index(count)//2)*height));keyframes.append(mapping[count])
  count+=1
 reader.stdout.close();decode_rc=reader.wait();assert decode_rc==0 and count==770
 montage.save(out/'DECODED_MONTAGE.png')
 verify_runtime(ident,c);assert all(sha(file)==h for file,h in hashes.items())
 save(out/'FRAME_MAPPING.json',mapping)
 manifest=dict(source_run=str(run),source_hashes=hashes,framebuffer_visual_only={'before':original_framebuffer,'after':[width,height]},video=str(video),video_sha256=sha(video),montage=str(out/'DECODED_MONTAGE.png'),montage_sha256=sha(out/'DECODED_MONTAGE.png'),frames=770,decoded_frames=count,ffmpeg_exit=rc,ffprobe_exit=probe.returncode,full_decode_exit=decode_rc,physical_seconds=15.4,playback_seconds=15.4,fps=50,rate=1.,mesh_count=71,all_mesh_alpha1=True,all_frame_fullbody_containment=True,silhouette_bounds=boxes,all_qpos_qvel_exact=True,all_heading_com_exact=True,source_unchanged=True,compiled_runtime_unchanged=True,physics_steps=0,integration_attempts=0,phase_frames={'settle':50,'teacher':240,'ANN':480},keyframes=keyframes,metrics=r['autonomous'],cycles=r['autonomous_cycles'],yaw_change_rad=r['yaw_change_rad'],engineering_pass=r['engineering_pass'],visual_review='pending actual montage opening',food_success=False,full_native_acceptance='NOT_PERFORMED')
 save(out/'VIDEO.json',manifest);print(json.dumps(dict(condition=a.condition,frames=count,video=str(video),montage=str(out/'DECODED_MONTAGE.png'))),flush=True)
if __name__=='__main__':main()
