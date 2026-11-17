"""Direct saved-state video. Default CODE_ONLY; no numerical imports until released."""
from pathlib import Path
import argparse,hashlib,os,subprocess,sys
from contracts import HERE,ROOT,BASE,IDS,B_IDS,read,sha,require,save_new,decision_at,events,audited_arrays,bind_functional,verify_unchanged

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',choices=IDS);p.add_argument('--execute-after-parent-release',action='store_true');a=p.parse_args()
    if not a.execute_after_parent_release:
        print('CODE_ONLY: no trajectory/model/physics/render read or execution');return
    require(a.condition is not None,'explicit condition required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(BASE));from physics import Core
    from identity import verify,verify_runtime
    data=bind_functional(audited_arrays(a.condition));t=data['trace'];d=data['decisions'];yaw=data['yaw'];r=data['result'];hashes=data['source_hashes']
    baseline=audited_arrays('B_noodor') if a.condition in B_IDS else None
    if baseline is not None:
        require(np.array_equal(t['time'],baseline['trace']['time']),'baseline time axis differs');hashes.update(baseline['source_hashes'])
    hashes[str(HERE/'render.py')]=sha(HERE/'render.py')
    verify(data['identity'])
    calls={'integration_attempts':0}
    def forbidden(*args,**kwargs):calls['integration_attempts']+=1;raise RuntimeError('visualization integration forbidden')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    c=Core(adhesion=False);m=c.plant;verify_runtime(data['identity'],c)
    require((m.nq,m.nv,m.nu)==(49,48,42),'physical body dimensions changed')
    head=m.body('Head').id
    head_audit_path=ROOT.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
    head_audit=read(head_audit_path);require(head_audit['status']=='PASS' and head==head_audit['head']['body_id'],'head mapping changed')
    local=np.asarray(head_audit['sensor_offsets']['local_xyz'],dtype=float);require(local.shape==(2,3),'two actual local sampling sites required');hashes[str(head_audit_path)]=sha(head_audit_path)
    mesh=m.geom_type==mj.mjtGeom.mjGEOM_MESH
    require(int(mesh.sum())==71 and np.all(m.geom_rgba[mesh,3]==1.),'all71 opaque meshes required')
    out=HERE/('VIDEO_'+a.condition);out.mkdir(exist_ok=False)
    width,height=960,640;before=[int(m.vis.global_.offwidth),int(m.vis.global_.offheight)]
    center=(t['com'].min(axis=0)+t['com'].max(axis=0))/2;radius=0.
    for i in range(0,15400,20):
        c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(m,c.d)
        radius=max(radius,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-center,axis=1)+m.geom_rbound[mesh])))
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,radius/np.sin(np.deg2rad(float(m.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=center
    opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;opt.geomgroup[:]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    video=out/'ACTUAL_STIMULUS.mp4';writer=None;renderer=None;mapping=[];boxes=[]
    m.vis.global_.offwidth=width;m.vis.global_.offheight=height
    try:
        renderer=mj.Renderer(m,height=height,width=width)
        writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        times=d['time'].tolist()
        for frame,i in enumerate(range(0,15400,20)):
            now=float(t['time'][i]);elapsed=now-5.8;source=int(t['control_source'][i]);j=decision_at(times,now)
            c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=now;mj.mj_forward(m,c.d)
            require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'pose altered during forward')
            require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'COM reconstruction mismatch')
            rot=c.d.xmat[c.th].reshape(3,3);angle=float(np.arctan2(rot[1,0],rot[0,0]));expected=float(data['geometry']['thorax_yaw'][i])
            require(abs(np.arctan2(np.sin(angle-expected),np.cos(angle-expected)))<1e-10,'yaw reconstruction mismatch')
            head_pos=c.d.xpos[head];head_rot=c.d.xmat[head].reshape(3,3);sensor=head_pos+local@head_rot.T
            for actual_value,saved_value in ((head_pos,data['geometry']['head_xyz'][i]),(head_rot,data['geometry']['head_rotation'][i]),(sensor,data['geometry']['sensor_xyz'][i]),(sensor.mean(0),data['geometry']['sensor_midpoint_xyz'][i])):
                require(np.allclose(actual_value,saved_value,rtol=0,atol=1e-10),'saved head/sampling sites not reconstructed from actual qpos')
            renderer.update_scene(c.d,camera=cam,scene_option=opt);pixels=renderer.render()
            renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
            mask=np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM));ys,xs=np.where(mask)
            require(len(xs)>0 and xs.min()>5 and xs.max()<width-6 and ys.min()>92 and ys.max()<height-100,'body cropped or hidden by overlay')
            boxes.append([int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
            phase={0:'SETTLE',2:'TEACHER WARMUP',3:'ANN ONLY'}[source]
            clock=f'ANN elapsed {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN'
            kind='CONCENTRATION_CLAMP / NOT NAVIGATION' if r['stage']=='B' else 'COMMAND PROBE / NOT ODOR RESPONSE'
            stimulus='NONE'
            if r['stage']=='B' and a.condition!='B_noodor' and elapsed>=.6:
                if a.condition in B_IDS[-2:]:stimulus='ON RAMP' if elapsed<.8 else 'FIRST SIDE' if elapsed<4.8 else 'SWAP RAMP' if elapsed<5. else 'SECOND SIDE'
                else:stimulus='ON RAMP' if elapsed<.8 else 'HELD' if elapsed<3.6 else 'OFF RAMP' if elapsed<3.8 else 'OFF'
            query=None
            if j>=0:
                require(float(d['time'][j])<=now,'future decision leak')
                z=d['receptors'][j];C=d['concentration'][j];turn=float(d['command'][j,1]);raw=float(d['raw_turn'][j])
                query=dict(index=j,time_s=float(d['time'][j]),receptors=z.tolist(),held_concentration=C.tolist(),raw_turn=raw,projected_command=d['command'][j].tolist())
                values=f'z causal L/R {z[0]:.4f}/{z[1]:.4f} | C held L/R {C[0]:.4f}/{C[1]:.4f} | turn raw/applied {raw:+.3f}/{turn:+.3f}'
            else:values='z / held C / turn: N/A (before first saved neural query)'
            body=float(np.degrees(yaw[i]));difference=None if baseline is None else float(np.degrees(yaw[i]-baseline['yaw'][i]))
            body_line=f'Body delta yaw {body:+.3f} deg (handover)' + (' | minus no-odor '+f'{difference:+.3f} deg' if difference is not None else ' | no-odor comparison N/A')
            original5=bool(r['autonomous']['all_gates']);functional=data['visual_functional_pass']
            flags='Original5 '+('PASS' if original5 else 'FAIL')+' | Functional '+('PENDING matched summary' if functional is None else 'PASS' if functional else 'FAIL')
            image=Image.fromarray(pixels);draw=ImageDraw.Draw(image);draw.rectangle((0,0,width,90),fill=(246,246,246));draw.rectangle((0,height-96,width,height),fill=(246,246,246))
            top=[f'{a.condition} | {phase} | source {source}',f'Actual {now:.3f}/15.400s | {clock} | 1.00x / 50fps',kind+' | stimulus '+stimulus,values]
            schedule='On .60-.80; pulse off 3.60-3.80; reversal 4.80-5.00 (ANN seconds)' if r['stage']=='B' else 'High-level probe override / NOT stimulus success / no behavioral advance'
            bottom=[body_line,f'Slip {r["autonomous"]["slip_ratio_max"]:.6f} (limit .5) | '+flags,schedule,'Saved actual physics / substitute ANN / bounded PD | no body receptor-cut in B6']
            for k,line in enumerate(top):draw.text((10,2+22*k),line,font=font,fill=(20,20,20))
            for k,line in enumerate(bottom):draw.text((10,height-94+23*k),line,font=font,fill=(20,20,20))
            raw_pixels=np.asarray(image).tobytes();writer.stdin.write(raw_pixels)
            mapping.append(dict(frame=frame,trace_index=i,time_s=now,ANN_elapsed_s=elapsed,source=source,decision=query,body_delta_yaw_deg=body,noodor_difference_deg=difference,qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(t['qvel'][i].tobytes()).hexdigest(),rendered_rgb_sha256=hashlib.sha256(raw_pixels).hexdigest()))
        writer.stdin.close();encode_exit=writer.wait();require(encode_exit==0,'ffmpeg failed')
    finally:
        if writer is not None and writer.poll() is None:
            writer.stdin.close();writer.wait()
        if renderer is not None:renderer.close()
        m.vis.global_.offwidth,m.vis.global_.offheight=before
    require(calls['integration_attempts']==0,'integration attempted');verify_runtime(data['identity'],c)
    require([int(m.vis.global_.offwidth),int(m.vis.global_.offheight)]==before,'visual framebuffer not restored')
    proc=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,check=True)
    import json
    meta=json.loads(proc.stdout);stream=meta['streams'][0]
    require(int(stream['nb_read_frames'])==770 and stream['r_frame_rate']=='50/1' and abs(float(meta['format']['duration'])-15.4)<.001,'video timing mismatch')
    # Include phase boundaries and stimulus ramps, not only early gait frames.
    chosen=[0,289,290,320,430,470,480,530,540,769]
    montage=Image.new('RGB',(width*2,height*5));reader=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
    count=0
    while True:
        raw=reader.stdout.read(width*height*3)
        if not raw:break
        require(len(raw)==width*height*3,'short decoded frame')
        if count in chosen:
            k=chosen.index(count);montage.paste(Image.frombytes('RGB',(width,height),raw),((k%2)*width,(k//2)*height))
        count+=1
    reader.stdout.close();decode_exit=reader.wait();require(decode_exit==0 and count==770,'full decode failed')
    montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path)
    verify_unchanged(hashes);save_new(out/'FFPROBE.json',meta);save_new(out/'FRAME_MAPPING.json',mapping)
    save_new(out/'VIDEO.json',dict(condition=a.condition,stage=r['stage'],source_run=str(data['run']),source_hashes=hashes,motor_sha256=r['checkpoint_sha256'],behavior_sha256=r['behavior_checkpoint_sha256'],state_layout=[128,64,64,32],state_total=288,video=str(video),video_sha256=sha(video),montage=str(montage_path),montage_sha256=sha(montage_path),frames=770,decoded_frames=count,fps=50,playback_seconds=15.4,physical_seconds=15.4,ffmpeg_exit=encode_exit,ffprobe_exit=proc.returncode,full_decode_exit=decode_exit,framebuffer_visual_only=dict(before=before,during=[width,height],after=before),mesh_count=71,all_mesh_alpha1=True,all_frame_fullbody_containment=True,silhouette_bounds=boxes,physics_steps=0,integration_attempts=0,all_qpos_qvel_COM_yaw_exact=True,all_head_and_sensor_geometry_tol1e10=True,source_unchanged=True,compiled_runtime_unchanged=True,fixed_camera=dict(center=center.tolist(),distance=cam.distance),engineering_pass=r['engineering_pass'],original_five_pass=original5,functional_pass=data['visual_functional_pass'],functional_source=data['visual_functional_source'],functional_flags=data.get('visual_functional_flags'),functional_metrics=data.get('visual_functional_metrics'),concentration_clamp_not_navigation=r['stage']=='B',body_receptor_cut_performed=False,actual_montage_review='PENDING_PARENT_OPENING'))
    print(str(out/'VIDEO.json'))
if __name__=='__main__':main()
