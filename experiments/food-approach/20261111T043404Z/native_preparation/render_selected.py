"""Render selected audited native same-run states. Explicit authorization, zero mj_step."""
from pathlib import Path
import argparse
import json
from contract import HERE, BASE, admitted, require, sha, load, save_new


def render(plan_path,checkpoint,digest,run,audit_path,out,distance):
    plan,proof=admitted(plan_path,checkpoint,digest)
    run=Path(run).resolve();out=Path(out).resolve();audit_path=Path(audit_path).resolve()
    require(run==HERE/'selected_native/EVAL_left','same selected native run required')
    require(out.is_relative_to(HERE/'videos') and not out.exists(),'fresh native videos output required')
    require(10.<=distance<=16.,'declared camera distance range10..16 required')
    report=load(audit_path)
    require(report['status']=='PASS' and not report['failures_or_unknowns'] and report['checkpoint_sha256']==digest and Path(report['run']).resolve()==run,'full current selected native audit PASS required')
    # Only this explicitly authorized rendering entry imports the physical loader.
    import sys
    import numpy as np
    import mujoco as mj
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    def forbidden(*a,**k):raise RuntimeError('physics integration forbidden in same-state renderer')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    sys.path.insert(0,str(BASE))
    from physics import Core,W,LEGS
    from identity import verify_runtime,verify
    paths=[run/n for n in ('RESULT.json','IDENTITY.json','HANDOVER.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz','NATIVE_STATIC.json')]+[audit_path,Path(plan_path),Path(__file__).resolve()]
    hashes={str(p):sha(p) for p in paths};r=load(run/'RESULT.json');ident=load(run/'IDENTITY.json');verify(ident)
    require(r['checkpoint_sha256']==digest and r['engineering_pass'] and r['native']['native_samples']==1232000,'native result identity/gates mismatch')
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
    with np.load(run/'geometry_1ms.npz',allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48),'full actual trajectory absent')
    c=Core(adhesion=False);verify_runtime(ident,c);mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
    require(int(mesh.sum())==71 and np.all(c.plant.geom_rgba[mesh,3]==1.),'complete71opaque model required')
    compiled={key:__import__('hashlib').sha256(getattr(c.plant,key).tobytes()).hexdigest() for key in ident['compiled_arrays']}
    out.mkdir(parents=True,exist_ok=False)
    video=out/'SELECTED_LEFT_SAME_NATIVE_RUN.mp4';indexes=list(range(19,15400,20));mapping=[];frame_silhouettes=[]
    renderer=mj.Renderer(c.plant,height=480,width=640);camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=distance;camera.azimuth=140.;camera.elevation=-25.;camera.lookat[:]=(t['com'].min(0)+t['com'].max(0))/2
    option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;option.geomgroup[:]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    writer=imageio.get_writer(video,fps=30,macro_block_size=1,codec='libx264',quality=8,ffmpeg_log_level='error')
    try:
        for f,i in enumerate(indexes):
            c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(c.plant,c.d)
            require(np.array_equal(c.d.qpos,t['qpos'][i]) and np.array_equal(c.d.qvel,t['qvel'][i]),'renderer changed saved physical pose')
            require(np.allclose(c.d.subtree_com[c.th],t['com'][i],atol=1e-10,rtol=0),'rendered COM differs')
            renderer.update_scene(c.d,camera=camera,scene_option=option);pixels=renderer.render()
            renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
            ys,xs=np.where(np.isin(seg[:,:,0],np.flatnonzero(mesh))&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))
            require(len(xs)>0 and xs.min()>5 and xs.max()<634 and ys.min()>80 and ys.max()<434,'fly silhouette clipped or covered by text; retain failure and use fresh video output with predeclared wider camera')
            frame_silhouettes.append([int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())])
            image=Image.new('RGB',(1040,480),(246,246,246));image.paste(Image.fromarray(pixels),(0,0));draw=ImageDraw.Draw(image)
            source=int(t['control_source'][i]);phase={0:'ZERO-RATE SETTLE',2:'ALGORITHM TEACHER',3:'PURE SUBSTITUTE ANN'}[source]
            draw.rectangle((0,0,639,73),fill=(246,246,246))
            for j,line in enumerate(['SELECTED LEFT NATIVE AUDIT PASS',f'{phase} | actual t={t["time"][i]:.3f}s','Food NOT evaluated | 20ms bins / 30fps / 1.667x slow']):draw.text((8,5+22*j),line,font=font,fill=(20,20,20))
            lines=['SAME ACTUAL NATIVE RECORD',f'Native rows {r["native"]["native_samples"]}',f'COM z {t["com"][i,2]:.6f}',f'Footload/W {t["loads"][i].sum()/W:.6f}',f'Nonfoot {t["nonfoot"][i]:.6f} native']
            lines += [f'{leg}: padz {height:+.4f} load/W {load/W:.4f}' for leg,height,load in zip(LEGS,t['padz'][i],t['loads'][i])]
            lines += ['', 'FULL ANN WINDOW 9.600s',f'Yaw {r["yaw_change_rad"]:+.6f} rad',f'Logical slip {r["autonomous"]["slip_ratio_max"]:.6f}',f'Native slip {report["engineering_native"]["slip"]["value"]:.6f}','D1-D8: see declared native methods','All71mesh opaque; extra forces0','Finite PD / native units only','SI / biology NOT ESTABLISHED']
            for j,line in enumerate(lines):draw.text((647,7+19*j),line,font=font,fill=(20,20,20))
            draw.rectangle((0,440,639,479),fill=(246,246,246));draw.text((8,444),'Same-run saved states; no motion correction / no physics step',font=font,fill=(20,20,20));draw.text((8,463),'Full body visible / continuous handover at actual 5.800s',font=font,fill=(20,20,20))
            writer.append_data(np.asarray(image));mapping.append(dict(frame=f,trace_index=i,actual_time_s=float(t['time'][i]),control_source=source))
    except Exception as exc:
        save_new(out/'VIDEO_FAILURE.json',dict(error=repr(exc),frames_attempted=len(mapping),source_hashes=hashes,physical_steps=0));raise
    finally:writer.close();renderer.close()
    key_indexes={min(range(len(mapping)),key=lambda j:abs(mapping[j]['actual_time_s']-time)) for time in (.02,1.,5.78,5.8,7.3,15.4)}
    reader=imageio.get_reader(video);frames=0;images=[];keys=[]
    try:
        metadata=reader.get_meta_data()
        for f,pixels in enumerate(reader):
            frames+=1
            if f in key_indexes:
                path=out/f'decoded_{f:04d}.png';im=Image.fromarray(pixels);im.save(path);images.append(im);keys.append(dict(path=str(path),sha256=sha(path),**mapping[f]))
    finally:reader.close()
    require(frames==770 and abs(metadata['fps']-30)<1e-8 and abs(metadata['duration']-770/30)<.04,'decoded complete video count/duration differs')
    require(abs(mapping[-1]['actual_time_s']-15.4)<1e-8,'video omitted final physical endpoint')
    montage=Image.new('RGB',(2080,1440),(246,246,246))
    for j,im in enumerate(images):montage.paste(im,((j%2)*1040,(j//2)*480))
    montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path)
    require(all(sha(p)==h for p,h in hashes.items()),'rendering input changed')
    require(all(__import__('hashlib').sha256(getattr(c.plant,k).tobytes()).hexdigest()==h for k,h in compiled.items()),'rendering changed compiled physical arrays')
    require(float(c.d.time)==float(t['time'][-1]),'rendered final state time differs')
    save_new(out/'FRAME_MAPPING.json',mapping)
    save_new(out/'VIDEO.json',dict(video=str(video),sha256=sha(video),frames=frames,decoded_frames=frames,source_run=str(run),
             native_audit=str(audit_path),native_audit_sha256=sha(audit_path),source_hashes=hashes,source_unchanged=True,
             camera=dict(distance=distance,azimuth=140.,elevation=-25.,fixed_world=True),mesh_geoms=71,all_mesh_opaque=True,
             all770_silhouettes_in_frame=True,silhouette_bounds=frame_silhouettes,all_qpos_qvel_exact=True,compiled_arrays_unchanged=True,
             physical_steps=0,frame_sampling='last endpoint of each20ms source bin; first.020,last15.400;50settle+240teacher+480ANN',
             physical_seconds=15.4,playback_seconds=770/30,slow_factor=1/(.020*30),keyframes=keys,montage=str(montage_path),
             parent_visual_review='PENDING actual opening; renderer cannot self-certify human/parent review',food_approach_established=False))
    print(json.dumps(dict(video=str(video),decoded_frames=frames,montage=str(montage_path),physical_steps=0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-plan',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--run',required=True);p.add_argument('--native-audit',required=True);p.add_argument('--out',required=True);p.add_argument('--camera-distance',type=float,default=10.);p.add_argument('--render-authorized',action='store_true')
    a=p.parse_args();require(a.render_authorized,'explicit parent --render-authorized required')
    render(a.run_plan,a.checkpoint,a.sha256,a.run,a.native_audit,a.out,a.camera_distance)
if __name__=='__main__':main()
