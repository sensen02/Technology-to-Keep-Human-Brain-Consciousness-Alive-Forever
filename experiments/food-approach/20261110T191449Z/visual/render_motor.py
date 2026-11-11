"""Same-run conditional motor rendering. No integration or motion correction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from physics import Core,W,LEGS
from identity import sha256,atomic_json,verify,verify_runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if not Path(args.run).is_absolute() or not Path(args.out).is_absolute():
        raise ValueError('absolute paths required')
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if not out.is_relative_to(HERE) or out==HERE:
        raise ValueError('new output must be under renderer visual directory')
    if out.exists():raise FileExistsError(out)
    paths=[run/name for name in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz')]
    paths += [Path(__file__).resolve(),HERE/'PREREG.json']
    hashes={str(path):sha256(path) for path in paths}
    result=json.loads(paths[0].read_text());identity=json.loads(paths[1].read_text())
    verify(identity)
    assert result['identity_sha256']==identity['identity_sha256']
    assert hashes[str(paths[2])]==result['trace']['sha256']
    assert hashes[str(paths[3])]==result['neural_decisions_sha256']
    assert result['transition']['all_exact'] and result['transition']['no_reset']
    assert result['teacher_calls_after_handover']==0 and result['extra_forces'] is False
    assert result['odor_or_food_success'] is False
    calls={'integration_attempts':0}
    def forbidden(*args,**kwargs):
        calls['integration_attempts']+=1
        raise RuntimeError('physics integration forbidden in renderer')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    core=Core(adhesion=False);verify_runtime(identity,core);model=core.plant
    assert (model.nq,model.nv,model.nu)==(49,48,42)
    mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1.)
    with np.load(paths[2],allow_pickle=False) as archive:z={key:archive[key] for key in archive.files}
    with np.load(paths[3],allow_pickle=False) as archive:d={key:archive[key] for key in archive.files}
    n=len(z['time']);assert n==15400 and z['qpos'].shape==(n,49) and z['qvel'].shape==(n,48)
    assert all(np.isfinite(value).all() for value in z.values())
    assert all(np.isfinite(value).all() for value in d.values())
    assert np.array_equal(z['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)])
    assert np.max(np.abs(z['time']-np.arange(1,n+1)*.001))<1e-8
    assert np.array_equal(z['actual_requested_action'],z['action'])
    assert all(np.max(np.abs(z[key]))==0 for key in ('external_body_force_max','external_generalized_force_max'))
    assert len(d['time'])==2880 and d['command'].shape==(2880,2)
    assert np.array_equal(z['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0))
    assert np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1])
    assert np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1])
    assert np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)])
    assert np.array_equal(d['command'][:960],np.zeros((960,2)))
    assert np.array_equal(d['command'][960:],np.tile(result['command'],(1920,1)))
    assert np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:])
    yaw=np.unwrap(d['yaw_after_each_control_ms'])
    assert len(yaw)==14400
    assert abs(float(yaw[-1]-yaw[4799])-result['yaw_change_rad'])<1e-12
    auto=result['autonomous'];counts=result['autonomous_cycles']['complete_per_leg']
    status='ENGINEERING GATES PASS' if auto['all_gates'] else 'ENGINEERING GATES FAIL'
    out.mkdir()
    video=out/'ACTUAL_MOTOR_HANDOVER.mp4'
    renderer=mj.Renderer(model,height=480,width=640)
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.
    camera.lookat[:]=z['com'][0]  # Fixed world camera; never follows/corrects motion.
    option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;option.geomgroup[:]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    mapping=[];yaw_restored=[]
    writer=imageio.get_writer(video,fps=30,macro_block_size=1,codec='libx264',quality=8,ffmpeg_log_level='error')
    try:
        for frame,i in enumerate(range(0,n,20)):
            time=float(z['time'][i]);code=int(z['control_source'][i])
            phase={0:'ZERO RATE SETTLE',2:'ALGORITHM TEACHER',3:'PURE CONDITIONAL ANN'}[code]
            core.d.qpos[:]=z['qpos'][i];core.d.qvel[:]=z['qvel'][i];core.d.time=time
            mj.mj_forward(model,core.d)
            assert np.array_equal(core.d.qpos,z['qpos'][i]) and np.array_equal(core.d.qvel,z['qvel'][i])
            assert np.allclose(core.d.subtree_com[core.th],z['com'][i],atol=1e-10,rtol=0)
            rotation=core.d.xmat[core.th].reshape(3,3)
            actual_yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
            if i>=1000:assert abs(np.arctan2(np.sin(actual_yaw-yaw[i-1000]),np.cos(actual_yaw-yaw[i-1000])))<1e-10
            command=np.zeros(2) if i<1000 else d['command'][(i-1000)//5]
            renderer.update_scene(core.d,camera=camera,scene_option=option)
            image=Image.new('RGB',(1040,480),(246,246,246));image.paste(Image.fromarray(renderer.render()),(0,0));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,639,76),fill=(246,246,246))
            color=(170,20,20) if not auto['all_gates'] else (20,100,20)
            for j,line in enumerate([status,f'{phase} | actual t={time:.3f}s','20ms saved states / 30fps / 1.667x slow']):draw.text((8,5+j*22),line,font=font,fill=color if j==0 else (20,20,20))
            lines=['ACTUAL SAME-RUN MEASUREMENTS',f'COM z {z["com"][i,2]:.5f} | foot load/W {z["loads"][i].sum()/W:.4f}',f'Nonfoot {z["nonfoot"][i]:.6f} native',f'Actual heading yaw {actual_yaw:+.6f} rad',f'Command forward {command[0]:+.3f} turn {command[1]:+.3f}',f'Applied 42-action RMS {np.sqrt(np.mean(z["action"][i]**2)):.5f}','PAD z / actual load/W']
            lines += [f'{leg}: {height:+.4f} / {load/W:.4f}' for leg,height,load in zip(LEGS,z['padz'][i],z['loads'][i])]
            lines += ['FULL PURE ANN WINDOW 9.600s',f'Forward {auto["forward_signed_native_per_s"]:+.6f}/s',f'Slip max {auto["slip_ratio_max"]:.6f} (limit .5)',f'Yaw change {result["yaw_change_rad"]:+.6f} rad',f'Requested command {result["command"]}','Cycles '+' '.join(f'{key}{value}' for key,value in counts.items()),'ENGINEERING ONLY / NO FOOD BEHAVIOR','Original full-native acceptance UNKNOWN','All 71 meshes opaque / extra forces ZERO','Fixed world camera / exact saved qpos qvel']
            for j,line in enumerate(lines):draw.text((647,7+j*19),line,font=font,fill=(20,20,20))
            draw.rectangle((0,440,639,479),fill=(246,246,246))
            draw.text((8,444),'Substitute ANN / derived full body / finite PD motors',font=font,fill=(20,20,20))
            draw.text((8,463),'No integration / no rendered motion correction / no food',font=font,fill=(20,20,20))
            writer.append_data(np.asarray(image))
            mapping.append(dict(frame=frame,trace_index=i,actual_time_s=time,control_source=code,command=command.tolist(),restored_heading_rad=actual_yaw))
            if frame%100==0:print(f'rendered {frame+1}/770',flush=True)
    finally:writer.close();renderer.close()
    selected={min(range(len(mapping)),key=lambda j:abs(mapping[j]['actual_time_s']-target)) for target in (0.,1.,5.78,5.8,7.3,15.38)}
    reader=imageio.get_reader(video);keyframes=[];images=[];count=0
    try:
        metadata=reader.get_meta_data()
        for frame,pixels in enumerate(reader):
            count+=1
            if frame in selected:
                path=out/f'decoded_{frame:04d}.png';image=Image.fromarray(pixels);image.save(path);images.append(image)
                keyframes.append(dict(path=str(path),sha256=sha256(path),**mapping[frame]))
    finally:reader.close()
    assert count==770 and abs(metadata['fps']-30)<1e-8 and abs(metadata['duration']-770/30)<.04
    montage=Image.new('RGB',(2080,1440),(246,246,246))
    for j,image in enumerate(images):montage.paste(image,((j%2)*1040,(j//2)*480))
    montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path)
    assert calls['integration_attempts']==0
    assert all(sha256(path)==digest for path,digest in hashes.items())
    verify_runtime(identity,core)
    atomic_json(out/'FRAME_MAPPING.json',mapping)
    artifacts={str(path):sha256(path) for path in [video,montage_path,out/'FRAME_MAPPING.json']+[Path(row['path']) for row in keyframes]}
    manifest=dict(video=str(video),sha256=sha256(video),source_run=str(run),source_hashes=hashes,artifacts_sha256=artifacts,frames=770,decoded_frames=count,physical_seconds=15.4,playback_seconds=770/30,slow_factor=1/(.02*30),sampling_ms=20,source_unchanged=True,compiled_arrays_unchanged=True,identity_runtime_verified=True,mesh_geoms=71,all_mesh_opaque=True,physics_steps=0,integration_attempts=0,blocked_entrypoints=['mj_step','mj_step1','mj_step2'],all_qpos_qvel_exact=True,restored_yaw_matches_trace=True,fixed_world_camera=True,phase_counts={'settle':1000,'teacher':4800,'ANN':9600},keyframes=keyframes,montage=str(montage_path),engineering_gates=auto['gates'],engineering_all_gates=auto['all_gates'],metrics=auto,yaw_change_rad=result['yaw_change_rad'],command=result['command'],cycles=counts,food_behavior=False,full_native_acceptance='UNKNOWN / not evaluated here',visual_review='pending actual opening')
    atomic_json(out/'VIDEO.json',manifest)
    print(json.dumps(dict(video=str(video),frames=count,montage=str(montage_path),status=status)),flush=True)

if __name__=='__main__':main()
