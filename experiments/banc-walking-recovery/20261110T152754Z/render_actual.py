"""Render completed same-run saved physical states; never integrate physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

SNAPSHOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
sys.path.insert(0,str(SNAPSHOT))
from physics import Core,W,LEGS
from identity import sha256,atomic_json,verify_runtime,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False)
    paths=[run/x for x in ('RESULT.json','IDENTITY.json','trace_1ms.npz','decisions_5ms.npz')]+[Path(__file__).resolve()]
    hashes={str(x):sha256(x) for x in paths}
    r=json.loads(paths[0].read_text());ident=json.loads(paths[1].read_text());verify(ident)
    assert hashes[str(paths[2])]==r['trace']['sha256'] and hashes[str(paths[3])]==r['decisions']['sha256']
    assert r['control']=='student' and not r['sampled'] and r['transition']['all_exact']
    def forbidden(*args,**kwargs):raise RuntimeError('physics integration forbidden in renderer')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    c=Core(adhesion=False);verify_runtime(ident,c);model=c.plant
    assert (model.nq,model.nv,model.nu)==(49,48,42)
    mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert mesh.sum()==71 and np.all(model.geom_rgba[mesh,3]==1.)
    compiled={k:hashlib.sha256(getattr(model,k).tobytes()).hexdigest() for k in ident['compiled_arrays']}
    with np.load(paths[2],allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    n=len(z['time']);assert n==15400 and z['qpos'].shape==(n,49) and z['qvel'].shape==(n,48)
    assert np.array_equal(z['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)])
    assert all(np.isfinite(v).all() for v in z.values())
    assert np.array_equal(z['actual_requested_action'],z['action'])
    assert all(np.max(np.abs(z[k]))==0 for k in ('external_body_force_max','external_generalized_force_max'))
    with np.load(paths[3],allow_pickle=False) as d:
        assert np.array_equal(z['action'][1000:],np.repeat(d['action'],5,axis=0))
        assert np.array_equal(d['X238'][1:,195:237],d['action'][:-1])
    ann=r['autonomous'];counts=r['autonomous_cycles']['complete_per_leg']
    status='ENGINEERING GATES PASS' if ann['all_gates'] else 'ENGINEERING GATES FAIL'
    native=r.get('native');native_rows=0 if native is None else native['native_samples']
    video=out/'ACTUAL_ANN_HANDOVER.mp4'
    renderer=mj.Renderer(model,height=480,width=640);camera=mj.MjvCamera()
    camera.type=mj.mjtCamera.mjCAMERA_FREE;camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.;camera.lookat[:]=z['com'][0]
    option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False;option.geomgroup[:]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    indexes=list(range(0,n,20));mapping=[]
    writer=imageio.get_writer(video,fps=30,macro_block_size=1,codec='libx264',quality=8,ffmpeg_log_level='error')
    try:
        for frame,i in enumerate(indexes):
            t=float(z['time'][i]);code=int(z['control_source'][i]);phase={0:'ZERO RATE SETTLE',2:'ALGORITHM TEACHER',3:'PURE SUBSTITUTE ANN'}[code]
            c.d.qpos[:]=z['qpos'][i];c.d.qvel[:]=z['qvel'][i];c.d.time=t;mj.mj_forward(model,c.d)
            assert np.array_equal(c.d.qpos,z['qpos'][i]) and np.array_equal(c.d.qvel,z['qvel'][i])
            assert np.allclose(c.d.subtree_com[c.th],z['com'][i],atol=1e-10,rtol=0)
            renderer.update_scene(c.d,camera=camera,scene_option=option)
            image=Image.new('RGB',(1000,480),(246,246,246));image.paste(Image.fromarray(renderer.render()),(0,0));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,639,73),fill=(246,246,246))
            for j,line in enumerate([status,f'{phase} | actual t={t:.3f}s','Same saved physical run | 1.667x slow | boundary 5.800s']):draw.text((8,5+j*22),line,font=font,fill=(20,20,20))
            lines=['ACTUAL MEASUREMENTS',f'COM z {z["com"][i,2]:.5f}',f'Foot load/W {z["loads"][i].sum()/W:.5f}',f'Nonfoot {z["nonfoot"][i]:.5f} native','PAD z / actual load/W']
            lines += [f'{leg}: {h:+.4f} / {load/W:.4f}' for leg,h,load in zip(LEGS,z['padz'][i],z['loads'][i])]
            lines += ['', 'FULL PURE ANN WINDOW 9.600s',f'Forward {ann["forward_signed_native_per_s"]:+.6f}/s',f'COM min {ann["com_height_min"]:.6f}',f'Nonfoot max {ann["nonfoot_force_max_native"]:.6f}',f'Slip max {ann["slip_ratio_max"]:.6f} (limit .5)','Cycles '+' '.join(f'{k}{v}' for k,v in counts.items()),f'Native rows: {native_rows}','All 71 opaque body meshes','Adhesion OFF; extra forces ZERO','Original acceptance: separate audit']
            for j,line in enumerate(lines):draw.text((647,7+j*19),line,font=font,fill=(20,20,20))
            draw.rectangle((0,440,639,479),fill=(246,246,246));draw.text((8,444),'Substitute ANN / derived full body / finite PD motors',font=font,fill=(20,20,20));draw.text((8,463),'Same-run trace only; no rendered motion correction',font=font,fill=(20,20,20))
            writer.append_data(np.asarray(image));mapping.append(dict(frame=frame,trace_index=i,actual_time_s=t,control_source=code))
    finally:writer.close();renderer.close()
    targets=[0.,1.,5.78,5.8,7.3,15.38];selected={min(range(len(mapping)),key=lambda j:abs(mapping[j]['actual_time_s']-t)):t for t in targets}
    reader=imageio.get_reader(video);keyframes=[];images=[];count=0
    try:
        meta=reader.get_meta_data()
        for f,pixels in enumerate(reader):
            count+=1
            if f in selected:
                path=out/f'decoded_{f:04d}.png';im=Image.fromarray(pixels);im.save(path);images.append(im);keyframes.append(dict(path=str(path),sha256=sha256(path),**mapping[f]))
    finally:reader.close()
    assert count==770 and abs(meta['fps']-30)<1e-8 and abs(meta['duration']-770/30)<.04
    montage=Image.new('RGB',(2000,1440),(246,246,246))
    for j,im in enumerate(images):montage.paste(im,((j%2)*1000,(j//2)*480))
    montage_path=out/'DECODED_MONTAGE.png';montage.save(montage_path)
    assert all(sha256(Path(k))==v for k,v in hashes.items())
    assert all(hashlib.sha256(getattr(model,k).tobytes()).hexdigest()==v for k,v in compiled.items());verify_runtime(ident,c)
    atomic_json(out/'FRAME_MAPPING.json',mapping)
    atomic_json(out/'VIDEO.json',dict(video=str(video),sha256=sha256(video),frames=770,decoded_frames=count,physical_seconds=15.4,playback_seconds=770/30,slow_factor=1/(.020*30),source_run=str(run),source_hashes=hashes,source_unchanged=True,compiled_arrays_unchanged=True,mesh_geoms=71,all_mesh_opaque=True,physics_steps=0,all_qpos_qvel_exact=True,keyframes=keyframes,montage=str(montage_path),engineering_gates=ann['gates'],metrics=ann,cycles=counts,native_rows=native_rows,full_goal_passed='see independent acceptance report',visual_review='pending actual opening by parent'))
    print(json.dumps(dict(video=str(video),frames=count,montage=str(montage_path),status=status)),flush=True)

if __name__=='__main__':main()
