"""Diagnostic rendering ONLY from saved native physical states; no new motion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from physics import Core,W
from identity import atomic_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--recording',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--stride',type=int,default=800)
    p.add_argument('--fps',type=int,default=30)
    a=p.parse_args()
    directory=Path(a.recording).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    files=sorted(directory.glob('native_*.npz'))
    if not files:raise ValueError('no native record chunks')
    core=Core();renderer=mj.Renderer(core.plant,height=480,width=640)
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.
    mesh_mask=core.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert np.all(core.plant.geom_rgba[mesh_mask,3]==1.)
    index=0;frames=[];times=[];states=[]
    slow=1/(a.stride*float(core.plant.opt.timestep)*a.fps)
    video=out/'diagnostic.mp4'
    writer=imageio.get_writer(video,fps=a.fps,macro_block_size=1)
    try:
        for file in files:
            with np.load(file,allow_pickle=False) as data:
                for row in range(len(data['time_after_s'])):
                    if index%a.stride:
                        index+=1;continue
                    core.d.qpos[:]=data['qpos_after'][row]
                    core.d.qvel[:]=data['qvel_after'][row]
                    mj.mj_forward(core.plant,core.d)
                    camera.lookat[:]=data['com_after'][row]
                    renderer.update_scene(core.d,camera=camera)
                    image=Image.fromarray(renderer.render())
                    draw=ImageDraw.Draw(image)
                    t=float(data['time_after_s'][row]);times.append(t)
                    lines=['DIAGNOSTIC: replacement GRU ANN / derived triad body',
                           'BC initialization; slip gate FAIL; NOT walking acceptance',
                           f'Saved physical state t={t:.5f}s; playback {slow:.3f}x slower',
                           'No continuous prewarm handover in this run',
                           f'COM={data["com_after"][row,2]:.3f}; nonfoot F_native={data["nonfoot_force_native"][row]:.4f}']
                    draw.rectangle((0,0,640,86),fill=(245,245,245))
                    for j,line in enumerate(lines):draw.text((6,4+j*16),line,fill=(25,25,25))
                    frame=np.asarray(image)
                    writer.append_data(frame)
                    if len(times) in (1,100,200):
                        image.save(out/f'frame_{len(times):04d}.png')
                    states.append(dict(frame=len(times)-1,native_index=index,time_s=t,chunk=str(file),row=row))
                    index+=1
    finally:
        writer.close();renderer.close()
    metadata=dict(video=str(video),sha256=sha256(video),frames=len(times),fps=a.fps,
                  sampling_stride_native=a.stride,native_dt_s=float(core.plant.opt.timestep),
                  playback_slow_factor=slow,recorded_first_s=times[0],recorded_last_s=times[-1],
                  state_source='saved native qpos/qvel only; mj_forward for rendering; no motion integration',
                  native_chunks={str(f):sha256(f) for f in files},frame_index=states,
                  visible_mesh_geoms=int(mesh_mask.sum()),status='diagnostic failure visualization, not final handover video',
                  visual_inspection='pending actual image open by parent')
    reader=imageio.get_reader(video)
    metadata['decoded_frame_count']=reader.count_frames()
    reader.close();assert metadata['decoded_frame_count']==metadata['frames']
    atomic_json(out/'VIDEO.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('native_chunks','frame_index')}),flush=True)

if __name__=='__main__':main()
