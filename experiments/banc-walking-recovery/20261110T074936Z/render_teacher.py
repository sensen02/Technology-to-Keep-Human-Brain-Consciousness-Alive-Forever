"""Teacher-only diagnostic video from independently saved real physical states."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from physics import Core,W,LEGS
from identity import HERE,atomic_json,sha256


def main():
    source=HERE/'teacher_native_replay'
    report=json.loads((source/'RESULT.json').read_text())
    trace=source/'TRACE_1MS.npz'
    assert sha256(trace)==report['trace_sha256'] and report['all_gates']
    out=HERE/'TEACHER_VIDEO';out.mkdir(exist_ok=False)
    c=Core(adhesion=False)
    renderer=mj.Renderer(c.plant,height=480,width=640)
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.
    option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
    mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert np.all(c.plant.geom_rgba[mesh,3]==1.)
    fps=30;stride=20;slow=1/(.001*stride*fps)
    frames=0;indexes=[];video=out/'ALGORITHM_TEACHER_NOT_NEURAL.mp4'
    with np.load(trace,allow_pickle=False) as z:
        camera.lookat[:]=z['com'][0]
        writer=imageio.get_writer(video,fps=fps,macro_block_size=1)
        try:
            for i in range(0,len(z['time']),stride):
                c.d.qpos[:]=z['qpos'][i];c.d.qvel[:]=z['qvel'][i]
                mj.mj_forward(c.plant,c.d)
                renderer.update_scene(c.d,camera=camera,scene_option=option)
                image=Image.new('RGB',(1000,480),(245,245,245))
                image.paste(Image.fromarray(renderer.render()),(0,0))
                draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,640,59),fill=(245,245,245))
                draw.text((6,4),'ALGORITHM TEACHER: coupled inverse statics + measured swing shapes',fill=(20,20,20))
                draw.text((6,21),'Derived full body / no adhesion; neural handover NOT shown here',fill=(20,20,20))
                draw.text((6,38),f'Actual simulated t={z["time"][i]:.3f}s; playback {slow:.3f}x slower',fill=(20,20,20))
                right=['ACTUAL CONTACT / NATIVE STATE',
                       f'COM height = {z["com"][i,2]:.4f}',
                       f'Foot support = {z["loads"][i].sum()/W:.4f} body weights',
                       f'Nonfoot force = {z["nonfoot"][i]:.5f} native','']
                for leg,load,height in zip(LEGS,z['loads'][i]/W,z['padz'][i]):
                    n=sum(e['type']=='complete' and e['leg']==leg and e['time']<=z['time'][i] for e in report['events'])
                    right.append(f'{leg}: load/W {load:.3f}  z {height:+.3f}  steps {n}')
                right.extend(['','WHOLE RUN ENGINEERING CHECKS',
                    f'Signed forward {report["forward_signed_native_per_s"]:.4f} native/s',
                    f'Max slip ratio {report["slip_ratio_max"]:.4f} <= 0.5',
                    f'Min COM {report["com_height_min"]:.4f} >= 1.2',
                    'All five engineering gates PASS',
                    'Strict complete steps: '+str(report['complete_per_leg']),
                    '', 'Not original complete acceptance.', 'Not connectome or ANN autonomy.'])
                for line,text in enumerate(right):draw.text((650,10+line*19),text,fill=(20,20,20))
                writer.append_data(np.asarray(image));frames+=1
                indexes.append(dict(frame=frames-1,trace_index=i,time_s=float(z['time'][i])))
                if frames in [1,250,500]:image.save(out/f'frame_{frames:04d}.png')
        finally:
            writer.close();renderer.close()
    reader=imageio.get_reader(video);decoded=reader.count_frames();reader.close();assert decoded==frames
    result=dict(video=str(video),video_sha256=sha256(video),source_trace=str(trace),source_sha256=sha256(trace),
      source_result=str(source/'RESULT.json'),frames=frames,fps=fps,physical_frame_dt_s=.020,slow_factor=slow,
      visible_mesh_geoms=int(mesh.sum()),tendon_overlay_hidden_for_visual_only=True,frame_index=indexes,
      interpretation='saved actual physical qpos only; algorithm teacher walking, NOT student/brain autonomy or final handover video',
      visual_review='pending actual opened frames by parent')
    atomic_json(out/'VIDEO.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='frame_index'}),flush=True)

if __name__=='__main__':main()
