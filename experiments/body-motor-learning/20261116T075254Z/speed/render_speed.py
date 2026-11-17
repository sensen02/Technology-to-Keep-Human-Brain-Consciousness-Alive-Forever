"""Frame-matched archived physical trajectories of distinct tested pace requests."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from speed_teacher import NOMINAL_V, WARM, stand
from walking_bench import read

WIDTH,HEIGHT,SCENE,FPS=480,340,265,30
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def trial(path):
    case=read(path/"RESULT.json")
    source=path/"TRACE.npz"
    if stand.sha(source)!=case["trace_sha256"]:
        raise ValueError("Physical trajectory unavailable")
    with np.load(source,allow_pickle=False) as z:
        frames={k:z[k] for k in ("qpos","qvel","com","command")}
    model=stand.physics.act_mod.load_full_model(stand.physics.PAD_XML)
    cam=mj.MjvCamera()
    cam.type=mj.mjtCamera.mjCAMERA_FREE
    cam.distance=5.8
    cam.azimuth=122
    cam.elevation=-14
    cam.lookat[:]=[-.35,0.,.77]
    return dict(run=str(path),source=str(source),case=case,frames=frames,model=model,
                data=mj.MjData(model),camera=cam)


def show(output,runs):
    cases=[trial(run) for run in runs]
    ends={len(q["frames"]["qpos"])-1 for q in cases}
    if len(ends)!=1:
        raise ValueError("Paired clip uses full common measured interval")
    finish=ends.pop()
    ms_times=sorted(set(range(0,finish+1,20))|{finish})
    marks=[500,max(1000,finish//3),finish-1200,finish]
    large=ImageFont.truetype(FONT,20)
    small=ImageFont.truetype(FONT,14)
    labels=["低速","中速","较高速"] if len(cases)==3 else [f"试验 {x+1}" for x in range(len(cases))]
    output.mkdir()
    video=output/"身体连续调速对照.mp4"
    montage=output/"身体连续调速关键帧.png"

    def no_integrator(*_,**__):
        raise RuntimeError("Offline renderer cannot integrate physics")

    mj.mj_step=mj.mj_step1=mj.mj_step2=no_integrator

    def frame(ms):
        image=Image.new("RGB",(WIDTH*len(cases),HEIGHT),(244,247,244))
        draw=ImageDraw.Draw(image)
        for i,(case,label) in enumerate(zip(cases,labels,strict=True)):
            p=case["frames"]
            d=case["data"]
            d.qpos[:]=p["qpos"][ms]
            d.qvel[:]=p["qvel"][ms]
            mj.mj_forward(case["model"],d)
            np.testing.assert_array_equal(d.qpos,p["qpos"][ms])
            case["render"].update_scene(d,camera=case["camera"])
            image.paste(Image.fromarray(case["render"].render()),(i*WIDTH,75))
            earliest=max(0,ms-480)
            distance=p["com"][ms,0]-p["com"][0,0]
            actual=(p["com"][ms,0]-p["com"][earliest,0])/.001/(ms-earliest) if ms else 0.
            walk=case["case"]["walk"]
            request=p["command"][ms,0]
            tag="联合物理通过" if case["case"]["native_joint_pass"] else "联合门未通过"
            x = i*WIDTH + 10
            draw.text((x,5),f"{label}  ·  {ms/1000:.3f}s  ·  {tag}",font=large,fill="#243a3a")
            draw.text((x,38),f"请求 {request:.4f}   近半秒实速 {actual:.4f}   前进 {distance:.3f}",font=small,fill="#314a4a")
            draw.text((x,56),f"全步行均速 {walk['signed_forward_native_s']:.5f}原生长度/s   滑移 {walk['max_ground_normalized_slip']:.3f}",
                      font=small,fill="#314a4a")
        return image

    with ExitStack() as stack:
        for c in cases:
            c["render"]=stack.enter_context(mj.Renderer(c["model"],width=WIDTH,height=SCENE))
        writer=stack.enter_context(imageio.get_writer(video,fps=FPS,codec="libx264",quality=8,
                                                     macro_block_size=2,ffmpeg_log_level="error"))
        for t in ms_times:
            writer.append_data(np.asarray(frame(t)))
        image=Image.new("RGB",(WIDTH*len(cases),HEIGHT*len(marks)))
        for i,t in enumerate(marks):
            image.paste(frame(t),(0,HEIGHT*i))
        image.save(montage)
    verified=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                              "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(video)],
                             text=True,check=True,capture_output=True)
    assert int(verified.stdout.strip())==len(ms_times)
    stand.put(output/"VIDEO.json",dict(actual_integrations=0,decoded_frames=len(ms_times),fps=FPS,
                 frame_times_ms=ms_times,montage_times_ms=marks,video=str(video),video_sha256=stand.sha(video),
                 montage=str(montage),montage_sha256=stand.sha(montage),source_runs=[str(x) for x in runs],
                 renderer_sha256=stand.sha(Path(__file__))))
    print(f"Decoded {len(ms_times)} archived frames",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("runs",type=Path,nargs="+")
    a=p.parse_args()
    show(a.output.resolve(),[q.resolve() for q in a.runs])
