"""Render verified saved poses at synchronized named native contact events."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_pair import open_measured
from support_experiment import W, bench, read, stan
from geometry import make_gap_model

WIDTH, HEIGHT, SCENE, FPS = 620, 380, 315, 30
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LEG_ZH = dict(LF="左前足",RH="右后足",RF="右前足",LH="左后足",RM="右中足",LM="左中足")


def camera(center, azimuth, elevation, distance):
    result = mj.MjvCamera()
    result.type = mj.mjtCamera.mjCAMERA_FREE
    result.lookat[:] = center
    result.azimuth,result.elevation,result.distance = azimuth,elevation,distance
    return result


def prepare(place):
    result,trace,decisions,native = open_measured(place)
    decisions.close()
    native.close()
    with trace:
        arrays = {name:trace[name] for name in ("qpos","qvel","com","loads","pair_pad_xyz","padz")}
    model = make_gap_model(result["scenario"])
    model.geom_friction[:,0] = bench.stand.physics.rc.MU
    return dict(place=place,result=result,arrays=arrays,model=model, data=mj.MjData(model))


def render(output,on,off,certificate,diagnostic=False):
    cert = read(certificate)
    experiments = [prepare(off),prepare(on)]
    if not cert["qualified"] and not diagnostic:
        raise ValueError("Renderer expects verified alternate landing and stop event times")
    t = cert["times"]
    final = cert["actual_ms"]
    trigger = int(t["trigger_ms"])
    landing = int(np.ceil(t["helper_on_load_ms"]))
    helper = bench.LEGS.index(cert["helper"])
    missing = bench.LEGS.index(cert["target"])
    hname,mname = LEG_ZH[cert["helper"]],LEG_ZH[cert["target"]]
    samples = [("辅助臂动作前",trigger-18),("辅助臂首次落脚命令",trigger),
               ("辅助臂首次持续承重",landing),("持续承重20 ms",landing+20),
               ("同一时刻的主动臂终点" if diagnostic else "两臂末段",final)]
    near = (trigger-70,min(final,landing+245))
    ticks = sorted(set(range(0,final+1,20)) | set(range(*near,5)) | {ms for _,ms in samples})
    fonts = [ImageFont.truetype(FONT,n) for n in (21,15)]
    for item in experiments:
        root = item["arrays"]["com"][trigger].copy()
        site = item["arrays"]["pair_pad_xyz"][landing,1]
        item["wide"] = camera(root - [0.,0.,.43],160,-24,7.0)
        item["close"] = camera([*site[:2],.14],227,-31,1.43)

    def never(*_, **__):
        raise RuntimeError("Video uses archived poses without physical stepping")

    mj.mj_step = mj.mj_step1 = mj.mj_step2 = never
    output.mkdir()
    movie = output/"换足补撑配对.mp4"
    still = output/"换足补撑关键帧.png"

    def pose(arm,ms,close):
        q = arm["arrays"]
        d,m = arm["data"],arm["model"]
        d.qpos[:] = q["qpos"][ms]
        d.qvel[:] = q["qvel"][ms]
        mj.mj_forward(m,d)
        np.testing.assert_array_equal(d.qpos,q["qpos"][ms])
        renderer=arm["renderer"]
        renderer.update_scene(d,camera=arm["close"] if close else arm["wide"])
        return Image.fromarray(renderer.render())

    def draw(ms,detail=False,stage=None):
        sheet = Image.new("RGB",(WIDTH*2,HEIGHT),(239,244,244))
        label = ImageDraw.Draw(sheet)
        for col,arm in enumerate(experiments):
            x = col*WIDTH
            sheet.paste(pose(arm,ms,detail),(x,65))
            if not detail and near[0]<=ms<=near[1]:
                zoom = pose(arm,ms,True).resize((224,121),Image.Resampling.LANCZOS)
                sheet.paste((240,245,245),(x+386,69,x+617,198))
                sheet.paste(zoom,(x+390,73))
            use = "开启" if col else "关闭"
            default_stage = "换足早期诊断，停稳失败" if diagnostic else "踏空换足与停稳"
            label.text((x+11,4),f"辅助足动作{use} · {stage or default_stage}",font=fonts[0],fill="#192526")
            f = arm["arrays"]["loads"][ms]/W
            label.text((x+11,36),f"{ms/1000:.3f} s  {mname} {f[missing]:.3f} W  {hname} {f[helper]:.3f} W  全足 {f.sum():.3f} W",
                       font=fonts[1],fill="#334a4d")
        return sheet

    with ExitStack() as handles:
        for arm in experiments:
            arm["renderer"] = handles.enter_context(mj.Renderer(arm["model"],height=SCENE,width=WIDTH))
        writer = handles.enter_context(imageio.get_writer(movie,fps=FPS,codec="libx264",quality=8,
                                                          macro_block_size=2,ffmpeg_log_level="error"))
        for ms in ticks:
            writer.append_data(np.asarray(draw(ms)))
        montage = Image.new("RGB",(WIDTH*2,HEIGHT*len(samples)))
        for i,(stage,ms) in enumerate(samples):
            montage.paste(draw(ms,detail=(i in (0,1,2,3)),stage=stage),(0,i*HEIGHT))
        montage.save(still)
    full = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                           "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)],
                          check=True,text=True,capture_output=True)
    assert int(full.stdout.strip()) == len(ticks)
    stan.put(output/"VIDEO.json",dict(video=str(movie),video_sha256=stan.sha(movie),montage=str(still),
                montage_sha256=stan.sha(still),decoded_frames=len(ticks),actual_integrations=0,fps=FPS,
                certificate=str(certificate),certificate_sha256=stan.sha(certificate),milliseconds=ticks,
                anchors=samples,source_paths=[str(item["place"]) for item in experiments],
                scope=cert["scope"],diagnostic_only=diagnostic,sustained_walk_claim=False))
    print(f"Decoded {len(ticks)} frames",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("on",type=Path)
    p.add_argument("off",type=Path)
    p.add_argument("certificate",type=Path)
    p.add_argument("--diagnostic-only",action="store_true")
    a=p.parse_args()
    render(*(v.resolve() for v in (a.output,a.on,a.off,a.certificate)),diagnostic=a.diagnostic_only)
