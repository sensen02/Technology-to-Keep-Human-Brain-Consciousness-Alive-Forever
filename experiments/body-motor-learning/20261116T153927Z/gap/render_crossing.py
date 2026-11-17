"""Render retained poses with the exact compiled shallow terrain, no integration."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from crossing import RF, W, read, stan
from geometry import make_gap_model

WIDTH,HEIGHT,IMAGE_HEIGHT,FPS=620,364,296,30
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def open_trial(path, caption):
    result = read(path/"RESULT.json")
    archive = path/"TRACE.npz"
    assert stan.sha(archive)==result["trace_sha256"]
    with np.load(archive,allow_pickle=False) as f:
        arrays={k:f[k] for k in ("qpos","qvel","loads","padz","rf_pad_xyz","com","all_pad_xyz")}
    model=make_gap_model(result["scenario"])
    model.geom_rgba[model.geom("gap_terrain_bottom").id]=[.44,.25,.16,1.]
    return dict(info=result,arrays=arrays,model=model,data=mj.MjData(model),caption=caption,path=path)


def camera(pos, close):
    c=mj.MjvCamera()
    c.type=mj.mjtCamera.mjCAMERA_FREE
    c.lookat[:]=pos
    c.azimuth=250 if close else 123
    c.elevation=-40 if close else -23
    c.distance=1.62 if close else 6.1
    return c


def render(output, cases, summary):
    output.mkdir()
    def deny(*args,**kw):
        raise RuntimeError("This renderer can only use replay")
    mj.mj_step=mj.mj_step1=mj.mj_step2=deny
    first_summary=summary["candidate"]
    loss=int(round(first_summary["first_slot_miss"]["ms"]))
    rise=int(round(first_summary["first_lift_after_loss"]["ms"]))
    far=int(round(first_summary["first_in_lane_far_support"]["ms"]))
    crossing_ms=next(t for t,edge in cases[1]["info"]["edges"] if edge=="far_rim_cleared")
    end=min(t["info"]["actual_ms"] for t in cases)
    xmin,xmax,ymin,ymax=cases[1]["info"]["scenario"]["opening"]
    detail=camera(((xmax+xmin)/2,(ymax+ymin)/2+.06,.10),True)
    phases=dict((edge,t) for t,edge in cases[1]["info"]["edges"])
    notes=[(loss,"实际踏空"),(rise,"右前足抬离缺口"),(crossing_ms,"右前足全垫过界"),
           (far,"远侧上平面承重")]
    if "lifted_clearance_for_near_side_settle" in phases:
        notes += [(phases["lifted_clearance_for_near_side_settle"],"二次抬足，准备返回近侧"),
                  (phases["whole_toe_cleared_near_lip"],"全足垫悬空退过近缘"),
                  (phases["near_side_settle"],"近侧安全制动")]
    if end>6450:
        notes += [(6450,"台架续行意图"),(end,"撤令后的末态")]
    else:
        notes += [(end,"远侧稳态支撑窗口末端")]
    detail_end=phases.get("whole_toe_cleared_near_lip",far)+130
    notes=[(t,name) for t,name in notes if t<=end]
    indices=sorted(set(range(0,end+1,40)) | set(range(loss-65,min(end,detail_end),8)) |
                   {i for i,_ in notes})
    fonts=tuple(ImageFont.truetype(FONT,size) for size in (21,16))
    def page(t, override=None):
        image=Image.new("RGB",(WIDTH*2,HEIGHT),"#e6ecea")
        d=ImageDraw.Draw(image)
        close=loss-100<=t<=detail_end
        for i,trial in enumerate(cases):
            a=trial["arrays"]
            data=trial["data"]
            data.qpos[:]=a["qpos"][t]
            data.qvel[:]=a["qvel"][t]
            mj.mj_forward(trial["model"],data)
            assert np.array_equal(data.qpos,a["qpos"][t])
            com=a["com"][t]
            r=trial["renderer"]
            r.update_scene(data,camera=detail if close else camera((com[0]+.25,com[1],.80),False))
            if close:
                # Small visual cues centered on the actual compiled near/far faces.
                for face, fill in ((xmin,[1.,.62,.14,.85]),(xmax,[.14,.85,.96,.85])):
                    geom=r.scene.geoms[r.scene.ngeom]
                    mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_BOX,[.0024,(ymax-ymin)/2,.0015],
                                    [face,(ymax+ymin)/2,.0016],np.eye(3).ravel(),fill)
                    r.scene.ngeom+=1
            x=i*WIDTH
            image.paste(Image.fromarray(r.render()),(x,HEIGHT-IMAGE_HEIGHT))
            caption=override or ("初始站立" if t<500 else "走向浅缺口" if t<loss else
                     "观察踏空与复踏" if t<=far else "恢复窗口" if t<6450 else "后续行走与撤令")
            if i==0 and t>=loss:
                caption="配对时刻" if override else "固定原策略"
            f=a["loads"][t,RF]/W
            rf=a["rf_pad_xyz"][t]
            d.text((x+11,8),f'{trial["caption"]} · {caption}',font=fonts[0],fill="#1f3430")
            d.text((x+11,40),f'{t/1000:.3f} s · 右前足 x {rf[0]:.3f} / 远缘 {xmax:.3f} · 支撑 {f:.3f} W',
                   font=fonts[1],fill="#2b4a43")
        return image
    video=output/"浅缺口远侧复踏配对.mp4"
    montage=output/"浅缺口远侧复踏关键帧.png"
    with ExitStack() as stack:
        for t in cases:
            t["renderer"]=stack.enter_context(mj.Renderer(t["model"],height=IMAGE_HEIGHT,width=WIDTH))
        sink=stack.enter_context(imageio.get_writer(video,fps=FPS,codec="libx264",quality=8,macro_block_size=2,
                                                   ffmpeg_log_level="error"))
        for t in indices:
            sink.append_data(np.asarray(page(t)))
        sheet=Image.new("RGB",(WIDTH*2,HEIGHT*len(notes)))
        for i,(t,note) in enumerate(notes):
            sheet.paste(page(t,note),(0,i*HEIGHT))
        sheet.save(montage)
    args=["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
          "stream=nb_read_frames","-of","csv=p=0",str(video)]
    frames=int(subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip())
    assert frames==len(indices)
    for case in cases:
        assert stan.sha(case["path"]/"TRACE.npz")==case["info"]["trace_sha256"]
    stan.put(output/"VIDEO.json",dict(decoded_frames=frames,video_sha256=stan.sha(video),
        montage_sha256=stan.sha(montage),no_integrations=True,filename=video.name,montage=montage.name,
        physical_ms_frames=indices,notes=notes,paired_sources=[str(t["path"]) for t in cases],
        renderer_sha256=stan.sha(Path(__file__)),display_only_geometry="amber near edge, cyan far edge"))
    print(json.dumps(dict(decoded=frames,moments=notes),ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("control",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("summary",type=Path)
    a=p.parse_args()
    render(a.output.resolve(),(open_trial(a.control.resolve(),"同缺口关闭反射"),
        open_trial(a.candidate.resolve(),"右前足收腿后向远侧复踏")),read(a.summary))
