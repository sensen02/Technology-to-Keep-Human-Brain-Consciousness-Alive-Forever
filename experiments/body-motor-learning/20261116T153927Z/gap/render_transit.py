"""Video and keyframes of the audited long run, using only retained states."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from crossing import read, stan
from geometry import make_gap_model
from hybrid_floor import applied
from render_crossing import camera, FONT


WIDTH, PHOTO, PAGE = 920, 310, 378


def main(summary, output, video):
    audit=read(summary)
    assert audit["whole_body_path_support_crossed_x_station"]
    run=Path(audit["extended"])
    info=read(run/"RESULT.json")
    trace=run/"TRACE.npz"
    assert stan.sha(trace)==info["trace_sha256"]
    x0,far=audit["gap_opening_xyz"]["x"]
    y0,y1=audit["gap_opening_xyz"]["y"]
    ft=audit["feet"]
    named=max(ft,key=lambda name:ft[name]["first_far_support_ms"])
    final=info["actual_ms"]
    com_ms=int(round(audit["first_COM_beyond_far_x_ms"]))
    last_foot_ms=int(round(ft[named]["first_far_support_ms"]))
    moments=(("质心通过缺口远缘对应的 x 线",com_ms), (f"最后越过且承重的足垫 {named}",last_foot_ms),
             ("撤掉推进后站稳",final))
    with applied():
        model=make_gap_model(info["scenario"])
    model.vis.global_.offwidth=max(WIDTH,model.vis.global_.offwidth)
    data=mj.MjData(model)
    with np.load(trace,allow_pickle=False) as f:
        values={k:f[k] for k in ("qpos","qvel","com","all_pad_xyz")}
    sheet=Image.new("RGB",(WIDTH,PAGE*len(moments)),"#e6ecea")
    title=ImageFont.truetype(FONT,23)
    label=ImageFont.truetype(FONT,16)
    fixed=camera((1.08,-.03,.63),False)
    fixed.azimuth=243
    fixed.elevation=-38
    close=camera(((x0+far)/2,(y0+y1)/2+.06,.1),True)
    first_renewal=audit["renewed_at_ms"]
    indices=[]
    decoded=None

    def draw(ren, ms, desc, use_detail=False):
        data.qpos[:]=values["qpos"][ms]
        data.qvel[:]=values["qvel"][ms]
        mj.mj_forward(model,data)
        assert np.array_equal(data.qpos,values["qpos"][ms])
        cx,cy,_=values["com"][ms]
        pan=camera((cx+.15,cy,.7),False)
        ren.update_scene(data,camera=close if use_detail else fixed if ms>=com_ms else pan)
        geom=ren.scene.geoms[ren.scene.ngeom]
        if use_detail:
            size=[.0035,(y1-y0)/2,.0025]
            origin=[far,(y0+y1)/2,.002]
        else:
            size=[.0035,1.75,.0025]
            origin=[far,0.,.002]
        mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_BOX,size,origin,np.eye(3).ravel(),[.13,.84,.93,1.])
        ren.scene.ngeom+=1
        img=Image.new("RGB",(WIDTH,PAGE),"#e6ecea")
        pen=ImageDraw.Draw(img)
        img.paste(Image.fromarray(ren.render()),(0,PAGE-PHOTO))
        pen.text((11,8),f"{desc} · {ms/1000:.3f} s",font=title,fill="#1f3430")
        x=values["all_pad_xyz"][ms,:,0].min()
        text=f"质心 x {cx:.3f} · 最后方足垫中心 x {x:.3f} · 青色为参考线 x={far:.3f}，实际缺口限右前足脚道"
        pen.text((11,43),text,font=label,fill="#2b4a43")
        return img

    with mj.Renderer(model,width=WIDTH,height=PHOTO) as ren:
        for i,(desc,ms) in enumerate(moments):
            sheet.paste(draw(ren,ms,desc),(0,PAGE*i))
        sheet.save(output)
        if video:
            indices=sorted(set(range(0,final+1,65))|set(range(2690,4320,12))|
                           {first_renewal,final,*[ms for _,ms in moments]})
            with imageio.get_writer(video,fps=30,codec="libx264",quality=8,macro_block_size=2,
                                    ffmpeg_log_level="error") as sink:
                for ms in indices:
                    desc=("原生接触下探与远侧复踏" if 2690<=ms<=4320 else
                          "确立稳定站姿" if ms<first_renewal else
                          "继续行走" if ms<last_foot_ms else "六足通过参考线后制动")
                    sink.append_data(np.asarray(draw(ren,ms,desc,use_detail=2690<=ms<=4320)))
    if video:
        args=["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
              "stream=nb_read_frames","-of","csv=p=0",str(video)]
        decoded=int(subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip())
        assert decoded==len(indices)
    stan.put(output.with_suffix(".json"),dict(source=str(run),trace_sha256=info["trace_sha256"],
              summary_sha256=stan.sha(summary),image_sha256=stan.sha(output),render_only=True,
              measured_keyframes=[ms for _,ms in moments],video_sha256=stan.sha(video) if video else None,
              decoded_frames=decoded,physical_video_frame_ms=indices,renderer_sha256=stan.sha(Path(__file__))))
    print(dict(image=str(output),video=str(video),decoded=decoded))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--video",type=Path)
    a=p.parse_args()
    main(a.summary,a.output,a.video)
