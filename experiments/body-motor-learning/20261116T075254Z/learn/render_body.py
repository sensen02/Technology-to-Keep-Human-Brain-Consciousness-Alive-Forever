"""Render frozen-body recorded poses and measured engineering state, without physics steps."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from physical import bench,sha,write

WIDTH,HEIGHT,VISUAL,FPS=620,426,348,30
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
NAMES={"stand":"静立","walking":"直走","low":"低速直走","accel":"加速","high":"加速后直行",
       "decel":"减速","slowed":"减速后直行","aligned_walk":"同速续走","turn_in":"转入弧线","arc":"走中转弯",
       "turn_out":"出弯","exit":"沿出弯朝向直走","brake":"制动","stop":"末端停稳"}


def spans(result,commands,final):
    if "course" in result:
        out=[]
        t=0
        for seg in result["course"]["segments"]:
            out.append((t,min(t+seg["duration_ms"],final),NAMES.get(seg["name"],seg["name"])))
            t+=seg["duration_ms"]
        return [row for row in out if row[0]<=final]
    phase=result.get("phase")
    if phase:
        warm=phase["stand_start_ms"]
        a,b,taper,coast=(phase[key] for key in ("arc_begin_ms","arc_end_ms","taper_complete_ms","stop_start_ms"))
        prefix=[(0,warm,"静立"),(warm,a,"低速直行与调速" if phase.get("speed_start_ms") is not None else "直行进入")]
        return [row for row in [*prefix,(a,b,"走中转弯"),(b,taper,"出弯"),(taper,coast,"沿出弯朝向直走"),
                (coast,final,"末端停稳")] if row[0]<=final]
    rise=np.flatnonzero(commands[:,0]>0.)
    begin=int(5*rise[0]) if len(rise) else final
    after=min(int(5*(rise[-1]+1)),final) if len(rise) else final
    peak=int(5*np.argmax(commands[:,0]))
    cuts=[(0,begin,"静立"),(begin,peak,"加速"),(peak,after,"减速"),(after,final,"末端停稳")]
    return [row for row in cuts if row[1]>row[0]]


def camera(x,y,z):
    view=mj.MjvCamera()
    view.type=mj.mjtCamera.mjCAMERA_FREE
    view.azimuth=160
    view.elevation=-21
    view.distance=5.3
    view.lookat[:]=[x,y,z]
    return view


def run(output,physical,caption,mode,audit=None):
    result=bench.read(physical/"RESULT.json")
    trace=physical/"TRACE.npz"
    decisions=physical/"DECISIONS.npz"
    assert result["trace_sha256"]==sha(trace)
    assert result.get("decision_sha256",result.get("decisions_sha256"))==sha(decisions)
    with np.load(trace,allow_pickle=False) as z:
        q,v,xy=(np.asarray(z[k]) for k in ("qpos","qvel","com"))
        if "yaw" in z.files:
            yaw=np.asarray(z["yaw"])
        else:
            rot=np.empty(9)
            angles=[]
            for quat in q[:,3:7]:
                mj.mju_quat2Mat(rot,quat)
                angles.append(np.arctan2(rot[3],rot[0]))
            yaw=np.asarray(angles)
    with np.load(decisions,allow_pickle=False) as z:
        commands=z["command"]
    out=output.resolve()
    out.mkdir()
    final=len(q)-1
    program=spans(result,commands,final)
    view=camera(*q[:,:2].mean(0),.82)
    model=bench.stand.physics.Core(adhesion=False).plant
    d=mj.MjData(model)
    def forbidden(*a,**kw):
        raise RuntimeError("Recorded view may not integrate any dynamics")
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    right=np.c_[np.cos(yaw),np.sin(yaw)]
    local=np.r_[np.sum(np.diff(xy[:,:2],axis=0)*right[:-1],axis=1),0.]
    integrated=np.r_[0.,np.cumsum(local)]
    span=np.max(np.ptp(xy[:,:2],axis=0))
    span=max(span+.15,.30)
    center=xy[:,:2].mean(0)
    fonts=[ImageFont.truetype(FONT,n) for n in (22,16,13)]

    def path_pixel(point):
        d1=point-center
        return 148+int(d1[0]/span*122),133-int(d1[1]/span*122)

    passed=result.get("native_joint_pass",result.get("passed",False))
    if audit is not None:
        audit_result=bench.read(audit)
        if audit_result["native_sha256"]!=sha(physical/"NATIVE.npz"):
            raise ValueError("Supplemental verification belongs to a different run")
        passed=passed and audit_result["passed"]
    default="联合验收通过" if passed else "本次验收未全过"
    stage_names=[stage for _,_,stage in program]
    goal_label="物理教师" if mode=="teacher" else "候选策略" if mode=="student" else "冻结对照"
    moments=sorted({min(max((a+b)//2,0),final) for a,b,_ in program}|{0,final})
    if len(moments)>9:
        select=np.linspace(0,len(moments)-1,9).round().astype(int)
        moments=[moments[i] for i in select]
    indices=sorted(set(range(0,final+1,35))|set(moments))

    def frame(t, renderer):
        d.qpos[:]=q[t]
        d.qvel[:]=v[t]
        mj.mj_forward(model,d)
        np.testing.assert_array_equal(d.qpos,q[t])
        renderer.update_scene(d,camera=view)
        img=Image.new("RGB",(WIDTH,HEIGHT),"#e9eeef")
        img.paste(Image.fromarray(renderer.render()),(0,HEIGHT-VISUAL))
        draw=ImageDraw.Draw(img)
        stage=next((s for a,b,s in program if a<=t<b),stage_names[-1] if stage_names else "末段")
        cmd=commands[min(t//5,len(commands)-1)]
        half=125
        a,b=max(0,t-half),min(final,t+half)
        estimate=(integrated[b]-integrated[a])/((b-a)*.001) if b>a else 0.
        draw.text((11,7),f"{caption}  {stage}  {t/1000:.3f}s",font=fonts[0],fill="#253339")
        draw.text((12,44),f"{goal_label}  {default}　推进指令 {cmd[0]:.3f}　实测切向速度 {estimate:+.3f}　"
                         f"偏航指令 {np.rad2deg(cmd[1]):+.1f}°/s",font=fonts[1],fill="#365257")
        draw.rounded_rectangle((71,86,227,180),radius=7,fill="#e4edef",outline="#7ca6a7")
        points=[path_pixel(p) for p in xy[::max(5,final//290),:2]]
        if len(points)>1:
            draw.line(points,fill="#7e9da0",width=2)
        p0=path_pixel(xy[t,:2])
        p1=(p0[0]+int(15*right[t,0]),p0[1]-int(15*right[t,1]))
        draw.line((p0,p1),fill="#dc5e40",width=3)
        draw.ellipse((p0[0]-3,p0[1]-3,p0[0]+3,p0[1]+3),fill="#dc5e40")
        draw.text((77,165),"实测质心轨迹与朝向",font=fonts[2],fill="#3b5962")
        return img

    video=out/"调速转向物理回放.mp4"
    montage=out/"关键帧.png"
    sheet=Image.new("RGB",(WIDTH*2,HEIGHT*((len(moments)+1)//2)),"#cfdbdc")
    with mj.Renderer(model,height=VISUAL,width=WIDTH) as renderer:
        with imageio.get_writer(video,fps=FPS,codec="libx264",quality=8,
                                macro_block_size=2,ffmpeg_log_level="error") as sink:
            for t in indices:
                img=frame(t,renderer)
                sink.append_data(np.asarray(img))
                if t in moments:
                    order=moments.index(t)
                    sheet.paste(img,(WIDTH*(order%2),HEIGHT*(order//2)))
    sheet.save(montage)
    decode=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
        "stream=nb_read_frames","-of","csv=p=0",str(video)],check=True,capture_output=True,text=True)
    decoded=int(decode.stdout.strip())
    assert decoded==len(indices) and sha(trace)==result["trace_sha256"]
    write(out/"VIDEO.json",dict(source=str(physical),trace_sha256=sha(trace),video_sha256=sha(video),
              montage_sha256=sha(montage),actual_integrations=0,decoded_frames=decoded,
              playback_fps=FPS,source_ms=indices,caption=caption,mode=mode,
              supplemental_audit_sha256=sha(audit) if audit is not None else None))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("physical",type=Path)
    p.add_argument("caption")
    p.add_argument("--mode",choices=("teacher","student","control"),default="student")
    p.add_argument("--audit",type=Path)
    a=p.parse_args()
    run(a.out,a.physical.resolve(),a.caption,a.mode,a.audit)
