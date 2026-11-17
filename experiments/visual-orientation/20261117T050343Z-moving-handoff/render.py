"""Render measured suffix traces and actual frozen body frames."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from replay import HERE, course, read
from snapshot import read_checkpoint


COLORS = {
    "teacher":("教师指令复放","#60737f"),
    "neutral":("中立对照","#9b764e"),
    "warm":("模仿模型","#317866"),
    "original":("旧奖励 SAC","#a94f61"),
    "route":("质心奖励 SAC","#385f9d"),
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FAMILY = FontProperties(fname=FONT)


def evidence():
    report = read(HERE/"RESULTS.json")
    assert read(HERE/"INDEPENDENT.json")["pass_controls"]
    out = HERE/"FIGURES"
    out.mkdir()
    fig, axs = plt.subplots(2,2,figsize=(13,7.8),layout="constrained")
    physical = course.BODY.act_mod.load_full_model(course.BODY.PAD_XML)
    data = mj.MjData(physical)
    original = read_checkpoint(HERE/"MOVING_CHECKPOINT.pkl",report["checkpoint"]["snapshot_sha256"])
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 139.
    camera.elevation = -43.
    camera.distance = 7.3
    canvas = Image.new("RGB",(1446,678),"#f4f5f6")
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT,17)
    with mj.Renderer(physical,width=476,height=295) as renderer:
        for i,r in enumerate(report["tests"]):
            name = r["arm"]
            label,color = COLORS[name]
            path = HERE/name/"SUFFIX.npz"
            assert course.sha(path)==r["episode_sha256"]
            with np.load(path,allow_pickle=False) as f:
                z = {k:f[k] for k in ("qpos","com","distance","tail_margin","delivered","goal")}
            pose,com,distance,tail,delivered,goal=(z[k] for k in ("qpos","com","distance","tail_margin","delivered","goal"))
            np.testing.assert_array_equal(pose[0],original["native_data"].qpos)
            t = .020*np.arange(len(com))
            radius = np.linalg.norm(com[:,:2]-goal[:2],axis=1)
            axs[0,0].plot(t,np.r_[r["initial_range"],distance],color=color,label=label)
            axs[0,1].plot(t,radius[0]-radius,color=color)
            # Take the last of four delivered slots to show autonomy after the
            # initial inherited 5 ms action has been replaced.
            axs[1,0].step(t[1:],delivered[:,-1,0],color=color,where="post")
            axs[1,1].plot(*com[:,:2].T,color=color)
            axs[1,1].scatter(*com[-1,:2],color=color,s=19)
            boxes = r["geometry"]["boxes"]
            if i == 0:
                axs[1,1].scatter(*com[0,:2],marker="o",c="#203437",s=27,zorder=3)
                axs[1,1].scatter(*goal[:2],marker="*",c="#b2323e",s=71,zorder=3)
                for x,y,_,a,b,_ in boxes:
                    axs[1,1].add_patch(Rectangle((x-a,y-b),2*a,2*b,color="#568769",clip_on=False))
            frames = [(i+1,-1,label)]
            if i==0:
                frames.insert(0,(0,0,"共同接管点"))
            for panel,at,title in frames:
                data.qpos[:] = pose[at]
                mj.mj_forward(physical,data)
                np.testing.assert_array_equal(data.qpos,pose[at])
                camera.lookat[:] = [com[at,0]+.25,com[at,1],.8]
                renderer.update_scene(data,camera=camera)
                scene = renderer.scene
                for x,y,z,a,b,c in boxes:
                    mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_BOX,np.array([a,b,c]),
                                    np.array([x,y,z]),np.eye(3).ravel(),np.array([.03,.9,.08,1.]))
                    scene.ngeom += 1
                mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,
                                np.array([course.WIDTH,1.09,0.]),goal,np.eye(3).ravel(),np.asarray(course.COLOR))
                scene.ngeom += 1
                x,y=(panel%3)*482,(panel//3)*339
                stamp=0. if at==0 else float(r["duration_s"])
                pen.text((x+10,y+10),f"{title} · 接管后 {stamp:.2f} 秒",font=font,fill="#24383a")
                canvas.paste(Image.fromarray(renderer.render()),(x+3,y+36))
    axs[0,0].axhspan(1.18,1.72,color="#3c8c77",alpha=.13)
    axs[0,0].legend(prop=FAMILY,frameon=False,ncol=2)
    axs[0,0].set_title("头部到目标的距离",fontproperties=FAMILY)
    axs[0,1].set_title("质心净接近目标",fontproperties=FAMILY)
    axs[1,0].set_title("到达身体的前进指令",fontproperties=FAMILY)
    axs[1,1].set_title("接管后的质心轨迹",fontproperties=FAMILY)
    axs[1,1].set_aspect("equal",adjustable="datalim")
    axs[1,1].set_xlabel("世界 X",fontproperties=FAMILY)
    axs[1,1].set_ylabel("世界 Y",fontproperties=FAMILY)
    for ax in (axs[0,0],axs[0,1],axs[1,0]):
        ax.set_xlabel("接管后仿真时间 / 秒",fontproperties=FAMILY)
    for ax in axs.flat:
        ax.grid(alpha=.15)
    fig.savefig(out/"接管后的运动与评分门.png",dpi=135)
    plt.close(fig)
    canvas.save(out/"同一快照的五组实际姿态.png")


if __name__=="__main__":
    evidence()
