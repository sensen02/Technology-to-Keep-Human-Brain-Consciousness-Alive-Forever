"""Show recorded pilot trajectories and full physical body poses."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from course import BODY, WIDTH, COLOR, HERE, sha

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP=FontProperties(fname=FONT)


def episode(root):
    meta=json.loads((root/"RESULT.json").read_text())
    path=root/"EPISODE.npz"
    assert sha(path)==meta["files"]["EPISODE.npz"]
    with np.load(path,allow_pickle=False) as z:
        row={k:z[k] for k in ("com","qpos","goal","distance","contact_W","delivered","reward")}
    return meta,row


def plot(out):
    pairs=[("SEED/cal_left","左侧绕箱示范","#165f56"),("SEED/cal_right","右侧绕箱示范","#946623"),
           ("LIVE/online_left","SAC 左箱采样","#6b7891"),("LIVE/online_right","SAC 右箱采样","#874d7b"),
           ("LIVE/phase_b_held_box","阶段 B 留出箱","#a44436")]
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    ax=axs[0,0]
    xs,ys=[],[]
    for path,label,color in pairs:
        meta,row=episode(HERE/path)
        xy=row["com"][:,:2]
        xs.extend([xy[:,0].min(),xy[:,0].max()])
        ys.extend([xy[:,1].min(),xy[:,1].max()])
        ax.plot(*xy.T,label=label,color=color)
        ax.scatter(*xy[-1],facecolors="white",edgecolors=color,s=24)
        goal=row["goal"][:2]
        xs.append(float(goal[0]))
        ys.append(float(goal[1]))
        ax.scatter(*goal,marker="*",color="#a12428",s=35)
        for x,y,_,a,b,_ in meta["geometry"]["boxes"]:
            ax.add_patch(Rectangle((x-a,y-b),2*a,2*b,linewidth=.8,ec="#365d3e",fc="#85b783",alpha=.35))
    ax.set_xlim(min(xs)-.15,max(xs)+.15)
    ax.set_ylim(min(-.6,min(ys)-.15),max(.6,max(ys)+.15))
    ax.set_aspect("equal")
    ax.set_title("单箱课程实际质心轨迹",fontproperties=FP)
    ax.legend(prop=FP,frameon=False,loc="lower right",fontsize=9)

    ax=axs[0,1]
    for path,label,color in pairs:
        meta,row=episode(HERE/path)
        t=(1+np.arange(len(row["distance"])))*.02
        ax.plot(t,row["distance"],label=label,color=color)
        ids=np.flatnonzero(row["contact_W"]>1.e-6)
        if len(ids):
            ax.scatter(t[ids[0]],row["distance"][ids[0]],marker="x",color=color,s=45,zorder=3)
    ax.set_title("实测目标距离 · × 为首触箱",fontproperties=FP)
    ax.set_xlabel("仿真时间 / s",fontproperties=FP)

    ax=axs[1,0]
    for path,label,color in (("PHASE_A","阶段 A","#55476e"),("PHASE_B","阶段 B","#17635d")):
        logs=[json.loads(line) for line in (HERE/path/"UPDATES.jsonl").read_text().splitlines()]
        t=np.asarray([row["update"] for row in logs])
        values=np.asarray([row["critic"] for row in logs])
        smoothed=np.convolve(values,np.ones(16)/16.,mode="valid")
        ax.plot(t[15:],smoothed,label=label,color=color)
    ax.set_title("双 Q 损失 · 16 次更新滑动平均",fontproperties=FP)
    ax.set_xlabel("SAC 更新",fontproperties=FP)
    ax.legend(prop=FP,frameon=False)

    ax=axs[1,1]
    for path,label,color in (("DIAGNOSTICS/warm_flat","初始模仿","#8894a5"),
                            ("DIAGNOSTICS/first_sac_flat","初始 SAC","#7f5b7c"),
                            ("LIVE/phase_a_held_flat","阶段 A","#254965"),
                            ("LIVE/phase_b_held_flat","阶段 B","#a5373f")):
        _,row=episode(HERE/path)
        t=(1+np.arange(len(row["distance"])))*.02
        ax.plot(t,row["distance"],label=label,color=color)
    ax.axhspan(1.75,2.34,alpha=.11,color="#2b7850")
    ax.set_title("留出平地的接近与停止",fontproperties=FP)
    ax.set_xlabel("仿真时间 / s",fontproperties=FP)
    ax.legend(prop=FP,frameon=False)
    for ax in axs.flat:
        ax.grid(alpha=.14)
    fig.savefig(out/"SAC实测结果.png",dpi=145)
    plt.close(fig)


def poses(out):
    trials=(("SEED/cal_left","绕箱示范"),("LIVE/phase_b_held_box","SAC 留出单箱"))
    model=BODY.act_mod.load_full_model(BODY.PAD_XML)
    data=mj.MjData(model)
    meshes=model.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert meshes.sum()==71 and np.all(model.geom_rgba[meshes,3]==1.)
    panel=Image.new("RGB",(1260,670),"#edf1f3")
    draw=ImageDraw.Draw(panel)
    font=ImageFont.truetype(FONT,17)
    camera=mj.MjvCamera()
    camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=7.0
    camera.azimuth=136
    camera.elevation=-49
    with mj.Renderer(model,width=620,height=292) as render:
        for row,(path,label) in enumerate(trials):
            meta,track=episode(HERE/path)
            terminal=len(track["qpos"])-1
            for col,i in enumerate((max(0,int(terminal*.38)),terminal)):
                data.qpos[:]=track["qpos"][i]
                mj.mj_forward(model,data)
                np.testing.assert_array_equal(data.qpos,track["qpos"][i])
                camera.lookat[:]=[*data.subtree_com[model.body("Thorax").id,:2],.6]
                render.update_scene(data,camera=camera)
                scene=render.scene
                for x,y,z,a,b,c in meta["geometry"]["boxes"]:
                    mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_BOX,np.array([a,b,c]),
                                    np.array([x,y,z]),np.eye(3).ravel(),np.array([.03,.9,.08,1.]))
                    scene.ngeom+=1
                mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,np.array([WIDTH,1.09,0.]),
                                track["goal"],np.eye(3).ravel(),np.asarray(COLOR))
                scene.ngeom+=1
                x,y=col*630,row*335
                draw.text((x+12,y+9),f"{label}   {i*.02:.2f} s",font=font,fill="#203047")
                panel.paste(Image.fromarray(render.render()),(x+5,y+35))
    panel.save(out/"SAC物理关键帧.png")


if __name__=="__main__":
    out=HERE/"FIGURES"
    out.mkdir()
    plot(out)
    poses(out)
