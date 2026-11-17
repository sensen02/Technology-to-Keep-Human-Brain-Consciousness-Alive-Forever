"""Plot the proof windows and render the saved physical checkpoints."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run import BODY, WIDTH, COLOR
from unified import HERE, sha

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP=FontProperties(fname=FONT)


def raw(case,result):
    path=HERE/"RUNS"/case/"TRACE.npz"
    assert sha(path)==result["hashes"]["TRACE.npz"]
    return np.load(path,allow_pickle=False)


def plot(cases,out):
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    with raw("straight",cases["straight"]) as z:
        t=np.arange(len(z["time"]))/1000.
        axs[0,0].plot(t,z["com"][:,0]-z["com"][0,0],color="#1f6c77")
        for a,b in ((2,10),(12,20),(22,30)):
            axs[0,0].axvspan(a,b,alpha=.12,color="#427f8f")
    axs[0,0].set_title("持续走与撤令后的停止",fontproperties=FP)
    axs[0,0].set_ylabel("前向位移",fontproperties=FP)
    for case,label,c in (("left","左转","#336379"),("right","右转","#aa5631")):
        with raw(case,cases[case]) as z:
            t=np.arange(len(z["time"]))/1000.
            yaw=np.degrees(np.unwrap(z["yaw"]))
            axs[0,1].plot(t,yaw-yaw[0],label=label,color=c)
    axs[0,1].set_title("左右转向及撤销后的实测偏航",fontproperties=FP)
    axs[0,1].set_ylabel("角度 / °",fontproperties=FP)
    axs[0,1].legend(prop=FP,frameon=False)
    with raw("pace",cases["pace"]) as z:
        d=z["decisions"]
        c=z["com"]
        velocity=(c[500:,0]-c[:-500,0])/.5
        axs[1,0].plot((np.arange(len(velocity))+250)/1000,velocity,label="实测前速 / 500 ms",color="#205360")
        axs[1,0].step(d[:,0]*.001,d[:,3],where="post",label="颈后速度请求",color="#a95837",alpha=.7)
    axs[1,0].set_title("同一脑的调速及停车",fontproperties=FP)
    axs[1,0].set_ylabel("原生长度 / s",fontproperties=FP)
    axs[1,0].legend(prop=FP,frameon=False)
    for case,label,c in (("an_intact","完整上行","#2a7875"),("an_cut","切断","#a95837"),
                         ("an_restore","恢复","#675692")):
        with raw(case,cases[case]) as z:
            d=z["decisions"]
            axs[1,1].step(d[:,0]*.001,d[:,3],label=label,color=c,where="post")
    axs[1,1].axvline(cases["an_intact"]["contact_ms"][0]*.001,color="#263238",alpha=.75,linestyle=":")
    axs[1,1].set_title("同一侧碰的颈后撤令对照",fontproperties=FP)
    axs[1,1].set_ylabel("颈后速度请求",fontproperties=FP)
    axs[1,1].legend(prop=FP,frameon=False)
    axs[1,1].set_xlim(2.24,2.37)
    for ax in axs.flat:
        ax.set_xlabel("物理时间 / s",fontproperties=FP)
        ax.grid(alpha=.20)
    fig.savefig(out/"联合脑物理对照.png",dpi=155)
    plt.close(fig)


def render_poses(cases,out):
    canvas=Image.new("RGB",(1260,680),"#eef1f3")
    text=ImageDraw.Draw(canvas)
    face=ImageFont.truetype(FONT,17)
    model=BODY.act_mod.load_full_model(BODY.PAD_XML)
    data=mj.MjData(model)
    visible=model.geom_type==mj.mjtGeom.mjGEOM_MESH
    assert int(visible.sum())==71 and np.all(model.geom_rgba[visible,3]==1.)
    with mj.Renderer(model,width=620,height=300) as renderer:
        for row,case in enumerate(("left","right")):
            record=cases[case]
            with raw(case,record) as z:
                d=z["decisions"]
                turning=d[d[:,4]!=0,0]
                walk=d[d[:,3]>0,0]
                turn_stamp=int(turning[-1]) if len(turning) else record["scene"]["on"]
                walk_stamp=int(walk[len(walk)//2]) if len(walk) else record["scene"]["end"]
                for column,(label,stamp) in enumerate((("转向撤令前",turn_stamp),("接回推进",walk_stamp))):
                    i=min(stamp//20,len(z["poses"])-1)
                    t=i*20
                    qpos=z["poses"][i]
                    data.qpos[:]=qpos
                    mj.mj_forward(model,data)
                    np.testing.assert_array_equal(data.qpos,qpos)
                    cam=mj.MjvCamera()
                    cam.type=mj.mjtCamera.mjCAMERA_FREE
                    cam.azimuth,cam.elevation,cam.distance=136,-46,6.5
                    cam.lookat[:]=[*z["com"][t,:2],.5]
                    renderer.update_scene(data,camera=cam)
                    if t<record["scene"]["off"]:
                        scene=renderer.scene
                        mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,np.array([WIDTH,1.09,0.]),
                                        z["goal"],np.eye(3).ravel(),np.asarray(COLOR,dtype=np.float32))
                        scene.ngeom+=1
                    x,y=column*630,row*340
                    canvas.paste(Image.fromarray(renderer.render()),(x+5,y+35))
                    motion={"left":"左转","right":"右转"}[case]
                    text.text((x+12,y+7),f"{motion} · {label} · {t/1000:.2f} s",font=face,fill="#243445")
    canvas.save(out/"左右转向接回推进关键帧.png")


def main():
    evidence=json.loads((HERE/"ACCEPTANCE.json").read_text())
    out=HERE/"FIGURES"
    out.mkdir()
    plot(evidence["cases"],out)
    render_poses(evidence["cases"],out)


if __name__=="__main__":
    main()
