"""Draw the measured trajectories and replay key physical poses without stepping."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from evaluate import mj, np, prepare, SCENES, WIDTH, COLOR, sha

HERE=Path(__file__).resolve().parent
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FIG_FONT=FontProperties(fname=FONT)
SCENE_LABELS={
    "near_left":"平地左侧目标", "near_right":"平地右侧目标", "flat_paired":"同向平地对照",
    "one_offset_box":"偏置单箱", "staggered_boxes":"错列双箱", "narrow_trench":"浅窄横坑",
    "wide_trench":"较宽横坑",
}
AGENT_LABELS={"approach":"冻结视觉趋近","detour":"冻结绕障","joint":"冻结联合脑","neck_probe":"颈后定速对照"}
PALETTE={"approach":"#00776b","detour":"#765093","joint":"#236298","neck_probe":"#b66635"}


def read_rows(tag,r):
    folder=HERE/"RUNS"/tag
    assert all(sha(folder/key)==digest for key,digest in r["artifacts"].items())
    return np.load(folder/"TRACE.npz",allow_pickle=False)


def plot(data,output):
    panels=list(SCENE_LABELS)
    fig,axes=plt.subplots(4,2,figsize=(12,14),layout="constrained")
    for ax,scene in zip(axes.flatten(),panels):
        cases=[(k,v) for k,v in data.items() if v["scene"]==scene]
        geometry=cases[0][1]["geometry"]
        for x,y,_,hx,hy,_ in geometry["boxes"]:
            ax.add_patch(Rectangle((x-hx,y-hy),hx*2,hy*2,fc="#87ba78",ec="#376a34",zorder=-1))
        if geometry["pit"]:
            pit=geometry["pit"]
            ax.axvspan(pit["near"],pit["far"],fc="#ccbecd",zorder=-2)
        x_limits = []
        y_limits = []
        for n,(tag,result) in enumerate(cases):
            policy=result["policy"]
            with read_rows(tag,result) as rows:
                com=rows["com"][::10]
                target=rows["goal"][:2]
            x_limits.extend([com[:,0].min(),com[:,0].max(),target[0]])
            y_limits.extend([com[:,1].min(),com[:,1].max(),target[1]])
            if n==0:
                ax.scatter(*target,marker="*",color="#b41625",s=125,zorder=5)
                ax.scatter(*com[0,:2],color="#22313c",s=22,zorder=5)
            line=PALETTE[policy]
            ax.plot(*com[:,:2].T,color=line,label=AGENT_LABELS[policy])
            ax.scatter(*com[-1,:2],facecolor="white",edgecolor=line,s=40,zorder=4)
            for i in np.linspace(0,len(com)-1,7,dtype=int)[1:-1]:
                if np.linalg.norm(com[min(i+1,len(com)-1),:2]-com[i,:2])>.0002:
                    ax.annotate("",xy=com[min(i+10,len(com)-1),:2],xytext=com[i,:2],
                                arrowprops=dict(arrowstyle="-|>",color=line,linewidth=.9))
        ax.set_ylim(min(-1.25,min(y_limits)-.3),max(1.25,max(y_limits)+.3))
        ax.set_xlim(min(x_limits)-.35,max(x_limits)+.35)
        ax.set_aspect("equal")
        ax.set_title(SCENE_LABELS[scene],fontproperties=FIG_FONT)
        ax.legend(loc="lower right" if scene=="near_left" else "upper right",prop=FIG_FONT,frameon=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=.15)
    ax=axes.flatten()[-1]
    for name in ("flat_paired__joint","flat_paired__neck_probe"):
        r=data[name]
        with read_rows(name,r) as rows:
            x=rows["com"][::20,0]
        ax.plot(np.arange(len(x))*.02,x-x[0],color=PALETTE[r["policy"]],label=AGENT_LABELS[r["policy"]])
    ax.axvspan(20.,30.,alpha=.15,color="#838e95")
    ax.set_title("同起点平地连续推进",fontproperties=FIG_FONT)
    ax.set_xlabel("物理时间 / s",fontproperties=FIG_FONT)
    ax.set_ylabel("质心 x 方向累计位移",fontproperties=FIG_FONT)
    ax.legend(loc="upper left",prop=FIG_FONT,frameon=False)
    ax.grid(alpha=.15)
    fig.suptitle("实测身体质心轨迹  |  黑点为初始站姿 · 空心点为终点 · 红星为目标 · 绿色为实体箱 · 灰紫条为横坑",
                 fontproperties=FIG_FONT,fontsize=12)
    fig.savefig(output/"导航场景实测轨迹.png",dpi=150)
    plt.close(fig)


def keyframes(data,output):
    cases=("near_left__approach","near_left__joint","one_offset_box__detour","staggered_boxes__detour",
           "narrow_trench__neck_probe","wide_trench__neck_probe")
    scene_table=json.loads(SCENES.read_text())["scenes"]
    sheet=Image.new("RGB",(1400,len(cases)*268),"#f0f3f5")
    font=ImageFont.truetype(FONT,17)
    for k,tag in enumerate(cases):
        result=data[tag]
        scene=result["scene"]
        # prepare restores the same terrain and standing snapshot. The following
        # loop only renders archived coordinates.
        body,geometry=prepare(output/f"render_body_{k}",scene_table[scene])
        assert geometry==result["geometry"]
        with read_rows(tag,result) as rows, mj.Renderer(body.plant,width=640,height=230) as renderer:
            stamps=rows["stamps"]
            hit=[i for i in result["first_contact_step"] if i>=0]
            first=min(hit)//80 if hit else int(stamps[-1]*.62)
            pit=geometry["pit"]
            if pit and not hit:
                first=int(stamps[np.argmax(rows["feet"][:,1]>=pit["near"])])
            indices=(int(np.abs(stamps-first).argmin()),len(stamps)-1)
            for j,index in enumerate(indices):
                t=int(stamps[index])
                body.d.qpos[:]=rows["qpos"][index]
                mj.mj_forward(body.plant,body.d)
                np.testing.assert_array_equal(body.d.qpos,rows["qpos"][index])
                cam=mj.MjvCamera()
                cam.type=mj.mjtCamera.mjCAMERA_FREE
                cam.azimuth=110
                cam.elevation=-56
                cam.distance=6.4
                cam.lookat[:]=[*rows["com"][t,:2],.5]
                renderer.update_scene(body.d,camera=cam)
                sc=renderer.scene
                if t>=500:
                    mj.mjv_initGeom(sc.geoms[sc.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,np.array([WIDTH,1.09,0.]),
                                    rows["goal"],np.eye(3).ravel(),np.asarray(COLOR,dtype=np.float32))
                    sc.ngeom+=1
                x=j*700
                y=k*268
                sheet.paste(Image.fromarray(renderer.render()),(x+30,y+34))
                caption=f"{SCENE_LABELS[scene]}  |  {AGENT_LABELS[result['policy']]}  |  {t/1000:.2f} s"
                ImageDraw.Draw(sheet).text((x+13,y+7),caption,fill="#1f3546",font=font)
    sheet.save(output/"物理回放关键帧.png")


if __name__=="__main__":
    output=HERE/"FIGURES"
    output.mkdir()
    data=json.loads((HERE/"BENCHMARK.json").read_text())["cases"]
    plot(data,output)
    keyframes(data,output)
