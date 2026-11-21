"""Create scientific views of the stored mesh solution and matched trajectories."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ("#263746","#8e5ba5","#2c8b77","#cf6947")
LABELS = ("Unloaded control", "Retained geometry", "Released geometry", "Released cable proxy")
MODES = ("unloaded_control","retained_geometry","unloaded_geometry","unloaded_cable_proxy")


def phases(ax):
    ax.axvspan(.45,.65,color="#d48844",alpha=.12)
    ax.axvspan(.65,1.05,color="#4b7fba",alpha=.09)
    ax.axvspan(1.05,1.20,color="#4ba589",alpha=.12)
    ax.grid(alpha=.18)
    ax.set_xlabel("Time (s)")


def physical():
    with np.load(HERE/"MECHANICS.npz") as solve, np.load(HERE/"TRANSPORT_PROFILES.npz") as outputs:
        fig, grid = plt.subplots(2,2,figsize=(11,7.5))
        mesh,cells = solve["mesh_xy"],solve["mesh_cells"]
        snapshots=solve["snapshot_displacement"]
        ax=grid[0,0]
        ax.triplot(*mesh,cells.T,color="#8b96a4",lw=.22,alpha=.45)
        for index,col in zip(range(3),("#d1774d","#377fa2","#488672"),strict=True):
            points=solve[f"axon{index}"]
            ax.plot(points[:,1],points[:,0],color=col,lw=1.5,label=("AN left at rest","AN right at rest","DN right at rest")[index])
        order=np.flatnonzero(np.isclose(mesh[1],0.))
        for step,label,col in ((1,"Wall after advance","#782444"),(3,"Wall at exit","#2965a1")):
            deformed=mesh+snapshots[step]
            ax.plot(deformed[0,order],deformed[1,order],color=col,lw=1.5,label=label)
        left = solve["axon0"]
        tri = Triangulation(mesh[0],mesh[1],cells.T)
        shift = np.column_stack([LinearTriInterpolator(tri, component)(left[:,1],left[:,0])
                                 for component in snapshots[1]])
        track = left[:,[1,0]] + shift
        ax.plot(track[:,0],track[:,1],color="#7e3524",lw=1.2,ls="--",label="AN left after advance")
        ax.set_title("Engineering mesh and axon traces in projection")
        ax.set(xlabel="BANC head–thorax axis (µm)",ylabel="Transverse depth (µm)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7,ncol=2,loc="upper right")
        ax=grid[0,1]
        for key,color,font in (("release",COLORS[2],"Release"),
                               ("retained",COLORS[1],"Retained"),
                               ("refined","#365687","Refined release")):
            data=solve[key]
            ax.plot(data[:,0],data[:,3],label=font,color=color,lw=1.6 if key!="refined" else .9)
        phases(ax)
        ax.set_title("Tissue proxy displacement")
        ax.set_ylabel("Max displacement (µm)")
        ax.legend(frameon=False,fontsize=9)
        ax=grid[1,0]
        for key,color in (("release",COLORS[2]),("retained",COLORS[1]),("stick_release","#986d3d")):
            r=solve[key]
            ax.plot(r[:,0],r[:,2],color=color,label=key.replace("_"," "),lw=1.5)
        phases(ax)
        ax.set_title("Normal contact reaction")
        ax.set_ylabel("Line reaction (kPa·µm)")
        ax.legend(frameon=False,fontsize=9)
        ax=grid[1,1]
        for lane in (0,2,3):
            ax.plot(outputs["time"],outputs["continuous_delay_ms"][:,lane,1],label=LABELS[lane],
                    color=COLORS[lane],lw=1.45)
        ax.axhline(1.30,ls="--",color="#566273",lw=.9,label="0.1 ms grid boundary")
        phases(ax)
        ax.set_title("Right AN arrival-time estimates")
        ax.set_ylabel("Continuous delay (ms)")
        ax.legend(frameon=False,fontsize=8)
        fig.suptitle("Virtual probe contact with a pre-opened access boundary",fontweight="bold")
        fig.tight_layout(rect=(0,.01,1,.95))
        fig.savefig(HERE/"MECHANICS_AND_DELAY.png",dpi=155)
        plt.close(fig)


def behavior():
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.5),sharex=True)
    for row,seed in enumerate((10901,11311,11701)):
        paths=HERE/"FORMAL"/f"seed{seed}"
        xy=[]
        yaw=[]
        for mode in MODES:
            with np.load(paths/mode/"TRACE.npz") as data:
                xy.append(data["position"][:,:2])
                angle=np.unwrap(data["yaw"])
                yaw.append(np.rad2deg(angle-angle[0]))
        time=np.arange(len(yaw[0]))*.0001
        for index in (1,2,3):
            style={"linestyle":"--","linewidth":.85} if index==1 else {"linewidth":1.4}
            axes[row,0].plot(time,yaw[index]-yaw[0],color=COLORS[index],label=LABELS[index],**style)
            axes[row,1].plot(time,np.linalg.norm(xy[index]-xy[0],axis=1),color=COLORS[index],**style)
        axes[row,0].text(.01,.90,f"Seed {seed}",transform=axes[row,0].transAxes,fontsize=9,color="#38465a")
        axes[row,0].set_ylabel("Yaw difference (°)")
        axes[row,1].set_ylabel("Position difference (mm)")
        for ax in axes[row]:
            phases(ax)
            ax.axvline(1.8,color="#333",ls=":",lw=.9)
            ax.set_xlim(0,2)
            if row<2:ax.set_xlabel("")
    axes[0,0].set_title("Angle vs matched unloaded control")
    axes[0,1].set_title("Path vs matched unloaded control")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.suptitle("Frozen Eon and body | virtual-contact scenarios",fontweight="bold")
    fig.tight_layout(rect=(0,.045,1,.95))
    fig.savefig(HERE/"MATCHED_FLAT.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("view",choices=("physical","behavior"))
    {"physical":physical,"behavior":behavior}[parser.parse_args().view]()
