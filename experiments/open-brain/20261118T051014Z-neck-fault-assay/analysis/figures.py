"""Plot the simulated geometry, verified event loss, and matched behavior."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fault_schedule import ARMS
from audit import angle, read

COLORS = ["#8f344d","#287d83","#295eaf","#b16e25","#725181"]
LABELS = ["AN tear","AN dropout","AN block","DN dropout","DN block"]
STYLES = ["-","-","--","-","--"]


def plot_neck(assay):
    embedding = ROOT.parent/"20261118T044947Z-neck-insertion-assay"/"MECHANICS.npz"
    fig, axes = plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    geometry = assay["geometry"]
    with np.load(embedding,allow_pickle=False) as data:
        for index,(name,color) in enumerate(zip(("Left front-touch AN","Right front-touch AN","Right DNg100"),
                                              ("#8f344d","#287d83","#295eaf"),strict=True)):
            path = data[f"axon{index}"]
            axes[0].plot(path[:,1],path[:,0],color=color,label=name)
    x,d = geometry["axial_um"], geometry["slit_depth_um"]
    axes[0].plot([x,x],[0,d],color="#343333",ls="--",lw=2,label="Prescribed slit")
    for row in geometry["trace_intersections"]:
        y=row["point_um"][0]
        axes[0].scatter(x,y,color="#343333" if row["cut_by_projected_slit"] else "white",
                       edgecolor="#343333",s=36,zorder=3)
    axes[0].set(title="Reference axon paths and simulated slit",xlabel="Head–thorax axis (µm)",
                ylabel="Transverse depth (µm)",ylim=(-3,82))
    axes[0].legend(frameon=False,loc="upper left",fontsize=8)
    axes[0].grid(alpha=.16)
    x = np.arange(len(ARMS)-1)
    for index,row in enumerate(assay["seed_runs"]):
        data = [row["arms"][a]["target_events"]["intervention"] for a in ARMS[1:]]
        rate = [100*p["lost"]/p["attempted"] for p in data]
        bars=axes[1].bar(x+(index-1)*.23,rate,.21,alpha=.56+.19*index,color=COLORS,
                         edgecolor="#354454",linewidth=.4)
        axes[1].bar_label(bars,labels=[str(p["lost"]) for p in data],padding=1,fontsize=6)
    axes[1].set(xticks=x,xticklabels=LABELS,ylim=(0,119),ylabel="Target pulses lost (%)",
                title="Verified pulse loss in the 0.65–1.20 s window")
    axes[1].text(.01,.965,"Left to right: 10901, 11311, 11701  ·  labels show counts",
                 transform=axes[1].transAxes,fontsize=8,color="#343333")
    axes[1].grid(axis="y",alpha=.18)
    fig.savefig(ROOT/"FAULT_GEOMETRY_AND_EVENTS.png",dpi=160)
    plt.close(fig)


def plot_behavior(assay):
    fig, axes = plt.subplots(3,2,figsize=(11,8),sharex=True)
    for row,data in enumerate(assay["seed_runs"]):
        folder=ROOT/"FORMAL"/f"seed{data['seed']}"
        with np.load(folder/"intact/TRACE.npz",allow_pickle=False) as trace:
            control_yaw, control_pos = angle(trace),trace["position"][:,:2]
        t=np.arange(len(control_yaw))*.0001
        for mode,label,col,style in zip(ARMS[1:],LABELS,COLORS,STYLES,strict=True):
            with np.load(folder/mode/"TRACE.npz",allow_pickle=False) as trace:
                axes[row,0].plot(t,angle(trace)-control_yaw,style,color=col,label=label,lw=1.3)
                axes[row,1].plot(t,np.linalg.norm(trace["position"][:,:2]-control_pos,axis=1),
                                 style,color=col,lw=1.3)
        axes[row,0].set_ylabel(f"{data['seed']}\nYaw (°)")
        axes[row,1].set_ylabel("Distance (mm)")
        for ax in axes[row]:
            ax.axvspan(.65,1.2,color="#516f93",alpha=.105,zorder=0)
            ax.axvline(1.8,color="#35363a",ls=":",lw=1)
            ax.grid(alpha=.17)
            ax.set_xlim(0,2)
        axes[row,0].axhline(5,color="#8292a5",ls=":",lw=.65)
        axes[row,0].axhline(-5,color="#8292a5",ls=":",lw=.65)
        axes[row,1].axhline(.5,color="#8292a5",ls=":",lw=.65)
    axes[0,0].set_title("Signed yaw difference from matched control")
    axes[0,1].set_title("Position difference from matched control")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc="lower center",frameon=False)
    fig.suptitle("BANC lesion sensitivities in the frozen flat-ground loop",fontweight="bold")
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(ROOT/"MATCHED_FAULT_BEHAVIOR.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    assay=read(ROOT/"FINAL_FAULT_ASSAY.json")
    plot_neck(assay)
    plot_behavior(assay)
