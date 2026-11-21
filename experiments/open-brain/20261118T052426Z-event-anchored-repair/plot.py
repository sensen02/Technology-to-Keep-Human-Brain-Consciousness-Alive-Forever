"""Compare archived behavior and the new matched trajectories and amplitudes."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"20261118T042507Z-observer-closed-loop"
SEEDS=(11701,11903)
RATES=10500,14000


def paths(parent,seed,arm):
    return parent/"FORMAL"/f"seed{seed}"/arm/"TRACE.npz"


def load_pose(path):
    with np.load(path,allow_pickle=False) as a:
        pos=a["position"][:,:2]
        yaw=np.rad2deg(np.unwrap(a["yaw"]))
    return pos-pos[0],yaw


def main():
    f,axes=plt.subplots(2,3,figsize=(12.4,7.5))
    for row,seed in enumerate(SEEDS):
        variants=((HERE,"clean_off","Normal reference","#253346",1.6,"-"),
                  (OLD,"drift_on","Old adaptive drift","#b96448",1.1,"-"),
                  (HERE,"drift_off","Drift / old rate path","#977b72",1.0,":"),
                  (HERE,"drift_on","Drift / new event path","#258766",1.1,"--"))
        all_yaw=[]
        for parent,arm,label,color,lw,line in variants:
            pose,yaw=load_pose(paths(parent,seed,arm))
            all_yaw.append(yaw)
            axes[row,0].plot(*pose.T,label=label,color=color,lw=lw,ls=line)
        t=np.arange(len(all_yaw[0]))/10000
        for i,(_,_,label,color,lw,line) in enumerate(variants):
            axes[row,1].plot(t,all_yaw[i]-all_yaw[0],label=label,color=color,lw=lw,ls=line)
        with np.load(HERE/"FORMAL"/f"seed{seed}"/"NEURAL.npz",allow_pickle=False) as n:
            prior=n["left_raw_hz"][:,3]-n["rates"][:,3,0]
            new=n["advisory_left_hz"][:,3]-n["rates"][:,3,0]
            axes[row,2].plot(t,prior,color="#b96448",label="Measured amplitude",lw=1.05)
            axes[row,2].plot(t,new,color="#258766",label="Advisory corrected",lw=1.15)
        axes[row,2].axvspan(.91,1.4,color="#a2afbf",alpha=.15)
        for a in axes[row]:
            a.grid(alpha=.18)
        axes[row,0].set(xlabel="Forward X (mm)",ylabel=f"Phase {seed}\nSide (mm)")
        axes[row,1].set(xlabel="Time (s)",ylabel="Yaw vs normal (deg)")
        axes[row,2].set(xlabel="Time (s)",ylabel="Error (Hz)")
    for a,title in zip(axes[0],("Actual body path","Heading error","Separate amplitude readout"),strict=True):
        a.set_title(title)
    axes[0,2].legend(fontsize=8,frameon=False,loc="lower left")
    handles,labels=axes[0,0].get_legend_handles_labels()
    f.legend(handles,labels,ncol=4,loc="lower center",frameon=False)
    f.suptitle("Preserve the causal event stream | two matched closed-loop phases",fontweight="bold")
    f.tight_layout(rect=(0,.065,1,.94))
    f.savefig(HERE/"EVENT_ANCHORED_REPAIR.png",dpi=155)


if __name__=="__main__":
    main()
