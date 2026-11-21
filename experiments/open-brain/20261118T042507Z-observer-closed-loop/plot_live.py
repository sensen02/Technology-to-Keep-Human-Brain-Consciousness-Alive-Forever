"""Plot the already saved closed-loop and same-body clean-signal traces."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from treatment import SEEDS, ARMS

HERE=Path(__file__).resolve().parent
COLORS=("#202e3c","#52708b","#bd553d","#267a60")
LABELS=("Clean | off","Clean | on","Drift | off","Drift | on")


def main():
    f, ax=plt.subplots(2,3,figsize=(13.5,7.1))
    for row,seed in enumerate(SEEDS):
        root=HERE/"FORMAL"/f"seed{seed}"
        traces=[]
        for i,arm in enumerate(ARMS):
            with np.load(root/arm/"TRACE.npz") as a:
                yaw=np.rad2deg(np.unwrap(a["yaw"]))
                pos=a["position"][:,:2]
            traces.append(yaw)
            ax[row,0].plot(pos[:,0]-pos[0,0],pos[:,1]-pos[0,1],label=LABELS[i],color=COLORS[i],lw=1.35)
            ax[row,0].plot(pos[-1,0]-pos[0,0],pos[-1,1]-pos[0,1],".",color=COLORS[i],ms=5)
        t=np.arange(len(traces[0]))/10000
        for i,yaw in enumerate(traces):
            ax[row,1].plot(t,yaw-traces[0],color=COLORS[i],lw=1.1,label=LABELS[i])
        ax[row,1].axhline(5,color=".55",lw=.7,ls=":")
        ax[row,1].axhline(-5,color=".55",lw=.7,ls=":")
        with np.load(root/"NEURAL.npz") as n:
            error=n["left_used_hz"]-n["rates"][:,:,0]
            for i in (1,2,3):
                ax[row,2].plot(t,error[:,i],color=COLORS[i],lw=1.05,label=LABELS[i])
        for panel in (ax[row,1],ax[row,2]):
            panel.axvspan(1.05,1.4,color="#91a3b4",alpha=.13)
            panel.set_xlabel("Time (s)")
        ax[row,0].set_ylabel(f"Phase {seed}\nSide (mm)")
        ax[row,0].set_xlabel("Forward X (mm)")
        ax[row,1].set_ylabel("Yaw error (degrees)")
        ax[row,2].set_ylabel("Rate error (Hz)")
        for panel in ax[row]:
            panel.grid(alpha=.16)
    for i,title in enumerate(("Body trajectory","Yaw vs clean / off","Left input vs own clean rate")):
        ax[0,i].set_title(title)
    handles,labels=ax[0,0].get_legend_handles_labels()
    f.legend(handles,labels,ncol=4,loc="lower center",frameon=False)
    f.suptitle("Frozen observer | matched closed-loop tests",fontweight="bold")
    f.tight_layout(rect=(0,.055,1,.95))
    f.savefig(HERE/"CLOSED_LOOP.png",dpi=155)


if __name__=="__main__":
    main()
