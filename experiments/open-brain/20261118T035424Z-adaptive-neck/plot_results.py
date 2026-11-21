"""Research plots from the completed and saved time series."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    f, axes=plt.subplots(3,1,figsize=(10.3,7.5))
    with np.load(HERE/"DYNAMIC/TRACE.npz") as data,np.load(HERE/"OBSERVER/TRACE.npz") as obs:
        t=np.arange(len(obs["truth"]))/10000
        for lane,(label,color) in enumerate((("Fixed", "#263346"),("Variable delay", "#2767b4"),("Blocked", "#b04f3d"))):
            x=t[data["arrivals"][:,lane,0]]
            axes[0].vlines(x,lane-.25,lane+.25,color=color,lw=1)
        axes[0].set(yticks=(0,1,2),yticklabels=("Fixed","Variable left delay","Left block"),ylim=(-.5,2.5),
                    xlim=(.45,.99),title="Recorded left AN arrivals through the same BANC axon")
        axes[0].axvline(.90,color=".35",lw=.8,ls="--",label="Gate reopens")
        axes[0].legend(loc="upper right",fontsize=8,frameon=False)
        for key,label,color in (("truth","Clean archival rate","#263346"),("drift_raw","Readout with drift","#ba5439"),("drift_estimate","Advisory corrected rate","#2767b4")):
            axes[1].plot(t,obs[key],label=label,color=color,lw=1,alpha=.85)
        axes[1].axvspan(1.05,1.40,color=".6",alpha=.1)
        axes[1].set(xlim=(.43,1.95),ylabel="Rate (Hz)",title="Held case | synthetic gain drift after the neck")
        axes[1].legend(loc="upper right",fontsize=8,frameon=False)
        axes[2].plot(t,obs["actual_synthetic_gain"],color=".30",label="True drift (audit only)",ls="--",lw=1)
        axes[2].plot(t,obs["drift_gain"],color="#2767b4",label="Online estimated gain",lw=1.45)
        axes[2].set(xlim=(.43,1.95),ylabel="Gain",xlabel="Time (s)",title="Calibration lags drift and its removal")
        axes[2].legend(loc="lower left",fontsize=8,frameon=False)
    for ax in axes:
        ax.grid(axis="x",alpha=.18)
    f.tight_layout()
    f.savefig(HERE/"ADAPTIVE_NECK.png",dpi=150)
    plt.close(f)

    arrays=[]
    for mode,label in (("FLIGHT_INTACT","Intact"),("FLIGHT_REOPEN","Short reopen")):
        p=HERE/mode/"TRACE.npy"
        if p.exists():
            arrays.append((np.load(p,allow_pickle=False),label))
    if arrays:
        f,axes=plt.subplots(1,2,figsize=(9.2,3.3))
        for a,label in arrays:
            axes[0].plot(a[:,0]*1000,(a[:,1]-a[0,1])*10,label=label)
            axes[1].plot(a[:,0]*1000,a[:,3]*10,label=label)
        axes[0].set(ylabel="Forward displacement (mm)")
        axes[1].set(ylabel="COM height (mm)")
        for ax in axes:
            ax.set_xlabel("Time (ms)")
            ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        f.suptitle("Fixed FlyBody policy replay | started airborne")
        f.tight_layout()
        f.savefig(HERE/"FLIGHT_BASELINE.png",dpi=150)
        plt.close(f)


if __name__=="__main__":
    main()
