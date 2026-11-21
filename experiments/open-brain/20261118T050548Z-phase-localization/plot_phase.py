"""Display the accumulated bias and the isolated threshold counterfactual."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from localize import PRIOR, read, threshold

HERE=Path(__file__).resolve().parent


def main():
    cases=read(HERE/"PHASE_AUDIT.json")["rows"]
    f,axes=plt.subplots(2,2,figsize=(10.5,6.7))
    for row,case in enumerate(cases):
        seed=case["seed"]
        with np.load(PRIOR/"FORMAL"/f"seed{seed}"/"NEURAL.npz",allow_pickle=False) as a:
            x=a["left_used_hz"][:,:2]
        p,_=threshold(x)
        first=case["first_an_tick"]
        t0=case["first_rate_tick"]-30
        s=slice(t0,first+1)
        axes[row,0].plot(np.arange(t0,first+1)*.1,(p[:,1]-p[:,0])[s],color="#2964af",lw=1.7)
        axes[row,0].axvline(case["first_rate_tick"]*.1,color="#777777",ls=":",lw=1)
        axes[row,0].axvline(first*.1,color="#a44c39",ls="--",lw=1)
        axes[row,0].set(xlabel="Time (ms)",ylabel=f"{seed} | advance (cycles)")
        axes[row,0].grid(alpha=.16)
        m=np.array(case["thresholds_phase_rows_rate_columns"])
        x=np.arange(4)
        for i,y in enumerate(m.ravel()):
            axes[row,1].vlines(i,1,y,color=("#3a6898" if i<2 else "#ae4f3c"),lw=2)
        axes[row,1].scatter(x,m.ravel(),c=["#3a6898"]*2+["#ae4f3c"]*2,s=28,zorder=3)
        axes[row,1].axhline(1,color="#263040",ls="--",lw=.8)
        axes[row,1].set(xticks=x,xticklabels=("Raw phase\nRaw rate","Raw phase\nAdapted rate",
                  "Adapted phase\nRaw rate","Adapted phase\nAdapted rate"),
                  ylabel="Threshold input (cycles)",ylim=(.975,1.014))
        axes[row,1].grid(axis="y",alpha=.16)
    axes[0,0].set_title("Phase difference before the first shifted spike")
    axes[0,1].set_title("Swap prior phase and current rate at that tick")
    f.suptitle("Healthy-input replay | isolate cumulative phase from the current sample",fontweight="bold")
    f.tight_layout(rect=(0,0,1,.94))
    f.savefig(HERE/"PHASE_CAUSE.png",dpi=155)


if __name__=="__main__":
    main()
