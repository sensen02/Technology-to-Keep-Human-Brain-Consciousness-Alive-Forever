"""Plot received activity against actual force for the same held crossed scene."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from separation import SCALE
from touch_runtime import turn


def plot(root,out):
    row=root/"group_12"
    figure, axes=plt.subplots(2,2,figsize=(9.4,5.),sharex=True,sharey=True,layout="constrained")
    for index,(authority,title) in enumerate((
              ("L0T0","Unloaded · no lateral contact"),("L0T1","Unloaded · lateral contact"),
              ("L1T0","Loaded · no lateral contact"),("L1T1","Loaded · lateral contact"))):
        panel=axes.flat[index]
        with np.load(row/authority/"SAMPLES.npz",allow_pickle=False) as z:
            actual=z["normal_load"][:,0]
            rx=z["received"]
            touch=z["truth_tactile"][:,0]
        ticks=np.arange(len(actual))
        begin,end=next(iter(np.flatnonzero(touch)),0), max([0,*np.flatnonzero(touch)])
        if end:
            panel.axvspan(begin,end,color="#fbbc7a",alpha=.35)
        panel.plot(ticks,actual,color="#a5aab0",label="Measured ground load",linewidth=2)
        panel.plot(ticks,rx[:,2]*SCALE,color="#147184",label="Received load")
        panel.plot(ticks,rx[:,0]*.13,color="#bc4e29",label="Received side touch (scaled)")
        panel.set_xlim(80,505)
        panel.set_ylim(-.004,.15)
        panel.spines[["right","top"]].set_visible(False)
        panel.set_title(title,loc="left",fontsize=10)
    axes[0,0].legend(loc="upper right",fontsize=7.5)
    for panel in axes[-1]:
        panel.set_xlabel("Simulation time (ms)")
    for panel in axes[:,0]:
        panel.set_ylabel("Load (body-weight fraction)")
    figure.suptitle("Held left-front leg · separate sensory arrivals",x=.03,ha="left",fontsize=15)
    figure.savefig(out,dpi=146)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    plot(a.root,a.out)
