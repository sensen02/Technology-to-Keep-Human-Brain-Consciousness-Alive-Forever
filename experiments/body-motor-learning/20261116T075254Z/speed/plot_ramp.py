"""Compare commanded pace with force-qualified measured forward displacement."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speed_teacher import WARM, stand
from walking_bench import read


def graph(run,out):
    case=read(run/"RESULT.json")
    trace=run/"TRACE.npz"
    if stand.sha(trace)!=case["trace_sha256"]:
        raise ValueError("Unverified ramp trace")
    with np.load(trace) as z:
        com=z["com"][WARM:WARM+case["walk_ms"]+1,0]
        request=z["command"][WARM:WARM+case["walk_ms"],0]
    phase=np.arange(len(request))*.001
    half=240
    centers=np.arange(half,len(request)-half,5)
    moving=(com[centers+half]-com[centers-half])/(.002*half)
    fig,ax=plt.subplots(figsize=(10.3,4.9),layout="constrained")
    ax.plot(phase,request,color="#dd7439",lw=2,label="Requested forward speed")
    ax.plot(centers*.001,moving,color="#426f83",lw=1.1,alpha=.75,label="Measured COM speed over 0.48 s")
    for key,interval in case["windows"].items():
        a,b=[int(x)*.001 for x in key.split("-")]
        velocity=interval["signed_forward_native_s"]
        ax.hlines(velocity,a,b,colors="#194c3b",linewidth=3,
                  label="Measured window mean" if a==0. else None)
        ax.annotate(f"{velocity:.4f}",((a+b)/2.,velocity),xytext=(0,-17),
                    textcoords="offset points",ha="center",color="#194c3b",fontsize=9)
    ax.set(xlabel="Time since walking request (s)", ylabel="Forward speed (native length/s)",
           title="Paced fly body physically accelerates and decelerates")
    ax.text(.995,.017,f"Whole walk: {case['walk']['signed_forward_native_s']:.5f} native length/s  •  native slip {case['walk']['max_ground_normalized_slip']:.3f}",
            color="#3f5855",fontsize=9,ha="right",va="bottom",transform=ax.transAxes)
    ax.grid(alpha=.25)
    ax.legend(loc="upper left",frameon=False,ncols=3)
    fig.savefig(out,dpi=150)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    graph(a.run.resolve(),a.output.resolve())
