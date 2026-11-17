"""Plot synchronized ground truth and causal forecasts from a held fixed-contact episode."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from touch_runtime import turn
from train import HORIZONS, Predictor, pack, rows
from train_zero_mass import ZeroMassForecast


def plot(data, model, output):
    held = [r for r in rows(data) if r["case"]["split"]=="held"]
    index = next(i for i,row in enumerate(held) if row["mode"]=="fixed" and any(row["touched_bins"]))
    episode = held[index]
    side = episode["case"]["side"]
    front = (0,3)[side]
    forecasts = {}
    evaluation = turn.read(model/"EVALUATION.json")
    revised = evaluation.get("same_held_groups_as_v1",False)
    torch.set_num_threads(1)
    for mode in ("sense","ordered","mixed"):
        checkpoint = torch.load(model/f"{mode}.pt", weights_only=True)
        net = (ZeroMassForecast() if revised else Predictor()).eval()
        net.load_state_dict(checkpoint["model"])
        x,*_ = pack(held, mode, checkpoint["center"].numpy(), checkpoint["scale"].numpy())
        with torch.inference_mode():
            event = net(x[index:index+1])[0]
            if revised:
                event = event+torch.tensor(evaluation[mode]["event_offset"])
        forecasts[mode] = event.sigmoid().numpy()[0,:,HORIZONS.index(100),side]
    samples = episode["state"]
    sec = episode["commands"][:,0]/1000
    fig, axes = plt.subplots(5,1,sharex=True,figsize=(9.2,8.2),layout="constrained")
    axes[0].plot(sec,samples[:,front],label="Front tibia angle",color="#21446c")
    axes[0].set_ylabel("rad")
    axes[1].plot(sec,samples[:,6+front],label="Front tibia angular velocity",color="#21446c")
    axes[1].set_ylabel("rad/s")
    axes[2].plot(sec,samples[:,12+front],label="Front foot load",color="#21446c")
    axes[2].set_ylabel("Body weight")
    axes[3].plot(sec,samples[:,18],label="Body forward velocity",color="#21446c")
    axes[3].set_ylabel("Native length/s")
    for mode, values in forecasts.items():
        axes[4].plot(sec[:len(values)],values,label=mode)
    axes[4].set_ylabel("Probability")
    axes[4].set_title("Causal forecast of contact in the next 100 ms",loc="left",fontsize=10)
    axes[4].legend(loc="upper right",ncols=3,fontsize=8)
    def active_spans(numbers):
        starts = np.flatnonzero(np.diff(np.r_[False,numbers].astype(int))==1)
        ends = np.flatnonzero(np.diff(np.r_[numbers,False].astype(int))==-1)
        return zip(starts,ends,strict=True)
    for ax in axes:
        for start, end in active_spans(episode["touch"][:,side]):
            ax.axvspan(sec[start],sec[end]+.005,color="#f3b063",alpha=.37)
        ax.spines[["right","top"]].set_visible(False)
    for ax in axes[:4]:
        ax.legend(loc="upper right",fontsize=8)
    axes[-1].set_xlabel("Simulated seconds • orange marks actual probe contact")
    fig.suptitle(f"Held group {episode['case']['group']} | fixed {'left' if side==0 else 'right'} forefoot contact",
                 x=.02, ha="left", fontsize=14)
    fig.savefig(output,dpi=145)
    print(episode["path"])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("model",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    plot(args.data,args.model,args.output)
