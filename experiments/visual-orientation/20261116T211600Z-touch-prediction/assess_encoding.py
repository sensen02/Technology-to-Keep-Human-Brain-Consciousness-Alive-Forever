"""Measure whether lateral contact is still separable if load shares the same AN scalars."""
import argparse
from pathlib import Path

import numpy as np

from touch_runtime import turn
from train import average_precision, rows


def pair_arrays(data, key):
    signal, targets, load = [], [], []
    for row in data:
        signal.append(row[key][1:])
        targets.append(row["touch"][1:] | row["touch"][:-1])
        load.append(row["state"][1:,[12,15]])
    return tuple(np.concatenate(series) for series in (signal,targets,load))


def report(path, out):
    records = rows(path)
    calibration = [row for row in records if row["case"]["split"]=="calibration"]
    held = [row for row in records if row["case"]["split"]=="held"]
    output = {}
    for key in ("activity","mixed"):
        ca, y, _ = pair_arrays(calibration,key)
        threshold = [max(.01,float(np.quantile(ca[~y[:,side],side],.985)))
                     for side in range(2)]
        view, actual, foot = pair_arrays(held,key)
        fire = view>threshold
        summary = {}
        for side in range(2):
            signal, truth, selected, force = (value[:,side] for value in (view,actual,fire,foot))
            summary["left" if side==0 else "right"] = dict(
                contact_average_precision=average_precision(truth,signal),
                false_positive_rate=float(selected[~truth].mean()),
                recall=float(selected[truth].mean()) if truth.any() else None,
                load_corr_without_contact=float(np.corrcoef(signal[~truth],force[~truth])[0,1])
                    if signal[~truth].std()>1.e-10 and force[~truth].std()>1.e-10 else None,
                samples_positive=int(truth.sum()), samples_negative=int((~truth).sum()))
        output[key] = dict(calibration_threshold=threshold,held=summary)
    turn.stand.put(out,output)
    print(output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    report(args.data,args.out)
